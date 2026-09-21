import asyncio
import base64
import hashlib
import json
import re
import shutil
from datetime import timezone
from pathlib import Path

from fastapi import HTTPException

from .config import ROOT, Settings
from .repository import Repository, now, uid
from .schemas import Performance
from .agents.wiki import WikiResearcher

DEFAULT_AUDITION = ('Вечер медленно опускался на город. Анна остановилась у окна и тихо сказала: '
    '«Я знала, что ты вернёшься». Но за дверью раздался незнакомый голос. '
    'Кто мог прийти в такой час? Она набралась смелости и сделала шаг навстречу.')


class Studio:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.root = settings.data_dir.resolve()
        self.files = self.root / 'artifacts'
        self.files.mkdir(parents=True, exist_ok=True)
        self.repo = Repository(self.root / 'studio.sqlite3')
        self.worker_task = None
        self.wake = asyncio.Event()
        self.active_job = None
        self.mutating_books = set()

    @property
    def provider(self):
        from .providers import DemoAdapter, ElevenLabsAdapter
        if self.settings.live:
            if not self.settings.elevenlabs_api_key:
                raise ValueError('В режиме live задайте ELEVENLABS_API_KEY в .env и перезапустите сервер.')
            return ElevenLabsAdapter(self.settings.elevenlabs_api_key)
        return DemoAdapter()

    @property
    def analyst(self):
        from .agents.pipeline import AnalysisPipeline
        return AnalysisPipeline(base_url=self.settings.llm_base_url,
            api_key=self.settings.llm_api_key, model=self.settings.llm_model,
            mode='llm' if self.settings.llm_configured else 'manual')

    @property
    def audio(self):
        from .services.audio import AudioEngine
        return AudioEngine(self.settings.ffmpeg_path)

    def require(self, kind, entity_id):
        item = self.repo.get(kind, entity_id)
        if item is None:
            raise HTTPException(404, 'Объект не найден.')
        return item

    def ensure_idle(self, book_id):
        if book_id in self.mutating_books or any(j['status'] in ('QUEUED', 'RUNNING') for j in self.repo.list('job', book_id)):
            raise HTTPException(409, 'Дождитесь завершения задания перед изменением книги.')

    def config(self):
        limits = json.loads((ROOT / 'config/provider_limits.json').read_text('utf-8')) if (ROOT / 'config/provider_limits.json').exists() else {}
        return {'mode': 'live' if self.settings.live else 'demo', 'version': '0.1.0',
                'llm_configured': self.settings.llm_configured,
                'tts_configured': bool(self.settings.elevenlabs_api_key),
                'asr_enabled': self.settings.asr_enabled, 'limits': {'dialogue_chars': limits.get('dialogue_chars',2000)},
                'provider_limits': limits, 'storage': 'SQLite / local files',
                'speaker_qc': 'manual', 'max_upload_mb': self.settings.max_upload_mb}

    def write_file(self, data: bytes, extension: str):
        if extension not in ('mp3','wav','flac','m4b','txt','epub','fb2','json'):
            raise ValueError('Неизвестный формат артефакта.')
        filename = f'{uid()}.{extension}'
        path = self.files / filename
        temporary = path.with_suffix('.tmp')
        temporary.write_bytes(data)
        temporary.replace(path)
        return filename

    def url(self, filename):
        return '/api/files/' + filename if filename else None

    def create_character(self, book_id, name, role='character', db=None):
        return self.repo.put('character', {'id': uid(), 'book_id': book_id,
            'canonical_name': name.strip(), 'aliases': [], 'gender': {'value':None,'confidence':0},
            'age_band': {'value':None,'confidence':0}, 'role':role, 'baseline_traits':[],
            'speech_profile':{}, 'evidence':[], 'voice_brief':{'language':'ru','baseline_delivery':'Natural, clear literary Russian narration with measured pacing.'},
            'voice':None, 'candidates':[], 'version':1}, db)

    def import_book(self, content, filename, title=None, author=None):
        from .parsers.books import parse_book
        parsed = parse_book(content, filename)
        source_file = self.write_file(content, Path(filename).suffix.lower().lstrip('.'))
        book_id = uid()
        book = {'id':book_id,'title':title or parsed['title'],'author':author or parsed.get('author',''),
                'language':parsed.get('language','ru'),'source_name':Path(filename).name,
                'source_file':source_file,'created_at':now(),'status':'PARSED', 'version':1,
                'analysis_mode':None, 'events':[],'relationships':[], 'character_states':[]}
        position = 0
        with self.repo.connect() as db:
            self.repo.put('book', book, db)
            narrator = self.create_character(book_id, 'Рассказчик', 'narrator', db)
            for ci, raw_chapter in enumerate(parsed['chapters']):
                chapter = {'id':uid(),'book_id':book_id,'parent_id':book_id,'position':ci,
                    'title':raw_chapter['title']}
                self.repo.put('chapter',chapter,db)
                for si, raw_scene in enumerate(raw_chapter['scenes']):
                    scene = {'id':uid(),'book_id':book_id,'parent_id':chapter['id'],'chapter_id':chapter['id'],
                        'position':si,'title':raw_scene.get('title') or f'Сцена {si+1}',
                        'summary':'','preview_approved':False,'version':1}
                    self.repo.put('scene',scene,db)
                    for pi, raw_paragraph in enumerate(raw_scene['paragraphs']):
                        paragraph = {'id':uid(),'book_id':book_id,'parent_id':scene['id'],
                            'position':pi,'text':raw_paragraph['text']}
                        self.repo.put('paragraph',paragraph,db)
                        for raw in raw_paragraph['utterances']:
                            is_narrator = raw.get('type') == 'NARRATOR'
                            utterance = {**raw,'id':uid(),'book_id':book_id,'parent_id':scene['id'],
                                'scene_id':scene['id'],'paragraph_id':paragraph['id'],'position':position,
                                'speaker_id':narrator['id'] if is_narrator else None,
                                'speaker_confidence':1.0 if is_narrator else raw.get('speaker_confidence',0),
                                'review_required':not is_narrator,'performance':Performance().model_dump(),
                                'status':'PENDING','generation_id':None,'mode':'auto','version':1,'manual_speaker':False}
                            self.repo.put('utterance',utterance,db)
                            position += 1
        if not position:
            raise ValueError('В книге нет текста для озвучки.')
        return self.detail(book_id)

    def demo(self):
        existing = next((b for b in self.repo.list('book') if b.get('is_demo')),None)
        if existing:
            return self.detail(existing['id'])
        text = (ROOT / 'examples/demo.txt').read_bytes()
        result = self.import_book(text,'demo.txt','Дом у маяка','Демонстрационная история')
        book_id = result['id']
        book = self.require('book',book_id)
        book['is_demo'] = True
        book['status'] = 'ANALYZED'
        book['analysis_mode'] = 'curated_demo'
        self.repo.put('book',book)
        anna = self.create_character(book_id,'Анна')
        mikhail = self.create_character(book_id,'Михаил')
        chars = self.repo.list('character',book_id)
        for char in chars:
            char['voice_brief'] = {'language':'ru','baseline_delivery':'Warm, natural literary Russian, clear articulation, subtle emotional expression.',
                                    'timbre':'soft and warm' if char['id']==anna['id'] else 'low and calm'}
            self.repo.put('character',char)
        for utterance in self.repo.list('utterance',book_id):
            if utterance['type'] == 'UNKNOWN':
                name = 'Анна' if any(cue in utterance['source_text'] for cue in ['Анна','Я думала','Тогда','Ты']) else 'Михаил'
                utterance.update(speaker_id=anna['id'] if name=='Анна' else mikhail['id'],
                    speaker_confidence=1.0,review_required=False,type='CHARACTER',manual_speaker=True,
                    attribution_reason='Разметка демонстрационного примера')
                self.repo.put('utterance',utterance)
        return self.detail(book_id)

    def generation_view(self, generation, active_id=None):
        return {**generation,'audio_url':self.url(generation.get('audio_file')),
                'active':generation['id']==active_id}

    def detail(self, book_id):
        book = self.require('book',book_id)
        chapters = self.repo.list('chapter',book_id)
        scenes = self.repo.list('scene',book_id)
        utterances = self.repo.list('utterance',book_id)
        generations = self.repo.list('generation',book_id)
        by_id = {g['id']:g for g in generations}
        histories = {}
        for g in generations:
            for utterance_id in g['utterance_ids']:
                histories.setdefault(utterance_id,[]).append(g)
        for utterance in utterances:
            gen = by_id.get(utterance.get('generation_id'))
            utterance['audio_url'] = self.url(gen.get('audio_file')) if gen else None
            utterance['qc'] = gen.get('qc') if gen else None
            utterance['generations'] = [self.generation_view(g, utterance.get('generation_id')) for g in histories.get(utterance['id'],[])]
        for scene in scenes:
            scene['utterances'] = [u for u in utterances if u['scene_id']==scene['id']]
        for chapter in chapters:
            chapter['scenes'] = [s for s in scenes if s['chapter_id']==chapter['id']]
        active = {u['generation_id'] for u in utterances if u.get('generation_id')}
        return {**book,'chapters':chapters,'characters':self.repo.list('character',book_id),
            'jobs':list(reversed(self.repo.list('job',book_id)))[0:50],
            'exports':self.repo.list('export',book_id),
            'metrics':{'total_utterances':len(utterances),
                'ready_utterances':sum(u['status']=='APPROVED' for u in utterances),
                'review_utterances':sum(u['review_required'] or u['status'] in ('QC_REVIEW','QC_FAIL') for u in utterances),
                'total_characters':sum(len(u['spoken_text']) for u in utterances),
                'generated_characters':sum(len(u['spoken_text']) for u in utterances if u.get('generation_id')),
                'tts_requests':sum(g.get('request_count',1) for g in generations),
                'audio_seconds':sum(g.get('duration_seconds',0) for g in generations if g['id'] in active),
                'retry_count':sum(g.get('attempt',1)>1 for g in generations),
                'estimated_cost':None}}

    def invalidate(self, book_id, utterance_ids=None, db=None):
        if db is None:
            with self.repo.connect() as conn:
                return self.invalidate(book_id, utterance_ids,conn)
        utterances = self.repo.list('utterance',book_id,db=db)
        ids = set(utterance_ids or [u['id'] for u in utterances])
        # A dialogue artifact is indivisible: changing one line invalidates the whole active block.
        group_ids = {u['generation_id'] for u in utterances if u['id'] in ids and u.get('generation_id')}
        touched_scenes = set()
        for utterance in utterances:
            if utterance['id'] in ids or utterance.get('generation_id') in group_ids:
                utterance.update(generation_id=None,status='PENDING')
                touched_scenes.add(utterance['scene_id'])
                self.repo.put('utterance',utterance,db)
        for scene_id in touched_scenes:
            scene = self.repo.get('scene',scene_id,db)
            scene['preview_approved'] = False
            scene['version'] += 1
            self.repo.put('scene',scene,db)
        for export in self.repo.list('export',book_id,db=db):
            self.repo.delete('export',export['id'],db)
        book = self.repo.get('book',book_id,db)
        book['version'] += 1
        book['status'] = 'DIRECTING'
        self.repo.put('book',book,db)

    def validate_ready(self, book_id, utterances, require_preview=False):
        chars = {c['id']:c for c in self.repo.list('character',book_id)}
        if not utterances:
            raise ValueError('Нет реплик для озвучки.')
        for u in utterances:
            if not u.get('speaker_id') or u.get('review_required') or u.get('speaker_confidence',0)<0.9:
                raise ValueError('Сначала подтвердите говорящего у всех выбранных реплик.')
            char = chars.get(u['speaker_id'])
            if not char or not char.get('voice') or char['voice']['status'] != 'LOCKED':
                raise ValueError(f"Зафиксируйте голос: {char['canonical_name'] if char else 'неизвестный персонаж'}.")
            expected_provider = 'elevenlabs' if self.settings.live else 'demo'
            if char['voice']['provider'] != expected_provider:
                raise ValueError('Голоса другого режима не используются. Повторите кастинг в текущем режиме.')
        if require_preview and not any(s['preview_approved'] for s in self.repo.list('scene',book_id)):
            raise ValueError('Сначала сгенерируйте, прослушайте и утвердите тестовую сцену.')

    def enqueue(self, kind, book_id, target_id=None, payload=None):
        self.require('book',book_id)
        if book_id in self.mutating_books:
            raise HTTPException(409,'Дождитесь завершения сохранения голоса.')
        for j in self.repo.list('job',book_id):
            if j['status'] in ('QUEUED','RUNNING'):
                if j['kind']==kind and j.get('target_id')==target_id:
                    return j
                raise HTTPException(409,'Для этой книги уже выполняется задание. Дождитесь завершения.')
        job = {'id':uid(),'book_id':book_id,'kind':kind,'target_id':target_id,
            'payload':payload or {},'status':'QUEUED','progress':0,'message':'Ожидает запуска',
            'error':None,'created_at':now(),'priority':0 if kind=='regenerate' else 10,
            'started_at':None,'finished_at':None,'updated_at':None,'progress_detail':{}}
        self.repo.put('job',job)
        self.wake.set()
        return job

    async def start(self):
        for j in self.repo.list('job'):
            if j['status']=='RUNNING':
                j.update(status='QUEUED',message='Возобновление после перезапуска')
                self.repo.put('job',j)
        self.worker_task = asyncio.create_task(self.worker())
        self.wake.set()

    async def stop(self):
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass

    def progress(self, job, progress, message, detail=None):
        from datetime import datetime
        updated = datetime.now(timezone.utc)
        started = datetime.fromisoformat(job.get('started_at')) if job.get('started_at') else None
        elapsed = (updated - started).total_seconds() if started else 0
        fraction = max(0.0, min(1.0, float(progress)))
        eta = (elapsed / fraction - elapsed) if fraction > 0.01 else None
        job.update(progress=round(fraction,4),message=message,updated_at=updated.isoformat(),
                   elapsed_seconds=round(elapsed,1),
                   eta_seconds=round(eta,1) if eta is not None else None,
                   progress_detail=detail or job.get('progress_detail') or {})
        self.repo.put('job',job)

    async def worker(self):
        while True:
            queued = sorted((j for j in self.repo.list('job') if j['status']=='QUEUED'),
                            key=lambda j:(j.get('priority',10),j['created_at']))
            if not queued:
                self.wake.clear()
                await self.wake.wait()
                continue
            job = queued[0]
            self.active_job = job['id']
            job.update(status='RUNNING',started_at=now(),error=None,progress_detail={})
            self.repo.put('job',job)
            try:
                await self.execute(job)
                job.update(status='COMPLETED',progress=1,message='Готово',finished_at=now())
            except asyncio.CancelledError:
                job.update(status='QUEUED',message='Остановлено; продолжится при запуске')
                self.repo.put('job',job)
                raise
            except Exception as exc:
                # Provider modules expose redacted errors. Remove keys defensively.
                message = str(exc)
                for secret in (self.settings.elevenlabs_api_key,self.settings.llm_api_key):
                    if secret:
                        message = message.replace(secret,'[secret]')
                job.update(status='FAILED',error=message[:1000],message='Требуется исправление и повторный запуск',finished_at=now())
                book = self.require('book',job['book_id'])
                book['status']='REVIEW'
                self.repo.put('book',book)
            self.repo.put('job',job)
            self.active_job = None

    async def execute(self, job):
        handlers = {'analyze':self.analyze,'pipeline':self.book_pipeline,'casting':self.casting,'direct':self.direct,
                    'preview':self.generate,'generate':self.generate,'regenerate':self.generate,'export':self.export}
        await handlers[job['kind']](job)

    def research_character(self, character_id: str) -> dict:
        character = self.require('character',character_id)
        result = self.wiki.search(character['canonical_name'])
        if result.get('summary'):
            # Apply researched age band when the book has not established one
            age = character.get('age_band', {})
            if result.get('age_band') and (age.get('value') in (None, 'unknown', '')):
                character['age_band'] = {'value': result['age_band'], 'confidence': 0.8}
            notes = character.get('research_notes') or []
            notes = [n for n in notes if n.get('source') != result['source']] + [result]
            character['research_notes'] = notes[-10:]
            # Enrich voice brief with canon knowledge (keeps locked voice intact)
            brief = character.get('voice_brief') or {}
            if not brief.get('customized'):
                summary = result['summary']
                canon = summary[:300]
                gender = character.get('gender',{}).get('value','unknown')
                role = character.get('role','unknown')
                name = character['canonical_name']
                age_now = character.get('age_band', {}).get('value', 'unknown')
                age_voice = {'child': 'light youthful voice of a child',
                             'young_adult': 'young adult voice in early twenties',
                             'adult': 'adult voice in thirties',
                             'middle_aged': 'vital mature voice of 30-40 years, ancient being who sounds timeless',
                             'elderly': 'deep elderly voice with gravitas'}.get(age_now, 'adult voice')
                if role == 'narrator':
                    delivery = f"Warm literary Russian narration. {age_voice}. Canon: {canon}"
                else:
                    delivery = f"Russian {gender} voice for {name}. {age_voice}. Canon background: {canon}"
                character['voice_brief'] = {**brief, 'language':'ru', 'baseline_delivery':delivery[:900],
                                            'research_source':result['url'] or result['source'], 'customized':True}
            self.repo.put('character',character)
        return result

    def delete_book(self, book_id: str) -> None:
        """Remove a book and all dependent records from the workspace."""
        self.require('book',book_id)
        # Single entities table: remove the book row and every row referencing it.
        with self.repo.connect() as db:
            db.execute("DELETE FROM entities WHERE book_id=?", (book_id,))
            db.execute("DELETE FROM entities WHERE id=?", (book_id,))
            db.commit()
        return {'deleted':book_id}

    def build_voice_brief(self, name, gender, age, role, traits):
        """Generate an individual voice brief from Character Bible demographics."""
        gender_map = {'male':'male','female':'female'}
        g = gender_map.get(gender, 'neutral')
        age_map = {'young_adult':'young adult in twenties','adult':'adult in thirties','middle_aged':'middle-aged, mature','elderly':'elderly'}
        a = age_map.get(age, 'adult')
        if role == 'narrator':
            base = f"Warm, measured literary Russian narrator. Clear diction, subtle emotional shading, consistent pacing."
        else:
            style = ' and '.join(traits[:3]) if traits else 'distinctive personality'
            base = f"Russian {g} voice, {a}. {style}. Natural conversational Russian with emotional range."
        return {'language':'ru','baseline_delivery':base,'gender':g,'age_band':a,
                'character_name':name,'customized':True}

    def _subjob(self, kind, book_id, target_id, payload=None):
        """Create a job record for pipeline sub-steps without queue checks."""
        j={'id':uid(),'book_id':book_id,'kind':kind,'target_id':target_id,
           'payload':payload or {},'status':'RUNNING','progress':0,'message':'',
           'error':None,'created_at':now(),'progress_detail':{},
           'parent_id':None,'attempts':1,'max_attempts':self.settings.max_attempts}
        return j

    async def book_pipeline(self, job):
        """One-click audiobook pipeline: analyze -> wiki -> direct -> cast -> generate."""
        book_id = job['book_id']
        target_chapter_id = (job.get('payload') or {}).get('chapter_id')
        book = self.require('book',book_id)
        chapters = self.repo.list('chapter',book_id)
        # Stage 1: analyze every chapter with persistent Character Bible
        self.progress(job,0.02,'Этап 1: анализ книги (персонажи, говорящие)')
        for i,ch in enumerate(chapters):
            self.progress(job,0.02+0.35*(i/max(1,len(chapters))),
                f'Анализ глав: {i+1}/{len(chapters)}',
                {'stage':1,'stages':5,'stage_name':'Анализ книги','current_chapter':ch['title']})
            aj = self._subjob('analyze',book_id,ch['id'],{'chapter_id':ch['id']})
            # Up to 3 attempts per chapter; LLM output varies between tries.
            chapter_error = None
            for attempt in range(1, 4):
                aj = self._subjob('analyze',book_id,ch['id'],{'chapter_id':ch['id'],'attempt':attempt})
                aj['status']='RUNNING'
                self.repo.put('job',aj)
                try:
                    await self.analyze(aj)
                    aj['status']='COMPLETED'
                    self.repo.put('job',aj)
                    chapter_error = None
                    break
                except Exception as exc:
                    chapter_error = str(exc)[:400]
                    aj['status']='FAILED'
                    aj['error']=chapter_error
                    self.repo.put('job',aj)
                    if attempt < 3:
                        await asyncio.sleep(3 * attempt)
            if chapter_error is not None:
                raise ValueError(f'Анализ главы {i+1} не выполнен за 3 попытки: {chapter_error}')
            await asyncio.sleep(2)
        # Stage 2: wiki research for all characters
        self.progress(job,0.40,'Этап 2: вики-исследование персонажей')
        chars = [c for c in self.repo.list('character',book_id) if isinstance(c,dict) and c.get('id')]
        for i,c in enumerate(chars):
            self.progress(job,0.40+0.08*(i/max(1,len(chars))),f'Вики: {i+1}/{len(chars)}',
                {'stage':2,'stages':5,'stage_name':'Вики-исследование'})
            try:
                self.research_character(c['id'])
            except Exception:
                continue
        # Stage 3: direct scenes of target chapters (unique per-utterance emotion)
        self.progress(job,0.50,'Этап 3: режиссура сцен (уникальные эмоции)')
        target = chapters
        if target_chapter_id:
            idx = next((i for i,c in enumerate(chapters) if c['id']==target_chapter_id), None)
            if idx is not None:
                target = chapters[:idx+1]
        scenes = [sc for sc in self.repo.list('scene',book_id) if isinstance(sc,dict) and sc.get('chapter_id')]
        scene_targets = [sc for sc in scenes if any(sc['chapter_id']==c['id'] for c in target)]
        for i,sc in enumerate(scene_targets):
            self.progress(job,0.50+0.20*(i/max(1,len(scene_targets))),
                f'Режиссура: {i+1}/{len(scene_targets)}',
                {'stage':3,'stages':5,'stage_name':'Режиссура'})
            dj = self._subjob('direct',book_id,sc['id'])
            dj['status']='RUNNING'
            self.repo.put('job',dj)
            try:
                await self.direct(dj)
                dj['status']='COMPLETED'
                self.repo.put('job',dj)
            except Exception:
                dj['status']='FAILED'
                self.repo.put('job',dj)
        # Stage 4: casting (voice brief + 3 candidates via LLM+TTS then auto-lock best)
        self.progress(job,0.72,'Этап 4: генерация голосов (3 варианта на персонажа)')
        chars = [c for c in self.repo.list('character',book_id) if isinstance(c,dict) and c.get('id')]
        for i,c in enumerate(chars):
            self.progress(job,0.72+0.10*(i/max(1,len(chars))),f'Голоса: {i+1}/{len(chars)}',
                {'stage':4,'stages':5,'stage_name':'Кастинг голосов'})
            if not c.get('voice') or c.get('voice',{}).get('status')=='LOCKED':
                continue
            cj = self._subjob('casting',book_id,c['id'])
            cj['status']='RUNNING'
            self.repo.put('job',cj)
            try:
                await self.casting(cj)
                cj['status']='COMPLETED'
                self.repo.put('job',cj)
            except Exception:
                cj['status']='FAILED'
                self.repo.put('job',cj)
        # Stage 5: generate preview+audio for target scenes
        self.progress(job,0.85,'Этап 5: озвучка глав')
        for i,sc in enumerate(scene_targets):
            self.progress(job,0.85+0.14*(i/max(1,len(scene_targets))),
                f'Озвучка: {i+1}/{len(scene_targets)}',
                {'stage':5,'stages':5,'stage_name':'Озвучка'})
            gj = self._subjob('preview',book_id,sc['id'])
            gj['status']='RUNNING'
            self.repo.put('job',gj)
            try:
                await self.generate(gj)
                gj['status']='COMPLETED'
                self.repo.put('job',gj)
            except Exception:
                gj['status']='FAILED'
                self.repo.put('job',gj)
        self.progress(job,1.0,'Пайплайн завершён')
        job['status']='COMPLETED'
        self.repo.put('job',job)

    async def analyze(self, job):
        book_id = job['book_id']
        book = self.require('book',book_id)
        scenes = self.repo.list('scene',book_id)
        chapter_id = (job.get('payload') or {}).get('chapter_id')
        if chapter_id:
            chapter = self.require('chapter',chapter_id)
            if chapter['book_id'] != book_id:
                raise ValueError('Глава не принадлежит этой книге.')
            scenes = [s for s in scenes if s['chapter_id'] == chapter_id]
            self.progress(job,0.05,f'Анализ главы: {chapter["title"]}',{'current_chapter':chapter['title'],'total_scenes':len(scenes),'processed_scenes':0})
        for scene in scenes:
            scene['utterances'] = self.repo.list('utterance',book_id,scene['id'])
        chars = self.repo.list('character',book_id)
        self.progress(job,0.1,'Анализ структуры и говорящих',{'total_scenes':len(scenes),'processed_scenes':0,'current_operation':'LLM analysis'})
        result = await self.analyst.analyze(book,chars,scenes)
        changed = []
        with self.repo.connect() as db:
            by_name = {c['canonical_name']:c for c in chars}
            for candidate in result.get('characters',[]):
                name = candidate['canonical_name']
                character = by_name.get(name)
                if character is None:
                    character = self.create_character(book_id,name,db=db)
                # Human edits and locked voices always survive re-analysis.
                if not character.get('manual_edit'):
                    character.update({k:v for k,v in candidate.items() if k in ('canonical_name','aliases','gender','age_band','role','baseline_traits','speech_profile','evidence')})
                    # Generate individual voice brief from Character Bible facts
                    gender = candidate.get('gender',{}).get('value','unknown')
                    age = candidate.get('age_band',{}).get('value','unknown')
                    role = candidate.get('role','unknown')
                    traits = candidate.get('baseline_traits',[])
                    if not character.get('voice') or not character.get('voice_brief',{}).get('customized'):
                        brief = self.build_voice_brief(name,gender,age,role,traits)
                        character['voice_brief'] = brief
                by_name[name]=character
                self.repo.put('character',character,db)
            # Post-processing: "сказал X" / "спросил X" / "X сказал" in the next narration
            # tags the preceding unknown dialogue utterance.
            import re as _re
            utterance_objs = [self.repo.get('utterance',u['id'],db) for scene in scenes for u in scene.get('utterances',[])]
            for i, utterance in enumerate(utterance_objs):
                if (utterance.get('speaker_id') or utterance.get('manual_speaker')
                        or utterance.get('type') not in ('UNKNOWN','CHARACTER')):
                    continue
                following = utterance_objs[i+1] if i+1 < len(utterance_objs) else None
                if not following or following.get('type')!='NARRATOR':
                    continue
                text = following.get('source_text','')
                matches = list(_re.finditer(r'(?:сказал|спросил|ответил|произнес|прошептал|крикнул|проговорил|промолвил)\s+([А-ЯЁ][а-яё]+)', text))
                if not matches:
                    matches = list(_re.finditer(r'([А-ЯЁ][а-яё]+)\s+(?:сказал|спросил|ответил|произнес|прошептал|крикнул|проговорил|промолвил)', text))
                if matches:
                    name = matches[0].group(1)
                    char = by_name.get(name)
                    if char:
                        utterance.update(speaker_id=char['id'], speaker_confidence=0.98,
                            review_required=False,
                            attribution_reason='POST_TAG: speech verb + name in following narration',
                            attribution_evidence=[following['id']])
                        self.repo.put('utterance',utterance,db)
            for attr in result.get('attributions',[]):
                utterance = self.repo.get('utterance',attr['utterance_id'],db)
                if not utterance or utterance['book_id']!=book_id or utterance.get('manual_speaker'):
                    continue
                character = by_name.get(attr.get('character_name'))
                speaker_id = character['id'] if character else None
                confidence = attr.get('confidence',0)
                if utterance['speaker_id'] != speaker_id or utterance['review_required'] != (speaker_id is None or confidence<0.9):
                    changed.append(utterance['id'])
                utterance.update(speaker_id=speaker_id,speaker_confidence=confidence,
                    review_required=speaker_id is None or confidence<0.9,
                    attribution_evidence=attr.get('evidence',[]),attribution_reason=attr.get('reason_code',''))
                if character:
                    utterance['type']='NARRATOR' if character['role']=='narrator' else 'CHARACTER'
                self.repo.put('utterance',utterance,db)
            for summary in result.get('scene_summaries',[]):
                scene = self.repo.get('scene',summary['scene_id'],db)
                if scene and scene['book_id']==book_id:
                    scene['summary']=summary['summary']
                    self.repo.put('scene',scene,db)
            if changed:
                self.invalidate(book_id,changed,db)
            book = self.repo.get('book',book_id,db)
            book.update(status='ANALYZED',analysis_mode=result.get('mode','manual'),
                events=result.get('events',[]),analysis_version=book.get('analysis_version',0)+1)
            self.repo.put('book',book,db)

    async def casting(self, job):
        char = self.require('character',job['target_id'])
        if char.get('voice'):
            raise ValueError('Голос уже зафиксирован. Для замены укажите новый voice_id и область регенерации.')
        payload = job['payload']
        brief = payload.get('voice_brief') or char['voice_brief']
        self.progress(job,0.2,'Создание трёх вариантов голоса')
        candidates = await self.provider.design_voice(brief,payload.get('audition_text') or DEFAULT_AUDITION)
        saved = []
        for candidate in candidates:
            audio = base64.b64decode(candidate['audio_base64'])
            extension = 'wav' if 'wav' in candidate.get('media_type','') else 'mp3'
            filename = self.write_file(audio,extension)
            saved.append({k:v for k,v in candidate.items() if k!='audio_base64'} | {'audio_url':self.url(filename),'audio_file':filename})
        char.update(candidates=saved,voice_brief=brief)
        self.repo.put('character',char)

    async def direct(self, job):
        scene = self.require('scene',job['target_id'])
        scene['utterances']=self.repo.list('utterance',job['book_id'],scene['id'])
        self.progress(job,0.1,'Режиссура сцены')
        plans = await self.analyst.direct(scene,self.repo.list('character',job['book_id']))
        with self.repo.connect() as db:
            touched=[]
            for plan in plans:
                u=self.repo.get('utterance',plan['utterance_id'],db)
                if not u or u['scene_id']!=scene['id'] or u.get('manual_performance'):
                    continue
                performance=Performance.model_validate({k:v for k,v in plan.items() if k!='utterance_id'}).model_dump()
                if u['performance']!=performance:
                    u['performance']=performance
                    u['version']+=1
                    self.repo.put('utterance',u,db)
                    touched.append(u['id'])
            if touched:
                self.invalidate(job['book_id'],touched,db)

    def spoken(self, utterance):
        text = utterance['spoken_text']
        for entry in self.repo.list('pronunciation',utterance['book_id']):
            if entry['context_signature']==utterance['id'] and entry['approved']:
                # Boundaries prevent a word override from rewriting a different word.
                text=re.sub(r'(?<!\w)'+re.escape(entry['lexical_form'])+r'(?!\w)',lambda _:entry['spoken_form'],text)
        return text

    async def ensure_directed(self, scene):
        """Auto-run Scene Director when performances are still defaults."""
        utterances = self.repo.list('utterance',scene['book_id'],scene['id'])
        defaults = Performance().model_dump()
        needs_directing = any(u.get('performance') == defaults and not u.get('manual_performance') for u in utterances)
        if not needs_directing:
            return
        scene['utterances'] = utterances
        plans = await self.analyst.direct(scene, self.repo.list('character',scene['book_id']))
        # Post-validation: detect template-like direction where every line reads identically.
        # In auto mode we warn and keep going; manual direct() enforces uniqueness strictly.
        if len(plans) >= 3:
            deliveries = [p.get('delivery','') for p in plans]
            unique = set(deliveries)
            if len(unique) == 1 and unique != {''}:
                import logging
                logging.getLogger('audiobook.studio').warning('Режиссура вернула одинаковую подачу для %d реплик', len(plans))
        with self.repo.connect() as db:
            for plan in plans:
                u = self.repo.get('utterance',plan['utterance_id'],db)
                if not u or u['scene_id']!=scene['id'] or u.get('manual_performance'):
                    continue
                performance = Performance.model_validate({k:v for k,v in plan.items() if k!='utterance_id'}).model_dump()
                u['performance'] = performance
                u['version'] += 1
                self.repo.put('utterance',u,db)

    def blocks(self, utterances, characters):
        """Conservative scene-aware grouping; tags included in the safe character budget."""
        limits = self.config()['provider_limits']
        dialogue_limit = limits.get('dialogue_chars',2000)
        voice_limit = limits.get('dialogue_voices',10)
        blocks, block, size, voices = [], [], 0, set()
        for u in utterances:
            text=self.spoken(u)
            chars=len(text)+sum(len(t)+3 for t in u['performance'].get('audio_tags',[]))
            if chars>dialogue_limit:
                raise ValueError(f'Реплика с тегами превышает {dialogue_limit} символов. Сократите нормализованный текст или теги.')
            voice=characters[u['speaker_id']]['voice']['voice_id']
            solo=(u.get('mode')=='single' or (u['type']=='NARRATOR' and (not self.settings.include_narrator_in_dialogue or self.settings.narration_model!='eleven_v3')))
            previous_solo=bool(block and (block[-1].get('mode')=='single' or (block[-1]['type']=='NARRATOR' and not self.settings.include_narrator_in_dialogue)))
            # Non-default exact pauses are implemented between artifacts, never implied inside dialogue.
            exact_pause=bool(block and (u['performance'].get('pause_before_ms',0)>0 or block[-1]['performance'].get('pause_after_ms',350)!=350))
            if block and (solo or previous_solo or exact_pause or u['scene_id']!=block[0]['scene_id'] or size+chars>dialogue_limit or len(voices|{voice})>voice_limit):
                blocks.append(block)
                block,size,voices=[],0,set()
            block.append(u)
            size+=chars
            voices.add(voice)
            if solo:
                blocks.append(block)
                block,size,voices=[],0,set()
        if block:
            blocks.append(block)
        return blocks

    def generation_blocks(self, utterances, characters):
        """Keep every existing artifact as an atomic boundary when re-planning nearby text."""
        planned, pending, seen = [], [], set()
        by_id = {u['id']:u for u in utterances}
        for u in utterances:
            generation_id = u.get('generation_id')
            if not generation_id:
                pending.append(u)
                continue
            if pending:
                planned.extend(self.blocks(pending,characters))
                pending=[]
            if generation_id in seen:
                continue
            seen.add(generation_id)
            gen = self.require('generation',generation_id)
            if any(x not in by_id or by_id[x].get('generation_id')!=generation_id for x in gen['utterance_ids']):
                raise ValueError('Обнаружен частично изменённый блок. Перегенерируйте сцену.')
            planned.append([by_id[x] for x in gen['utterance_ids']])
        if pending:
            planned.extend(self.blocks(pending,characters))
        return planned

    async def finish_qc(self, generation, provider):
        """Resume QC from saved paid audio; never synthesize again to recover a QC failure."""
        from .services.audio import qc_text,evaluate_qc
        path=self.files/generation['audio_file']
        acoustic=await asyncio.to_thread(self.audio.inspect,path)
        text_metrics=None
        transcript=None
        asr_error=None
        if generation['provider']=='elevenlabs' and self.settings.asr_enabled:
            try:
                transcript=await provider.transcribe(path.read_bytes(),path.name)
                text_metrics=qc_text(' '.join(generation['spoken_texts']),transcript)
            except Exception:
                asr_error='ASR недоступен; требуется ручная проверка текста.'
        qc=evaluate_qc(acoustic,text_metrics,None,demo=generation['provider']=='demo')
        if asr_error:
            qc['issues'].append(asr_error)
        generation.update(qc=qc,transcript=transcript,
            duration_seconds=acoustic.get('duration_seconds',acoustic.get('duration',0)))
        self.repo.put('generation',generation)
        self.select_generation(generation)
        return generation

    async def generate(self, job):
        from .providers.base import ProviderError
        book_id=job['book_id']
        all_utterances=self.repo.list('utterance',book_id)
        utterances=all_utterances
        if job['kind']=='preview':
            utterances=[u for u in utterances if u['scene_id']==job['target_id']]
        elif job['kind']=='regenerate':
            ids=set(job['payload']['utterance_ids'])
            utterances=[u for u in utterances if u['id'] in ids]
        if job['kind']=='preview' and job.get('target_id'):
            self.progress(job,0.02,'Автоматическая режиссура сцены')
            scene=self.require('scene',job['target_id'])
            await self.ensure_directed(scene)
            utterances=self.repo.list('utterance',book_id)
            utterances=[u for u in utterances if u['scene_id']==job['target_id']]
            self.progress(job,0.05,'Режиссура завершена')
        self.validate_ready(book_id,utterances,require_preview=job['kind']=='generate')
        characters={c['id']:c for c in self.repo.list('character',book_id)}
        block_list=self.generation_blocks(utterances,characters)
        provider=self.provider
        book=self.require('book',book_id)
        book['status']='SYNTHESIZING'
        self.repo.put('book',book)
        for index, block in enumerate(block_list):
            self.progress(job,index/len(block_list),f'Озвучка блока {index+1} из {len(block_list)}',{'total_blocks':len(block_list),'processed_blocks':index,'current_operation':'TTS synthesis','attempt':1})
            if all(u.get('generation_id') for u in block):
                active=self.require('generation',block[0]['generation_id'])
                if active['qc'].get('pending'):
                    await self.finish_qc(active,provider)
                continue
            lines=[{'text':self.spoken(u),'voice_id':characters[u['speaker_id']]['voice']['voice_id'],'performance':u['performance']} for u in block]
            mode='text_to_dialogue' if len(block)>1 else 'single'
            text=lines[0]['text']
            model=self.settings.narration_model if all(u['type']=='NARRATOR' for u in block) and mode=='single' else 'eleven_v3'
            if model not in ('eleven_v3','eleven_multilingual_v2'):
                raise ValueError('NARRATION_MODEL должен быть eleven_v3 или eleven_multilingual_v2.')
            signature={'utterances':[(u['id'],u['version']) for u in block],'lines':lines,
                'voices':[characters[u['speaker_id']]['voice'] for u in block], 'mode':mode,'model':model,
                'provider':'elevenlabs' if self.settings.live else 'demo','prompt_version':'scene_director_v1',
                'regeneration':job['id'] if job['kind']=='regenerate' else None}
            cache_key=hashlib.sha256(json.dumps(signature,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
            cached=next((g for g in reversed(self.repo.list('generation',book_id)) if g['cache_key']==cache_key and (self.files/g['audio_file']).exists() and g['qc']['status'] in ('PASS','APPROVED','REVIEW')),None)
            if cached:
                self.select_generation(cached)
                if cached['qc'].get('pending'):
                    await self.finish_qc(cached,provider)
                continue
            generation=None
            for attempt in range(1,self.settings.max_attempts+1):
                seed=int(hashlib.sha256(f'{cache_key}:{attempt}'.encode()).hexdigest()[:8],16)
                try:
                    if mode=='text_to_dialogue':
                        artifact=await provider.synthesize_dialogue(lines,seed)
                    else:
                        artifact=await provider.synthesize_single(text,lines[0]['voice_id'],lines[0]['performance'],seed,model)
                except ProviderError as exc:
                    if not exc.retryable or attempt==self.settings.max_attempts:
                        raise
                    self.progress(job,index/len(block_list),f'Повтор после ошибки провайдера: попытка {attempt+1}',{'total_blocks':len(block_list),'processed_blocks':index,'current_operation':'retry','attempt':attempt+1})
                    await asyncio.sleep(max(0,float(exc.retry_after)) if exc.retry_after is not None else min(2**attempt,60))
                    continue
                filename=self.write_file(artifact.audio,artifact.extension)
                generation={'id':uid(),'book_id':book_id,'utterance_ids':[u['id'] for u in block],
                    'scene_id':block[0]['scene_id'],'cache_key':cache_key,'provider':signature['provider'],
                    'mode':mode,'model_id':artifact.model_id,'provider_voice_ids':[l['voice_id'] for l in lines],
                    'seed':seed,'provider_parameters':artifact.parameters,'prompt_version':'scene_director_v1',
                    'attempt':attempt,'audio_file':filename,'qc':{'status':'REVIEW','pending':True,
                        'issues':['Проверка аудио ожидает выполнения.'],'metrics':{}},'transcript':None,
                    'spoken_texts':[l['text'] for l in lines], 'source_versions':signature['utterances'],
                    'created_at':now(),'request_id':artifact.request_id,'request_count':artifact.parameters.get('http_attempts',1),
                    'duration_seconds':0,
                    'pause_before_ms':block[0]['performance'].get('pause_before_ms',0),
                    'pause_after_ms':block[-1]['performance'].get('pause_after_ms',350)}
                self.repo.put('generation',generation)
                self.select_generation(generation)
                generation=await self.finish_qc(generation,provider)
                if generation['qc']['status']!='FAIL':
                    break
                if attempt==2:
                    # Retry with restrained, validated delivery while preserving words and locked voice.
                    for line in lines:
                        line['performance']={**line['performance'],'audio_tags':[]}
            if generation:
                self.select_generation(generation)
            await asyncio.sleep(self.settings.request_gap_seconds)
        book=self.require('book',book_id)
        book['status']='QC'
        self.repo.put('book',book)

    def select_generation(self,generation):
        with self.repo.connect() as db:
            for utterance_id in generation['utterance_ids']:
                u=self.repo.get('utterance',utterance_id,db)
                u['generation_id']=generation['id']
                u['status']='GENERATED' if generation['qc'].get('pending') else {'PASS':'APPROVED','APPROVED':'APPROVED','FAIL':'QC_FAIL','REVIEW':'QC_REVIEW'}[generation['qc']['status']]
                self.repo.put('utterance',u,db)

    async def export(self,job):
        book=self.require('book',job['book_id'])
        utterances=self.repo.list('utterance',book['id'])
        if not utterances or any(u['status']!='APPROVED' or u['review_required'] for u in utterances):
            raise ValueError('Экспорт доступен после озвучки и подтверждения QC всех реплик.')
        generations={g['id']:g for g in self.repo.list('generation',book['id'])}
        chapters=[]
        for chapter in self.repo.list('chapter',book['id']):
            scene_ids={s['id'] for s in self.repo.list('scene',book['id'],chapter['id'])}
            segments=[]
            seen=set()
            for u in utterances:
                if u['scene_id'] in scene_ids and u['generation_id'] not in seen:
                    seen.add(u['generation_id'])
                    g=generations[u['generation_id']]
                    if set(g['utterance_ids'])!={v['id'] for v in utterances if v['generation_id']==g['id']}:
                        raise ValueError('Блок диалога изменён частично. Повторите генерацию сцены.')
                    segments.append({'path':self.files/g['audio_file'], 'pause_before_ms':g['pause_before_ms'],'pause_after_ms':g['pause_after_ms']})
            if segments:
                chapters.append({'id':chapter['id'],'title':chapter['title'],'segments':segments})
        self.progress(job,0.2,'Монтаж глав, пауз и метаданных',{'current_operation':'assembly'})
        book['status']='MASTERING'
        self.repo.put('book',book)
        result=await asyncio.to_thread(self.audio.export,chapters,self.root/'exports'/job['id'],book['title'],book.get('author',''),job['payload']['formats'])
        with self.repo.connect() as db:
            for old in self.repo.list('export',book['id'],db=db):
                self.repo.delete('export',old['id'],db)
            for output in result['files']:
                path=Path(output['path'])
                filename=f'{uid()}{path.suffix.lower()}'
                shutil.copyfile(path,self.files/filename)
                self.repo.put('export',{'id':uid(),'book_id':book['id'],'filename':output.get('filename',path.name),
                    'format':output['format'],'url':self.url(filename),'audio_file':filename,'created_at':now()},db)
            book['status']='COMPLETED'
            self.repo.put('book',book,db)
