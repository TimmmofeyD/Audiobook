"""HTTP API and local single-worker application entry point."""
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .config import ROOT, Settings
from .repository import uid, now
from .schemas import (Approval, BookPatch, CandidatesRequest, CharacterCreate,
    CharacterPatch, ExportRequest, Performance, PronunciationCreate, UtterancePatch, VoiceLock)
from .studio import Studio


def create_app(settings: Settings | None = None):
    studio = Studio(settings or Settings())

    @asynccontextmanager
    async def lifespan(app):
        await studio.start()
        yield
        await studio.stop()

    app = FastAPI(title='Голос — студия аудиокниг', version='0.1.0', lifespan=lifespan)
    app.state.studio = studio
    app.add_middleware(TrustedHostMiddleware,allowed_hosts=['127.0.0.1','localhost','[::1]','testserver'])

    @app.middleware('http')
    async def local_origin(request: Request, call_next):
        # No CORS; browser writes are same-origin. A localhost app must also reject foreign HTML forms.
        origin = request.headers.get('origin')
        if request.method not in ('GET','HEAD','OPTIONS') and origin:
            parsed = urlsplit(origin)
            if parsed.netloc != request.headers.get('host') or parsed.scheme not in ('http','https'):
                return JSONResponse({'detail':'Запрос разрешён только из интерфейса приложения.'},status_code=403)
        response = await call_next(request)
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Referrer-Policy']='no-referrer'
        response.headers['X-Frame-Options']='DENY'
        return response

    @app.exception_handler(ValueError)
    async def bad_input(request, exc):
        return JSONResponse({'detail':str(exc)},status_code=400)

    from .providers.base import ProviderError
    @app.exception_handler(ProviderError)
    async def provider_error(request, exc):
        return JSONResponse({'detail':str(exc)},status_code=502)

    @app.get('/api/config')
    async def config():
        return studio.config()

    @app.get('/api/health')
    async def health():
        return {'ok':True,'worker_running':bool(studio.worker_task and not studio.worker_task.done()),
            'mode':'live' if studio.settings.live else 'demo'}

    @app.get('/api/books')
    async def books():
        return list(reversed(studio.repo.list('book')))

    @app.post('/api/books',status_code=201)
    async def upload(file: UploadFile = File(...),title: str|None=Form(None),author: str|None=Form(None)):
        filename=Path(file.filename or 'book.txt').name
        if Path(filename).suffix.lower() not in ('.txt','.fb2','.epub'):
            raise HTTPException(400,'Поддерживаются TXT, FB2 и EPUB.')
        maximum=studio.settings.max_upload_mb*1024*1024
        content=await file.read(maximum+1)
        await file.close()
        if len(content)>maximum:
            raise HTTPException(413,f'Файл превышает {studio.settings.max_upload_mb} МБ.')
        if not content:
            raise HTTPException(400,'Загружен пустой файл.')
        return studio.import_book(content,filename,title,author)

    @app.post('/api/demo',status_code=201)
    async def demo():
        return studio.demo()

    @app.get('/api/books/{book_id}')
    async def book_detail(book_id: str):
        return studio.detail(book_id)

    @app.patch('/api/books/{book_id}')
    async def book_update(book_id: str,body: BookPatch):
        book=studio.require('book',book_id)
        studio.ensure_idle(book_id)
        book.update(body.model_dump(exclude_unset=True,exclude_none=True))
        with studio.repo.connect() as db:
            studio.repo.put('book',book,db)
            for export in studio.repo.list('export',book_id,db=db):
                studio.repo.delete('export',export['id'],db)
        return book

    @app.post('/api/books/{book_id}/analyze',status_code=202)
    async def analyze(book_id: str):
        return studio.enqueue('analyze',book_id)

    @app.delete('/api/books/{book_id}')
    async def delete_book(book_id: str):
        studio.require('book',book_id)
        # Refuse deletion while background work is running or queued for this book.
        if book_id in studio.mutating_books:
            raise HTTPException(409,'Для этой книги выполняется задание. Дождитесь завершения.')
        active=[j for j in studio.repo.list('job',book_id) if j['status'] in ('QUEUED','RUNNING')]
        if active:
            raise HTTPException(409,'Для этой книги уже выполняется задание. Дождитесь завершения.')
        studio.delete_book(book_id)
        return {'deleted':book_id}

    @app.post('/api/books/{book_id}/pipeline',status_code=202)
    async def pipeline(book_id: str, body: dict | None = None):
        payload = body or {}
        return studio.enqueue('pipeline',book_id,payload.get('chapter_id'),{'chapter_id':payload.get('chapter_id')})

    @app.post('/api/chapters/{chapter_id}/analyze',status_code=202)
    async def analyze_chapter(chapter_id: str):
        chapter=studio.require('chapter',chapter_id)
        return studio.enqueue('analyze',chapter['book_id'],chapter_id,{'chapter_id':chapter_id})

    @app.get('/api/books/{book_id}/characters')
    async def characters(book_id: str):
        studio.require('book',book_id)
        return studio.repo.list('character',book_id)

    @app.post('/api/books/{book_id}/characters',status_code=201)
    async def character_create(book_id: str,body: CharacterCreate):
        studio.require('book',book_id)
        studio.ensure_idle(book_id)
        if not body.canonical_name.strip():
            raise HTTPException(400,'Введите имя персонажа.')
        if any(c['canonical_name'].casefold()==body.canonical_name.strip().casefold() for c in studio.repo.list('character',book_id)):
            raise HTTPException(409,'Персонаж с таким именем уже существует.')
        return studio.create_character(book_id,body.canonical_name)

    @app.patch('/api/characters/{character_id}')
    async def character_update(character_id: str,body: CharacterPatch):
        char=studio.require('character',character_id)
        studio.ensure_idle(char['book_id'])
        patch=body.model_dump(exclude_unset=True,exclude_none=True)
        if 'canonical_name' in patch:
            patch['canonical_name']=patch['canonical_name'].strip()
            if not patch['canonical_name']:
                raise HTTPException(400,'Имя не может быть пустым.')
            if any(c['id']!=char['id'] and c['canonical_name'].casefold()==patch['canonical_name'].casefold() for c in studio.repo.list('character',char['book_id'])):
                raise HTTPException(409,'Такое имя уже используется.')
        if char['role']=='narrator' and patch.get('role','narrator')!='narrator':
            raise HTTPException(400,'Сохраните роль рассказчика.')
        char.update(patch,manual_edit=True,version=char['version']+1)
        studio.repo.put('character',char)
        return char

    @app.post('/api/characters/{character_id}/voice-candidates',status_code=202)
    async def voice_candidates(character_id: str,body: CandidatesRequest):
        char=studio.require('character',character_id)
        if char.get('voice'):
            raise HTTPException(409,'Голос зафиксирован. Для замены используйте явную смену voice_id с регенерацией.')
        return studio.enqueue('casting',char['book_id'],character_id,body.model_dump(exclude_none=True))

    @app.post('/api/characters/{character_id}/voice-lock')
    async def voice_lock(character_id: str,body: VoiceLock):
        char=studio.require('character',character_id)
        studio.ensure_idle(char['book_id'])
        if not body.audition_confirmed:
            raise HTTPException(400,'Подтвердите прослушивание голоса.')
        if bool(body.voice_id)==bool(body.generated_voice_id):
            raise HTTPException(400,'Укажите один voice_id или один выбранный кандидат.')
        if char.get('voice') and not (body.replace and body.regeneration_scope=='all'):
            raise HTTPException(409,'LOCK защищает голос. Для замены подтвердите replace и regeneration_scope=all.')
        if body.generated_voice_id and not any(c['generated_voice_id']==body.generated_voice_id for c in char['candidates']):
            raise HTTPException(400,'Кандидат не принадлежит этому персонажу.')
        studio.mutating_books.add(char['book_id'])
        try:
            provider=studio.provider
            if body.generated_voice_id:
                metadata=await provider.save_voice(body.generated_voice_id,char['canonical_name'],str(char['voice_brief'])[:1000])
                voice_id=metadata['voice_id']
            else:
                metadata=await provider.get_voice_metadata(body.voice_id)
                voice_id=body.voice_id
            previous=char.get('voice')
            voice={'voice_id':voice_id,'provider':'elevenlabs' if studio.settings.live else 'demo',
                'status':'LOCKED','provenance':body.provenance,'version':previous['version']+1 if previous else 1,
                'approved_at':now(),'audition_version':1,'metadata':metadata}
            with studio.repo.connect() as db:
                if previous:
                    char.setdefault('voice_history',[]).append(previous)
                    studio.invalidate(char['book_id'],db=db)
                char['voice']=voice
                studio.repo.put('character',char,db)
                chars=studio.repo.list('character',char['book_id'],db=db)
                book=studio.repo.get('book',char['book_id'],db)
                book['status']='CAST_LOCKED' if all(c.get('voice') for c in chars) else 'CASTING'
                studio.repo.put('book',book,db)
            return char
        finally:
            studio.mutating_books.discard(char['book_id'])

    @app.patch('/api/utterances/{utterance_id}')
    async def utterance_update(utterance_id: str,body: UtterancePatch):
        utterance=studio.require('utterance',utterance_id)
        studio.ensure_idle(utterance['book_id'])
        patch=body.model_dump(exclude_unset=True)
        if 'speaker_id' in patch:
            char=studio.require('character',patch['speaker_id']) if patch['speaker_id'] else None
            if char and char['book_id']!=utterance['book_id']:
                raise HTTPException(400,'Персонаж относится к другой книге.')
            patch.update(speaker_confidence=1.0 if char else 0.0,review_required=not bool(char),manual_speaker=True,
                type=('NARRATOR' if char['role']=='narrator' else 'CHARACTER') if char else 'UNKNOWN')
        if 'spoken_text' in patch and (patch['spoken_text'] is None or not patch['spoken_text'].strip()):
            raise HTTPException(400,'Текст реплики не может быть пустым.')
        if 'performance' in patch and patch['performance'] is not None:
            patch['performance']=Performance.model_validate({**utterance['performance'],**patch['performance']}).model_dump()
            allowed=set(studio.config()['provider_limits'].get('audio_tags',[]))
            if set(patch['performance']['audio_tags'])-allowed:
                raise HTTPException(400,'Неизвестный аудиотег. Используйте разрешённые теги из настроек.')
            patch['manual_performance']=True
        patch={k:v for k,v in patch.items() if v is not None or k=='speaker_id'}
        if patch:
            with studio.repo.connect() as db:
                studio.invalidate(utterance['book_id'],[utterance_id],db)
                utterance=studio.repo.get('utterance',utterance_id,db)
                utterance.update(patch,version=utterance['version']+1)
                studio.repo.put('utterance',utterance,db)
        return utterance

    @app.post('/api/characters/{character_id}/research')
    async def research(character_id: str):
        return studio.research_character(character_id)

    @app.post('/api/scenes/{scene_id}/direct',status_code=202)
    async def direct(scene_id: str):
        scene=studio.require('scene',scene_id)
        return studio.enqueue('direct',scene['book_id'],scene_id)

    @app.post('/api/scenes/{scene_id}/generate-preview',status_code=202)
    async def preview(scene_id: str):
        scene=studio.require('scene',scene_id)
        studio.validate_ready(scene['book_id'],studio.repo.list('utterance',scene['book_id'],scene_id))
        return studio.enqueue('preview',scene['book_id'],scene_id)

    @app.post('/api/scenes/{scene_id}/approve-preview')
    async def approve_preview(scene_id: str,body: Approval):
        scene=studio.require('scene',scene_id)
        studio.ensure_idle(scene['book_id'])
        utterances=studio.repo.list('utterance',scene['book_id'],scene_id)
        if body.approved and (not utterances or any(not u.get('generation_id') or u['status'] in ('QC_FAIL','GENERATED') for u in utterances)):
            raise HTTPException(409,'Сначала сгенерируйте полную сцену и исправьте ошибки QC.')
        scene['preview_approved']=body.approved
        scene['preview_approved_at']=now() if body.approved else None
        studio.repo.put('scene',scene)
        return scene

    @app.post('/api/books/{book_id}/generate',status_code=202)
    async def generate(book_id: str):
        studio.require('book',book_id)
        studio.validate_ready(book_id,studio.repo.list('utterance',book_id),require_preview=True)
        return studio.enqueue('generate',book_id)

    def regenerate(kind,entity_id):
        item=studio.require(kind,entity_id)
        book_id=item['book_id']
        studio.ensure_idle(book_id)
        all_u=studio.repo.list('utterance',book_id)
        if kind=='utterance':
            selected=[item]
        elif kind=='scene':
            selected=[u for u in all_u if u['scene_id']==entity_id]
        else:
            scenes={s['id'] for s in studio.repo.list('scene',book_id,entity_id)}
            selected=[u for u in all_u if u['scene_id'] in scenes]
        old_generations={u['generation_id'] for u in selected if u.get('generation_id')}
        ids={u['id'] for u in selected}|{u['id'] for u in all_u if u.get('generation_id') in old_generations}
        studio.validate_ready(book_id,[u for u in all_u if u['id'] in ids])
        # Keep previous A/B artifacts in history; detach the entire affected dialogue block.
        studio.invalidate(book_id,list(ids))
        return studio.enqueue('regenerate',book_id,entity_id,{'utterance_ids':list(ids)})

    @app.post('/api/utterances/{utterance_id}/regenerate',status_code=202)
    async def regenerate_utterance(utterance_id: str):
        return regenerate('utterance',utterance_id)

    @app.post('/api/scenes/{scene_id}/regenerate',status_code=202)
    async def regenerate_scene(scene_id: str):
        return regenerate('scene',scene_id)

    @app.post('/api/chapters/{chapter_id}/regenerate',status_code=202)
    async def regenerate_chapter(chapter_id: str):
        return regenerate('chapter',chapter_id)

    @app.get('/api/generations/{generation_id}')
    async def generation_detail(generation_id: str):
        return studio.generation_view(studio.require('generation',generation_id))

    @app.get('/api/utterances/{utterance_id}/generations')
    async def generation_history(utterance_id: str):
        u=studio.require('utterance',utterance_id)
        return [studio.generation_view(g,u.get('generation_id')) for g in studio.repo.list('generation',u['book_id']) if utterance_id in g['utterance_ids']]

    @app.post('/api/generations/{generation_id}/approve')
    async def generation_approve(generation_id: str,body: Approval):
        generation=studio.require('generation',generation_id)
        studio.ensure_idle(generation['book_id'])
        active=[u for u in studio.repo.list('utterance',generation['book_id']) if u.get('generation_id')==generation_id]
        if not active:
            raise HTTPException(409,'Это архивный вариант; утвердите текущую генерацию.')
        if generation['qc'].get('pending'):
            raise HTTPException(409,'Проверка аудио ещё не завершена. Возобновите задание QC.')
        generation['qc']['automatic_status']=generation['qc'].get('automatic_status',generation['qc']['status'])
        generation['qc'].update(status='APPROVED' if body.approved else generation['qc']['automatic_status'],
            approved_by='local_user' if body.approved else None,approved_at=now() if body.approved else None)
        if not body.approved:
            for export in studio.repo.list('export',generation['book_id']):
                studio.repo.delete('export',export['id'])
        studio.repo.put('generation',generation)
        studio.select_generation(generation)
        return studio.generation_view(generation)

    @app.get('/api/jobs/{job_id}')
    async def job_detail(job_id: str):
        return studio.require('job',job_id)

    @app.post('/api/jobs/{job_id}/cancel')
    async def job_cancel(job_id: str):
        job = studio.require('job',job_id)
        if job.get('status') not in ('QUEUED','RUNNING'):
            raise HTTPException(409,'Задача уже завершена')
        job.update(status='CANCELLED',message='Отменено пользователем',error=None)
        studio.repo.put('job',job)
        return job

    @app.post('/api/jobs/{job_id}/retry',status_code=202)
    async def job_retry(job_id: str):
        job=studio.require('job',job_id)
        if job['status']!='FAILED':
            raise HTTPException(409,'Повтор доступен для завершившегося с ошибкой задания.')
        studio.ensure_idle(job['book_id'])
        job.update(status='QUEUED',error=None,progress=0,message='Повторный запуск',manual_retries=job.get('manual_retries',0)+1)
        studio.repo.put('job',job)
        studio.wake.set()
        return job

    @app.post('/api/books/{book_id}/export',status_code=202)
    async def export(book_id: str,body: ExportRequest):
        studio.require('book',book_id)
        utterances=studio.repo.list('utterance',book_id)
        if not utterances or any(u['status']!='APPROVED' or u['review_required'] for u in utterances):
            raise HTTPException(409,'Все реплики должны быть озвучены и пройти QC или ручное утверждение.')
        return studio.enqueue('export',book_id,payload=body.model_dump())

    @app.get('/api/books/{book_id}/pronunciations')
    async def pronunciations(book_id: str):
        studio.require('book',book_id)
        return studio.repo.list('pronunciation',book_id)

    @app.post('/api/books/{book_id}/pronunciations',status_code=201)
    async def pronunciation_create(book_id: str,body: PronunciationCreate):
        studio.require('book',book_id)
        studio.ensure_idle(book_id)
        utterance=studio.require('utterance',body.context_signature)
        if utterance['book_id']!=book_id:
            raise HTTPException(400,'Контекст должен указывать на реплику этой книги.')
        import re
        if not re.search(r'(?<!\w)'+re.escape(body.lexical_form)+r'(?!\w)',utterance['spoken_text']):
            raise HTTPException(400,'Указанное слово не найдено в этой реплике.')
        with studio.repo.connect() as db:
            previous=next((e for e in studio.repo.list('pronunciation',book_id,db=db) if e['lexical_form']==body.lexical_form and e['context_signature']==body.context_signature),None)
            entry={**body.model_dump(),'id':previous['id'] if previous else uid(),'book_id':book_id,
                'version':previous['version']+1 if previous else 1,'approved_by':'local_user','created_at':now()}
            studio.repo.put('pronunciation',entry,db)
            studio.invalidate(book_id,[utterance['id']],db)
        return entry

    @app.delete('/api/pronunciations/{entry_id}')
    async def pronunciation_delete(entry_id: str):
        entry=studio.require('pronunciation',entry_id)
        studio.ensure_idle(entry['book_id'])
        with studio.repo.connect() as db:
            studio.repo.delete('pronunciation',entry_id,db)
            studio.invalidate(entry['book_id'],[entry['context_signature']],db)
        return {'deleted':True}

    @app.get('/api/files/{filename}')
    async def file_download(filename: str):
        import re
        if not re.fullmatch(r'[0-9a-f-]{36}\.(?:wav|mp3|m4b|flac|txt|fb2|epub|json)',filename):
            raise HTTPException(404,'Файл не найден.')
        path=studio.files/filename
        if not path.is_file():
            raise HTTPException(404,'Файл не найден.')
        types={'.wav':'audio/wav','.mp3':'audio/mpeg','.m4b':'audio/mp4','.flac':'audio/flac'}
        return FileResponse(path,media_type=types.get(path.suffix,'application/octet-stream'))

    app.mount('/static',StaticFiles(directory=ROOT/'app/static'),name='static')

    @app.get('/',include_in_schema=False)
    async def index():
        return FileResponse(ROOT/'app/static/index.html',headers={'Cache-Control':'no-cache'})

    return app


app=create_app()
