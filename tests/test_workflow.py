import time
from pathlib import Path

from fastapi.testclient import TestClient
from app.config import Settings
from app.main import create_app


def settings(tmp_path):
    return Settings(_env_file=None,data_dir=tmp_path,app_mode='demo',request_gap_seconds=0,
                    llm_api_key='',llm_model='',elevenlabs_api_key='')


def done(client, response):
    assert response.status_code in (200,201,202),response.text
    job=response.json()
    deadline=time.monotonic()+30
    while time.monotonic()<deadline:
        job=client.get('/api/jobs/'+job['id']).json()
        if job['status'] in ('COMPLETED','FAILED'):
            assert job['status']=='COMPLETED',job
            return job
        time.sleep(.03)
    raise AssertionError(job)


def utterances(book):
    return [u for c in book['chapters'] for s in c['scenes'] for u in s['utterances']]


def cast(client,book):
    for char in book['characters']:
        response=client.post(f"/api/characters/{char['id']}/voice-lock",json={
            'voice_id':'demo-'+char['id'],'provenance':'designed','audition_confirmed':True})
        assert response.status_code==200,response.text


def approve_all(client,book):
    ids={u['generation_id'] for u in utterances(book) if u['generation_id']}
    for generation in ids:
        result=client.post(f'/api/generations/{generation}/approve',json={'approved':True})
        assert result.status_code==200,result.text


def test_end_to_end_export_cache_and_lock(tmp_path):
    app=create_app(settings(tmp_path))
    with TestClient(app) as client:
        book=client.post('/api/demo').json()
        book_id=book['id']
        cast(client,book)
        # A full run is forbidden before a listened preview.
        assert client.post(f'/api/books/{book_id}/generate',json={}).status_code==400
        first=book['chapters'][0]['scenes'][0]['id']
        done(client,client.post(f'/api/scenes/{first}/generate-preview',json={}))
        assert client.post(f'/api/scenes/{first}/approve-preview',json={'approved':True}).status_code==200
        done(client,client.post(f'/api/books/{book_id}/generate',json={}))
        book=client.get('/api/books/'+book_id).json()
        assert all(u['generation_id'] for u in utterances(book))
        assert all(u['qc']['status']=='REVIEW' for u in utterances(book))
        assert client.post(f'/api/books/{book_id}/export',json={'formats':['m4b']}).status_code==409
        approve_all(client,book)
        before=client.get('/api/books/'+book_id).json()['metrics']['tts_requests']
        done(client,client.post(f'/api/books/{book_id}/generate',json={}))
        assert client.get('/api/books/'+book_id).json()['metrics']['tts_requests']==before
        done(client,client.post(f'/api/books/{book_id}/export',json={'formats':['wav','mp3','m4b']}))
        book=client.get('/api/books/'+book_id).json()
        assert book['status']=='COMPLETED'
        assert {'wav','mp3','m4b'}=={x['format'] for x in book['exports']}
        for output in book['exports']:
            audio=client.get(output['url'])
            assert audio.status_code==200 and len(audio.content)>100
        character=book['characters'][0]
        assert client.post(f"/api/characters/{character['id']}/voice-lock",json={
            'voice_id':'demo-new','audition_confirmed':True}).status_code==409
        changed=client.post(f"/api/characters/{character['id']}/voice-lock",json={
            'voice_id':'demo-new','audition_confirmed':True,'replace':True,'regeneration_scope':'all'})
        assert changed.status_code==200,changed.text
        revised=client.get('/api/books/'+book_id).json()
        assert all(u['generation_id'] is None for u in utterances(revised))
        assert not revised['exports']
        assert all(not s['preview_approved'] for c in revised['chapters'] for s in c['scenes'])


def test_edit_invalidates_whole_dialogue_and_preserves_history(tmp_path):
    with TestClient(create_app(settings(tmp_path))) as client:
        book=client.post('/api/demo').json()
        cast(client,book)
        scene=book['chapters'][0]['scenes'][0]
        done(client,client.post(f"/api/scenes/{scene['id']}/generate-preview",json={}))
        book=client.get('/api/books/'+book['id']).json()
        u=utterances(book)[0]
        active=u['generation_id']
        grouped=[v for v in utterances(book) if v['generation_id']==active]
        assert len(grouped)>1
        response=client.patch('/api/utterances/'+u['id'],json={'spoken_text':'Исправленный текст.'})
        assert response.status_code==200,response.text
        book=client.get('/api/books/'+book['id']).json()
        for v in utterances(book):
            if v['id'] in {x['id'] for x in grouped}:
                assert v['generation_id'] is None
                assert v['generations'][-1]['id']==active
        assert utterances(book)[0]['source_text']==u['source_text']
        assert client.post(f'/api/generations/{active}/approve',json={'approved':True}).status_code==409


def test_unknown_speaker_gated_and_manual_analysis(tmp_path):
    with TestClient(create_app(settings(tmp_path))) as client:
        book=client.post('/api/books',files={'file':('test.txt','— Кто здесь?\n\nНочь была тихой.'.encode(),'text/plain')}).json()
        done(client,client.post(f"/api/books/{book['id']}/analyze",json={}))
        book=client.get('/api/books/'+book['id']).json()
        unknown=next(u for u in utterances(book) if u['type']=='UNKNOWN')
        assert unknown['speaker_id'] is None and unknown['review_required']
        scene=book['chapters'][0]['scenes'][0]
        assert client.post(f"/api/scenes/{scene['id']}/generate-preview",json={}).status_code==400
        char=client.post(f"/api/books/{book['id']}/characters",json={'canonical_name':'Гость'}).json()
        assert client.patch(f"/api/utterances/{unknown['id']}",json={'speaker_id':char['id']}).status_code==200
        book=client.get('/api/books/'+book['id']).json()
        cast(client,book)
        done(client,client.post(f"/api/scenes/{scene['id']}/direct",json={}))
        done(client,client.post(f"/api/scenes/{scene['id']}/generate-preview",json={}))


def test_pronunciation_context_and_web_origin(tmp_path):
    with TestClient(create_app(settings(tmp_path))) as client:
        book=client.post('/api/books',files={'file':('book.txt','Старый замок.\n\nДверной замок.'.encode(),'text/plain')}).json()
        us=utterances(book)
        result=client.post(f"/api/books/{book['id']}/pronunciations",json={
            'lexical_form':'замок','spoken_form':'за́мок','context_signature':us[0]['id'],'approved':True})
        assert result.status_code==201,result.text
        studio=client.app.state.studio
        assert 'за́мок' in studio.spoken(us[0])
        assert studio.spoken(us[1])=='Дверной замок.'
        assert client.post('/api/demo',headers={'origin':'https://foreign.example'}).status_code==403
        assert client.get('/api/config',headers={'host':'foreign.example'}).status_code==400
        assert client.get('/api/files/not-a-valid-file.wav').status_code==404


def test_queued_job_survives_restart(tmp_path):
    app=create_app(settings(tmp_path))
    studio=app.state.studio
    book=studio.import_book('Спокойное море.'.encode(),'book.txt')
    job=studio.enqueue('analyze',book['id'])
    job['status']='RUNNING'
    studio.repo.put('job',job)
    with TestClient(create_app(settings(tmp_path))) as client:
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            status=client.get('/api/jobs/'+job['id']).json()
            if status['status']=='COMPLETED':
                break
            time.sleep(.03)
        assert status['status']=='COMPLETED',status
        assert client.get('/api/books/'+book['id']).json()['status']=='ANALYZED'
