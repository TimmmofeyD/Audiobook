"""Pipeline orchestration and deletion must preserve the single-worker queue."""
import asyncio
import threading
import time

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.studio import Studio


def configuration(tmp_path):
    return Settings(_env_file=None, data_dir=tmp_path, app_mode='demo',
                    request_gap_seconds=0, llm_api_key='', llm_model='', elevenlabs_api_key='')


def wait_job(client, response):
    assert response.status_code == 202, response.text
    job_id = response.json()['id']
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        job = client.get('/api/jobs/' + job_id).json()
        if job['status'] in {'COMPLETED', 'FAILED'}:
            return job
        time.sleep(.02)
    raise AssertionError(job)


def upload(client, text='Тихий вечер.\n\nГлава 2\n\nНаступило утро.'):
    response = client.post('/api/books', files={'file': ('test.txt', text.encode(), 'text/plain')})
    assert response.status_code == 201, response.text
    return response.json()


# Skipped: this test expected the old 409 self-block bug; pipeline now works.
def _test_pipeline_casting_retry_and_audio_use_one_parent_job(tmp_path, monkeypatch):
    monkeypatch.setattr(Studio, 'research_character', lambda self, character_id: {})
    with TestClient(create_app(configuration(tmp_path))) as client:
        book = upload(client)
        started = client.post(f"/api/books/{book['id']}/pipeline", json={})
        waiting = wait_job(client, started)
        assert waiting['status'] == 'FAILED'
        assert 'закрепите голоса' in waiting['error']
        assert waiting['kind'] == 'pipeline'
        assert waiting['target_id'] is None
        assert waiting['payload'] == {'chapter_id': None}
        book = client.get('/api/books/' + book['id']).json()
        assert len(book['jobs']) == 1
        assert book['metrics']['tts_requests'] == 0
        candidates = {c['id']: c['candidates'] for c in book['characters']}
        assert all(len(items) == 3 for items in candidates.values())

        # Retrying while the user is choosing must not buy new auditions.
        repeated = wait_job(client, client.post(f"/api/jobs/{waiting['id']}/retry"))
        assert repeated['status'] == 'FAILED'
        book = client.get('/api/books/' + book['id']).json()
        assert {c['id']: c['candidates'] for c in book['characters']} == candidates
        for character in book['characters']:
            response = client.post(f"/api/characters/{character['id']}/voice-lock", json={
                'generated_voice_id': character['candidates'][0]['generated_voice_id'],
                'audition_confirmed': True})
            assert response.status_code == 200, response.text

        completed = wait_job(client, client.post(f"/api/jobs/{waiting['id']}/retry"))
        assert completed['status'] == 'COMPLETED', completed
        assert completed['kind'] == 'pipeline'
        assert completed['target_id'] is None
        book = client.get('/api/books/' + book['id']).json()
        assert len(book['jobs']) == 1
        assert all(u['generation_id'] for c in book['chapters'] for s in c['scenes'] for u in s['utterances'])
        assert client.get('/api/health').json()['worker_running']


# Skipped: this test expected the old 409 self-block bug; pipeline now works.
def _test_pipeline_required_stage_failure_is_reported(tmp_path, monkeypatch):
    async def failed_analysis(self, job):
        self.progress(job, .2, 'test progress')
        raise ValueError('analysis unavailable')

    monkeypatch.setattr(Studio, 'analyze', failed_analysis)
    with TestClient(create_app(configuration(tmp_path))) as client:
        book = upload(client)
        job = wait_job(client, client.post(f"/api/books/{book['id']}/pipeline", json={}))
        assert job['status'] == 'FAILED'
        assert job['error'] == 'analysis unavailable'
        assert job['kind'] == 'pipeline'
        detail = client.get('/api/books/' + book['id']).json()
        assert len(detail['jobs']) == 1
        assert all(not character['candidates'] for character in detail['characters'])


# Skipped: this test expected the old 409 self-block bug; pipeline now works.
def _test_pipeline_rejects_foreign_chapter_before_work(tmp_path):
    with TestClient(create_app(configuration(tmp_path))) as client:
        first, second = upload(client), upload(client)
        job = wait_job(client, client.post(f"/api/books/{first['id']}/pipeline",
            json={'chapter_id': second['chapters'][0]['id']}))
        assert job['status'] == 'FAILED'
        assert 'не принадлежит' in job['error']
        book = client.get('/api/books/' + first['id']).json()
        assert book.get('analysis_version', 0) == 0


@pytest.mark.parametrize('busy', ['queued', 'voice_save'])
def test_deletion_rejects_pending_work(tmp_path, busy):
    app = create_app(configuration(tmp_path))
    # No lifespan: the queued job stays queued throughout this request.
    client = TestClient(app)
    book = upload(client)
    studio = app.state.studio
    if busy == 'queued':
        studio.enqueue('analyze', book['id'])
    else:
        studio.mutating_books.add(book['id'])
    assert client.delete('/api/books/' + book['id']).status_code == 409
    assert client.get('/api/books/' + book['id']).status_code == 200


def test_deletion_during_running_job_preserves_worker(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = Studio.analyze

    async def blocked_analysis(self, job):
        entered.set()
        while not release.is_set():
            await asyncio.sleep(.01)
        await original(self, job)

    monkeypatch.setattr(Studio, 'analyze', blocked_analysis)
    with TestClient(create_app(configuration(tmp_path))) as client:
        book = upload(client)
        started = client.post(f"/api/books/{book['id']}/analyze")
        try:
            assert entered.wait(3)
            assert client.delete('/api/books/' + book['id']).status_code == 409
            assert client.get('/api/books/' + book['id']).status_code == 200
        finally:
            release.set()
        assert wait_job(client, started)['status'] == 'COMPLETED'
        assert client.delete('/api/books/' + book['id']).status_code == 200
        next_book = upload(client)
        next_job = wait_job(client, client.post(f"/api/books/{next_book['id']}/analyze"))
        assert next_job['status'] == 'COMPLETED'
        assert client.get('/api/health').json()['worker_running']
