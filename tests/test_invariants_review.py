"""Regression checks for editing boundaries and durable paid-audio checkpoints."""
import asyncio
import time

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.providers.demo import DemoAdapter
from app.services.audio import AudioEngine
from app.studio import Studio


def configuration(tmp_path):
    return Settings(_env_file=None, data_dir=tmp_path, app_mode="demo",
                    request_gap_seconds=0, llm_api_key="", llm_model="",
                    elevenlabs_api_key="", include_narrator_in_dialogue=True)


def units(book):
    return [u for chapter in book["chapters"] for scene in chapter["scenes"] for u in scene["utterances"]]


def finish(client, response):
    assert response.status_code == 202, response.text
    job_id = response.json()["id"]
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        job = client.get("/api/jobs/" + job_id).json()
        if job["status"] in {"COMPLETED", "FAILED"}:
            assert job["status"] == "COMPLETED", job
            return job
        time.sleep(.02)
    raise AssertionError(f"Job did not complete: {job}")


def test_shortening_line_preserves_complete_active_dialogue_artifacts(tmp_path):
    with TestClient(create_app(configuration(tmp_path))) as client:
        text = "\n\n".join(letter * 900 for letter in "АБВГ")
        response = client.post("/api/books", files={"file": ("boundaries.txt", text.encode(), "text/plain")})
        assert response.status_code == 201, response.text
        book = response.json()
        assert len(units(book)) == 4
        scene_id = book["chapters"][0]["scenes"][0]["id"]
        narrator_id = book["characters"][0]["id"]
        locked = client.post(f"/api/characters/{narrator_id}/voice-lock", json={"voice_id": "demo-narrator", "audition_confirmed": True})
        assert locked.status_code == 200, locked.text
        finish(client, client.post(f"/api/scenes/{scene_id}/generate-preview", json={}))
        book = client.get("/api/books/" + book["id"]).json()
        first_ids = {u["generation_id"] for u in units(book)}
        assert len(first_ids) == 2
        edited = client.patch("/api/utterances/" + units(book)[1]["id"], json={"spoken_text": "Б" * 100})
        assert edited.status_code == 200, edited.text
        finish(client, client.post(f"/api/scenes/{scene_id}/generate-preview", json={}))
        book = client.get("/api/books/" + book["id"]).json()
        active = units(book)
        for generation_id in {u["generation_id"] for u in active}:
            record = client.get("/api/generations/" + generation_id).json()
            selected = {u["id"] for u in active if u["generation_id"] == generation_id}
            assert selected == set(record["utterance_ids"]), "An active dialogue artifact cannot be selected for only part of its text."
            approved = client.post(f"/api/generations/{generation_id}/approve", json={"approved": True})
            assert approved.status_code == 200, approved.text
        # Approval cannot silently re-select an old overlapping dialogue block.
        after = client.get("/api/books/" + book["id"]).json()
        assert [u["generation_id"] for u in units(after)] == [u["generation_id"] for u in active]
        finish(client, client.post(f"/api/books/{book['id']}/export", json={"formats": ["wav"]}))


def test_partial_performance_patch_preserves_other_directing_fields(tmp_path):
    with TestClient(create_app(configuration(tmp_path))) as client:
        response = client.post("/api/books", files={"file": ("performance.txt", "Она тихо закрыла дверь.".encode(), "text/plain")})
        assert response.status_code == 201, response.text
        utterance = units(response.json())[0]
        performance = {**utterance["performance"], "audio_tags": ["whispers"], "delivery": "quiet, measured", "intent": "conceal presence"}
        updated = client.patch("/api/utterances/" + utterance["id"], json={"performance": performance})
        assert updated.status_code == 200, updated.text
        updated = client.patch("/api/utterances/" + utterance["id"], json={"performance": {"pause_after_ms": 850}})
        assert updated.status_code == 200, updated.text
        actual = updated.json()["performance"]
        assert actual == {**performance, "pause_after_ms": 850}


def test_interrupted_qc_reuses_persisted_audio_after_restart(tmp_path, monkeypatch):
    settings = configuration(tmp_path)
    studio = Studio(settings)
    book = studio.import_book("Одна короткая спокойная фраза.".encode(), "checkpoint.txt")
    narrator = studio.repo.list("character", book["id"])[0]
    narrator["voice"] = {"voice_id": "demo-checkpoint", "provider": "demo", "status": "LOCKED", "version": 1}
    studio.repo.put("character", narrator)
    scene_id = book["chapters"][0]["scenes"][0]["id"]
    job = studio.enqueue("preview", book["id"], scene_id)
    job["status"] = "RUNNING"
    studio.repo.put("job", job)
    call_count = 0
    synthesize = DemoAdapter.synthesize_single
    inspect = AudioEngine.inspect

    async def counted_synthesize(self, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        return await synthesize(self, *args, **kwargs)

    def interrupted_inspection(self, path):
        raise asyncio.CancelledError("simulated shutdown during acoustic QC")

    monkeypatch.setattr(DemoAdapter, "synthesize_single", counted_synthesize)
    monkeypatch.setattr(AudioEngine, "inspect", interrupted_inspection)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(studio.generate(job))
    records = studio.repo.list("generation", book["id"])
    assert len(records) == 1, "Persist returned audio before awaiting acoustic or ASR QC."
    pending = records[0]
    assert pending["qc"].get("pending") is True
    assert (studio.files / pending["audio_file"]).is_file()
    assert studio.repo.list("utterance", book["id"])[0]["generation_id"] == pending["id"]
    assert call_count == 1

    monkeypatch.setattr(AudioEngine, "inspect", inspect)
    # Reopening the application loads the RUNNING job from disk and resumes QC.
    with TestClient(create_app(settings)) as client:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            recovered = client.get("/api/jobs/" + job["id"]).json()
            if recovered["status"] in {"COMPLETED", "FAILED"}:
                break
            time.sleep(.02)
        assert recovered["status"] == "COMPLETED", recovered
        detail = client.get("/api/books/" + book["id"]).json()
        assert units(detail)[0]["generation_id"] == pending["id"]
        generation = client.get("/api/generations/" + pending["id"]).json()
        assert not generation["qc"].get("pending")
        assert generation["qc"]["status"] == "REVIEW"
        assert call_count == 1, "Restarting QC must not issue another paid TTS request."
