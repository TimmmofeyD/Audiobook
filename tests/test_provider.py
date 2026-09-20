import asyncio
import base64
import json

import httpx
import pytest

from app.providers import DemoAdapter, ElevenLabsAdapter, ProviderError
from app.providers.demo import tone_wav
from app.providers.elevenlabs import load_limits, plan_blocks, render_text


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def mock_http(monkeypatch):
    original = httpx.AsyncClient

    def install(handler):
        transport = httpx.MockTransport(handler)
        monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(transport=transport, **kwargs))
    return install


def test_official_voice_design_and_save_contract(mock_http):
    requests = []

    def handler(request):
        requests.append(request)
        assert request.headers["xi-api-key"] == "test-key"
        payload = json.loads(request.content)
        if request.url.path.endswith("/design"):
            assert payload["model_id"] == "eleven_ttv_v3"
            assert payload["auto_generate_text"] is False
            assert 100 <= len(payload["text"]) <= 1000
            return httpx.Response(200, json={"previews": [{"audio_base_64": base64.b64encode(tone_wav()).decode(), "generated_voice_id": "test_candidate", "media_type": "audio/wav", "duration_secs": 1.2}]})
        assert request.url.path == "/v1/text-to-voice"
        assert payload == {"generated_voice_id": "test_candidate", "voice_name": "Анна", "voice_description": "Soft literary voice"}
        return httpx.Response(200, json={"voice_id": "saved_anna", "name": "Анна"})

    mock_http(handler)
    adapter = ElevenLabsAdapter("test-key")
    previews = run(adapter.design_voice({"description": "Warm natural Russian female voice for literary narration."}, "Проверка голоса персонажа. " * 6))
    assert previews[0]["audio_base64"]
    assert run(adapter.save_voice(previews[0]["generated_voice_id"], "Анна", "Soft literary voice"))["voice_id"] == "saved_anna"
    assert len(requests) == 2


def test_tts_dialogue_and_transcription_payloads(mock_http):
    seen = []

    def handler(request):
        seen.append(request)
        if request.url.path == "/v1/speech-to-text":
            assert request.headers["content-type"].startswith("multipart/form-data")
            assert b"scribe_v2" in request.content
            assert b'name="file"; filename="test.wav"' in request.content
            assert b'name="tag_audio_events"\r\n\r\nfalse' in request.content
            return httpx.Response(200, json={"text": "Тихо, Анна."})
        payload = json.loads(request.content)
        assert payload["seed"] == 42
        assert payload["model_id"] == "eleven_v3"
        assert request.url.params["output_format"] == "mp3_44100_128"
        if request.url.path.endswith("/text-to-dialogue"):
            assert payload["inputs"] == [{"text": "[whispers] Тихо.", "voice_id": "anna"}, {"text": "Я здесь.", "voice_id": "boris"}]
        else:
            assert request.url.path == "/v1/text-to-speech/anna"
            assert payload["text"] == "[whispers] Тихо."
        return httpx.Response(200, content=b"ID3-test-audio", headers={"content-type": "audio/mpeg", "request-id": "req42", "character-cost": "10"})

    mock_http(handler)
    adapter = ElevenLabsAdapter("test-key")
    single = run(adapter.synthesize_single("Тихо.", "anna", {"audio_tags": ["whispers"]}, 42))
    dialogue = run(adapter.synthesize_dialogue([{"text": "Тихо.", "voice_id": "anna", "performance": {"audio_tags": ["whispers"]}}, {"text": "Я здесь.", "voice_id": "boris"}], 42))
    assert single.request_id == dialogue.request_id == "req42"
    assert single.parameters["voice_id"] == "anna"
    assert single.parameters["character_cost"] == "10"
    assert "test-key" not in json.dumps(single.parameters)
    assert run(adapter.transcribe(tone_wav(), "test.wav")) == "Тихо, Анна."
    assert len(seen) == 3


def test_guards_run_before_network(mock_http):
    mock_http(lambda request: pytest.fail("Invalid request must not reach provider"))
    adapter = ElevenLabsAdapter("test-key")
    with pytest.raises(ProviderError, match="аудиотеги"):
        run(adapter.synthesize_single("Слова", "voice", {"audio_tags": ["invented-tag"]}, 0))
    with pytest.raises(ProviderError, match="dialogue_chars"):
        run(adapter.synthesize_dialogue([{"text": "а" * 1995, "voice_id": "voice", "performance": {"audio_tags": ["whispers"]}}], 0))
    with pytest.raises(ProviderError, match="число"):
        run(adapter.synthesize_dialogue([{"text": "Слова", "voice_id": f"voice{i}"} for i in range(11)], 0))
    with pytest.raises(ProviderError, match="seed"):
        run(adapter.synthesize_single("Слова", "voice", {}, -1))
    with pytest.raises(ProviderError, match="только"):
        run(adapter.synthesize_single("Слова", "voice", {"audio_tags": ["whispers"]}, 0, "eleven_multilingual_v2"))
    with pytest.raises(ProviderError, match="SSML"):
        run(adapter.synthesize_single('Слова <break time="1s"/>', "voice", {}, 0))
    with pytest.raises(ProviderError, match="скобки"):
        run(adapter.synthesize_single("[unapproved command] Слова", "voice", {}, 0))
    with pytest.raises(ProviderError, match="прослушивания"):
        run(adapter.design_voice({"description": "Natural clear literary narration voice"}, "Коротко"))


def test_configured_chunking_preserves_lines_and_counts_tags(tmp_path):
    limits_file = tmp_path / "limits.json"
    limits_file.write_text(json.dumps({"dialogue_chars": 30, "dialogue_voices": 2}), encoding="utf-8")
    limits = load_limits(limits_file)
    lines = [{"text": "Длинная фраза.", "voice_id": "a"}, {"text": "Ответ.", "voice_id": "b", "performance": {"audio_tags": ["whispers"]}}, {"text": "Ещё реплика.", "voice_id": "c"}]
    blocks = plan_blocks(lines, limits)
    assert [line for block in blocks for line in block] == lines
    assert len(blocks) >= 2
    assert all(sum(len(render_text(l["text"], l.get("performance", {}), limits)) for l in block) <= 30 for block in blocks)


def test_rate_limit_retry_metadata_and_ambiguous_failure(mock_http, monkeypatch):
    attempts = []
    waits = []

    async def sleep(seconds):
        waits.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", sleep)

    def handler(request):
        attempts.append(request)
        if len(attempts) == 1:
            return httpx.Response(429, headers={"retry-after": "0"})
        return httpx.Response(200, content=b"ID3-audio", headers={"content-type": "audio/mpeg"})

    mock_http(handler)
    artifact = run(ElevenLabsAdapter("test-key").synthesize_single("Привет", "voice", {}, 10))
    assert artifact.parameters["http_attempts"] == 2
    assert waits == [0.0]


def test_ambiguous_post_is_not_retried_and_secrets_are_redacted(mock_http):
    attempts = []

    def handler(request):
        attempts.append(request)
        return httpx.Response(503, json={"detail": {"status": "test-key unavailable"}}, headers={"request-id": "server-id"})

    mock_http(handler)
    with pytest.raises(ProviderError) as caught:
        run(ElevenLabsAdapter("test-key").synthesize_single("Привет", "voice", {}, 10))
    assert not caught.value.retryable
    assert "test-key" not in str(caught.value)
    assert "неизвестен" in str(caught.value)
    assert len(attempts) == 1


def test_demo_is_explicit_non_speech_and_has_no_fake_asr():
    adapter = DemoAdapter()
    candidates = run(adapter.design_voice({}, ""))
    assert len(candidates) == 3
    assert all("ДЕМО" in c["description"] for c in candidates)
    artifact = run(adapter.synthesize_single("Любые слова", candidates[0]["generated_voice_id"], {}, 2))
    assert artifact.audio.startswith(b"RIFF")
    assert artifact.model_id == "demo-tone-v1"
    assert artifact.parameters["speech"] is False
    with pytest.raises(ProviderError, match="ASR"):
        run(adapter.transcribe(artifact.audio, "tone.wav"))
