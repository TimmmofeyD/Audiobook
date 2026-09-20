import asyncio
import json
from copy import deepcopy

import httpx
import pytest

from app.agents.pipeline import AnalysisPipeline, AnalysisResponse, Character, DirectorResponse


def scene(scene_id="s1", uid="u1", text="— Здравствуй, — сказала Анна."):
    return {"id": scene_id, "title": "Сцена", "utterances": [{"id": uid, "source_text": text, "spoken_text": text, "type": "UNKNOWN"}]}


def response(scene_id="s1", uid="u1", *, characters=None, speaker="Анна"):
    if characters is None:
        characters = [Character(canonical_name="Анна", evidence=[uid]).model_dump(exclude_none=True)]
    return {"characters": characters, "attributions": [{"utterance_id": uid, "character_name": speaker, "confidence": 0.99 if speaker else 0.0,
            "evidence": [uid] if speaker else [], "reason_code": "EXPLICIT_SPEECH_CUE" if speaker else "AMBIGUOUS_SPEAKER"}],
            "scene_summaries": [{"scene_id": scene_id, "summary": "Анна приветствует собеседника."}], "events": []}


def stub(pipeline, output, calls=None):
    async def request(version, payload, schema):
        if calls is not None:
            calls.append(deepcopy(payload))
        raw = output(payload) if callable(output) else output
        return schema.model_validate(raw)
    pipeline._request = request


def live():
    return AnalysisPipeline("https://llm.example/v1", "test-secret", "test-model", "llm")


def test_manual_keeps_ambiguous_speakers_unknown_and_does_not_invent_cast():
    narrator = {"id": "n", "canonical_name": "Рассказчик", "role": "narrator", "gender": {"value": None, "confidence": 0}, "voice": {"voice_id": "locked-voice"}}
    input_scene = scene(text="— Ты придёшь?\n— Не знаю.")
    input_scene["utterances"].append({"id": "n1", "source_text": "Ночь наступила.", "spoken_text": "Ночь наступила.", "type": "NARRATOR"})
    original = deepcopy(input_scene)
    result = asyncio.run(AnalysisPipeline().analyze({}, [narrator], [input_scene]))
    assert result["mode"] == "manual"
    assert [char["canonical_name"] for char in result["characters"]] == ["Рассказчик"]
    assert result["characters"][0]["id"] == "n"
    assert result["attributions"][0]["character_name"] is None
    assert result["attributions"][1]["character_name"] == "Рассказчик"
    assert input_scene == original


def test_manual_does_not_promote_prior_uncertain_attribution_to_human_override():
    current = scene()
    current["utterances"][0].update(speaker_id="a", speaker_confidence=0.55)
    characters = [{"id": "a", "canonical_name": "Анна"}]
    result = asyncio.run(AnalysisPipeline().analyze({}, characters, [current]))
    assert result["attributions"][0]["character_name"] is None
    current["utterances"][0]["manual_speaker"] = True
    result = asyncio.run(AnalysisPipeline().analyze({}, characters, [current]))
    assert result["attributions"][0]["character_name"] == "Анна"


def test_llm_canonical_memory_and_ids_persist_across_scenes():
    pipeline = live()
    calls = []
    existing = Character(id="anna-id", canonical_name="Анна", aliases=["Аня"]).model_dump(exclude_none=True)
    def output(payload):
        current = payload["current_scene"]
        return response(current["id"], current["utterances"][0]["id"], characters=[existing])
    stub(pipeline, output, calls)
    result = asyncio.run(pipeline.analyze({}, [existing], [scene(), scene("s2", "u2", "— Да, — ответила Аня.")]))
    assert len([item for item in result["characters"] if item["canonical_name"] == "Анна"]) == 1
    assert result["characters"][0]["canonical_name"] == "Рассказчик"
    anna = next(item for item in calls[1]["characters"] if item["canonical_name"] == "Анна")
    assert anna["id"] == "anna-id"
    assert calls[1]["previous_summaries"][0]["scene_id"] == "s1"


@pytest.mark.parametrize("mutation", [
    lambda output: output["characters"][0].update(id="invented-id"),
    lambda output: output["attributions"][0].update(character_name="Михаил"),
    lambda output: output["attributions"][0].update(evidence=["unknown-ref"]),
    lambda output: output["characters"][0].update(evidence=[]),
    lambda output: output["characters"][0].update(canonical_name="Борис"),
    lambda output: output["scene_summaries"][0].update(scene_id="other-scene"),
    lambda output: output["attributions"].append(deepcopy(output["attributions"][0])),
    lambda output: output["attributions"].clear(),
    lambda output: output["attributions"][0].update(character_name=None, confidence=0.9),
    lambda output: output["characters"][0].update(voice_id="forbidden"),
])
def test_analysis_fails_closed_on_bad_schema_or_references(mutation):
    pipeline = live()
    output = response()
    mutation(output)
    stub(pipeline, output)
    with pytest.raises(ValueError):
        asyncio.run(pipeline.analyze({}, [], [scene()]))


def test_alias_collision_and_demographic_drift_rejected():
    pipeline = live()
    existing = Character(id="anna-id", canonical_name="Анна", aliases=["Аня"], gender={"value": "female", "confidence": 0.99}).model_dump(exclude_none=True)
    output = response(characters=[Character(canonical_name="Аня", evidence=["u1"]).model_dump(exclude_none=True)], speaker="Аня")
    stub(pipeline, output)
    with pytest.raises(ValueError, match="alias"):
        asyncio.run(pipeline.analyze({}, [existing], [scene(text="Аня заговорила.")]))
    changed = deepcopy(existing)
    changed["gender"] = {"value": "male", "confidence": 0.95}
    stub(pipeline, response(characters=[changed]))
    with pytest.raises(ValueError, match="характеристику"):
        asyncio.run(pipeline.analyze({}, [existing], [scene()]))


def test_existing_character_cannot_steal_another_alias():
    pipeline = live()
    anna = Character(id="a", canonical_name="Анна").model_dump(exclude_none=True)
    boris = Character(id="b", canonical_name="Борис", aliases=["Боря"]).model_dump(exclude_none=True)
    candidate = {**anna, "aliases": ["Боря"], "evidence": ["u1"]}
    stub(pipeline, response(characters=[candidate]))
    with pytest.raises(ValueError, match="Alias"):
        asyncio.run(pipeline.analyze({}, [anna, boris], [scene()]))


def test_reanalysis_accepts_persisted_evidence_from_later_chapters():
    pipeline = live()
    anna = Character(id="a", canonical_name="Анна", evidence=["u2"]).model_dump(exclude_none=True)
    def output(payload):
        current = payload["current_scene"]
        return response(current["id"], current["utterances"][0]["id"], characters=[anna])
    stub(pipeline, output)
    result = asyncio.run(pipeline.analyze({}, [anna], [scene(), scene("s2", "u2")]))
    assert next(item for item in result["characters"] if item["canonical_name"] == "Анна")["evidence"] == ["u2"]


def test_bounded_analysis_processes_every_utterance_without_silent_truncation():
    pipeline = live()
    calls = []
    current = scene()
    current["utterances"] = [{"id": f"u{index}", "source_text": "Неизвестная реплика. " * 60, "type": "UNKNOWN"} for index in range(63)]
    def output(payload):
        data = payload["current_scene"]
        return {"characters": [], "attributions": [{"utterance_id": item["id"], "character_name": None, "confidence": 0.0, "evidence": [], "reason_code": "AMBIGUOUS_SPEAKER"} for item in data["utterances"]],
                "scene_summaries": [{"scene_id": data["id"], "summary": "Неизвестные говорящие."}], "events": []}
    stub(pipeline, output, calls)
    result = asyncio.run(pipeline.analyze({}, [], [current]))
    assert len(result["attributions"]) == 63
    assert len(calls) > 1
    assert all(sum(len(item["source_text"]) for item in call["current_scene"]["utterances"]) <= 14000 for call in calls)


def test_director_keeps_originals_and_rejects_rewrite_foreign_id_and_tags():
    current = scene()
    original = deepcopy(current)
    plans = asyncio.run(AnalysisPipeline().direct(current, []))
    assert plans[0]["utterance_id"] == "u1"
    assert plans[0]["audio_tags"] == []
    assert current == original
    pipeline = live()
    for edits in [{"utterance_id": "unknown"}, {"spoken_text": "Переписанный текст"}, {"audio_tags": ["<break/>"]}, {"emotion_internal": {"anger": 1.5}}, {"pause_after_ms": -1}]:
        invalid = [{**plans[0], **edits}]
        stub(pipeline, {"plans": invalid})
        with pytest.raises(ValueError):
            asyncio.run(pipeline.direct(current, []))


def test_http_contract_and_truncated_or_malformed_responses(monkeypatch):
    real_client = httpx.AsyncClient
    seen = []
    current_response = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(response())}}]}
    def handle(request):
        seen.append(request)
        return httpx.Response(200, json=current_response)
    transport = httpx.MockTransport(handle)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: real_client(transport=transport, **kwargs))
    pipeline = live()
    result = asyncio.run(pipeline.analyze({}, [], [scene()]))
    assert result["mode"] == "llm"
    request_body = json.loads(seen[0].content)
    assert seen[0].url.path == "/v1/chat/completions"
    assert request_body["response_format"] == {"type": "json_object"}
    assert "untrusted" in request_body["messages"][0]["content"]
    current_response["choices"][0]["finish_reason"] = "length"
    with pytest.raises(ValueError, match="не завершил"):
        asyncio.run(pipeline.analyze({}, [], [scene()]))
    current_response["choices"][0]["finish_reason"] = "stop"
    current_response["choices"][0]["message"]["content"] = 'not JSON'
    with pytest.raises(ValueError, match="JSON"):
        asyncio.run(pipeline.analyze({}, [], [scene()]))


def test_http_errors_do_not_expose_provider_body_or_key(monkeypatch):
    real_client = httpx.AsyncClient
    transport = httpx.MockTransport(lambda _: httpx.Response(401, text="test-secret should never escape"))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: real_client(transport=transport, **kwargs))
    with pytest.raises(ValueError, match="HTTP 401") as caught:
        asyncio.run(live().analyze({}, [], [scene()]))
    assert "test-secret" not in str(caught.value)


def test_unsafe_remote_http_rejected_but_local_servers_supported():
    with pytest.raises(ValueError, match="HTTPS"):
        AnalysisPipeline("http://remote.example/v1", "secret", "model", "llm")
    assert AnalysisPipeline("http://localhost:11434/v1", "", "local-model", "llm").configured
