"""Stateless services; the orchestrator persists canonical identities and results.

Uses a generic Chat Completions-compatible HTTP endpoint. JSON mode is combined
with strict local schema/reference validation; no model output is trusted as code.
See https://developers.openai.com/api/docs/guides/structured-outputs .
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

ANALYSIS_PROMPT_VERSION = "book_analyst_v1"
DIRECTOR_PROMPT_VERSION = "scene_director_v1"
PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"
MAX_BATCH_CHARS = 14_000
MAX_BATCH_UTTERANCES = 8
LLM_RETRY_ATTEMPTS = 3
MAX_REQUEST_CHARS = 160_000
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
ALLOWED_AUDIO_TAGS = {"whispers", "shouting", "laughs", "sighs", "crying", "excited", "sad", "angry", "curious", "sarcastic", "thoughtful", "calm"}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Fact(StrictModel):
    value: str = Field(min_length=1, max_length=80)
    confidence: float = Field(ge=0, le=1)

    @field_validator("value", mode="before")
    @classmethod
    def fact_value(cls, value):
        # Local models sometimes emit empty strings; treat as unknown, not an error
        return value if isinstance(value, str) and value.strip() else "unknown"


class Character(StrictModel):
    id: str | None = None
    canonical_name: str = Field(min_length=1, max_length=160)

    @model_validator(mode="before")
    @classmethod
    def normalize_local_model_output(cls, data):
        if isinstance(data, dict):
            # Drop known hallucinated extra fields from local/cloud models
            for junk in ("evidence_note", "note", "comment", "notes"):
                data.pop(junk, None)
            # Empty fact objects from local models become unknown facts
            for field in ("gender", "age_band"):
                fact = data.get(field)
                if isinstance(fact, dict) and not fact.get("value", "").strip():
                    data[field] = {"value": "unknown", "confidence": 0.0}
            name = data.get("canonical_name", "")
            if isinstance(name, str):
                key = name.strip().lower()
                if key in {"narrator", "the narrator", "narrator voice"}:
                    data["canonical_name"] = "Рассказчик"
                elif key == "norlev":
                    data["canonical_name"] = "Норлев"
        return data
    aliases: list[str] = Field(default_factory=list, max_length=40)
    gender: Fact = Field(default_factory=lambda: Fact(value="unknown", confidence=0.0))
    age_band: Fact = Field(default_factory=lambda: Fact(value="unknown", confidence=0.0))
    role: str = Field(default="unknown", max_length=120)
    baseline_traits: list[str] = Field(default_factory=list, max_length=20)
    speech_profile: dict[str, str] = Field(default_factory=dict)
    evidence: list[str] = Field(default_factory=list, max_length=100)

    @field_validator("canonical_name")
    @classmethod
    def clean_name(cls, value):
        if value != value.strip() or not value.strip():
            raise ValueError("canonical_name must not contain outer whitespace")
        return value

    @field_validator("aliases", "baseline_traits", "evidence")
    @classmethod
    def bounded_strings(cls, values):
        if any(not value.strip() or len(value) > 200 for value in values):
            raise ValueError("Empty or oversized list item")
        return list(dict.fromkeys(values))


class Attribution(StrictModel):
    utterance_id: str = Field(min_length=1)
    character_name: str | None
    confidence: float = Field(ge=0, le=1)
    evidence: list[str] = Field(default_factory=list, max_length=30)
    reason_code: str = Field(min_length=1, max_length=400)


class SceneSummary(StrictModel):
    model_config = ConfigDict(extra="ignore", strict=True, allow_inf_nan=False)
    scene_id: str
    summary: str = Field(max_length=2000)


class Event(StrictModel):
    scene_id: str
    participants: list[str] = Field(default_factory=list, max_length=30)
    summary: str = Field(min_length=1, max_length=2000)
    evidence: list[str] = Field(min_length=1, max_length=30)


class AnalysisResponse(StrictModel):
    characters: list[Character] = Field(max_length=100)
    attributions: list[Attribution]
    scene_summaries: list[SceneSummary]
    events: list[Event] = Field(default_factory=list, max_length=50)


class Performance(StrictModel):
    utterance_id: str
    intent: str = Field(max_length=300)
    emotion_internal: dict[str, float] = Field(default_factory=dict)
    emotion_expressed: dict[str, float] = Field(default_factory=dict)
    delivery: str = Field(max_length=1000)
    audio_tags: list[str] = Field(default_factory=list, max_length=3)
    pause_before_ms: int = Field(default=0, ge=0, le=5000)
    pause_after_ms: int = Field(default=350, ge=0, le=5000)

    @field_validator("audio_tags")
    @classmethod
    def tags(cls, values):
        if any(value not in ALLOWED_AUDIO_TAGS for value in values):
            raise ValueError("Unsupported audio tag")
        return list(dict.fromkeys(values))

    @field_validator("emotion_internal", "emotion_expressed")
    @classmethod
    def emotions(cls, values):
        if len(values) > 8 or any(not key or len(key) > 50 or not math.isfinite(value) or not 0 <= value <= 1 for key, value in values.items()):
            raise ValueError("Invalid emotion intensity")
        return values


class DirectorResponse(StrictModel):
    plans: list[Performance]


def _key(value: str) -> str:
    # Normalize for entity comparison: case, ё/e, apostrophes, hyphens, spaces.
    v = unicodedata.normalize("NFKC", value).casefold()
    v = v.replace("ё", "е").replace("й", "и")
    v = re.sub(r"[\u2019\u02bc'`\u2010-\u2015-]+", "", v)
    return re.sub(r"\s+", " ", v).strip()


def _character(value: dict) -> dict:
    # DB character objects also contain voice registry fields. They are deliberately
    # excluded from the LLM contract so the LLM cannot alter them.
    allowed = Character.model_fields.keys()
    clean = {key: deepcopy(item) for key, item in value.items() if key in allowed}
    for field in ("gender", "age_band"):
        if field in clean and clean[field].get("value") is None:
            clean[field] = {"value": "unknown", "confidence": 0.0}
    return Character.model_validate(clean).model_dump(exclude_none=True)


def _narrator() -> dict:
    return Character(canonical_name="Рассказчик", role="narrator").model_dump(exclude_none=True)


def _characters(values: list[dict]) -> list[dict]:
    result = [_character(value) for value in values]
    names = [_key(value["canonical_name"]) for value in result]
    ids = [value["id"] for value in result if value.get("id")]
    if len(set(names)) != len(names) or len(set(ids)) != len(ids):
        raise ValueError("Character Bible содержит дубли имён или ID")
    if _key("Рассказчик") not in names:
        result.insert(0, _narrator())
    return result


def _batches(utterances: list[dict]):
    result, size = [], 0
    for utterance in utterances:
        text_size = len(utterance.get("source_text", ""))
        if text_size > MAX_BATCH_CHARS:
            raise ValueError("Слишком длинная реплика для LLM; разделите исходный абзац")
        if result and (size + text_size > MAX_BATCH_CHARS or len(result) >= MAX_BATCH_UTTERANCES):
            yield result
            result, size = [], 0
        result.append(utterance)
        size += text_size
    if result:
        yield result


def _scene_data(scene: dict, utterances: list[dict]) -> dict:
    return {"id": scene["id"], "title": scene.get("title", ""), "summary": scene.get("summary", ""),
            "utterances": [{key: value for key, value in utterance.items() if key in {"id", "source_text", "spoken_text", "type", "speaker_id"}} for utterance in utterances]}


class AnalysisPipeline:
    def __init__(self, base_url: str = "", api_key: str = "", model: str = "", mode: str = "manual"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.mode = mode
        if mode not in {"manual", "llm", "auto"}:
            raise ValueError("LLM mode должен быть manual, auto или llm")
        if self.base_url:
            parsed = urlsplit(self.base_url)
            if parsed.scheme not in {"https", "http"} or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError("Некорректный LLM_BASE_URL")
            if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
                raise ValueError("Для удалённого LLM endpoint требуется HTTPS")
        if mode == "llm" and not (self.base_url and model):
            raise ValueError("LLM_BASE_URL и LLM_MODEL обязательны для режима llm")

    @property
    def configured(self):
        return self.mode != "manual" and bool(self.base_url and self.model)

    async def _request(self, prompt_version: str, payload: dict, schema: type[BaseModel]) -> BaseModel:
        prompt = (PROMPT_DIR / f"{prompt_version}.txt").read_text(encoding="utf-8")
        message = json.dumps({"input": payload, "output_schema": schema.model_json_schema()}, ensure_ascii=False)
        if len(message) > MAX_REQUEST_CHARS:
            raise ValueError("Character Bible и контекст превышают лимит LLM-запроса; требуется сокращение контекста")
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        endpoint = self.base_url if self.base_url.endswith("/chat/completions") else self.base_url + "/chat/completions"
        # Generic JSON mode maximizes compatibility; local Pydantic validation is
        # mandatory, because JSON syntax alone does not establish schema adherence.
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(900, connect=30), follow_redirects=False) as client:
                async with client.stream("POST", endpoint, headers=headers, json={
                    "model": self.model, "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": message}],
                    "response_format": {"type": "json_object"},
                }) as response:
                    if response.status_code >= 400:
                        # Never expose provider response bodies or headers: they can contain secrets.
                        raise ValueError(f"LLM API вернул HTTP {response.status_code}; проверьте конфигурацию/квоту")
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body.extend(chunk)
                        if len(body) > MAX_RESPONSE_BYTES:
                            raise ValueError("Ответ LLM превышает безопасный размер")
        except httpx.HTTPError as exc:
            raise ValueError("LLM endpoint недоступен или истёк таймаут") from exc
        try:
            raw = json.loads(body)
            choice = raw["choices"][0]
            if choice.get("finish_reason") != "stop" or choice["message"].get("refusal"):
                raise ValueError("LLM не завершил структурированный ответ")
            content = choice["message"]["content"]
            if not isinstance(content, str):
                raise ValueError("LLM вернул неподдерживаемый формат")
            # Save last raw response for debugging (no secrets, only model output)
            debug_dir = Path("data/llm_debug")
            debug_dir.mkdir(parents=True, exist_ok=True)
            (debug_dir / "last_response.txt").write_text(content, encoding="utf-8")
            # Try direct parse; fallback: extract JSON from markdown code fences
            try:
                return schema.model_validate_json(content)
            except (json.JSONDecodeError, ValidationError):
                # Strip markdown code fences: ```json ... ```
                cleaned = content.strip()
                if cleaned.startswith("```"):
                    lines = cleaned.split("\n")
                    lines = [l for l in lines if not l.strip().startswith("```")]
                    cleaned = "\n".join(lines).strip()
                # Find outermost { }
                start = cleaned.find("{")
                end = cleaned.rfind("}")
                if start >= 0 and end > start:
                    cleaned = cleaned[start:end+1]
                try:
                    return schema.model_validate_json(cleaned)
                except (json.JSONDecodeError, ValidationError):
                    pass
                # Repair pass 1: truncate duplicated JSON object (model sometimes emits two)
                try:
                    decoder = json.JSONDecoder()
                    obj, idx = decoder.raw_decode(cleaned)
                    if idx < len(cleaned.strip()) - 1:
                        cleaned = json.dumps(obj, ensure_ascii=False)
                    return schema.model_validate_json(cleaned)
                except (json.JSONDecodeError, ValueError):
                    pass
                # Repair pass 2: close unterminated strings/brackets from truncation
                repaired = cleaned
                if repaired.count("{") > repaired.count("}"):
                    repaired += "}" * (repaired.count("{") - repaired.count("}"))
                if repaired.count("[") > repaired.count("]"):
                    repaired += "]" * (repaired.count("[") - repaired.count("]"))
                # Truncate to last complete object when truncated mid-value
                last_brace = repaired.rfind("}")
                if last_brace > 0:
                    candidate = repaired[:last_brace+1]
                    try:
                        obj = json.loads(candidate)
                        return schema.model_validate_json(json.dumps(obj, ensure_ascii=False))
                    except (json.JSONDecodeError, ValidationError):
                        pass
                return schema.model_validate_json(repaired)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError, ValidationError) as exc:
            raise ValueError(f"Ответ LLM не соответствует JSON-контракту: {type(exc).__name__}: {str(exc)[:300]}") from exc

    @staticmethod
    def _validate_references(response: AnalysisResponse, scene_id: str, batch: list[dict], memory: list[dict], known_texts: dict[str, str]) -> list[dict]:
        expected = {utterance["id"] for utterance in batch}
        actual = [item.utterance_id for item in response.attributions]
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError("LLM должен вернуть каждую реплику ровно один раз")
        summaries = response.scene_summaries
        if len(summaries) != 1:
            raise ValueError("LLM вернул неверное число summaries")
        # Local models sometimes alter UUID case or make typos; accept a valid UUID
        # shape and normalize it, but reject non-UUID values as untrusted output.
        # Scene id is normalized: orchestrator owns scene identity. Plain short labels
        # like "1", "scene 1" or a UUID are local-model formatting artifacts, while
        # arbitrary strings remain rejected as untrusted output.
        returned_scene_id = summaries[0].scene_id
        if returned_scene_id != scene_id:
            if re.fullmatch(r"[0-9a-fA-F-]{32,36}|scene[-_ ]?\d+|\d+", returned_scene_id, re.IGNORECASE):
                summaries[0].scene_id = scene_id
            else:
                raise ValueError("LLM вернул неизвестную или пропущенную сцену")
        by_name = {value["canonical_name"]: deepcopy(value) for value in memory}
        by_id = {value["id"]: value["canonical_name"] for value in memory if value.get("id")}
        emitted = set()

        def evidence(refs, required=False):
            if required and not refs:
                raise ValueError("LLM вернул утверждение без evidence")
            if any(ref not in known_texts for ref in refs):
                raise ValueError("LLM ссылается на неизвестное доказательство")

        for item in response.characters:
            # Character evidence may be quotes; normalize to known IDs
            if item.evidence:
                normalized = []
                for ref in item.evidence:
                    if ref in known_texts:
                        normalized.append(ref)
                    else:
                        matches = [uid for uid, text in known_texts.items() if ref and ref in text]
                        if len(matches) == 1:
                            normalized.append(matches[0])
                item.evidence = list(dict.fromkeys(normalized))
            candidate = item.model_dump(exclude_none=True)
            name = candidate["canonical_name"]
            # Local models may use English narrator labels; canonical is Рассказчик
            if _key(name) in {_key("narrator"), _key("the narrator"), _key("narrator voice")}:
                name = candidate["canonical_name"] = "Рассказчик"
            if _key(name) in emitted:
                raise ValueError("LLM вернул дубли персонажей")
            emitted.add(_key(name))
            if item.id and by_id.get(item.id) not in (None, name):
                raise ValueError("LLM создал неизвестный ID или изменил имя персонажа")
            if item.id and by_id.get(item.id) is None:
                # Local models sometimes mangle an existing UUID into another UUID.
                # Entity resolution remains authoritative by canonical_name.
                if re.fullmatch(r"[0-9a-fA-F-]{36}", item.id):
                    candidate.pop("id", None)
                else:
                    raise ValueError("LLM создал некорректный ID персонажа")
            existing = by_name.get(name)
            evidence(item.evidence, required=existing is None)
            for field in ("gender", "age_band"):
                fact = candidate[field]
                if fact["value"] == "unknown" and fact["confidence"] != 0:
                    raise ValueError("Неизвестная характеристика должна иметь confidence=0")
                if fact["value"] != "unknown" and (existing is None or existing[field]["value"] == "unknown"):
                    evidence(item.evidence, required=True)
            if existing is None:
                evidence_text = _key("\n".join(known_texts[ref] for ref in item.evidence))
                aliases = [_key(a) for a in [name, *item.aliases]]
                # Confirmation across three mention forms:
                # 1) full normalized name
                # 2) any significant token (len>=4) — first name OR surname
                # 3) name with apostrophes/hyphens already stripped by _key
                tokens = [t for a in aliases for t in a.split() if len(t) >= 4]
                confirmed = (
                    any(a in evidence_text for a in aliases)
                    or any(t in evidence_text for t in tokens)
                    or any(a.replace(" ", "") in evidence_text.replace(" ", "") for a in aliases)
                )
                if not confirmed:
                    raise ValueError("Имя нового персонажа не подтверждено исходным текстом")
                # A new entity may not steal another canonical name or an alias.
                labels = {_key(name), *(_key(alias) for alias in item.aliases)}
                if any(labels.intersection({_key(old["canonical_name"]), *(_key(alias) for alias in old["aliases"])}) for old in by_name.values()):
                    raise ValueError("Новый персонаж конфликтует с существующей identity/alias")
                by_name[name] = candidate
            else:
                # Persistent identity and established demographic facts are immutable
                # for the analyst. Corrections belong to the explicit human editor.
                for field in ("gender", "age_band"):
                    old = existing[field]
                    new = candidate[field]
                    if old["value"] != "unknown" and new["value"] not in {old["value"], "unknown"}:
                        raise ValueError("LLM изменил устойчивую характеристику персонажа")
                    if old["value"] == "unknown" and new["value"] != "unknown":
                        evidence(item.evidence, required=True)
                        existing[field] = new
                added_aliases = {_key(alias) for alias in candidate["aliases"] if alias not in existing["aliases"]}
                if any(added_aliases.intersection({_key(old["canonical_name"]), *(_key(alias) for alias in old["aliases"])}) for old_name, old in by_name.items() if old_name != name):
                    raise ValueError("Alias персонажа конфликтует с другой identity")
                if added_aliases or (candidate["baseline_traits"] and not existing["baseline_traits"]) or (candidate["speech_profile"] and not existing["speech_profile"]):
                    evidence(item.evidence, required=True)
                existing["aliases"] = list(dict.fromkeys(existing["aliases"] + candidate["aliases"]))
                existing["evidence"] = list(dict.fromkeys(existing["evidence"] + candidate["evidence"]))[-100:]
                if existing["role"] == "unknown":
                    existing["role"] = candidate["role"]
                if not existing["baseline_traits"]:
                    existing["baseline_traits"] = candidate["baseline_traits"]
                if not existing["speech_profile"]:
                    existing["speech_profile"] = candidate["speech_profile"]
        # Local models sometimes mention a character in attributions but omit them
        # from the characters list. Register such characters when the name is present
        # in the evidence-backed source text; otherwise fail closed.
        # Normalize English narrator labels from local models
        for item in response.attributions:
            if item.character_name and _key(item.character_name) in {_key("narrator"), _key("the narrator"), _key("narrator voice")}:
                item.character_name = "Рассказчик"
        # Normalize evidence from local models: keep known utterance IDs, convert
        # quoted text to matching IDs, and always anchor attribution to its own utterance.
        def normalize_evidence(refs, anchor):
            normalized = []
            invalid = []
            for ref in refs or []:
                if ref in known_texts:
                    normalized.append(ref)
                else:
                    matches = [uid for uid, text in known_texts.items() if ref and ref in text]
                    if len(matches) == 1:
                        normalized.append(matches[0])
                    else:
                        invalid.append(ref)
            if invalid:
                # Fail closed on references that are neither IDs nor resolvable quotes
                raise ValueError("LLM ссылается на неизвестное доказательство")
            if not normalized and anchor in known_texts:
                normalized.append(anchor)
            return list(dict.fromkeys(normalized))

        for item in response.attributions:
            item.evidence = normalize_evidence(item.evidence, item.utterance_id)
            if item.character_name is not None and item.character_name not in by_name:
                batch_texts = " ".join(known_texts.get(ref, "") for ref in item.evidence)
                if item.character_name in batch_texts:
                    new_character = Character(canonical_name=item.character_name,
                                              evidence=item.evidence).model_dump(exclude_none=True)
                    by_name[item.character_name] = new_character
                else:
                    raise ValueError("LLM назначил неизвестного персонажа")
            if item.character_name is None and item.confidence != 0:
                raise ValueError("Неизвестный говорящий должен иметь confidence=0")
        for event in response.events:
            if event.scene_id != scene_id or any(name not in by_name for name in event.participants):
                raise ValueError("LLM-событие содержит неизвестную сцену или персонажа")
            evidence(event.evidence, required=True)
        return list(by_name.values())

    async def analyze(self, book: dict, characters: list, scenes: list) -> dict:
        memory = _characters(characters)
        all_utterances = [utterance for scene in scenes for utterance in scene.get("utterances", [])]
        ids = [utterance["id"] for utterance in all_utterances]
        scene_ids = [scene["id"] for scene in scenes]
        if len(set(ids)) != len(ids) or len(set(scene_ids)) != len(scene_ids):
            raise ValueError("Дублирующиеся ID сцен или реплик")
        if not self.configured:
            by_id = {item.get("id"): item["canonical_name"] for item in memory if item.get("id")}
            attributions = []
            for utterance in all_utterances:
                assigned = by_id.get(utterance.get("speaker_id")) if utterance.get("manual_speaker") else None
                narrator = utterance.get("type") == "NARRATOR"
                attributions.append({"utterance_id": utterance["id"], "character_name": assigned or ("Рассказчик" if narrator else None),
                                     "confidence": 1.0 if assigned or narrator else 0.0, "evidence": [utterance["id"]] if assigned or narrator else [],
                                     "reason_code": "HUMAN_ASSIGNMENT" if assigned else ("NARRATION_STRUCTURE" if narrator else "MANUAL_REVIEW_REQUIRED")})
            return {"characters": memory, "attributions": attributions, "scene_summaries": [
                {"scene_id": scene["id"], "summary": scene.get("summary") or "Ручной режим: художественный анализ сцены ещё не выполнен."} for scene in scenes], "events": [], "mode": "manual"}
        attributions, summaries, events = [], [], []
        # Evidence already stored in a Character Bible may reference later chapters
        # during re-analysis. Validate against the whole immutable book ID index.
        known_texts: dict[str, str] = {utterance["id"]: utterance.get("source_text", "") for utterance in all_utterances}
        for scene in scenes:
            parts = []
            for batch in _batches(scene.get("utterances", [])):
                payload = {
                    "book": {"title": book.get("title", ""), "author": book.get("author", ""), "language": book.get("language", "ru")},
                    "characters": memory, "current_scene": _scene_data(scene, batch),
                    "previous_summaries": summaries[-4:], "recent_events": events[-8:],
                }
                response = None
                for attempt in range(LLM_RETRY_ATTEMPTS):
                    response = await self._request(ANALYSIS_PROMPT_VERSION, payload, AnalysisResponse)
                    # Completeness check: retry only for missing/extra attributions, not structural errors
                    expected = {u["id"] for u in batch}
                    actual = {item.utterance_id for item in response.attributions}
                    if expected == actual and len(response.attributions) == len(expected):
                        break
                    if attempt < LLM_RETRY_ATTEMPTS - 1:
                        payload["mandatory_utterance_ids"] = sorted(expected)
                        payload["missing_from_last_attempt"] = sorted(expected - actual)
                        payload["extra_from_last_attempt"] = sorted(actual - expected)
                        payload["reminder"] = "Return exactly one attribution object per utterance_id. No omissions, no duplicates, no extras."
                    else:
                        raise ValueError(f"LLM должен вернуть каждую реплику ровно один раз: получено {len(actual)}, ожидается {len(expected)}")
                # Structural validation always runs after completeness is confirmed
                memory = self._validate_references(response, scene["id"], batch, memory, known_texts)
                attributions.extend(item.model_dump() for item in response.attributions)
                parts.extend(item.summary for item in response.scene_summaries)
                events.extend(item.model_dump() for item in response.events)
            summaries.append({"scene_id": scene["id"], "summary": "\n".join(parts)})
        return {"characters": memory, "attributions": attributions, "scene_summaries": summaries, "events": events, "mode": "llm"}

    async def direct(self, scene: dict, characters: list) -> list[dict]:
        utterances = scene.get("utterances", [])
        ids = [item["id"] for item in utterances]
        if len(ids) != len(set(ids)):
            raise ValueError("Дублирующиеся ID реплик")
        if not self.configured:
            return [Performance(utterance_id=item["id"], intent="Нейтральное чтение; ручная режиссура", delivery="Нейтрально, естественно, без дополнительной эмоциональной интерпретации.").model_dump() for item in utterances]
        memory = _characters(characters)
        results = []
        previous = []
        for batch in _batches(utterances):
            response = await self._request(DIRECTOR_PROMPT_VERSION, {"current_scene": _scene_data(scene, batch), "characters": memory, "context_before": previous[-3:]}, DirectorResponse)
            expected = [item["id"] for item in batch]
            actual = [item.utterance_id for item in response.plans]
            if len(actual) != len(expected) or set(actual) != set(expected):
                raise ValueError("Scene Director должен вернуть каждую реплику ровно один раз")
            plans = {item.utterance_id: item.model_dump() for item in response.plans}
            results.extend(plans[item_id] for item_id in expected)
            previous = _scene_data(scene, batch)["utterances"]
        return results


