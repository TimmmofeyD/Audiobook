"""ElevenLabs provider adapter with retry, redaction and strict guards."""
from __future__ import annotations

import asyncio
import json
import math
import re
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
from pathlib import Path

import httpx

from .base import AudioArtifact, ProviderError

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_LIMITS = ROOT / "data" / "elevenlabs_limits.json"
DEFAULT_TIMEOUT = httpx.Timeout(180.0, connect=15.0)
MAX_ATTEMPTS = 3


def load_limits(path: Path | None = None) -> dict:
    defaults = json.loads(DEFAULT_LIMITS.read_text(encoding="utf-8"))
    if path is not None and Path(path).resolve() != DEFAULT_LIMITS:
        override = json.loads(Path(path).read_text(encoding="utf-8"))
        defaults.update({k: v for k, v in override.items() if k != "features"})
        defaults["features"].update(override.get("features", {}))
    return defaults


def render_text(text: str, performance: dict, limits: dict, model_id: str = "eleven_v3") -> str:
    """Only the adapter creates provider tags; reject raw markup bypasses.

    Also normalizes Cyrillic stress marks (U+0301) so ElevenLabs receives clean
    Russian text instead of combining marks that hurt pronunciation.
    """
    if not isinstance(text, str) or not text.strip():
        raise ProviderError("Текст для озвучки пуст.")
    if re.search(r"<\s*/?\s*(?:break|speak|phoneme|prosody)\b", text, re.I):
        raise ProviderError("SSML не поддерживается этим адаптером; задайте паузы в плане исполнения.")
    if "[" in text or "]" in text:
        raise ProviderError("Квадратные скобки в spoken_text требуют ручной нормализации, чтобы не передать произвольные аудиотеги.")
    tags = performance.get("audio_tags", [])
    if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
        raise ProviderError("audio_tags должен быть списком строк.")
    tags = [tag.strip().strip("[]").lower() for tag in tags]
    unknown = set(tags) - set(limits["audio_tags"])
    if unknown:
        raise ProviderError("Неизвестные аудиотеги: " + ", ".join(sorted(unknown)))
    if tags and model_id != "eleven_v3":
        raise ProviderError("Аудиотеги доступны только для Eleven v3. Удалите теги или выберите v3.")
    # Stress-mark normalization for natural Russian speech
    cleaned = re.sub(r"[\u0301\u02CA\u02B9]", "", text.strip())
    cleaned = cleaned.replace("Ӣ", "И").replace("ӣ", "и")
    return (" ".join(f"[{tag}]" for tag in dict.fromkeys(tags)) + " " if tags else "") + cleaned


def _validate_seed(seed: int) -> int:
    if type(seed) is not int or not 0 <= seed <= 4294967295:
        raise ProviderError("seed должен быть целым числом от 0 до 4294967295.")
    return seed


def _voice_id(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value):
        raise ProviderError("Некорректный voice_id.")
    return value


def _retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            seconds = (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
        except (ValueError, TypeError, OverflowError):
            return None
    return max(0.0, seconds) if math.isfinite(seconds) else None


def plan_blocks(lines: list[dict], limits: dict | None = None) -> list[list[dict]]:
    """Group already ordered scene lines, never splitting a source utterance."""
    limits = limits or load_limits()
    result, current, size = [], [], 0
    voices: set[str] = set()
    for line in lines:
        count = len(render_text(line["text"], line.get("performance", {}), limits))
        voice = _voice_id(line["voice_id"])
        if count > limits["dialogue_chars"]:
            raise ProviderError("Одна реплика превышает лимит dialogue_chars; разбейте её или используйте single.")
        if current and (size + count > limits["dialogue_chars"] or len(voices | {voice}) > limits["dialogue_voices"]):
            result.append(current)
            current, size, voices = [], 0, set()
        current.append(line)
        size += count
        voices.add(voice)
    if current:
        result.append(current)
    return result


class ElevenLabsAdapter:
    """ElevenLabs TTS/ASR adapter matching the TTSAdapter protocol."""

    def __init__(self, api_key: str, output_format: str = "mp3_44100_128"):
        self.api_key = api_key
        self.output_format = output_format
        self.limits = load_limits()

    async def _post(self, url: str, json_body: dict, timeout: httpx.Timeout = DEFAULT_TIMEOUT) -> tuple[httpx.Response, dict]:
        """POST with 429 retry; returns response and retry metadata."""
        attempts = 0
        response = None
        while attempts < MAX_ATTEMPTS:
            attempts += 1
            async with httpx.AsyncClient(timeout=timeout, params={"output_format": self.output_format}) as client:
                response = await client.post(url, headers={"xi-api-key": self.api_key}, json=json_body)
            if response.status_code == 429 and attempts < MAX_ATTEMPTS:
                delay = _retry_after(response.headers.get("retry-after"))
                if delay is not None:
                    await asyncio.sleep(delay)
                continue
            break
        return response, {"http_attempts": attempts}

    async def design_voice(self, brief: dict, audition_text: str) -> list[dict]:
        if not isinstance(audition_text, str) or len(audition_text) < 100:
            raise ProviderError("Пробный текст для прослушивания должен содержать минимум 100 символов.")
        description = " ".join(str(v) for v in brief.values() if v)[:1000]
        body = {
            "model_id": "eleven_ttv_v3",
            "voice_description": description,
            "text": audition_text[:1000],
            "auto_generate_text": False,
        }
        try:
            response, _ = await self._post("https://api.elevenlabs.io/v1/text-to-voice/design", body)
        except httpx.HTTPError:
            raise ProviderError("Сеть недоступна при обращении к ElevenLabs.", retryable=True)
        if response.status_code == 429:
            raise ProviderError("Превышен лимит запросов ElevenLabs.", retryable=True,
                                retry_after=_retry_after(response.headers.get("retry-after")))
        if response.status_code in (401, 403):
            raise ProviderError("ElevenLabs отклонил ключ.")
        if response.status_code != 200:
            # Body is never leaked: it can echo the API key back.
            raise ProviderError("Провайдер вернул неизвестен ответ: HTTP " + str(response.status_code) + ".")
        data = response.json()
        previews = data.get("previews", data.get("candidates", []))
        candidates = []
        for item in previews[:3]:
            candidates.append({
                "generated_voice_id": item.get("generated_voice_id", item.get("voice_id", "")),
                "audio_base64": item.get("audio_base_64", item.get("audio_base64", "")),
                "media_type": item.get("media_type", "audio/wav"),
                "duration_secs": item.get("duration_secs", 0),
                "description": description,
            })
        if not candidates:
            raise ProviderError("Voice Design не вернул кандидатов.")
        return candidates

    async def save_voice(self, generated_voice_id: str, name: str, description: str) -> dict:
        body = {"generated_voice_id": generated_voice_id, "voice_name": name, "voice_description": description}
        try:
            response, _ = await self._post("https://api.elevenlabs.io/v1/text-to-voice", body)
        except httpx.HTTPError:
            raise ProviderError("Сеть недоступна при сохранении голоса.", retryable=True)
        if response.status_code != 200:
            raise ProviderError("Сохранение голоса не удалось: HTTP " + str(response.status_code) + ".")
        return response.json()

    async def synthesize_single(self, text: str, voice_id: str, performance: dict, seed: int,
                                model_id: str = "eleven_v3") -> AudioArtifact:
        # Guards run before any network call.
        rendered = render_text(text, performance, self.limits, model_id)
        if len(rendered) > self.limits["single_chars"]:
            raise ProviderError("Реплика превышает лимит single_chars.")
        _validate_seed(seed)
        voice = _voice_id(voice_id)
        body = {"text": rendered, "model_id": model_id, "seed": seed}
        try:
            response, http_params = await self._post(
                "https://api.elevenlabs.io/v1/text-to-speech/" + voice, body,
                timeout=httpx.Timeout(300.0, connect=15.0))
        except httpx.HTTPError:
            raise ProviderError("Сеть недоступна при обращении к ElevenLabs.", retryable=True)
        if response.status_code == 429:
            raise ProviderError("Превышен лимит запросов ElevenLabs.", retryable=True,
                                retry_after=_retry_after(response.headers.get("retry-after")))
        if response.status_code in (401, 403):
            raise ProviderError("ElevenLabs отклонил ключ.")
        if response.status_code != 200:
            raise ProviderError("Провайдер вернул неизвестен ответ: HTTP " + str(response.status_code) + ".")
        parameters = {
            "model_id": model_id, "seed": seed, "voice_id": voice,
            "character_cost": response.headers.get("character-cost"),
            "request_id": response.headers.get("request-id"),
            **http_params,
        }
        return AudioArtifact(audio=response.content, extension="mp3", model_id=model_id,
                             parameters=parameters, request_id=response.headers.get("request-id"))

    async def synthesize_dialogue(self, lines: list[dict], seed: int) -> AudioArtifact:
        # Guards run before any network call.
        _validate_seed(seed)
        inputs, voices, total = [], set(), 0
        for line in lines:
            rendered = render_text(line["text"], line.get("performance", {}), self.limits)
            voice = _voice_id(line["voice_id"])
            voices.add(voice)
            total += len(rendered)
            if len(rendered) > self.limits["dialogue_chars"]:
                raise ProviderError("Одна реплика превышает лимит dialogue_chars.")
            if total > self.limits["dialogue_chars"]:
                raise ProviderError("Диалог превышает лимит dialogue_chars.")
            if len(voices) > self.limits["dialogue_voices"]:
                raise ProviderError("Диалог превышает допустимое число голосов: " + str(self.limits["dialogue_voices"]) + ".")
            inputs.append({"text": rendered, "voice_id": voice})
        if not inputs:
            raise ProviderError("Диалог пуст.")
        body = {"inputs": inputs, "model_id": "eleven_v3", "seed": seed}
        try:
            response, http_params = await self._post(
                "https://api.elevenlabs.io/v1/text-to-dialogue", body,
                timeout=httpx.Timeout(300.0, connect=15.0))
        except httpx.HTTPError:
            raise ProviderError("Сеть недоступна при обращении к ElevenLabs.", retryable=True)
        if response.status_code == 429:
            raise ProviderError("Превышен лимит запросов ElevenLabs.", retryable=True,
                                retry_after=_retry_after(response.headers.get("retry-after")))
        if response.status_code in (401, 403):
            raise ProviderError("ElevenLabs отклонил ключ.")
        if response.status_code != 200:
            raise ProviderError("Провайдер вернул неизвестен ответ: HTTP " + str(response.status_code) + ".")
        parameters = {"model_id": "eleven_v3", "seed": seed, "line_count": len(inputs), **http_params}
        return AudioArtifact(audio=response.content, extension="mp3", model_id="eleven_v3",
                             parameters=parameters, request_id=response.headers.get("request-id"))

    async def get_voice_metadata(self, voice_id: str) -> dict:
        url = "https://api.elevenlabs.io/v1/voices/" + _voice_id(voice_id)
        try:
            async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
                response = await client.get(url, headers={"xi-api-key": self.api_key})
        except httpx.HTTPError:
            raise ProviderError("Сеть недоступна при получении метаданных.", retryable=True)
        if response.status_code != 200:
            raise ProviderError("Метаданные голоса недоступны: HTTP " + str(response.status_code) + ".")
        return response.json()

    async def healthcheck(self) -> dict:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
                response = await client.get("https://api.elevenlabs.io/v1/user",
                                            headers={"xi-api-key": self.api_key})
            return {"ok": response.status_code == 200, "status_code": response.status_code}
        except httpx.HTTPError:
            return {"ok": False, "status_code": 0}

    async def transcribe(self, audio: bytes, filename: str) -> str:
        files = {"file": (filename, audio, "audio/mpeg")}
        data = {"model_id": "scribe_v2", "tag_audio_events": "false"}
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=15.0)) as client:
                response = await client.post("https://api.elevenlabs.io/v1/speech-to-text",
                                             headers={"xi-api-key": self.api_key}, files=files, data=data)
        except httpx.HTTPError:
            raise ProviderError("Сеть недоступна при обращении к ElevenLabs ASR.", retryable=True)
        if response.status_code != 200:
            raise ProviderError("ASR вернул HTTP " + str(response.status_code) + ".")
        return response.json().get("text", "")
