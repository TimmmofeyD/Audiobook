"""Offline plumbing demonstration. Every artifact is a tone, never speech."""
from __future__ import annotations

import base64
import hashlib
import io
import math
import struct
import uuid
import wave
from pathlib import Path

from .base import AudioArtifact, ProviderError
from .elevenlabs import load_limits, plan_blocks, render_text, _validate_seed


def tone_wav(duration: float = 1.2, frequency: float = 440, sample_rate: int = 24000) -> bytes:
    buffer = io.BytesIO()
    count = int(duration * sample_rate)
    samples = bytearray()
    for i in range(count):
        envelope = min(1.0, i / (sample_rate * .02), (count - i - 1) / (sample_rate * .02))
        # Pulses make the intentionally non-speech nature immediately audible.
        pulse = .18 if i % int(sample_rate * .4) < sample_rate * .25 else 0
        samples.extend(struct.pack("<h", int(32767 * pulse * envelope * math.sin(2 * math.pi * frequency * i / sample_rate))))
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples)
    return buffer.getvalue()


class DemoAdapter:
    name = "demo"
    demo = True

    def __init__(self, api_key: str = "", limits_path: Path | None = None):
        self.limits = load_limits(limits_path)

    async def design_voice(self, brief: dict, audition_text: str) -> list[dict]:
        return [{"generated_voice_id": "demo-" + uuid.uuid4().hex, "audio_base64": base64.b64encode(tone_wav(frequency=freq)).decode(), "media_type": "audio/wav", "duration_secs": 1.2, "description": f"ДЕМО: тестовый тон {freq} Гц. Это не голос и не озвучка текста."} for freq in (330, 440, 550)]

    async def save_voice(self, generated_voice_id: str, name: str, description: str) -> dict:
        if not generated_voice_id.startswith("demo-"):
            raise ProviderError("В деморежиме допустимы только demo-голоса.")
        return {"voice_id": generated_voice_id, "name": name, "description": "ДЕМО — тестовый тон", "category": "demo", "provider": "demo"}

    async def synthesize_single(self, text: str, voice_id: str, performance: dict, seed: int, model_id: str = "eleven_v3") -> AudioArtifact:
        rendered = render_text(text, performance, self.limits, model_id)
        if len(rendered) > self.limits["single_chars"]:
            raise ProviderError("Реплика превышает лимит single_chars.")
        if not voice_id.startswith("demo-"):
            raise ProviderError("В деморежиме допустимы только demo-голоса.")
        _validate_seed(seed)
        frequency = 220 + int(hashlib.sha256(voice_id.encode()).hexdigest()[:4], 16) % 440
        duration = min(3.0, max(.8, len(text) / 90))
        return AudioArtifact(tone_wav(duration, frequency), "wav", "demo-tone-v1", {"demo": True, "speech": False, "text": rendered, "voice_id": voice_id, "seed": seed, "frequency_hz": frequency}, "demo-" + uuid.uuid4().hex)

    async def synthesize_dialogue(self, lines: list[dict], seed: int) -> AudioArtifact:
        # The orchestrator owns block planning; validate limits without re-chunking.
        if not lines:
            raise ProviderError("Диалог пуст.")
        rendered = [render_text(line["text"], line.get("performance", {}), self.limits) for line in lines]
        if any(len(r) > self.limits["dialogue_chars"] for r in rendered):
            raise ProviderError("Одна реплика превышает лимит dialogue_chars.")
        if sum(len(r) for r in rendered) > self.limits["dialogue_chars"]:
            raise ProviderError("Диалог превышает лимит dialogue_chars.")
        if len({line["voice_id"] for line in lines}) > self.limits["dialogue_voices"]:
            raise ProviderError("Диалог превышает допустимое число голосов: " + str(self.limits["dialogue_voices"]) + ".")
        if any(not line["voice_id"].startswith("demo-") for line in lines):
            raise ProviderError("В деморежиме допустимы только demo-голоса.")
        _validate_seed(seed)
        return AudioArtifact(tone_wav(min(5.0, max(1.0, len(lines) * .7))), "wav", "demo-tone-v1", {"demo": True, "speech": False, "inputs": lines, "seed": seed}, "demo-" + uuid.uuid4().hex)

    async def get_voice_metadata(self, voice_id: str) -> dict:
        if not voice_id.startswith("demo-"):
            raise ProviderError("В деморежиме допустимы только demo-голоса.")
        return {"voice_id": voice_id, "category": "demo", "description": "ДЕМО — тестовый тон", "demo": True}

    async def healthcheck(self) -> dict:
        return {"provider": "demo", "ok": True, "demo": True, "speech": False}

    async def transcribe(self, audio: bytes, filename: str) -> str:
        raise ProviderError("В деморежиме речь и ASR отсутствуют. Требуется ручная проверка.")
