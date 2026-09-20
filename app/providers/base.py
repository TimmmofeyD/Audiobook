"""Provider-neutral speech contracts. No database or UI dependencies."""
from dataclasses import dataclass
from typing import Protocol, TypedDict


class ProviderError(RuntimeError):
    def __init__(self, message: str, retryable: bool = False, retry_after: float | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.retry_after = retry_after


@dataclass
class AudioArtifact:
    audio: bytes
    extension: str
    model_id: str
    parameters: dict
    request_id: str | None = None


class VoiceCandidate(TypedDict):
    generated_voice_id: str
    audio_base64: str
    media_type: str
    duration_secs: float
    description: str


class TTSAdapter(Protocol):
    async def design_voice(self, brief: dict, audition_text: str) -> list[dict]: ...
    async def save_voice(self, generated_voice_id: str, name: str, description: str) -> dict: ...
    async def synthesize_single(self, text: str, voice_id: str, performance: dict, seed: int, model_id: str = "eleven_v3") -> AudioArtifact: ...
    async def synthesize_dialogue(self, lines: list[dict], seed: int) -> AudioArtifact: ...
    async def get_voice_metadata(self, voice_id: str) -> dict: ...
    async def healthcheck(self) -> dict: ...
    async def transcribe(self, audio: bytes, filename: str) -> str: ...
