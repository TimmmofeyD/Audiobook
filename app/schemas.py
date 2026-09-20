from typing import Literal
from pydantic import BaseModel, Field, ConfigDict, field_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid',allow_inf_nan=False)


class Fact(Strict):
    value: str | None = None
    confidence: float = Field(default=0, ge=0, le=1)


class CharacterCreate(Strict):
    canonical_name: str = Field(min_length=1, max_length=100)


class CharacterPatch(Strict):
    canonical_name: str | None = Field(default=None, min_length=1, max_length=100)
    aliases: list[str] | None = None
    gender: Fact | None = None
    age_band: Fact | None = None
    role: str | None = None
    baseline_traits: list[str] | None = None
    speech_profile: dict | None = None
    voice_brief: dict | None = None


class Performance(Strict):
    intent: str = ''
    emotion_internal: dict[str, float] = Field(default_factory=dict)
    emotion_expressed: dict[str, float] = Field(default_factory=dict)
    delivery: str = ''
    audio_tags: list[str] = Field(default_factory=list, max_length=5)
    pause_before_ms: int = Field(default=0, ge=0, le=10000)
    pause_after_ms: int = Field(default=350, ge=0, le=10000)

    @field_validator('emotion_internal','emotion_expressed')
    @classmethod
    def emotion_range(cls,value):
        if any(not 0 <= intensity <= 1 for intensity in value.values()):
            raise ValueError('Интенсивность эмоций должна быть от 0 до 1.')
        return value


class UtterancePatch(Strict):
    speaker_id: str | None = None
    spoken_text: str | None = Field(default=None, min_length=1, max_length=1800)
    type: Literal['NARRATOR','CHARACTER','INNER_THOUGHT','LETTER','DIARY','PHONE','RADIO','ANNOUNCEMENT','GROUP','UNKNOWN'] | None = None
    performance: Performance | None = None
    mode: Literal['auto','single','dialogue'] | None = None


class VoiceLock(Strict):
    generated_voice_id: str | None = None
    voice_id: str | None = None
    provenance: Literal['designed','library','licensed','consented_clone'] = 'designed'
    audition_confirmed: bool = False
    replace: bool = False
    regeneration_scope: Literal['all'] | None = None


class CandidatesRequest(Strict):
    voice_brief: dict | None = None
    audition_text: str | None = Field(default=None, min_length=100, max_length=1000)


class Approval(Strict):
    approved: bool = True


class ExportRequest(Strict):
    formats: list[Literal['mp3','m4b','wav']] = Field(default_factory=lambda: ['mp3','m4b','wav'], min_length=1)


class PronunciationCreate(Strict):
    lexical_form: str = Field(min_length=1, max_length=100)
    spoken_form: str = Field(min_length=1, max_length=200)
    context_signature: str
    approved: bool = True


class BookPatch(Strict):
    title: str | None = Field(default=None,min_length=1,max_length=300)
    author: str | None = Field(default=None,max_length=300)
