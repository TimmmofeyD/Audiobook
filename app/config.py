from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(ROOT / '.env'), env_file_encoding='utf-8', extra='ignore')
    app_mode: Literal['demo','live'] = 'demo'
    data_dir: Path = ROOT / 'data'
    elevenlabs_api_key: str = ''
    llm_base_url: str = 'https://api.openai.com/v1'
    llm_api_key: str = ''
    llm_model: str = ''
    ffmpeg_path: str = ''
    asr_enabled: bool = True
    max_upload_mb: int = Field(default=25, ge=1, le=25)
    max_attempts: int = Field(default=3, ge=1, le=5)
    narration_model: str = 'eleven_v3'
    include_narrator_in_dialogue: bool = True
    request_gap_seconds: float = Field(default=0.5, ge=0)

    @property
    def live(self):
        return self.app_mode.lower() == 'live'

    @property
    def llm_configured(self):
        local = urlsplit(self.llm_base_url).hostname in ('localhost','127.0.0.1','::1')
        return bool(self.llm_model and (self.llm_api_key or local))
