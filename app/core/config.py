from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ENV: str = "development"
    DEBUG: bool = False

    JWT_SECRET_KEY: str = Field(..., min_length=32)
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost:8080"

    SUPABASE_URL: str
    SUPABASE_KEY: str

    # Manual scraping source
    BASE_URL: str

    DEEPSEEK_API_KEY: str
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    DEEPSEEK_MODEL: str = "deepseek-flash"

    WHISPER_MODEL: str = "small"
    CLAP_MODEL: str = "laion/clap-htsat-unfused"
    SOUND_MIN_SCORE: float = 0.2

    MAX_FRAMES: int = 8
    MAX_MEDIA_SECONDS: int = 60
    MAX_MEDIA_MB: int = 100
    MAX_IMAGES: int = 4
    MAX_IMAGE_MB: int = 10

    KB_TOP_K: int = 8
    HISTORY_MESSAGES: int = 10
    MEDIA_BUCKET: str = "media"

    def get_allowed_origins(self) -> List[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]

    def is_production(self) -> bool:
        return self.ENV.lower() == "production"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
