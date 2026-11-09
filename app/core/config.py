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

    DATABASE_URL: str  # postgresql://user:password@host:5432/db; never log it

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
    MEDIA_DIR: str = "/data/media"
    MEDIA_MIN_FREE_MB: int = 2048  # uploads are refused when the media volume has less free space
    MEDIA_USER_QUOTA_MB: int = 500  # per-user cap on stored originals; 0 turns the cap off

    # When set, sign-up needs this code (register's invite_code) and Google sign-in no longer creates accounts.
    # Empty leaves registration open; a public deployment should set it.
    REGISTRATION_CODE: str = ""

    ENABLE_DOCS: bool = False  # serves /docs, /redoc and /openapi.json; off for a public deployment
    DAILY_MESSAGE_LIMIT: int = 200  # messages per account per UTC day, counted in memory per process; 0 turns the cap off
    MAX_CONCURRENT_MEDIA_TURNS: int = 2  # media turns running at once in this process, one per account; 0 turns the limits off

    # Comma-separated Google OAuth client ids accepted as an ID token audience; empty turns Google sign-in off.
    GOOGLE_CLIENT_IDS: str = ""

    # Manual discovery: Brave Search finds extra manual PDFs besides startmycar; empty key keeps startmycar only.
    BRAVE_API_KEY: str = ""
    MANUAL_MAX_SOURCES: int = 3  # manuals indexed per car, startmycar included
    MANUAL_MAX_MB: int = 80  # larger downloads are abandoned

    # Vehicle visuals: stored artwork, the public https origin used in signed download urls, and worker limits.
    VISUALS_DIR: str = "/data/visuals"
    PUBLIC_BASE_URL: str = ""  # e.g. https://api.example.com; empty or non-https means no signed urls are issued
    VISUAL_URL_TTL_SECONDS: int = 900
    VISUAL_PREPARE_PER_HOUR: int = 20
    VISUAL_JOBS_PER_HOUR: int = 60
    VISUAL_POLL_SECONDS: int = 10
    VISUAL_PROVIDER_ENABLED: bool = False
    VISUAL_PROVIDER_DAILY_BUDGET: int = 20
    VISUAL_AUTO_APPROVE: bool = False  # publish provider images that pass quality_problem without an operator
    CLOUDFLARE_ACCOUNT_ID: str = ""
    CLOUDFLARE_API_TOKEN: str = ""
    VISUAL_CURATED_DIR: str = "/data/visuals/curated"

    def get_allowed_origins(self) -> List[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]

    def get_google_client_ids(self) -> List[str]:
        return [c.strip() for c in self.GOOGLE_CLIENT_IDS.split(",") if c.strip()]

    def is_production(self) -> bool:
        return self.ENV.lower() == "production"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
