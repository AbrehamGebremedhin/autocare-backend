from pydantic import BaseSettings
from functools import lru_cache
import os

class Settings(BaseSettings):
    # Application settings
    APP_NAME: str = "AutoCare API"
    DEBUG: bool = False
    ENV: str = os.getenv("ENV", "development")

    # Supabase settings
    SUPABASE_URL: str
    SUPABASE_KEY: str

    # WebSocket settings
    WEBSOCKET_URL: str = "ws://localhost:8080"

    # Car data base URL
    BASE_URL: str = "https://example.com/"  # Change to your actual base URL

    # Gemini API key
    GEMINI_KEY: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache()
def get_settings() -> Settings:
    return Settings()
