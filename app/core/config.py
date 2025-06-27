from pydantic_settings import BaseSettings
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
    SUPABASE_PASSWORD: str

    # WebSocket settings
    WEBSOCKET_URL: str = "ws://localhost:8080"

    # Car data base URL
    BASE_URL: str = "https://example.com/"  # Change to your actual base URL

    # Gemini API key
    GEMINI_KEY: str
    GEMINI_MODEL_1: str
    GEMINI_MODEL_2: str

    # YouTube API key
    YOUTUBE_API_KEY: str

    # Milvus settings
    MILVUS_HOST: str = "milvus"
    MILVUS_PORT: str = "19530"

    # Redis settings
    REDIS_HOST: str = "redis"
    REDIS_PORT: str = "6379"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

@lru_cache()
def get_settings() -> Settings:
    return Settings()
