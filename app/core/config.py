from pydantic_settings import BaseSettings
from pydantic import Field, validator
from functools import lru_cache
import os
from typing import List, Optional

class Settings(BaseSettings):
    # Application settings
    APP_NAME: str = "AutoCare API"
    DEBUG: bool = Field(default=False, description="Enable debug mode")
    ENV: str = Field(default="development", description="Environment (development, staging, production)")
    
    # Security settings
    JWT_SECRET_KEY: str = Field(default="change-this-in-production", description="JWT secret key")
    ALLOWED_ORIGINS: str = Field(default="http://localhost:3000,http://localhost:8080", description="Comma-separated allowed origins")
    
    # Supabase settings
    SUPABASE_URL: str = Field(..., description="Supabase project URL")
    SUPABASE_KEY: str = Field(..., description="Supabase anon key")
    SUPABASE_PASSWORD: str = Field(..., description="Supabase password")

    # WebSocket settings
    WEBSOCKET_URL: str = Field(default="ws://localhost:8080", description="WebSocket URL")

    # Car data base URL
    BASE_URL: str = Field(default="https://autocare.yourdomain.com/", description="Base URL for the application")

    # Gemini API key
    GEMINI_KEY: str = Field(..., description="Google Gemini API key")
    GEMINI_MODEL_1: str = Field(default="gemini-pro", description="Primary Gemini model")
    GEMINI_MODEL_2: str = Field(default="gemini-pro-vision", description="Secondary Gemini model")

    # YouTube API key
    YOUTUBE_API_KEY: str = Field(..., description="YouTube Data API key")

    # Milvus settings
    MILVUS_HOST: str = Field(default="localhost", description="Milvus host")
    MILVUS_PORT: str = Field(default="19530", description="Milvus port")

    # Redis settings
    REDIS_HOST: str = Field(default="localhost", description="Redis host")
    REDIS_PORT: str = Field(default="6379", description="Redis port")
    
    # Rate limiting settings
    RATE_LIMIT_ENABLED: bool = Field(default=True, description="Enable rate limiting")
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = Field(default=60, description="Requests per minute limit")
    RATE_LIMIT_REQUESTS_PER_HOUR: int = Field(default=1000, description="Requests per hour limit")
    
    # Security settings
    SECURITY_HEADERS_ENABLED: bool = Field(default=True, description="Enable security headers")
    AUDIT_LOGGING_ENABLED: bool = Field(default=True, description="Enable audit logging")
    
    @validator('ENV')
    def validate_environment(cls, v):
        allowed_envs = ['development', 'staging', 'production']
        if v not in allowed_envs:
            raise ValueError(f'ENV must be one of {allowed_envs}')
        return v
    
    @validator('JWT_SECRET_KEY')
    def validate_jwt_secret(cls, v, values):
        env = values.get('ENV', 'development')
        if env == 'production' and v == 'change-this-in-production':
            raise ValueError('JWT_SECRET_KEY must be changed in production')
        if len(v) < 32:
            raise ValueError('JWT_SECRET_KEY must be at least 32 characters long')
        return v
    
    @validator('SUPABASE_URL')
    def validate_supabase_url(cls, v):
        if not v or not v.startswith('https://'):
            raise ValueError('SUPABASE_URL must be a valid HTTPS URL')
        return v
    
    @validator('MILVUS_PORT', 'REDIS_PORT')
    def validate_ports(cls, v):
        port = int(v)
        if not 1 <= port <= 65535:
            raise ValueError('Port must be between 1 and 65535')
        return v
    
    @validator('RATE_LIMIT_REQUESTS_PER_MINUTE', 'RATE_LIMIT_REQUESTS_PER_HOUR')
    def validate_rate_limits(cls, v):
        if v <= 0:
            raise ValueError('Rate limits must be positive integers')
        return v
    
    def get_allowed_origins(self) -> List[str]:
        """Get list of allowed origins"""
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(',') if origin.strip()]
    
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.ENV == 'production'
    
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.ENV == 'development'

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    try:
        return Settings()
    except Exception as e:
        print(f"Configuration error: {str(e)}")
        print("Please check your .env file and ensure all required variables are set.")
        raise
