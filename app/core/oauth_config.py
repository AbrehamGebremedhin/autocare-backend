from pydantic_settings import BaseSettings
from pydantic import Field, validator
from functools import lru_cache
from typing import List
import os

class OAuthSettings(BaseSettings):
    """OAuth configuration settings"""
    
    # Google OAuth
    GOOGLE_CLIENT_ID: str = Field(..., description="Google OAuth Client ID")
    GOOGLE_CLIENT_SECRET: str = Field(..., description="Google OAuth Client Secret")
    GOOGLE_REDIRECT_URI: str = Field(default="http://localhost:8080/auth/google/callback", description="Google OAuth Redirect URI")
    
    # OAuth settings
    OAUTH_STATE_SECRET: str = Field(default="change-this-oauth-state-secret", description="Secret for OAuth state verification")
    OAUTH_SESSION_TIMEOUT: int = Field(default=300, description="OAuth session timeout in seconds")
    
    # Frontend URLs
    FRONTEND_SUCCESS_URL: str = Field(default="http://localhost:3000/dashboard", description="Frontend success redirect URL")
    FRONTEND_ERROR_URL: str = Field(default="http://localhost:3000/login?error=oauth", description="Frontend error redirect URL")
    
    @validator('GOOGLE_CLIENT_ID', 'GOOGLE_CLIENT_SECRET')
    def validate_google_credentials(cls, v):
        if not v or v == "":
            raise ValueError('Google OAuth credentials must be provided')
        return v
    
    @validator('OAUTH_STATE_SECRET')
    def validate_oauth_secret(cls, v):
        if len(v) < 32:
            raise ValueError('OAuth state secret must be at least 32 characters long')
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
        case_sensitive = True

@lru_cache()
def get_oauth_settings() -> OAuthSettings:
    """Get cached OAuth settings instance"""
    try:
        return OAuthSettings()
    except Exception as e:
        # Log to proper logger instead of print in production
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"OAuth Configuration error: {str(e)}")
        logger.error("Please check your .env file and ensure all required OAuth variables are set.")
        raise
