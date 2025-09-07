"""
Enhanced configuration for concurrent user support.
Optimized settings for high-performance, multi-user scenarios.
"""
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from functools import lru_cache
import os
from typing import List, Optional


class Settings(BaseSettings):
    # Application settings
    APP_NAME: str = "AutoCare API - Concurrent Edition"
    DEBUG: bool = Field(default=False, description="Enable debug mode")
    ENV: str = Field(default="production", description="Environment (development, staging, production)")
    VERSION: str = Field(default="2.0.0", description="API version")
    
    # Security settings
    JWT_SECRET_KEY: str = Field(..., description="JWT secret key")
    ALLOWED_ORIGINS: str = Field(
        default="http://localhost:3000,http://localhost:8080,https://autocare.yourdomain.com",
        description="Comma-separated allowed origins"
    )
    
    # Database settings
    SUPABASE_URL: str = Field(..., description="Supabase project URL")
    SUPABASE_KEY: str = Field(..., description="Supabase anon key")
    SUPABASE_PASSWORD: str = Field(..., description="Supabase password")
    
    # Enhanced database connection pool settings
    DB_POOL_MIN_SIZE: int = Field(default=10, description="Minimum database connections")
    DB_POOL_MAX_SIZE: int = Field(default=100, description="Maximum database connections")
    DB_POOL_MAX_IDLE_TIME: int = Field(default=3600, description="Max idle time for connections (seconds)")
    DB_POOL_MAX_USES: int = Field(default=1000, description="Max uses per connection")
    DB_CONNECTION_TIMEOUT: int = Field(default=30, description="Connection timeout (seconds)")
    
    # Redis settings for session management and caching
    REDIS_HOST: str = Field(default="localhost", description="Redis host")
    REDIS_PORT: str = Field(default="6379", description="Redis port")
    REDIS_PASSWORD: str = Field(default="", description="Redis password")
    REDIS_DB: int = Field(default=0, description="Redis database number")
    REDIS_POOL_SIZE: int = Field(default=20, description="Redis connection pool size")
    REDIS_SOCKET_TIMEOUT: int = Field(default=5, description="Redis socket timeout")
    
    # Milvus settings
    MILVUS_HOST: str = Field(default="localhost", description="Milvus host")
    MILVUS_PORT: str = Field(default="19530", description="Milvus port")
    
    # Session management settings
    MAX_CONCURRENT_SESSIONS: int = Field(default=10000, description="Maximum concurrent sessions")
    MAX_SESSIONS_PER_USER: int = Field(default=5, description="Maximum sessions per user")
    SESSION_TTL_HOURS: int = Field(default=24, description="Session TTL in hours")
    SESSION_CLEANUP_INTERVAL: int = Field(default=3600, description="Session cleanup interval (seconds)")
    
    # WebSocket settings
    WEBSOCKET_URL: str = Field(default="ws://localhost:8000", description="WebSocket URL")
    WEBSOCKET_MAX_CONNECTIONS: int = Field(default=1000, description="Maximum WebSocket connections")
    WEBSOCKET_MAX_CONNECTIONS_PER_USER: int = Field(default=5, description="Max WebSocket connections per user")
    WEBSOCKET_HEARTBEAT_INTERVAL: int = Field(default=30, description="WebSocket heartbeat interval (seconds)")
    WEBSOCKET_CONNECTION_TIMEOUT: int = Field(default=3600, description="WebSocket connection timeout (seconds)")
    
    # Agent pool settings
    AGENT_POOL_SIZE: int = Field(default=50, description="Total agent pool size")
    ORCHESTRATOR_AGENTS_MIN: int = Field(default=2, description="Minimum orchestrator agents")
    ORCHESTRATOR_AGENTS_MAX: int = Field(default=10, description="Maximum orchestrator agents")
    SYMPTOM_AGENTS_MIN: int = Field(default=3, description="Minimum symptom extractor agents")
    SYMPTOM_AGENTS_MAX: int = Field(default=15, description="Maximum symptom extractor agents")
    DIAGNOSTIC_AGENTS_MIN: int = Field(default=3, description="Minimum diagnostic agents")
    DIAGNOSTIC_AGENTS_MAX: int = Field(default=15, description="Maximum diagnostic agents")
    USER_INTERACTION_AGENTS_MIN: int = Field(default=2, description="Minimum user interaction agents")
    USER_INTERACTION_AGENTS_MAX: int = Field(default=10, description="Maximum user interaction agents")
    
    # Enhanced rate limiting settings
    RATE_LIMIT_ENABLED: bool = Field(default=True, description="Enable rate limiting")
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = Field(default=100, description="Requests per minute limit")
    RATE_LIMIT_REQUESTS_PER_HOUR: int = Field(default=2000, description="Requests per hour limit")
    RATE_LIMIT_BURST_THRESHOLD: int = Field(default=50, description="Burst threshold for rapid requests")
    ADAPTIVE_RATE_LIMITING: bool = Field(default=True, description="Enable adaptive rate limiting")
    
    # Caching settings
    CACHE_TTL_SECONDS: int = Field(default=3600, description="Default cache TTL (seconds)")
    LOCAL_CACHE_TTL_SECONDS: int = Field(default=300, description="Local cache TTL (seconds)")
    CONVERSATION_CACHE_SIZE: int = Field(default=1000, description="Maximum conversations in cache")
    
    # Performance settings
    MAX_CONCURRENT_REQUESTS: int = Field(default=1000, description="Maximum concurrent requests")
    REQUEST_TIMEOUT: int = Field(default=300, description="Request timeout (seconds)")
    BACKGROUND_TASK_WORKERS: int = Field(default=4, description="Background task workers")
    
    # LLM API settings
    GEMINI_KEY: str = Field(..., description="Google Gemini API key")
    GEMINI_MODEL_1: str = Field(default="gemini-pro", description="Primary Gemini model")
    GEMINI_MODEL_2: str = Field(default="gemini-pro-vision", description="Secondary Gemini model")
    DEEPSEEK_API_KEY: str = Field(default="", description="DeepSeek AI API key")
    DEEPSEEK_DEFAULT_MODEL: str = Field(default="deepseek-chat", description="Default DeepSeek model")
    
    # External API settings
    YOUTUBE_API_KEY: str = Field(..., description="YouTube Data API key")
    BASE_URL: str = Field(
        default="https://autocare.yourdomain.com/",
        description="Base URL for the application"
    )
    
    # Monitoring and logging settings
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    ENABLE_METRICS: bool = Field(default=True, description="Enable metrics collection")
    METRICS_PORT: int = Field(default=9090, description="Metrics endpoint port")
    
    # Security settings
    SECURITY_HEADERS_ENABLED: bool = Field(default=True, description="Enable security headers")
    AUDIT_LOGGING_ENABLED: bool = Field(default=True, description="Enable audit logging")
    CORS_MAX_AGE: int = Field(default=3600, description="CORS preflight cache time")
    
    # Horizontal scaling settings
    ENABLE_HORIZONTAL_SCALING: bool = Field(default=False, description="Enable horizontal scaling features")
    SERVICE_DISCOVERY_ENABLED: bool = Field(default=False, description="Enable service discovery")
    LOAD_BALANCER_ENABLED: bool = Field(default=False, description="Enable load balancer")
    
    @validator('ENV')
    @classmethod
    def validate_environment(cls, v):
        allowed_envs = ['development', 'staging', 'production']
        if v not in allowed_envs:
            raise ValueError(f'ENV must be one of {allowed_envs}')
        return v
    
    @validator('JWT_SECRET_KEY')
    @classmethod
    def validate_jwt_secret(cls, v, values):
        env = values.get('ENV', 'development') if values else 'development'
        if env == 'production' and v == 'change-this-in-production':
            raise ValueError('JWT_SECRET_KEY must be changed in production')
        if len(v) < 32:
            raise ValueError('JWT_SECRET_KEY must be at least 32 characters long')
        return v
    
    @validator('SUPABASE_URL')
    @classmethod
    def validate_supabase_url(cls, v):
        if not v or not v.startswith('https://'):
            raise ValueError('SUPABASE_URL must be a valid HTTPS URL')
        return v
    
    @validator('MILVUS_PORT', 'REDIS_PORT')
    @classmethod
    def validate_ports(cls, v):
        port = int(v)
        if not 1 <= port <= 65535:
            raise ValueError('Port must be between 1 and 65535')
        return v
    
    @validator('DB_POOL_MIN_SIZE', 'DB_POOL_MAX_SIZE')
    @classmethod
    def validate_db_pool_sizes(cls, v, values):
        if v <= 0:
            raise ValueError('Pool size must be positive')
        
        # Validate min <= max
        if values and 'DB_POOL_MIN_SIZE' in values:
            min_size = values.get('DB_POOL_MIN_SIZE', 1)
            if v < min_size and 'MAX' in str(v):
                raise ValueError('DB_POOL_MAX_SIZE must be >= DB_POOL_MIN_SIZE')
        
        return v
    
    @validator('MAX_CONCURRENT_SESSIONS', 'MAX_SESSIONS_PER_USER')
    @classmethod
    def validate_session_limits(cls, v):
        if v <= 0:
            raise ValueError('Session limits must be positive integers')
        return v
    
    @validator('RATE_LIMIT_REQUESTS_PER_MINUTE', 'RATE_LIMIT_REQUESTS_PER_HOUR')
    @classmethod
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
    
    def get_redis_url(self) -> str:
        """Get Redis connection URL"""
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
    def get_database_config(self) -> dict:
        """Get database configuration dictionary"""
        return {
            "url": self.SUPABASE_URL,
            "key": self.SUPABASE_KEY,
            "pool_config": {
                "min_size": self.DB_POOL_MIN_SIZE,
                "max_size": self.DB_POOL_MAX_SIZE,
                "max_idle_time": self.DB_POOL_MAX_IDLE_TIME,
                "max_uses": self.DB_POOL_MAX_USES,
                "connection_timeout": self.DB_CONNECTION_TIMEOUT
            }
        }
    
    def get_agent_pool_config(self) -> dict:
        """Get agent pool configuration"""
        return {
            "orchestrator": {
                "min": self.ORCHESTRATOR_AGENTS_MIN,
                "max": self.ORCHESTRATOR_AGENTS_MAX,
                "max_uses": 50
            },
            "symptom_extractor": {
                "min": self.SYMPTOM_AGENTS_MIN,
                "max": self.SYMPTOM_AGENTS_MAX,
                "max_uses": 100
            },
            "diagnostic": {
                "min": self.DIAGNOSTIC_AGENTS_MIN,
                "max": self.DIAGNOSTIC_AGENTS_MAX,
                "max_uses": 100
            },
            "user_interaction": {
                "min": self.USER_INTERACTION_AGENTS_MIN,
                "max": self.USER_INTERACTION_AGENTS_MAX,
                "max_uses": 200
            }
        }
    
    def get_performance_config(self) -> dict:
        """Get performance-related configuration"""
        return {
            "max_concurrent_sessions": self.MAX_CONCURRENT_SESSIONS,
            "max_concurrent_requests": self.MAX_CONCURRENT_REQUESTS,
            "max_websocket_connections": self.WEBSOCKET_MAX_CONNECTIONS,
            "request_timeout": self.REQUEST_TIMEOUT,
            "cache_ttl": self.CACHE_TTL_SECONDS,
            "session_ttl_hours": self.SESSION_TTL_HOURS
        }

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
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Configuration error: {str(e)}")
        logger.error("Please check your .env file and ensure all required variables are set.")
        raise


# Alias for backward compatibility
get_concurrent_settings = get_settings
