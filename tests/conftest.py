"""
Test configuration and fixtures for the AutoCare backend application.
Enhanced with security, authentication, and comprehensive testing utilities.
"""
import pytest
import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
from typing import Dict, Any, List
import json
from dotenv import load_dotenv

# Load test environment before importing app modules
test_env_path = Path(__file__).parent.parent / ".env.test"
if test_env_path.exists():
    load_dotenv(test_env_path, override=True)
else:
    # Set minimal test environment if .env.test doesn't exist
    os.environ.setdefault("JWT_SECRET_KEY", "test-jwt-secret-key-for-unit-tests-must-be-32-chars-minimum")
    os.environ.setdefault("TESTING", "true")
    os.environ.setdefault("DATABASE_URL", "sqlite:///./test_autocare.db")
    os.environ.setdefault("REDIS_URL", "redis://localhost:6379/1")

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test event loop configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

# Test environment setup
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment variables and configuration."""
    test_env = {
        "ENVIRONMENT": "test",
        "DEBUG": "false",
        "LOG_LEVEL": "WARNING",
        
        # Database settings (test values)
        "SUPABASE_URL": "https://test.supabase.co",
        "SUPABASE_ANON_KEY": "test_anon_key",
        "DATABASE_URL": "postgresql://test:test@localhost:5432/test_db",
        
        # Redis settings (test values)
        "REDIS_HOST": "localhost",
        "REDIS_PORT": "6379",
        "REDIS_DB": "1",  # Use different DB for tests
        
        # Security settings (test values)
        "SECRET_KEY": "test_secret_key_for_testing_only",
        "JWT_SECRET_KEY": "test_jwt_secret_key",
        "ALLOWED_ORIGINS": "http://localhost:3000,http://localhost:8080",
        
        # Rate limiting (relaxed for tests)
        "RATE_LIMIT_ENABLED": "false",
        "RATE_LIMIT_PER_MINUTE": "1000",
        "RATE_LIMIT_PER_HOUR": "10000",
    }
    
    # Store original values
    original_env = {}
    for key, value in test_env.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    yield
    
    # Restore original values
    for key, original_value in original_env.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value

# Mock data fixtures
@pytest.fixture
def sample_user_data():
    """Sample user data for testing."""
    return {
        "id": "test-user-123",
        "email": "test@example.com",
        "created_at": datetime.now(timezone.utc),
        "phone": "+1234567890",
        "user_metadata": {"name": "Test User"},
        "app_metadata": {"role": "user"},
        "confirmed_at": datetime.now(timezone.utc),
        "last_sign_in_at": datetime.now(timezone.utc),
        "role": "user",
        "cars": ["car-123", "car-456"]
    }

@pytest.fixture
def sample_car_data():
    """Sample car data for testing."""
    return {
        "id": "car-123",
        "make": "Toyota",
        "model": "Camry",
        "year": 2020,
        "text": "Toyota Camry 2020 sedan vehicle",
        "owner_manual_url": "https://example.com/manual.pdf",
        "service_manual_url": "https://example.com/service.pdf",
        "car_guide_links": ["https://example.com/guide1", "https://example.com/guide2"]
    }

@pytest.fixture
def sample_chat_session_data():
    """Sample chat session data for testing."""
    return {
        "id": "session-123",
        "user_id": "test-user-123",
        "car_id": "car-123",
        "title": "Engine Issue Troubleshooting",
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
        "messages": [
            {"role": "user", "content": "My car won't start"},
            {"role": "assistant", "content": "Let me help you troubleshoot that issue."}
        ]
    }

@pytest.fixture
def sample_diagnosis_data():
    """Sample diagnosis data for testing."""
    return {
        "id": "diagnosis-123",
        "user_id": "test-user-123",
        "car_id": "car-123",
        "symptoms": ["won't start", "clicking sound"],
        "diagnosis": "Battery issue",
        "confidence": 0.85,
        "recommendations": ["Check battery connections", "Test battery voltage"],
        "created_at": datetime.now(timezone.utc)
    }

# Mock service fixtures
@pytest.fixture
def mock_websocket_manager():
    """Mock WebSocket manager for testing."""
    manager = AsyncMock()
    manager.connect = AsyncMock()
    manager.disconnect = AsyncMock()
    manager.send_personal_message = AsyncMock()
    manager.broadcast = AsyncMock()
    manager.close = AsyncMock()
    return manager

@pytest.fixture
def mock_db_handler():
    """Mock database handler for testing."""
    db_handler = AsyncMock()
    db_handler.create = AsyncMock()
    db_handler.read = AsyncMock()
    db_handler.update = AsyncMock()
    db_handler.delete = AsyncMock()
    db_handler.close = AsyncMock()
    return db_handler

@pytest.fixture
def mock_logger():
    """Mock logger for testing."""
    logger = AsyncMock()
    logger.info = AsyncMock()
    logger.error = AsyncMock()
    logger.warning = AsyncMock()
    logger.debug = AsyncMock()
    logger.close = AsyncMock()
    return logger

@pytest.fixture
def mock_redis_cache():
    """Mock Redis cache for testing."""
    cache = AsyncMock()
    cache.get = AsyncMock()
    cache.set = AsyncMock()
    cache.delete = AsyncMock()
    cache.exists = AsyncMock()
    cache.close = AsyncMock()
    return cache

@pytest.fixture
def mock_llm_service():
    """Mock LLM service for testing."""
    service = AsyncMock()
    service.generate_response = AsyncMock()
    service.extract_symptoms = AsyncMock()
    service.diagnose = AsyncMock()
    return service

@pytest.fixture
def mock_embedding_service():
    """Mock embedding service for testing."""
    service = AsyncMock()
    service.embed_text = AsyncMock()
    service.search_similar = AsyncMock()
    return service

# Mock external dependencies
@pytest.fixture
def mock_supabase_client():
    """Mock Supabase client for testing."""
    client = MagicMock()
    client.table = MagicMock()
    
    # Mock table operations
    table_mock = MagicMock()
    table_mock.select = MagicMock(return_value=table_mock)
    table_mock.insert = MagicMock(return_value=table_mock)
    table_mock.update = MagicMock(return_value=table_mock)
    table_mock.delete = MagicMock(return_value=table_mock)
    table_mock.eq = MagicMock(return_value=table_mock)
    table_mock.execute = MagicMock()
    
    client.table.return_value = table_mock
    return client

@pytest.fixture
def mock_milvus_client():
    """Mock Milvus client for testing."""
    client = MagicMock()
    client.search = MagicMock()
    client.insert = MagicMock()
    client.delete = MagicMock()
    client.create_collection = MagicMock()
    client.drop_collection = MagicMock()
    return client

# Test data generators
@pytest.fixture
def generate_test_messages():
    """Generate test messages for various scenarios."""
    def _generate(count: int = 5, message_type: str = "chat") -> List[Dict[str, Any]]:
        messages = []
        for i in range(count):
            msg = {
                "type": message_type,
                "source": "user" if i % 2 == 0 else "assistant",
                "content": f"Test message {i+1}",
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                "session_id": "test-session-123"
            }
            messages.append(msg)
        return messages
    return _generate

@pytest.fixture
def generate_test_embeddings():
    """Generate test embeddings for testing."""
    def _generate(dimension: int = 768, count: int = 1) -> List[List[float]]:
        import random
        embeddings = []
        for _ in range(count):
            embedding = [random.random() for _ in range(dimension)]
            embeddings.append(embedding)
        return embeddings if count > 1 else embeddings[0]
    return _generate

# Environment setup fixtures
@pytest.fixture
def mock_environment_variables():
    """Mock environment variables for testing."""
    env_vars = {
        "SUPABASE_URL": "https://test.supabase.co",
        "SUPABASE_KEY": "test-key",
        "REDIS_URL": "redis://localhost:6379",
        "MILVUS_URI": "http://localhost:19530",
        "OPENAI_API_KEY": "test-openai-key",
        "ANTHROPIC_API_KEY": "test-anthropic-key",
        "CELERY_BROKER_URL": "redis://localhost:6379/0",
        "CELERY_RESULT_BACKEND": "redis://localhost:6379/0"
    }
    
    with patch.dict("os.environ", env_vars):
        yield env_vars

# Async test helpers
@pytest.fixture
def async_test_client():
    """Create a test client for FastAPI application."""
    from fastapi.testclient import TestClient
    from main import app
    
    with TestClient(app) as client:
        yield client

# Database fixtures
@pytest.fixture
async def mock_db_response():
    """Mock database response structure."""
    def _create_response(data: Any = None, error: Any = None):
        response = MagicMock()
        response.data = data if data is not None else []
        response.error = error
        return response
    return _create_response

# Utility fixtures
@pytest.fixture
def json_serializer():
    """JSON serialization utility for tests."""
    def _serialize(obj: Any) -> str:
        if isinstance(obj, datetime):
            return obj.isoformat()
        return json.dumps(obj, default=str)
    return _serialize

@pytest.fixture
def datetime_now():
    """Fixed datetime for consistent testing."""
    return datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

# Security test fixtures
@pytest.fixture
def mock_logger():
    """Mock logger for testing."""
    logger = MagicMock()
    logger.info = AsyncMock()
    logger.warning = AsyncMock()
    logger.error = AsyncMock()
    logger.debug = AsyncMock()
    return logger

@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    settings = MagicMock()
    settings.ENVIRONMENT = "test"
    settings.DEBUG = False
    settings.SECRET_KEY = "test_secret"
    settings.SUPABASE_URL = "https://test.supabase.co"
    settings.SUPABASE_ANON_KEY = "test_key"
    settings.REDIS_HOST = "localhost"
    settings.REDIS_PORT = "6379"
    settings.ALLOWED_ORIGINS = ["http://localhost:3000"]
    return settings

@pytest.fixture
def mock_supabase_client():
    """Enhanced mock Supabase client for testing."""
    client = MagicMock()
    
    # Mock table operations
    table_mock = MagicMock()
    table_mock.select = MagicMock(return_value=table_mock)
    table_mock.insert = MagicMock(return_value=table_mock)
    table_mock.update = MagicMock(return_value=table_mock)
    table_mock.delete = MagicMock(return_value=table_mock)
    table_mock.eq = MagicMock(return_value=table_mock)
    table_mock.execute = AsyncMock()
    
    client.table = MagicMock(return_value=table_mock)
    client.from_ = MagicMock(return_value=table_mock)
    
    # Mock auth operations
    auth_mock = MagicMock()
    auth_mock.sign_up = AsyncMock()
    auth_mock.sign_in_with_password = AsyncMock()
    auth_mock.get_user = AsyncMock()
    client.auth = auth_mock
    
    return client

@pytest.fixture
def mock_redis_client():
    """Enhanced mock Redis client for testing."""
    redis = MagicMock()
    redis.get = AsyncMock()
    redis.set = AsyncMock()
    redis.delete = AsyncMock()
    redis.ping = AsyncMock(return_value=True)
    redis.info = AsyncMock(return_value={
        "connected_clients": 1,
        "used_memory": 1024000,
        "keyspace_hits": 100,
        "keyspace_misses": 10
    })
    redis.close = AsyncMock()
    return redis

@pytest.fixture
def mock_jwt_payload():
    """Mock JWT payload for testing."""
    return {
        "sub": "test_user_id",
        "email": "test@example.com",
        "role": "user",
        "permissions": ["read", "write"],
        "exp": 9999999999,  # Far future
        "iat": 1000000000,
        "jti": "test_token_id"
    }

@pytest.fixture
def mock_request():
    """Mock FastAPI request for testing."""
    from starlette.requests import Request
    
    request = MagicMock(spec=Request)
    request.client = MagicMock()
    request.client.host = "127.0.0.1"
    request.headers = {}
    request.url = MagicMock()
    request.url.path = "/test"
    request.method = "GET"
    request.state = MagicMock()
    return request

# Test utilities
class TestHelper:
    """Helper class for common test operations."""
    
    @staticmethod
    def create_mock_response(status_code=200, json_data=None, text_data=None):
        """Create a mock HTTP response."""
        response = MagicMock()
        response.status_code = status_code
        response.json.return_value = json_data or {}
        response.text = text_data or ""
        response.headers = {}
        return response
    
    @staticmethod
    def create_mock_exception(exception_type, message="Test exception"):
        """Create a mock exception for testing error handling."""
        return exception_type(message)
    
    @staticmethod
    def assert_correlation_id_in_response(response_data):
        """Assert that correlation ID is present in response."""
        assert "correlation_id" in response_data
        assert isinstance(response_data["correlation_id"], str)
        assert len(response_data["correlation_id"]) > 0
    
    @staticmethod
    def assert_security_headers_present(headers):
        """Assert that security headers are present."""
        security_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options", 
            "X-XSS-Protection",
            "Strict-Transport-Security",
            "Content-Security-Policy"
        ]
        
        for header in security_headers:
            assert header in headers, f"Security header {header} is missing"

@pytest.fixture
def test_helper():
    """Provide TestHelper instance for tests."""
    return TestHelper()

# Error simulation fixtures
@pytest.fixture
def database_error():
    """Simulate database connection error."""
    return ConnectionError("Database connection failed")

@pytest.fixture 
def redis_error():
    """Simulate Redis connection error."""
    return ConnectionError("Redis connection failed")

@pytest.fixture
def external_api_error():
    """Simulate external API error."""
    import requests
    return requests.RequestException("External API unavailable")

# Performance test fixtures
@pytest.fixture
def performance_monitor():
    """Monitor for performance testing."""
    import time
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
        
        @property
        def duration(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
        
        def assert_duration_less_than(self, max_duration):
            assert self.duration is not None, "Timer not started/stopped"
            assert self.duration < max_duration, f"Duration {self.duration}s exceeds {max_duration}s"
    
    return PerformanceMonitor()

# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "security: mark test as a security test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test paths."""
    for item in items:
        # Add markers based on test file path
        if "test_security" in item.nodeid:
            item.add_marker(pytest.mark.security)
        if "test_auth" in item.nodeid:
            item.add_marker(pytest.mark.security)
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)

# Async test decorator
def async_test(coro):
    """Decorator to run async test functions."""
    def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro(*args, **kwargs))
    return wrapper

# Performance testing fixtures
@pytest.fixture
def performance_timer():
    """Timer for performance testing."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
        
        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    return Timer()

# Error simulation fixtures
@pytest.fixture
def simulate_errors():
    """Simulate various error conditions for testing."""
    class ErrorSimulator:
        @staticmethod
        def connection_error():
            from requests.exceptions import ConnectionError
            raise ConnectionError("Connection failed")
        
        @staticmethod
        def timeout_error():
            from requests.exceptions import Timeout
            raise Timeout("Request timed out")
        
        @staticmethod
        def validation_error():
            from pydantic import ValidationError
            raise ValidationError("Validation failed", None)
        
        @staticmethod
        def http_error(status_code: int = 500):
            from fastapi import HTTPException
            raise HTTPException(status_code=status_code, detail="HTTP Error")
    
    return ErrorSimulator()
