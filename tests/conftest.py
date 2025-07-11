"""
Test configuration and fixtures for the AutoCare backend application.
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
from typing import Dict, Any, List
import json

# Test event loop configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

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
