import os

# Required settings come from env vars so tests never need a real .env.
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-0123456789-abcdefghijklmnop")
os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_KEY", "test-supabase-key")
os.environ.setdefault("BASE_URL", "http://localhost")
os.environ.setdefault("DEEPSEEK_API_KEY", "test-deepseek-key")
os.environ.setdefault("ENV", "development")

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    import main

    with TestClient(main.app) as c:
        yield c


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
        "owner_manual_url": "https://example.com/manual.pdf",
        "service_manual_url": "https://example.com/service.pdf",
        "car_guide_links": [
            {"link": "https://example.com/guide1", "summary": "Guide one"},
            {"link": "https://example.com/guide2"},
        ]
    }


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
def mock_settings():
    """Mock settings using the real Settings field names."""
    settings = MagicMock()
    settings.ENV = "test"
    settings.DEBUG = False
    settings.JWT_SECRET_KEY = "test-secret-key-0123456789-abcdefghijklmnop"
    settings.SUPABASE_URL = "https://test.supabase.co"
    settings.SUPABASE_KEY = "test_key"
    settings.BASE_URL = "http://localhost"
    settings.ALLOWED_ORIGINS = "http://localhost:3000"
    return settings


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
