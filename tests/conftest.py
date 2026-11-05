import os

# Required settings come from env vars so tests never need a real .env.
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-0123456789-abcdefghijklmnop")
os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_KEY", "test-supabase-key")
os.environ.setdefault("BASE_URL", "http://localhost")
os.environ.setdefault("DEEPSEEK_API_KEY", "test-deepseek-key")
os.environ.setdefault("ENV", "development")

import asyncio
import json
import re
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    import main

    with patch("main.llm.ping", AsyncMock(return_value=True)), patch(
        "main.check_supabase", AsyncMock(return_value=True)
    ), TestClient(main.app) as c:
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


# --- Chat sessions and diagnosis turns ----------------------------------------
# Offline fakes: no Supabase, no DeepSeek. The store round-trips through JSON like jsonb.

USER_ID = "11111111-1111-4111-8111-111111111111"
OTHER_USER_ID = "22222222-2222-4222-8222-222222222222"


def _json_copy(value):
    return json.loads(json.dumps(value, ensure_ascii=False))


class FakeSessionStore:
    """In-memory stand-in for ChatSessionCRUD."""

    def __init__(self):
        self.rows: dict[str, dict] = {}

    def add(self, user_id: str, car_id: str, **fields) -> dict:
        now = datetime.now(timezone.utc).isoformat()
        row = {
            "id": str(uuid4()), "user_id": user_id, "car_id": car_id, "title": "New diagnosis",
            "messages": [], "state": {}, "created_at": now, "updated_at": now, **fields,
        }
        self.rows[row["id"]] = row
        return _json_copy(row)

    def messages(self, session_id: str) -> list[dict]:
        return self.rows[session_id]["messages"]

    async def create(self, data: dict) -> list[dict]:
        return [self.add(**data)]

    async def get_for_user(self, session_id: str, user_id: str) -> dict | None:
        row = self.rows.get(session_id)
        return _json_copy(row) if row and row["user_id"] == user_id else None

    async def list_for_user(self, user_id: str) -> list[dict]:
        return [
            {"id": r["id"], "car_id": r["car_id"], "title": r["title"], "updated_at": r["updated_at"],
             "message_count": len(r["messages"])}
            for r in self.rows.values() if r["user_id"] == user_id
        ]

    async def append_messages(self, session_id: str, messages: list[dict]) -> None:
        self.rows[session_id]["messages"] += _json_copy(messages)

    async def save_state(self, session_id: str, state: dict, title: str) -> None:
        self.rows[session_id].update(state=_json_copy(state), title=title)

    async def delete(self, filters: dict) -> list[dict]:
        gone = [r for r in self.rows.values() if all(r.get(k) == v for k, v in filters.items())]
        for r in gone:
            del self.rows[r["id"]]
        return gone


class FakeKB:
    def __init__(self):
        self.passages: list[dict] = []
        self.error: Exception | None = None
        self.queries: list[tuple] = []

    async def search(self, query: str, car_id: str, k: int) -> list[dict]:
        self.queries.append((query, car_id, k))
        if self.error is not None:
            raise self.error
        return self.passages


class FakeLLM:
    """Counts calls; `fail_stream_after=n` raises before token n (0 = before the first token)."""

    def __init__(self):
        from app.schemas.session import DiagnosisState, SuspectedCause

        self.calls = 0
        self.reply = "Likely the fan relay. Check it with the engine cold."
        self.fail_stream_after: int | None = None
        self.state: DiagnosisState | None = DiagnosisState(
            suspected_causes=[SuspectedCause(cause="Fan relay", likelihood=0.6)],
            title="Fan cycling on and off",
        )
        self.gate: asyncio.Event | None = None
        self.prompt: list[dict] | None = None
        self.conversation: list[dict] | None = None

    async def stream_reply(self, messages: list[dict]):
        self.calls += 1
        self.prompt = messages
        if self.gate is not None:
            await self.gate.wait()
        for token in re.findall(r"\S+\s*", self.reply)[: self.fail_stream_after]:
            yield token
        if self.fail_stream_after is not None:
            raise RuntimeError("stream broke")

    async def extract_state(self, conversation: list[dict], previous):
        self.calls += 1
        self.conversation = conversation
        return self.state if self.state is not None else previous


@pytest.fixture
def user_id():
    return USER_ID


@pytest.fixture
def other_user_id():
    return OTHER_USER_ID


@pytest.fixture
def car():
    return {"id": "toyota-echo-2001", "make": "Toyota", "model": "Echo", "year": 2001}


@pytest.fixture
def fake_store(monkeypatch):
    from app.services import diagnosis

    store = FakeSessionStore()
    monkeypatch.setattr(diagnosis, "sessions", store)
    return store


@pytest.fixture
def fake_kb(monkeypatch):
    from app.services import kb

    fake = FakeKB()
    monkeypatch.setattr(kb, "search", fake.search)
    return fake


@pytest.fixture
def fake_llm(monkeypatch, fake_kb):
    from app.services import llm

    fake = FakeLLM()
    monkeypatch.setattr(llm, "stream_reply", fake.stream_reply)
    monkeypatch.setattr(llm, "extract_state", fake.extract_state)
    return fake


@pytest.fixture
def slow_llm(fake_llm):
    """The reply stream waits on `slow_llm.gate` before its first token."""
    fake_llm.gate = asyncio.Event()
    return fake_llm


@pytest.fixture
def session(fake_store, car):
    return fake_store.add(USER_ID, car["id"])


@pytest.fixture
def fake_garage(monkeypatch, car):
    """USER_ID owns `car`; nobody else owns any car."""
    from app.api.v1 import session_route

    async def get_user_cars(user_id):
        return [car] if user_id == USER_ID else []

    async def get_car_by_id(car_id):
        return car if car_id == car["id"] else None

    monkeypatch.setattr(session_route.profiles, "get_user_cars", get_user_cars)
    monkeypatch.setattr(session_route.profiles.car_crud, "get_car_by_id", get_car_by_id)


@pytest.fixture
def existing_session(session, fake_garage):
    return session


def _bearer(monkeypatch, user_id: str) -> dict:
    from app.utils import auth_middleware

    monkeypatch.setattr(auth_middleware, "validate_user_with_supabase", AsyncMock(return_value=True))
    token = auth_middleware.jwt_handler.create_access_token({"sub": user_id})
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def auth_headers(monkeypatch):
    return _bearer(monkeypatch, USER_ID)


@pytest.fixture
def auth_headers_other_user(monkeypatch):
    return _bearer(monkeypatch, OTHER_USER_ID)
