import os

# Required settings come from env vars so tests never need a real .env.
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-0123456789-abcdefghijklmnop")
# .invalid never resolves: nothing can reach a real database even if the fake pool were bypassed.
os.environ.setdefault("DATABASE_URL", "postgresql://autocare:test@db.invalid:5432/autocare")
os.environ.setdefault("DEEPSEEK_API_KEY", "test-deepseek-key")
os.environ.setdefault("GOOGLE_CLIENT_IDS", "")  # Google sign-in off unless a test turns it on
os.environ.setdefault("ENV", "development")

import asyncio
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import asyncpg
import pytest
from fastapi.testclient import TestClient


class _FakeTransaction:
    def __init__(self, pool):
        self.pool = pool

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.pool.transactions.append("rollback" if exc_type else "commit")
        if exc_type is None and self.pool.commit_error is not None:
            raise self.pool.commit_error
        return False


class _FakeAcquire:
    def __init__(self, pool):
        self.pool = pool

    async def __aenter__(self):
        self.pool.acquired += 1
        return self.pool  # the fake pool doubles as its own connection

    async def __aexit__(self, exc_type, exc, tb):
        self.pool.acquired -= 1
        return False


class FakePool:
    """Stand-in for an asyncpg pool (and its connections): records (method, sql, args) for every statement.
    `rows` answers fetch/fetchrow, `value` answers fetchval, `status` answers execute; `error` makes every statement raise."""

    def __init__(self):
        self.calls: list[tuple[str, str, tuple]] = []
        self.rows: list[dict] = []
        self.value = None
        self.status = "OK"
        self.error: Exception | None = None
        self.error_on_executemany = 0  # n: the nth executemany call raises
        self.transactions: list[str] = []
        self.acquired = 0
        self.acquire_timeouts: list = []
        self.acquire_error: Exception | None = None  # acquire() raises it (pool closing, timeout)
        self.commit_error: Exception | None = None  # a transaction's COMMIT raises it
        self.closed = False

    def _run(self, method: str, sql: str, args: tuple):
        self.calls.append((method, sql, args))
        if self.error is not None:
            raise self.error

    async def fetch(self, sql, *args):
        self._run("fetch", sql, args)
        return [dict(r) for r in self.rows]

    async def fetchrow(self, sql, *args):
        self._run("fetchrow", sql, args)
        return dict(self.rows[0]) if self.rows else None

    async def fetchval(self, sql, *args):
        self._run("fetchval", sql, args)
        return self.value

    async def execute(self, sql, *args):
        self._run("execute", sql, args)
        return self.status

    async def executemany(self, sql, args_list):
        self._run("executemany", sql, tuple(tuple(a) for a in args_list))
        if self.error_on_executemany and len([c for c in self.calls if c[0] == "executemany"]) == self.error_on_executemany:
            raise RuntimeError("batch failed")

    def acquire(self, timeout=None):
        self.acquire_timeouts.append(timeout)
        if self.acquire_error is not None:
            raise self.acquire_error
        return _FakeAcquire(self)

    def transaction(self):
        return _FakeTransaction(self)

    async def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def fake_pool():
    """Every test runs against a fake pool, so no test can open a real database connection."""
    from app.db import pool

    fake = FakePool()
    pool.set_pool_for_tests(fake)
    yield fake
    pool.set_pool_for_tests(None)


@pytest.fixture(autouse=True)
def _no_google_network(monkeypatch):
    """Google ID token checks never fetch Google's keys in tests; a test that needs one patches it again."""
    from google.oauth2 import id_token

    def offline(*args, **kwargs):
        raise RuntimeError("network disabled in tests")

    monkeypatch.setattr(id_token, "verify_oauth2_token", offline)


@pytest.fixture(autouse=True)
def _reset_rate_limits():
    """The limiter, the daily message counts and the media slots live in memory for the whole run; every test starts from zero."""
    from app.services import usage_limits
    from app.utils.limiter import limiter

    limiter.reset()
    usage_limits._counts.clear()
    usage_limits._media_users.clear()


class FakeUserStore:
    """In-memory stand-in for UserCRUD, with the users table's unique indexes on lower(email) and google_sub.
    tokens_valid_after is epoch seconds (float), compared with a token's iat like the SQL does."""

    def __init__(self):
        self.rows: dict[str, dict] = {}

    def _check_unique(self, email: str | None = None, google_sub: str | None = None, skip: str | None = None) -> None:
        for row in self.rows.values():
            if row["id"] == skip:
                continue
            if (email is not None and row["email"].lower() == email.lower()) or (
                google_sub is not None and row["google_sub"] == google_sub
            ):
                raise asyncpg.UniqueViolationError("duplicate key value violates unique constraint")

    def add(self, email: str, password_hash: str | None = None, google_sub: str | None = None) -> dict:
        self._check_unique(email, google_sub)
        row = {
            "id": str(uuid4()), "email": email, "password_hash": password_hash, "google_sub": google_sub,
            "tokens_valid_after": time.time(),
        }
        self.rows[row["id"]] = row
        return row

    def by_email(self, email: str) -> dict | None:
        return next((r for r in self.rows.values() if r["email"].lower() == email), None)

    @staticmethod
    def _public(row: dict | None) -> dict | None:
        return None if row is None else {"id": row["id"], "email": row["email"]}

    async def get_by_email(self, email: str) -> dict | None:
        row = self.by_email(email)
        return dict(row) if row else None

    async def get_by_google_sub(self, google_sub: str) -> dict | None:
        row = next((r for r in self.rows.values() if r["google_sub"] == google_sub), None)
        return dict(row) if row else None

    async def exists(self, user_id: str, issued_at: float) -> bool:
        row = self.rows.get(user_id)
        return row is not None and row["tokens_valid_after"] <= issued_at

    async def revoke_tokens(self, user_id: str, cutoff: float) -> None:
        row = self.rows.get(user_id)
        if row is not None:
            row["tokens_valid_after"] = max(row["tokens_valid_after"], cutoff)

    async def create_password_user(self, email: str, password_hash: str) -> dict:
        return self._public(self.add(email, password_hash=password_hash))

    async def create_google_user(self, email: str, google_sub: str) -> dict:
        return self._public(self.add(email, google_sub=google_sub))

    async def link_google(self, email: str, google_sub: str) -> dict | None:
        row = self.by_email(email)
        if row is None or row["google_sub"] is not None:
            return None
        self._check_unique(google_sub=google_sub, skip=row["id"])
        row.update(google_sub=google_sub, password_hash=None, tokens_valid_after=max(row["tokens_valid_after"], time.time()))
        return self._public(row)


@pytest.fixture
def strict_clock(monkeypatch):
    """time.time() that never repeats a value (Windows' clock can tick in ~16 ms steps), so a token issued
    before a tokens_valid_after bump and one issued after it never share a timestamp in tests."""
    real, last = time.time, [0.0]

    def now() -> float:
        last[0] = max(real(), last[0] + 1e-6)
        return last[0]

    monkeypatch.setattr(time, "time", now)


@pytest.fixture
def fake_users(monkeypatch, strict_clock):
    from app.services import auth_service

    store = FakeUserStore()
    monkeypatch.setattr(auth_service, "users", store)
    return store


@pytest.fixture
def client():
    import main

    with patch("main.llm.ping", AsyncMock(return_value=True)), patch(
        "main.check_database", AsyncMock(return_value=True)
    ), TestClient(main.app) as c:
        yield c


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
# Offline fakes: no database, no DeepSeek. The store round-trips through JSON like jsonb.

USER_ID = "11111111-1111-4111-8111-111111111111"
OTHER_USER_ID = "22222222-2222-4222-8222-222222222222"


def _json_copy(value):
    return json.loads(json.dumps(value, ensure_ascii=False))


class FakeSessionStore:
    """In-memory stand-in for ChatSessionCRUD, with the busy_until/busy_by lease (on the monotonic clock)."""

    def __init__(self):
        self.rows: dict[str, dict] = {}
        self.leases: dict[str, tuple[float, str]] = {}  # session id -> (expiry, holder)
        self.renewals = 0

    def busy(self, session_id: str) -> bool:
        return session_id in self.leases and self.leases[session_id][0] > time.monotonic()

    def holder(self, session_id: str) -> str | None:
        return self.leases[session_id][1] if session_id in self.leases else None

    def hold(self, session_id: str, seconds: float = 300) -> str:
        """Another worker holds the lease (negative seconds: it held one that has expired). Its holder token."""
        holder = str(uuid4())
        self.leases[session_id] = (time.monotonic() + seconds, holder)
        return holder

    async def claim(self, session_id: str, user_id: str, seconds: float = 300) -> str | None:
        row = self.rows.get(session_id)
        if row is None or row["user_id"] != user_id or self.busy(session_id):
            return None
        return self.hold(session_id, seconds)

    async def renew(self, session_id: str, holder: str, seconds: float = 300) -> bool:
        if self.holder(session_id) != holder:
            return False
        self.renewals += 1
        self.leases[session_id] = (time.monotonic() + seconds, holder)
        return True

    async def release(self, session_id: str, holder: str) -> None:
        await asyncio.sleep(0)  # a database round trip suspends, so a cancelled caller would never get here
        if self.holder(session_id) == holder:
            del self.leases[session_id]

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

    async def create(self, data: dict) -> dict:
        return self.add(**data)

    async def get_for_user(self, session_id: str, user_id: str) -> dict | None:
        row = self.rows.get(session_id)
        return _json_copy(row) if row and row["user_id"] == user_id else None

    async def list_for_user(self, user_id: str) -> list[dict]:
        rows = sorted(
            (r for r in self.rows.values() if r["user_id"] == user_id), key=lambda r: r["updated_at"], reverse=True
        )
        return [
            {"id": r["id"], "car_id": r["car_id"], "title": r["title"], "updated_at": r["updated_at"],
             "message_count": len(r["messages"])}
            for r in rows
        ]

    # Like the real CRUD: writes to a missing row are no-ops, and every write bumps updated_at.
    def _write(self, session_id: str, **fields) -> None:
        if session_id in self.rows:
            self.rows[session_id].update(fields, updated_at=datetime.now(timezone.utc).isoformat())

    async def append_messages(self, session_id: str, messages: list[dict]) -> None:
        if session_id in self.rows:
            self._write(session_id, messages=[*self.rows[session_id]["messages"], *_json_copy(messages)])

    async def update_message(self, session_id: str, message_id: str, patch: dict) -> bool:
        if session_id not in self.rows or not any(m["id"] == message_id for m in self.rows[session_id]["messages"]):
            return False
        msgs = [{**m, **_json_copy(patch)} if m["id"] == message_id else m for m in self.rows[session_id]["messages"]]
        self._write(session_id, messages=msgs)
        return True

    async def save_state(self, session_id: str, state: dict, title: str) -> None:
        self._write(session_id, state=_json_copy(state), title=title)

    async def delete(self, filters: dict) -> list[dict]:
        gone = [r for r in self.rows.values() if all(r.get(k) == v for k, v in filters.items())]
        for r in gone:
            del self.rows[r["id"]]
            self.leases.pop(r["id"], None)  # the lease is a column of the row
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
def media_root(monkeypatch, tmp_path):
    """A real MediaStore rooted in tmp_path, swapped in for the one diagnosis uses."""
    from app.services import diagnosis
    from app.services.media_store import MediaStore

    root = tmp_path / "media-root"
    monkeypatch.setattr(diagnosis, "media_store", MediaStore(root))
    return root


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
def session(fake_store, media_root, car):
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

    monkeypatch.setattr(auth_middleware, "validate_user", AsyncMock(return_value=True))
    token = auth_middleware.jwt_handler.create_access_token({"sub": user_id})
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def auth_headers(monkeypatch):
    return _bearer(monkeypatch, USER_ID)


@pytest.fixture
def auth_headers_other_user(monkeypatch):
    return _bearer(monkeypatch, OTHER_USER_ID)
