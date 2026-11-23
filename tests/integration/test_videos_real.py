"""Real Postgres check of the video tables: cache TTL and refresh, feedback uniqueness, per-user exclusion.
Runs only with `-m db` and TEST_DATABASE_URL pointing at a throwaway database. Synthetic data only."""
import os
from pathlib import Path
from uuid import uuid4

import asyncpg
import pytest

from app.db import pool
from app.services.videos import store

pytestmark = [
    pytest.mark.db,
    pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL is not set"),
]

SCHEMA = Path(__file__).resolve().parents[2] / "app" / "db" / "sql" / "schema.sql"
ITEMS = [{"video_id": "a1", "title": "t", "channel": "c", "duration_seconds": 600, "views": 9000, "verdict": "match", "fetched_at": "x"}]


@pytest.fixture
async def real_pool():
    pool.set_pool_for_tests(None)  # drop the autouse fake
    await pool.init_pool(os.environ["TEST_DATABASE_URL"])
    await pool.execute(SCHEMA.read_text(encoding="utf-8"))
    try:
        yield
    finally:
        await pool.close_pool()


async def _user() -> str:
    return await pool.fetchval("INSERT INTO users (email, password_hash) VALUES ($1, 'x') RETURNING id", f"{uuid4().hex}@example.test")


async def test_cache_round_trip_ttl_and_refresh(real_pool):
    key = f"k-{uuid4().hex}"
    assert await store.cached(key, "job", 30) is None
    await store.save(key, "job", ITEMS)
    assert await store.cached(key, "job", 30) == ITEMS
    await pool.execute("UPDATE video_suggestions SET fetched_at = now() - interval '31 days' WHERE car_key = $1", key)
    assert await store.cached(key, "job", 30) is None  # older than the maximum: ignored
    await pool.execute("UPDATE video_suggestions SET fetched_at = now() - interval '8 days' WHERE car_key = $1", key)
    assert await store.cached(key, "job", 30) == ITEMS and await store.cached(key, "job", 7) is None
    await store.save(key, "job", [])  # a refresh replaces the row and its age
    assert await store.cached(key, "job", 7) == []
    assert await pool.fetchval("SELECT count(*) FROM video_suggestions WHERE car_key = $1", key) == 1


async def test_feedback_is_idempotent_and_per_user_car_and_job(real_pool):
    u1, u2 = await _user(), await _user()
    await store.add_feedback(u1, "car-a", "job", "v1")
    await store.add_feedback(u1, "car-a", "job", "v1")  # twice: still one row, no error
    await store.add_feedback(u1, "car-a", "job", "v2")
    assert await store.excluded(u1, "car-a", "job") == {"v1", "v2"}
    assert await store.excluded(u2, "car-a", "job") == set()  # a second user still sees them
    assert await store.excluded(u1, "car-b", "job") == set() and await store.excluded(u1, "car-a", "other") == set()
    assert await pool.fetchval("SELECT count(*) FROM video_feedback WHERE user_id = $1", u1) == 2


async def test_feedback_needs_a_real_user_and_goes_with_the_account(real_pool):
    with pytest.raises(asyncpg.ForeignKeyViolationError):
        await store.add_feedback(str(uuid4()), "car-a", "job", "v1")
    u = await _user()
    await store.add_feedback(u, "car-a", "job", "v1")
    await pool.execute("DELETE FROM users WHERE id = $1", u)
    assert await store.excluded(u, "car-a", "job") == set()
