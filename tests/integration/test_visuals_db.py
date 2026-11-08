"""Real Postgres check of vehicle_visuals.sql: idempotent apply, dedupe, and the claim/finalize/publish functions.

Runs only with `-m db` and TEST_DATABASE_URL pointing at a throwaway database.
"""
import os
from pathlib import Path
from uuid import uuid4

import pytest

from app.db import pool

pytestmark = [
    pytest.mark.db,
    pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL is not set"),
]

SQL_DIR = Path(__file__).resolve().parents[2] / "app" / "db" / "sql"


@pytest.fixture
async def real_pool():
    pool.set_pool_for_tests(None)  # drop the autouse fake
    await pool.init_pool(os.environ["TEST_DATABASE_URL"])
    try:
        yield
    finally:
        await pool.close_pool()


async def _apply():
    for name in ("schema.sql", "vehicle_visuals.sql"):
        await pool.execute((SQL_DIR / name).read_text(encoding="utf-8"))


async def _insert(variant: str, state: str = "queued", **extra) -> str:
    cols = {"variant_key": variant, "style_version": "s1", "state": state, **extra}
    names = ", ".join(cols)
    marks = ", ".join(f"${i + 1}" for i in range(len(cols)))
    return str(await pool.fetchval(f"insert into vehicle_visuals ({names}) values ({marks}) returning id", *cols.values()))


async def test_visuals_schema_applies_twice_and_dedupes_variant(real_pool):
    await _apply()
    await _apply()
    variant = f"test-{uuid4().hex[:8]}"
    try:
        for _ in range(2):
            await pool.execute(
                "insert into vehicle_visuals (variant_key, style_version, state) values ($1, 's1', 'queued') "
                "on conflict do nothing", variant)
        assert await pool.fetchval("select count(*) from vehicle_visuals where variant_key = $1", variant) == 1
    finally:
        await pool.execute("delete from vehicle_visuals where variant_key = $1", variant)


async def test_claim_finalize_publish_cycle(real_pool):
    await _apply()
    await pool.execute("delete from vehicle_visuals")
    variant = f"test-{uuid4().hex[:8]}"
    try:
        vid = await _insert(variant)
        row = await pool.fetchrow("select * from claim_due_visual_job(60)")
        assert str(row["id"]) == vid and row["state"] == "generating"
        assert row["fencing_token"] == 1 and row["attempt_count"] == 1 and row["lease_expires_at"] is not None
        empty = await pool.fetchrow("select * from claim_due_visual_job(60)")
        assert empty["id"] is None  # nothing due: the lease is live

        assert await pool.fetchval("select finalize_visual_job($1, 99, 'review', 'abc', null, null)", row["id"]) is False
        assert await pool.fetchval("select finalize_visual_job($1, 1, 'review', 'abc', null, null)", row["id"]) is True
        assert await pool.fetchval("select finalize_visual_job($1, 1, 'review', 'abc', null, null)", row["id"]) is False
        done = await pool.fetchrow("select * from vehicle_visuals where id = $1", row["id"])
        assert done["state"] == "review" and done["draft_sha256"] == "abc" and done["lease_expires_at"] is None

        assert await pool.fetchval("select publish_visual_revision($1, 5, 'f' , 1024, 420)", row["id"]) is False
        assert await pool.fetchval("select publish_visual_revision($1, 0, 'f', 1024, 420)", row["id"]) is True
        assert await pool.fetchval("select publish_visual_revision($1, 0, 'f', 1024, 420)", row["id"]) is False
        pub = await pool.fetchrow("select * from vehicle_visuals where id = $1", row["id"])
        assert (pub["state"], pub["sha256"], pub["width"], pub["height"], pub["published_revision"]) == ("ready", "f", 1024, 420, 1)
    finally:
        await pool.execute("delete from vehicle_visuals where variant_key = $1", variant)


async def test_expired_lease_is_reclaimed_with_a_new_token(real_pool):
    await _apply()
    await pool.execute("delete from vehicle_visuals")
    variant = f"test-{uuid4().hex[:8]}"
    try:
        await _insert(variant, "generating", fencing_token=4, attempt_count=1)
        await pool.execute("update vehicle_visuals set lease_expires_at = now() - interval '1 minute' where variant_key = $1", variant)
        row = await pool.fetchrow("select * from claim_due_visual_job(60)")
        assert row["variant_key"] == variant and row["fencing_token"] == 5 and row["attempt_count"] == 2
    finally:
        await pool.execute("delete from vehicle_visuals where variant_key = $1", variant)
