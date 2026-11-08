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


# --- vehicle_visual_crud over the real functions ---------------------------------------------------

import asyncio  # noqa: E402

from app.CRUD import vehicle_visual_crud as crud  # noqa: E402


async def _fresh():
    await _apply()
    await pool.execute("delete from vehicle_visuals")


def _variant() -> str:
    return f"test-{uuid4().hex[:8]}"


async def _make_user_with_car(car_id: str) -> str:
    uid = str(await pool.fetchval(
        "insert into users (email, password_hash) values ($1, 'x') returning id", f"{uuid4().hex}@example.com"))
    await pool.execute(
        """insert into "Car" (id, make, model, year) values ($1, 'm', 'x', 2020) on conflict do nothing""", car_id)
    await pool.execute("insert into user_profiles (user_id, cars) values ($1::uuid, $2)", uid, [car_id])
    return uid


async def test_concurrent_ensure_job_makes_one_row(real_pool):
    await _fresh()
    v = _variant()
    results = await asyncio.gather(*[crud.ensure_job(v) for _ in range(20)])
    assert sum(1 for _, created in results if created) == 1
    assert {r["id"] for r, _ in results} == {results[0][0]["id"]}
    assert all(r["state"] == "queued" for r, _ in results)
    assert await pool.fetchval("select count(*) from vehicle_visuals where variant_key = $1", v) == 1


async def test_claim_is_exclusive_and_lease_expiry_reclaims(real_pool):
    await _fresh()
    v = _variant()
    await crud.ensure_job(v)
    first, second = await asyncio.gather(crud.claim_job(60), crud.claim_job(60))
    assert (first is None) != (second is None)
    won = first or second
    assert won["variant_key"] == v and won["fencing_token"] == 1
    await pool.execute("update vehicle_visuals set lease_expires_at = now() - interval '1 minute' where variant_key = $1", v)
    again = await crud.claim_job(60)
    assert again["id"] == won["id"] and again["fencing_token"] > won["fencing_token"]


async def test_stale_token_cannot_finalize_or_renew(real_pool):
    await _fresh()
    v = _variant()
    await crud.ensure_job(v)
    old = await crud.claim_job(60)
    await pool.execute("update vehicle_visuals set lease_expires_at = now() - interval '1 minute' where variant_key = $1", v)
    new = await crud.claim_job(60)
    assert new["fencing_token"] > old["fencing_token"]
    before = await crud.get_visual(v)
    assert await crud.renew_lease(old["id"], old["fencing_token"], 600) is False
    assert await crud.finalize_job(old["id"], old["fencing_token"], "review", "stale", None, None) is False
    assert await crud.get_visual(v) == before  # nothing changed, not even updated_at
    assert await crud.renew_lease(new["id"], new["fencing_token"], 600) is True
    assert await crud.finalize_job(new["id"], new["fencing_token"], "failed", None, "boom", 120) is True
    done = await crud.get_visual(v)
    assert done["state"] == "failed" and done["last_error_code"] == "boom" and done["lease_expires_at"] is None
    assert await pool.fetchval(
        "select next_attempt_at > now() + interval '100 seconds' from vehicle_visuals where variant_key = $1", v)


async def test_publish_requires_review_and_expected_revision(real_pool):
    await _fresh()
    v = _variant()
    row, _ = await crud.ensure_job(v)
    assert await crud.publish(row["id"], 0, "s", 10, 20) is False  # still queued
    await crud.seed_review(v, "draft", 10, 20)
    assert await crud.publish(row["id"], 3, "s", 10, 20) is False  # stale revision
    assert await crud.publish(row["id"], 0, "s", 10, 20) is True
    got = await crud.get_visual(v)
    assert (got["state"], got["published_revision"], got["sha256"]) == ("ready", 1, "s")


async def test_choice_is_per_user_car(real_pool):
    await _apply()
    car = f"car-{uuid4().hex[:6]}"
    u1, u2 = await _make_user_with_car(car), await _make_user_with_car(car)
    try:
        await crud.set_choice(u1, car, "a")
        await crud.set_choice(u1, car, "b")
        assert await crud.get_choice(u1, car) == "b"
        assert await crud.get_choice(u2, car) is None
        await crud.set_choice(u2, car, "c")
        await crud.clear_choice(u1, car)
        assert await crud.get_choice(u1, car) is None and await crud.get_choice(u2, car) == "c"
    finally:
        await pool.execute("delete from users where id = any($1::uuid[])", [u1, u2])
        await pool.execute('delete from "Car" where id = $1', car)


async def test_owned_car_is_none_for_other_user(real_pool):
    await _apply()
    car = f"car-{uuid4().hex[:6]}"
    owner, other = await _make_user_with_car(car), await _make_user_with_car(f"{car}-z")
    try:
        assert (await crud.owned_car(owner, car))["id"] == car
        assert await crud.owned_car(other, car) is None
        assert await crud.owned_car("nope", car) is None
    finally:
        await pool.execute("delete from users where id = any($1::uuid[])", [owner, other])
        await pool.execute('delete from "Car" where id = any($1)', [car, f"{car}-z"])


async def test_requeue_only_from_failed_or_review(real_pool):
    await _fresh()
    v = _variant()
    await crud.ensure_job(v)
    assert await crud.requeue(v) is False  # queued
    await pool.execute(
        "update vehicle_visuals set state = 'failed', attempt_count = 3, last_error_code = 'x' where variant_key = $1", v)
    assert await crud.requeue(v) is True
    row = await crud.get_visual(v)
    assert (row["state"], row["attempt_count"], row["last_error_code"]) == ("queued", 0, None)
    await pool.execute("update vehicle_visuals set state = 'review' where variant_key = $1", v)
    assert await crud.requeue(v) is True
    await pool.execute("update vehicle_visuals set state = 'ready' where variant_key = $1", v)
    assert await crud.requeue(v) is False
    await pool.execute("update vehicle_visuals set state = 'generating' where variant_key = $1", v)
    assert await crud.requeue(v) is False


async def test_reject_only_from_review(real_pool):
    await _fresh()
    v = _variant()
    await crud.ensure_job(v)
    assert await crud.reject(v) is False
    await crud.seed_review(v, "draft", 1, 2)
    assert await crud.reject(v) is True
    got = await crud.get_visual(v)
    assert (got["state"], got["last_error_code"], got["draft_sha256"]) == ("failed", "rejected", None)
    await pool.execute("update vehicle_visuals set state = 'ready', sha256 = 'pub' where variant_key = $1", v)
    assert await crud.reject(v) is False
    assert (await crud.get_visual(v))["sha256"] == "pub"


async def test_referenced_shas(real_pool):
    await _fresh()
    await _insert(_variant(), "ready", sha256="r1", draft_sha256="ignored-draft-of-ready")
    await _insert(_variant(), "review", draft_sha256="d1")
    await _insert(_variant(), "queued")
    await _insert(_variant(), "failed", draft_sha256=None)
    assert await crud.referenced_shas() == {"r1", "d1"}


async def test_seed_review_does_not_touch_published_revision(real_pool):
    await _fresh()
    v = _variant()
    row, _ = await crud.ensure_job(v)
    await crud.seed_review(v, "d1", 10, 20)
    assert await crud.publish(row["id"], 0, "pub", 10, 20) is True
    seeded = await crud.seed_review(v, "d2", 11, 21)
    assert (seeded["state"], seeded["draft_sha256"], seeded["published_revision"], seeded["sha256"]) == ("review", "d2", 1, "pub")
    assert await crud.publish(row["id"], 1, "pub2", 11, 21) is True
    await pool.execute(
        "update vehicle_visuals set state = 'generating', lease_expires_at = now() + interval '1 minute' where variant_key = $1", v)
    with pytest.raises(ValueError):
        await crud.seed_review(v, "d3", 1, 1)
    await pool.execute("update vehicle_visuals set lease_expires_at = now() - interval '1 minute' where variant_key = $1", v)
    assert (await crud.seed_review(v, "d3", 1, 1))["state"] == "review"
