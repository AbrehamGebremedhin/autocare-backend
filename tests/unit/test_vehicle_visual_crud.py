"""vehicle_visual_crud over a FakePool: bind parameters only, and the Python-side adaptation."""
import re
from uuid import uuid4

import pytest

from app.CRUD import vehicle_visual_crud as crud

UID = str(uuid4())


def _only(fake):
    assert len(fake.calls) == 1
    return fake.calls[0]


def test_style_version_constant():
    assert crud.STYLE_VERSION == "workshop-v1"


async def test_owned_car_uses_the_exact_join_and_str_user_id(fake_pool):
    fake_pool.rows = [{"id": "c1"}]
    assert await crud.owned_car(UID, "c1") == {"id": "c1"}
    method, sql, args = _only(fake_pool)
    assert method == "fetchrow"
    assert " ".join(sql.split()) == (
        'select c.* from user_profiles p join "Car" c on c.id = any(p.cars) where p.user_id = $1::uuid and c.id = $2')
    assert args == (UID, "c1")


async def test_owned_car_invalid_uuid_returns_none_without_query(fake_pool):
    assert await crud.owned_car("not-a-uuid", "c1") is None
    assert fake_pool.calls == []


async def test_get_visual_binds_variant_and_style(fake_pool):
    assert await crud.get_visual("v1") is None
    _, sql, args = _only(fake_pool)
    assert "$1" in sql and "$2" in sql and args == ("v1", crud.STYLE_VERSION)


async def test_ensure_job_created(fake_pool):
    fake_pool.rows = [{"variant_key": "v1", "state": "queued"}]
    row, created = await crud.ensure_job("v1")
    assert created is True and row["state"] == "queued"
    _, sql, args = _only(fake_pool)
    assert "on conflict (variant_key, style_version) do nothing" in sql.lower()
    assert args == ("v1", crud.STYLE_VERSION, "curated")


async def test_ensure_job_existing_selects_the_row(fake_pool):
    answers = [None, {"variant_key": "v1", "state": "ready"}]
    orig = fake_pool.fetchrow

    async def fetchrow(sql, *args):
        await orig(sql, *args)
        return answers.pop(0)

    fake_pool.fetchrow = fetchrow
    row, created = await crud.ensure_job("v1", "provider")
    assert created is False and row["state"] == "ready"
    assert len(fake_pool.calls) == 2
    assert fake_pool.calls[0][2] == ("v1", crud.STYLE_VERSION, "provider")
    assert fake_pool.calls[1][2] == ("v1", crud.STYLE_VERSION)


async def test_jobs_created_last_hour(fake_pool):
    fake_pool.value = 3
    assert await crud.jobs_created_last_hour() == 3
    _, sql, args = _only(fake_pool)
    assert "interval '1 hour'" in sql and args == ()


async def test_choice_functions_are_parameterised(fake_pool):
    fake_pool.value = "v1"
    assert await crud.get_choice(UID, "c1") == "v1"
    await crud.set_choice(UID, "c1", "v1")
    await crud.clear_choice(UID, "c1")
    assert [c[0] for c in fake_pool.calls] == ["fetchval", "execute", "execute"]
    assert "on conflict (user_id, car_id) do update" in fake_pool.calls[1][1].lower()
    assert all(c[2][:2] == (UID, "c1") for c in fake_pool.calls)


async def test_claim_job_all_null_composite_is_none(fake_pool):
    fake_pool.rows = [{"id": None, "state": None}]
    assert await crud.claim_job(60) is None
    _, sql, args = _only(fake_pool)
    assert "claim_due_visual_job($1)" in sql and args == (60,)


async def test_claim_job_returns_row(fake_pool):
    fake_pool.rows = [{"id": "j1", "fencing_token": 1}]
    assert await crud.claim_job(60) == {"id": "j1", "fencing_token": 1}


async def test_renew_lease_reports_whether_a_row_changed(fake_pool):
    fake_pool.rows = [{"id": "j1"}]
    assert await crud.renew_lease("j1", 2, 90) is True
    fake_pool.rows = []
    assert await crud.renew_lease("j1", 2, 90) is False
    _, sql, args = fake_pool.calls[0]
    assert "fencing_token = $2" in sql and "state = 'generating'" in sql
    assert args == ("j1", 2, 90)


@pytest.mark.parametrize("seconds", [None, 30])
async def test_finalize_job_passes_seconds_or_null(fake_pool, seconds):
    fake_pool.value = True
    assert await crud.finalize_job("j1", 3, "failed", None, "boom", seconds) is True
    _, sql, args = _only(fake_pool)
    assert "finalize_visual_job(" in sql and "make_interval" in sql
    assert args == ("j1", 3, "failed", None, "boom", seconds)


async def test_publish_passes_all_values(fake_pool):
    fake_pool.value = False
    assert await crud.publish("j1", 0, "sha", 1024, 420) is False
    _, sql, args = _only(fake_pool)
    assert "publish_visual_revision($1::uuid, $2, $3, $4, $5)" in sql
    assert args == ("j1", 0, "sha", 1024, 420)


async def test_list_review_filters_state(fake_pool):
    fake_pool.rows = [{"variant_key": "v1"}]
    assert await crud.list_review() == [{"variant_key": "v1"}]
    assert "'review'" in fake_pool.calls[0][1]


async def test_reject_and_requeue_report_changed_rows(fake_pool):
    fake_pool.rows = [{"id": "j1"}]
    assert await crud.reject("v1") is True
    assert await crud.requeue("v1") is True
    fake_pool.rows = []
    assert await crud.reject("v1") is False
    assert await crud.requeue("v1") is False
    reject_sql, requeue_sql = fake_pool.calls[0][1], fake_pool.calls[1][1]
    assert "state = 'review'" in reject_sql and "'rejected'" in reject_sql
    assert "state in ('failed', 'review')" in requeue_sql


async def test_provider_ledger(fake_pool):
    await crud.record_provider_call("v1")
    fake_pool.value = 7
    assert await crud.provider_calls_last_day() == 7
    assert fake_pool.calls[0][2] == ("v1",)
    assert "interval '1 day'" in fake_pool.calls[1][1]


async def test_referenced_shas(fake_pool):
    fake_pool.rows = [{"sha": "a"}, {"sha": "b"}, {"sha": "a"}]
    assert await crud.referenced_shas() == {"a", "b"}
    sql = fake_pool.calls[0][1]
    assert "sha256" in sql and "draft_sha256" in sql and "is not null" in sql.lower()


async def test_seed_review_refuses_live_generating_lease(fake_pool):
    fake_pool.rows = []  # the guarded upsert changed nothing
    with pytest.raises(ValueError):
        await crud.seed_review("v1", "sha", 10, 20)


async def test_seed_review_upserts(fake_pool):
    fake_pool.rows = [{"variant_key": "v1", "state": "review"}]
    row = await crud.seed_review("v1", "sha", 10, 20)
    assert row["state"] == "review"
    _, sql, args = _only(fake_pool)
    assert args == ("v1", crud.STYLE_VERSION, "sha", 10, 20)
    assert "published_revision" not in sql.split("do update")[1]


def test_no_string_built_sql():
    src = open(crud.__file__, encoding="utf-8").read()
    assert not re.search(r"\bf[\"']", src) and ".format(" not in src and "% (" not in src
