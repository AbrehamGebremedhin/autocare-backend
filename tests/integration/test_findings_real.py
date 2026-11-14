"""Real Postgres check of car_findings: the SQL of FindingCRUD, the constraints and cascades, the write rules end to end,
and the backfill twice. Runs only with `-m db` and TEST_DATABASE_URL pointing at a throwaway database."""
import importlib.util
import os
from pathlib import Path
from uuid import uuid4

import asyncpg
import pytest

from app.db import pool
from app.schemas.session import DiagnosisState, Fact, RuledOut, SuspectedCause
from app.services import findings
from app.services.user_profile_service import UserProfileService

pytestmark = [
    pytest.mark.db,
    pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL is not set"),
]

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = ROOT / "app" / "db" / "sql" / "schema.sql"
_spec = importlib.util.spec_from_file_location("backfill_findings", ROOT / "scripts" / "backfill_findings.py")
backfill = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(backfill)
CAR = "toyota-echo-2001"


@pytest.fixture
async def real_pool():
    pool.set_pool_for_tests(None)  # drop the autouse fake
    await pool.init_pool(os.environ["TEST_DATABASE_URL"])
    await pool.execute(SCHEMA.read_text(encoding="utf-8"))
    await pool.execute(SCHEMA.read_text(encoding="utf-8"))  # a second run is a no-op
    try:
        yield
    finally:
        await pool.close_pool()


async def _user(car=CAR) -> str:
    uid = await pool.fetchval("INSERT INTO users (email, password_hash) VALUES ($1, 'x') RETURNING id", f"{uuid4().hex}@example.test")
    await pool.execute("INSERT INTO user_profiles (user_id, cars) VALUES ($1, $2)", uid, [car])
    return uid


async def _session(uid, state=None, messages=None, car=CAR) -> str:
    return await pool.fetchval(
        'INSERT INTO "Chatsession" (user_id, car_id, state, messages) VALUES ($1, $2, $3, $4) RETURNING id',
        uid, car, state or {}, messages or [],
    )


async def _all(uid):
    return await pool.fetch("SELECT * FROM car_findings WHERE user_id = $1 ORDER BY created_at, id", uid)


async def test_a_session_writes_and_the_next_session_reads_it(real_pool):
    uid, other = await _user(), await _user()
    s1, s2 = await _session(uid), await _session(uid)
    state = DiagnosisState(suspected_causes=[SuspectedCause(cause="ABS tone ring damaged")])
    await findings.record_turn(uid, CAR, s1, [Fact(repair="new rear-left hub fitted", quote="replaced the hub", basis="stated")],
                               "I replaced the hub", state)
    await findings.record_turn(uid, CAR, s1, [], "", state)  # the same turn again: nothing new
    rows = await _all(uid)
    assert sorted((r["kind"], r["source"]) for r in rows) == [("repair_done", "user_confirmed"), ("suspected_cause", "model_suggested")]
    assert next(r for r in rows if r["kind"] == "repair_done")["evidence_quote"] == "replaced the hub"

    assert await findings.car_history(uid, CAR, s1) == []  # the current session is excluded
    assert len(await findings.car_history(uid, CAR, s2)) == 2
    block = findings.render_history(await findings.car_history(uid, CAR, s2))
    assert block.index("new rear-left hub fitted") < block.index("ABS tone ring damaged")

    ids = [r["id"] for r in rows]  # nothing crosses users
    assert await findings.car_history(other, CAR, None) == [] and await findings.crud.listing(other, CAR, None) == []
    assert await findings.crud.get(other, CAR, ids[0]) is None
    assert await findings.crud.retract(other, CAR, ids[0]) is None and await findings.crud.delete_one(other, CAR, ids[0]) is False
    assert await findings.crud.delete_all(other, CAR) == 0 and len(await _all(uid)) == 2


async def test_ruled_out_supersedes_and_links_and_the_model_cannot_resurrect_it(real_pool):
    uid = await _user()
    s = await _session(uid)
    await findings.record_turn(uid, CAR, s, [], "", DiagnosisState(suspected_causes=[SuspectedCause(cause="Loose wheel nut")]))
    await findings.record_turn(uid, CAR, s, [], "", DiagnosisState(ruled_out=[RuledOut(cause="loose wheel nut", reason="torqued")]))
    await findings.record_turn(uid, CAR, s, [], "", DiagnosisState(suspected_causes=[SuspectedCause(cause="Loose wheel nut")]))
    rows = {r["kind"]: r for r in await _all(uid)}
    assert len(rows) == 2 and rows["suspected_cause"]["status"] == "superseded"
    assert rows["suspected_cause"]["superseded_by"] == rows["ruled_out"]["id"] and rows["ruled_out"]["status"] == "active"


async def test_confirm_resolve_and_the_constraints(real_pool):
    uid = await _user()
    s = await _session(uid)
    await findings.record_turn(uid, CAR, s, [], "", DiagnosisState(confirmed_findings=["rear-left bearing worn"]))
    [row] = await _all(uid)
    assert row["source"] == "model_suggested" and row["evidence_quote"] is None
    confirmed = await findings.crud.edit(uid, CAR, row["id"], None, None, findings.APP_CONFIRMED)
    assert confirmed["source"] == "user_confirmed"
    fixed = await findings.crud.resolve(uid, CAR, row["id"], {"id": str(uuid4()), "text": "replaced the hub", "norm": "replaced the hub", "evidence_quote": "replaced the hub"})
    old = await findings.crud.get(uid, CAR, row["id"])
    assert fixed["kind"] == "repair_done" and old["status"] == "superseded" and old["superseded_by"] == fixed["id"]
    assert await findings.crud.resolve(uid, CAR, row["id"], {"id": str(uuid4()), "text": "x", "norm": "x", "evidence_quote": "x"}) is None
    with pytest.raises(asyncpg.CheckViolationError):  # the database refuses a confirmation with no words behind it
        await pool.execute("INSERT INTO car_findings (user_id, car_id, kind, text, norm, source) VALUES ($1, $2, 'ruled_out', 'a', 'a', 'user_confirmed')", uid, CAR)
    with pytest.raises(asyncpg.UniqueViolationError):
        await pool.execute("INSERT INTO car_findings (user_id, car_id, kind, text, norm, source, evidence_quote) VALUES ($1, $2, 'repair_done', 'r', 'replaced the hub', 'user_confirmed', 'q')", uid, CAR)


async def test_deleting_a_session_keeps_findings_and_deleting_the_user_or_the_car_removes_them(real_pool):
    uid = await _user()
    s = await _session(uid)
    await findings.record_turn(uid, CAR, s, [], "", DiagnosisState(suspected_causes=[SuspectedCause(cause="Tone ring")]))
    await pool.execute('DELETE FROM "Chatsession" WHERE id = $1', s)
    [row] = await _all(uid)
    assert row["session_id"] is None
    assert (await UserProfileService().remove_car_from_user(uid, CAR))["success"] is True
    assert await _all(uid) == []

    uid2 = await _user()
    await findings.record_turn(uid2, CAR, await _session(uid2), [], "", DiagnosisState(suspected_causes=[SuspectedCause(cause="Tone ring")]))
    await pool.execute("DELETE FROM users WHERE id = $1", uid2)
    assert await _all(uid2) == []


async def test_backfill_twice_adds_nothing_the_second_time_and_respects_retraction(real_pool):
    uid = await _user()
    state = {"suspected_causes": [{"cause": "Tone ring", "rank": "likely"}], "confirmed_findings": ["bearing worn"],
             "tests_done": [{"test": "spun the hub", "result": "rough"}]}
    msgs = [{"id": "m", "role": "user", "content": "I spun the hub", "observations": {"facts": [{"test": "spun the hub", "result": "rough"}]}}]
    await _session(uid, state, msgs)
    await _session(uid, state, [], car="not-in-my-garage")
    first = await backfill.backfill(dry_run=False)
    assert first[1] == 3 and len(await _all(uid)) == 3
    rows = {r["kind"]: r for r in await _all(uid)}
    assert rows["test_done"]["source"] == "user_confirmed" and rows["confirmed_finding"]["source"] == "model_suggested"
    await findings.crud.retract(uid, CAR, rows["suspected_cause"]["id"])
    assert (await backfill.backfill(dry_run=False))[1] == 0
    assert len(await _all(uid)) == 3 and (await findings.crud.get(uid, CAR, rows["suspected_cause"]["id"]))["status"] == "retracted"
    assert (await backfill.backfill(dry_run=True))[1] == 0


async def test_after_the_car_leaves_the_garage_a_later_turn_writes_and_the_owner_can_still_list_and_delete(real_pool):
    uid = await _user()
    old = await _session(uid)
    assert (await UserProfileService().remove_car_from_user(uid, CAR))["success"] is True
    await findings.record_turn(uid, CAR, old, [], "", DiagnosisState(suspected_causes=[SuspectedCause(cause="Tone ring")]))
    [row] = await findings.crud.listing(uid, CAR, None)  # no garage involved
    assert row["text"] == "Tone ring"
    assert await findings.crud.delete_one(uid, CAR, row["id"]) is True and await findings.crud.listing(uid, CAR, None) == []


async def test_resolve_never_touches_another_owners_row(real_pool):
    uid, other = await _user(), await _user()
    await findings.record_turn(uid, CAR, await _session(uid), [], "", DiagnosisState(confirmed_findings=["bearing worn"]))
    [mine] = await _all(uid)
    new_row = {"id": str(uuid4()), "text": "replaced the hub", "norm": "replaced the hub", "evidence_quote": "q"}
    assert await findings.crud.resolve(other, CAR, mine["id"], new_row) is None
    assert (await findings.crud.get(uid, CAR, mine["id"]))["status"] == "active"
