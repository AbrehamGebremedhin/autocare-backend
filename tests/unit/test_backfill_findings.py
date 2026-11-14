"""scripts/backfill_findings.py: what a session's state becomes, and that a rerun adds nothing (fake pool). Synthetic data only."""
import importlib.util
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("backfill_findings", ROOT / "scripts" / "backfill_findings.py")
backfill = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(backfill)

U = "11111111-1111-1111-1111-111111111111"
S1 = "22222222-2222-2222-2222-222222222222"
AT = datetime(2026, 11, 2, 10, tzinfo=timezone.utc)

STATE = {
    "suspected_causes": [{"cause": "ABS tone ring damaged", "rank": "likely"}],
    "confirmed_findings": ["rear-left wheel bearing worn"],
    "tests_done": [{"test": "spun the hub by hand", "result": "rough"}, {"test": "checked the fuse", "result": "fine"}],
    "ruled_out": [{"cause": "loose wheel nut", "reason": "torque checked"}],
}
USER_FACTS = [{"content": "I took the wheel off and spun the hub by hand", "facts": [{"test": "Spun the hub by hand.", "result": "rough"}]}]


def session(**extra):
    return {"id": S1, "user_id": U, "car_id": "toyota-echo-2001", "state": STATE, "user_facts": USER_FACTS, "at": AT, **extra}


def by_kind(rows):
    return {(r["kind"], r["norm"]): r for r in rows}


def test_each_part_of_the_state_becomes_the_right_kind_and_source():
    rows = by_kind(backfill.rows_for(session()))
    assert rows[("confirmed_finding", "rear left wheel bearing worn")]["source"] == "model_suggested"  # nothing proves the user said it
    assert rows[("suspected_cause", "abs tone ring damaged")]["source"] == "model_suggested"
    assert rows[("ruled_out", "loose wheel nut")]["text"] == "loose wheel nut (torque checked)"
    spun = rows[("test_done", "spun the hub by hand")]
    assert spun["source"] == "user_confirmed" and spun["evidence_quote"] == "I took the wheel off and spun the hub by hand"
    fuse = rows[("test_done", "checked the fuse")]  # no user message carries it: not provably the user's
    assert fuse["source"] == "model_suggested" and fuse["evidence_quote"] is None
    assert all(r["created_at"] == AT and r["session_id"] == S1 for r in rows.values())


def test_a_user_confirmed_row_always_has_a_quote():
    for r in backfill.rows_for(session(user_facts=[{"content": "", "facts": [{"test": "spun the hub by hand"}]}])):
        assert r["source"] == "model_suggested" or r["evidence_quote"]


def test_legacy_states_with_float_likelihoods_and_string_flags_still_map():
    legacy = {"suspected_causes": [{"cause": "Fan relay", "likelihood": 0.7}], "safety_flags": ["Do not drive"], "title": "Fan"}
    assert [(r["kind"], r["text"]) for r in backfill.rows_for(session(state=legacy, user_facts=[]))] == [("suspected_cause", "Fan relay")]
    assert backfill.rows_for(session(state={}, user_facts=None)) == []
    assert backfill.rows_for(session(state=None)) == []


def test_keys_are_stable_so_a_second_run_conflicts_on_every_row():
    first, second = backfill.rows_for(session()), backfill.rows_for(session())
    assert [r["backfill_key"] for r in first] == [r["backfill_key"] for r in second]
    assert len({r["backfill_key"] for r in first}) == len(first)
    assert "ON CONFLICT DO NOTHING" in backfill._INSERT_SQL


def test_the_owners_own_rows_go_first():
    other = session(id="33333333-3333-3333-3333-333333333333", user_facts=[], at=datetime(2026, 10, 1, tzinfo=timezone.utc))
    ordered = backfill.plan([other, session()])
    assert ordered[0]["source"] == "user_confirmed"
    assert [r["source"] for r in ordered] == sorted((r["source"] for r in ordered), key=lambda s: s != "user_confirmed")


def test_only_cars_still_in_the_garage_are_read():
    assert "JOIN user_profiles p ON p.user_id = s.user_id AND s.car_id = ANY(p.cars)" in backfill._SESSIONS_SQL


async def test_dry_run_reads_and_writes_nothing(fake_pool):
    fake_pool.rows = [session()]
    assert await backfill.backfill(dry_run=True) == (5, 0)
    assert [c[0] for c in fake_pool.calls] == ["fetch"]


async def test_a_run_inserts_each_row_once_and_counts_only_new_ones(fake_pool):
    fake_pool.rows = [session()]
    fake_pool.status = "INSERT 0 1"
    assert await backfill.backfill(dry_run=False) == (5, 5)
    fake_pool.calls.clear()
    fake_pool.status = "INSERT 0 0"  # the second run: every row conflicts
    assert await backfill.backfill(dry_run=False) == (5, 0)
    assert [c[0] for c in fake_pool.calls] == ["fetch"] + ["execute"] * 5
