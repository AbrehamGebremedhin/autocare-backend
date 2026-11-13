"""The Car history block in the prompt, and the turn's reads and writes of it. Synthetic data only."""
from datetime import datetime, timezone

import pytest

from app.schemas.session import DiagnosisState, Fact, SuspectedCause
from app.services import findings
from app.services.diagnosis import run_turn
from app.services.prompts import SYSTEM_PROMPT, build_messages

CAR = {"id": "toyota-echo-2001", "make": "Toyota", "model": "Echo", "year": 2001}
BEARING = "rear-left wheel bearing worn, confirmed by wheel-off spin test"
HUB = "new rear-left hub fitted"


def _row(id, kind, text, source, day, **extra):
    return {"id": id, "kind": kind, "text": text, "norm": findings._norm(text), "status": "active", "source": source,
            "evidence_quote": "q" if source == "user_confirmed" else None,
            "created_at": datetime(2026, 11, day, 10, tzinfo=timezone.utc), **extra}


SESSION_ONE = [
    _row("a", "confirmed_finding", BEARING, "user_confirmed", 2),
    _row("b", "repair_done", HUB, "user_confirmed", 6),
    _row("c", "suspected_cause", "ABS tone ring damaged", "model_suggested", 6),
]


def _text(findings_rows):
    msgs = build_messages(car=CAR, state=DiagnosisState(suspected_causes=[SuspectedCause(cause="Wheel speed sensor fault")]),
                          history=[], text="ABS light comes on over bumps after the new hub", observations=None,
                          passages=[], images=[], findings=findings_rows)
    return msgs[-1]["content"][0]["text"]


def test_the_block_has_both_confirmed_items_with_dates_above_any_hypothesis():
    text = _text(SESSION_ONE)
    block = text.split("## Car history")[1].split("##")[0]
    assert f"- 2026-11-02: {BEARING}" in block and f"- 2026-11-06 (repair): {HUB}" in block
    assert block.index(BEARING) < block.index("ABS tone ring damaged") and block.index(HUB) < block.index("ABS tone ring damaged")
    assert "Unverified hypotheses" in block
    assert text.index("## Car history") < text.index("## Safety flags") < text.index("Current suspected causes")


def test_no_history_means_no_block_and_the_prompt_is_unchanged():
    assert "Car history" not in _text([])
    assert _text([]) == _text(())


def test_the_block_is_byte_identical_when_nothing_changed():
    assert _text(SESSION_ONE) == _text(list(reversed(SESSION_ONE)))


def test_system_prompt_carries_the_history_rules():
    for phrase in ("not ground truth for the current complaint", "do not assume which is right", "ask the user",
                   "Use the confirmed repairs when you reason", "ABS\n  sensor or tone-ring fault",
                   "Never present a hypothesis from the history as a fact"):
        assert phrase in SYSTEM_PROMPT, phrase
    assert "The same goes for the Car history" in SYSTEM_PROMPT


class FakeFindings:
    def __init__(self, rows=()):
        self.rows, self.applied, self.history_args, self.error = list(rows), [], None, None

    async def history(self, user_id, car_id, exclude_session):
        self.history_args = (user_id, car_id, exclude_session)
        if self.error:
            raise self.error
        return self.rows

    async def for_car(self, user_id, car_id):
        if self.error:
            raise self.error
        return []

    async def apply(self, user_id, car_id, session_id, ops):
        self.applied.append((user_id, car_id, session_id, ops))


@pytest.fixture
def fake_findings(monkeypatch):
    fake = FakeFindings(SESSION_ONE)
    monkeypatch.setattr(findings, "crud", fake)
    return fake


async def _events(session, text):
    return [x async for x in run_turn(session, CAR, text, [])]


async def test_a_new_session_prompt_gets_the_history_of_other_sessions(fake_llm, fake_store, session, fake_findings):
    await _events(session, "ABS light comes on over bumps after the new hub")
    assert fake_findings.history_args == (session["user_id"], CAR["id"], session["id"])  # the current session is excluded
    prompt = fake_llm.prompt[-1]["content"][0]["text"]
    assert HUB in prompt and BEARING in prompt


async def test_the_turn_writes_user_facts_as_confirmed_and_state_as_suggestions(fake_llm, fake_store, session, fake_findings):
    fake_llm.facts = [Fact(repair="new rear-left hub fitted", quote="I replaced the rear-left hub", basis="stated")]
    out = await _events(session, "I replaced the rear-left hub last week and now the ABS light flickers")
    assert out[-1][0] == "done"
    [(user_id, car_id, session_id, ops)] = fake_findings.applied
    assert (user_id, car_id, session_id) == (session["user_id"], CAR["id"], session["id"])
    inserted = {o[1]["kind"]: o[1] for o in ops if o[0] == "insert"}
    assert inserted["repair_done"]["source"] == "user_confirmed"
    assert inserted["repair_done"]["evidence_quote"] == "I replaced the rear-left hub"
    assert inserted["suspected_cause"]["source"] == "model_suggested" and inserted["suspected_cause"]["evidence_quote"] is None


async def test_a_findings_outage_never_fails_the_turn(fake_llm, fake_store, session, fake_findings):
    fake_findings.error = RuntimeError("db down")
    out = await _events(session, "ABS light flickers")
    assert [e for e, _ in out][-2:] == ["state", "done"]
    assert "Car history" not in fake_llm.prompt[-1]["content"][0]["text"]
