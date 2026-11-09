"""Replay of a real exported session (Toyota Platz, rear bearing, then an intermittent ABS light after a hub swap).

The model is stubbed, so this checks everything the code controls about the final turn: what the model is shown
(the open critical hub-nut flag first, the new hub already established, no float likelihoods, clean citation
labels) and what is stored and sent back (clean citations, no raw floats, no synthetic user message)."""
import json
import re
from pathlib import Path

from app.schemas.session import Fact
from app.services.diagnosis import run_turn

EXPORT = json.loads((Path(__file__).parent.parent / "fixtures" / "platz_abs_session.json").read_text(encoding="utf-8"))
SYNTHETIC = "Please continue from my last saved message. Do not repeat it."
FINAL_USER = "it is a new wheel hub on the car"
FLOAT = re.compile(r"likelihood|\b0\.\d+\b")


def _seed(fake_store, session):
    row = fake_store.rows[session["id"]]
    kept = [m for m in EXPORT["messages"] if m["content"] != SYNTHETIC]  # a retry never writes this one
    kept = kept[: [m["content"] for m in kept].index(FINAL_USER)]  # the transcript up to the final turn
    row["messages"] = [{"id": f"m{i}", "media": [], "observations": None, "created_at": "2026-11-09T00:00:00+00:00",
                        "status": "ok", "retry_of": None, **m} for i, m in enumerate(kept)]
    row["state"] = EXPORT["state"]  # the old shape: float likelihoods, string flags
    row["title"] = EXPORT["state"]["title"]
    return row


async def _final_turn(fake_llm, fake_kb, fake_store, session):
    car = EXPORT["car"]
    row = _seed(fake_store, session)
    fake_kb.passages = [{"source_type": "manual", "title": "Toyota Platz 2001 manual (s3.amazonaws.com)", "page": 140,
                         "content": "ABS warning light: conventional brakes still work."}]
    fake_llm.reply = ("Check the hub nut torque before you drive again. Brakes still work "
                      "(<Toyota Platz 2001 manual (s3.amazonaws.com), p. 140>).")
    fake_llm.facts = [Fact(test="fitted a new wheel hub", result="new hub on the car", part="rear left hub")]
    fake_llm.state = None  # the state call keeps the migrated state
    events = [x async for x in run_turn(row, car, FINAL_USER, [])]
    return row, events


def test_the_export_had_the_two_consecutive_user_messages():
    roles = [m["role"] for m in EXPORT["messages"]]
    assert roles[2:4] == ["user", "user"] and EXPORT["messages"][3]["content"] == SYNTHETIC


async def test_final_turn_prompt_leads_with_the_hub_nut_flag(fake_llm, fake_kb, fake_store, session):
    await _final_turn(fake_llm, fake_kb, fake_store, session)
    text = fake_llm.prompt[-1]["content"][0]["text"]
    flags = text.split("## Safety flags")[1].split("##")[0].strip().splitlines()[1:]
    assert flags[0].startswith("- CRITICAL:") and "hub nut" in flags[0]
    assert all("CRITICAL" in f for f in flags[:3]) and len(flags) <= 5
    assert "first paragraph of your reply states it" in fake_llm.prompt[0]["content"]


async def test_final_turn_has_no_float_likelihoods_anywhere(fake_llm, fake_kb, fake_store, session):
    row, events = await _final_turn(fake_llm, fake_kb, fake_store, session)
    text = fake_llm.prompt[-1]["content"][0]["text"]
    causes = text.split("## Current suspected causes")[1].split("##")[0]
    assert "(most likely)" in causes and not FLOAT.search(causes)
    sent_state = next(d for e, d in events if e == "state")
    assert not FLOAT.search(json.dumps(sent_state)) and not FLOAT.search(json.dumps(fake_store.rows[session["id"]]["state"]))


async def test_final_turn_citations_are_clean(fake_llm, fake_kb, fake_store, session):
    row, events = await _final_turn(fake_llm, fake_kb, fake_store, session)
    assert "[Toyota Platz 2001 manual, p. 140]" in fake_llm.prompt[-1]["content"][0]["text"]
    stored = fake_store.messages(session["id"])[-1]["content"]
    assert "(Toyota Platz 2001 manual, p. 140)" in stored
    assert not re.search(r"amazonaws|[<>]", stored)


async def test_the_new_hub_is_established_for_the_next_turn_and_nothing_synthetic_is_stored(
    fake_llm, fake_kb, fake_store, session
):
    row, _ = await _final_turn(fake_llm, fake_kb, fake_store, session)
    saved = fake_store.rows[session["id"]]
    assert saved["state"]["tests_done"] == [{"test": "fitted a new wheel hub", "result": "new hub on the car"}]
    assert saved["title"] == EXPORT["state"]["title"]  # the title did not move
    assert SYNTHETIC not in [m["content"] for m in saved["messages"]]
    roles = [m["role"] for m in saved["messages"]]
    assert all(a != b for a, b in zip(roles, roles[1:]))  # strictly alternating
    await _next(fake_llm, fake_store, session)
    follow_up = fake_llm.prompt[-1]["content"][0]["text"]
    assert "test done: fitted a new wheel hub -> new hub on the car" in follow_up
    assert "do not ask the user for anything listed there" in fake_llm.prompt[0]["content"]


async def _next(fake_llm, fake_store, session):
    fake_llm.facts = []
    return [x async for x in run_turn(fake_store.rows[session["id"]], EXPORT["car"], "the light is still on", [])]
