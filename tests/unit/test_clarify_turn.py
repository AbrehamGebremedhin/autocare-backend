# ruff: noqa: F811  (the fixtures imported below are used by name as test arguments)
"""The clarify gate through run_turn: the event contract, storage and replay, the limits, the safety note and the
findings it must not write. All offline fakes, synthetic data."""
import json
import logging
import re

import pytest

from app.schemas.session import DiagnosisState, Fact, Message, SafetyFlag, SuspectedCause
from app.services import clarify, findings
from app.services.diagnosis import run_turn
from tests.unit.test_car_history import FakeFindings
from tests.unit.test_turn_media import (  # noqa: F401  (fixtures)
    _clean, audio_upload, fake_all, fake_hearing, fake_media, make_upload, video_upload,
)

LINE = {"reason": "frames_unrelated_to_noise", "question_id": "when", "safety_note": "", "seen": "a door seam, close up"}
NOISE = "there is a noise when I drive"


def _ask(fake_llm, **over):
    fake_llm.reply = "CLARIFY " + json.dumps({**LINE, **over})


async def _run(session, car, text, uploads=()):
    return [x async for x in run_turn(session, car, text, list(uploads))]


def _types(events):
    return [e for e, _ in events]


def _prompt(fake_llm) -> str:
    return fake_llm.prompt[-1]["content"][0]["text"]


async def _fresh(fake_store, session):
    return await fake_store.get_for_user(session["id"], session["user_id"])


# --- the contract ----------------------------------------------------------------------------------------------

async def test_a_question_streams_as_plain_text_then_clarify_then_state_then_done(fake_all, session, car, video_upload):
    _ask(fake_all.llm, options=[{"id": "braking", "label": "When braking"}, {"id": "bumps", "label": "Over bumps"}])
    events = await _run(session, car, NOISE, [video_upload])
    types = _types(events)
    assert types[:2] == ["stage", "stage"] and types[-3:] == ["clarify", "state", "done"] and "error" not in types
    text = "".join(d["text"] for e, d in events if e == "token")
    assert text == "When does it happen?\n1. When braking\n2. Over bumps\n" + clarify.TAIL  # an older client shows a normal message
    [(_, payload)] = [x for x in events if x[0] == "clarify"]
    assert payload == {"question": "When does it happen?", "options": [{"id": "braking", "label": "When braking"}, {"id": "bumps", "label": "Over bumps"}],
                       "reason": "frames_unrelated_to_noise"}
    assert "CLARIFY" not in text and "question_id" not in text


async def test_the_question_is_stored_on_the_assistant_message_and_the_state_is_pending(fake_all, fake_store, session, car, video_upload):
    _ask(fake_all.llm)
    events = await _run(session, car, NOISE, [video_upload])
    user, assistant = fake_store.messages(session["id"])
    msg = Message.model_validate(assistant)
    assert msg.clarify.question_id == "when" and msg.clarify.reason == "frames_unrelated_to_noise"
    assert [o.id for o in msg.clarify.options] == ["accelerating", "braking", "turning", "bumps"]
    assert msg.content.startswith("When does it happen?") and events[-1] == ("done", {"message_id": assistant["id"]})
    pending = DiagnosisState.model_validate(fake_store.rows[session["id"]]["state"]).pending_clarification
    assert (pending.message_id, pending.media_message_id, pending.question_id) == (assistant["id"], user["id"], "when")
    assert user["observations"]["seen"] == "a door seam, close up"


async def test_a_clarify_turn_still_runs_the_state_model_so_safety_flags_persist(fake_all, fake_store, session, car, video_upload):
    fake_all.llm.state = DiagnosisState(safety_flags=[SafetyFlag(text="Do not drive until the brakes are checked", severity="critical")])
    _ask(fake_all.llm)
    await _run(session, car, "a noise when I brake", [video_upload])
    saved = DiagnosisState.model_validate(fake_store.rows[session["id"]]["state"])
    assert fake_all.llm.calls == 2 and saved.safety_flags[0].severity == "critical" and saved.pending_clarification is not None


async def test_the_next_turn_knows_it_is_answering_and_reasons_over_the_earlier_media(fake_all, fake_store, session, car, video_upload):
    _ask(fake_all.llm)
    await _run(session, car, NOISE, [video_upload])
    fake_all.llm.reply = "A rear wheel bearing fits a hum over bumps."
    events = await _run(await _fresh(fake_store, session), car, "over bumps")
    prompt = _prompt(fake_all.llm)
    assert "## Pending clarification" in prompt and '"When does it happen?"' in prompt and "Clarify check" not in prompt
    history = " ".join(m["content"] for m in fake_all.llm.prompt if isinstance(m["content"], str))
    assert "earlier model-written description of the frames, not a fact: a door seam, close up" in history
    assert "When does it happen?" in history and "(no longer shown)" in history  # no re-upload: the note stands in
    assert "clarify" not in _types(events) and _types(events)[-1] == "done"
    assert DiagnosisState.model_validate(fake_store.rows[session["id"]]["state"]).pending_clarification is None


async def test_the_answering_turn_cannot_ask_again_even_if_the_model_tries(fake_all, fake_store, session, car, video_upload):
    _ask(fake_all.llm)
    await _run(session, car, NOISE, [video_upload])
    _ask(fake_all.llm, question_id="side")  # the model ignores the closed block
    events = await _run(await _fresh(fake_store, session), car, "I don't know")
    assert "clarify" not in _types(events) and _types(events)[-1] == "done"
    assert DiagnosisState.model_validate(fake_store.rows[session["id"]]["state"]).pending_clarification is None


async def test_the_extractor_sees_the_question_the_user_is_answering(fake_all, fake_store, session, car, video_upload):
    _ask(fake_all.llm)
    await _run(session, car, NOISE, [video_upload])
    fake_all.llm.reply = "ok"
    await _run(await _fresh(fake_store, session), car, "over bumps")
    message, context = fake_all.llm.facts_input
    assert message == "over bumps" and "When does it happen?" in context[-1]["content"] and context[-1]["role"] == "assistant"


def test_an_option_answer_cannot_confirm_a_conclusion():
    facts = [Fact(finding="the wheel bearing is worn", quote="Over bumps", basis="answered_question")]
    [p] = findings.user_proposals(facts, "Over bumps")
    assert p.source == findings.MODEL


# --- when it does not ask --------------------------------------------------------------------------------------

async def test_a_text_only_turn_gets_no_gate_text_and_cannot_ask(fake_llm, fake_store, session, car):
    _ask(fake_llm)
    events = await _run(session, car, "grinding from the front wheels when I brake, pulls to one side")
    prompt = _prompt(fake_llm)
    assert "Clarify check" not in prompt and "CLARIFY" not in prompt and "Pending clarification" not in prompt
    assert "clarify" not in _types(events) and _types(events)[-1] == "done"


async def test_a_media_turn_without_a_question_is_a_normal_streamed_reply(fake_all, session, car, video_upload):
    fake_all.llm.reply = "Likely a belt squeal at start-up."
    events = await _run(session, car, "squeals when I start it", [video_upload])
    assert "## Clarify check" in _prompt(fake_all.llm) and "clarify" not in _types(events)
    assert "".join(d["text"] for e, d in events if e == "token") == fake_all.llm.reply


async def test_the_session_cap_removes_the_gate_text(fake_all, fake_store, session, car, video_upload, monkeypatch):
    asked = [{"id": str(i), "role": "assistant", "content": "q", "created_at": "2026-11-14T10:00:00+00:00",
              "clarify": {"question": "q", "options": [], "reason": "tags_weak", "question_id": "when"}} for i in range(2)]
    _ask(fake_all.llm)
    events = await _run({**session, "messages": asked}, car, NOISE, [video_upload])
    assert "Clarify check" not in _prompt(fake_all.llm) and "clarify" not in _types(events)
    monkeypatch.setenv("MAX_CLARIFICATIONS_PER_SESSION", "0")
    from app.core.config import get_settings
    get_settings.cache_clear()
    try:
        await _run(session, car, NOISE, [video_upload])
        assert "Clarify check" not in _prompt(fake_all.llm)
    finally:
        monkeypatch.delenv("MAX_CLARIFICATIONS_PER_SESSION")
        get_settings.cache_clear()


async def test_a_second_upload_after_an_answered_question_may_ask_again(fake_all, fake_store, session, car, video_upload, make_upload):
    _ask(fake_all.llm)
    await _run(session, car, NOISE, [video_upload])
    fake_all.llm.reply = "ok"
    await _run(await _fresh(fake_store, session), car, "over bumps")
    _ask(fake_all.llm, question_id="side")
    events = await _run(await _fresh(fake_store, session), car, "here is another clip", [make_upload("video", b"v2", duration=3.0, has_audio=True)])
    assert "clarify" in _types(events)


# --- a bad line never fails the turn ----------------------------------------------------------------------------

def _script(monkeypatch, replies):
    """stream_reply answers with the next reply each call (tokens split on spaces); an Exception in the list is raised."""
    from app.services import llm

    seen = []

    async def stream_reply(messages):
        seen.append(messages)
        reply = replies[len(seen) - 1]
        if isinstance(reply, Exception):
            raise reply
        for token in re.findall(r"\S+\s*", reply):
            yield token

    monkeypatch.setattr(llm, "stream_reply", stream_reply)
    return seen


def _streamed(events):
    return "".join(d["text"] for e, d in events if e == "token")


@pytest.mark.parametrize("reply,code", [
    ('CLARIFY {"reason": "tags_weak", "question_id": "when", "options": [{"id": "braking", "la', "bad_json"),
    ("CLARIFY " + json.dumps({**LINE, "options": [{"id": "braking", "label": "x"}, {"id": "nope", "label": "y"}]}), "unknown_option"),
    ("CLARIFY " + json.dumps({**LINE, "options": "braking"}), "bad_options"),
])
async def test_a_bad_line_that_names_a_bank_question_shows_the_bank_question_never_the_json(fake_all, fake_store, session, car, video_upload, caplog, reply, code):
    fake_all.llm.reply = reply
    with caplog.at_level(logging.INFO):
        events = await _run(session, car, NOISE, [video_upload])
    text = _streamed(events)
    assert text == "When does it happen?\n1. When accelerating\n2. When braking\n3. When turning\n4. Over bumps\n" + clarify.TAIL
    assert "{" not in text and "question_id" not in text and "CLARIFY" not in text
    assert _types(events)[-3:] == ["clarify", "state", "done"] and "error" not in _types(events)
    assert f"bad_line={code}/bank" in caplog.text
    assistant = fake_store.messages(session["id"])[1]
    assert assistant["clarify"]["question_id"] == "when" and assistant["clarify"]["reason"] == "other"
    assert DiagnosisState.model_validate(fake_store.rows[session["id"]]["state"]).pending_clarification is not None


@pytest.mark.parametrize("reply,code", [
    ("CLARIFY " + json.dumps({**LINE, "question_id": "colour"}), "unknown_question"),
    ("CLARIFY " + json.dumps({**LINE, "question_id": ["when"]}), "unknown_question"),
    ('CLARIFY {"reason": "tags_weak"', "bad_json"),
])
async def test_a_bad_line_with_no_bank_question_reruns_the_turn_once_with_the_gate_off(fake_all, fake_store, session, car, video_upload, caplog, monkeypatch, reply, code):
    seen = _script(monkeypatch, [reply, "A wheel bearing hum fits a noise that follows road speed."])
    with caplog.at_level(logging.INFO):
        events = await _run(session, car, NOISE, [video_upload])
    assert len(seen) == 2 and "## Clarify check" in seen[0][-1]["content"][0]["text"] and "Clarify check" not in seen[1][-1]["content"][0]["text"]
    assert _streamed(events) == "A wheel bearing hum fits a noise that follows road speed."
    assert "clarify" not in _types(events) and "error" not in _types(events) and _types(events)[-1] == "done"
    assert f"bad_line={code}/rerun" in caplog.text
    assistant = fake_store.messages(session["id"])[1]
    assert assistant["clarify"] is None and assistant["content"] == _streamed(events)
    assert DiagnosisState.model_validate(fake_store.rows[session["id"]]["state"]).pending_clarification is None


@pytest.mark.parametrize("second", [RuntimeError("provider down"), "", "CLARIFY " + json.dumps(LINE), 'CLARIFY {"question_id": "colour"}'])
async def test_if_the_rerun_also_fails_the_user_gets_the_plain_error_and_no_json(fake_all, fake_store, session, car, video_upload, monkeypatch, second):
    seen = _script(monkeypatch, ["CLARIFY " + json.dumps({**LINE, "question_id": "colour"}), second])
    events = await _run(session, car, NOISE, [video_upload])
    assert len(seen) == 2 and _types(events)[-1] == "error" and "clarify" not in _types(events)
    [(_, error)] = [x for x in events if x[0] == "error"]
    assert error["message"] == "The diagnosis service failed, please retry." and error["retry_message_id"]
    assert "{" not in _streamed(events) and "question_id" not in _streamed(events)
    user, failed = fake_store.messages(session["id"])
    assert failed["status"] == "failed" and failed["retry_of"] == user["id"]


# --- safety is never withheld -----------------------------------------------------------------------------------

async def test_the_backstop_note_is_added_only_when_the_model_left_it_out(fake_all, session, car, video_upload):
    _ask(fake_all.llm)
    events = await _run(session, car, "the ABS light is on and there is a noise", [video_upload])
    text = "".join(d["text"] for e, d in events if e == "token")
    assert text.startswith("Safety first: if the brakes feel soft")
    _ask(fake_all.llm, safety_note="Stop driving and have the ABS checked.")
    events = await _run(session, car, "the ABS light is on and there is a noise", [video_upload])
    text = "".join(d["text"] for e, d in events if e == "token")
    assert text.startswith("Stop driving and have the ABS checked.\n\n") and "Safety first" not in text


async def test_no_safety_topic_means_no_note(fake_all, session, car, video_upload):
    _ask(fake_all.llm)
    text = "".join(d["text"] for e, d in await _run(session, car, NOISE, [video_upload]) if e == "token")
    assert text.startswith("When does it happen?")


# --- findings ----------------------------------------------------------------------------------------------------

@pytest.fixture
def fake_findings(monkeypatch):
    fake = FakeFindings()
    monkeypatch.setattr(findings, "crud", fake)
    return fake


def _inserted(fake):
    return [o[1] for _, _, _, ops in fake.applied for o in ops if o[0] == "insert"]


async def test_a_clarify_turn_about_a_door_seam_video_writes_no_suspected_cause(fake_all, session, car, video_upload, fake_findings):
    fake_all.llm.state = DiagnosisState(suspected_causes=[SuspectedCause(cause="Panel gap at the door seam")], confirmed_findings=["door seam gap"])
    fake_all.llm.facts = [Fact(repair="replaced the rear-left hub", quote="I replaced the rear-left hub", basis="stated")]
    _ask(fake_all.llm)
    events = await _run(session, car, "there is a noise, I replaced the rear-left hub last week", [video_upload])
    assert "clarify" in _types(events)
    kinds = {r["kind"]: r["source"] for r in _inserted(fake_findings)}
    assert kinds == {"repair_done": "user_confirmed"}  # the user's own fact stays; no model-proposed row


async def test_the_same_state_on_a_normal_turn_is_stored_as_before(fake_all, session, car, video_upload, fake_findings):
    fake_all.llm.state = DiagnosisState(suspected_causes=[SuspectedCause(cause="Panel gap at the door seam")])
    fake_all.llm.reply = "A belt squeal is likely."
    await _run(session, car, "squeals when I start it", [video_upload])
    assert {r["kind"] for r in _inserted(fake_findings)} == {"suspected_cause"}


# --- the log: counts and numbers, never content -------------------------------------------------------------------

async def test_one_log_line_per_media_turn_with_numbers_and_no_content(fake_all, session, car, video_upload, caplog):
    _ask(fake_all.llm, question="Secret wording of the question?", seen="a very private driveway")
    with caplog.at_level(logging.INFO):
        await _run(session, car, "my private complaint text", [video_upload])
    [line] = [r.getMessage() for r in caplog.records if "clarify gate" in r.getMessage()]
    assert "fired=True" in line and "reason=frames_unrelated_to_noise" in line and "asked=1/2" in line
    assert "top_scores=[0.7]" in line and "margin=None" in line and "rms_dbfs=" in line
    for secret in ("Secret wording", "private driveway", "private complaint", "squeals when I start"):
        assert secret not in line and secret not in caplog.text


async def test_a_text_turn_logs_no_gate_line(fake_llm, session, car, caplog):
    with caplog.at_level(logging.INFO):
        await _run(session, car, "fan clicks on and off")
    assert "clarify gate" not in caplog.text
