from datetime import datetime

from app.core.config import get_settings
from app.schemas.session import DiagnosisState
from app.services import llm
from app.services.diagnosis import run_turn, sse

LLM_ERROR = "The diagnosis service failed, please retry."


async def _run(session, car, text):
    return [x async for x in run_turn(session, car, text, [])]


async def test_event_order_and_two_llm_calls(fake_llm, fake_store, session, car):
    events = [e async for e, _ in run_turn(session, car, "fan clicks on and off", [])]
    assert events[0] == "stage" and events[-2:] == ["state", "done"] and "token" in events
    assert fake_llm.calls == 2


async def test_tokens_saved_reply_state_and_done(fake_llm, fake_store, session, car):
    out = await _run(session, car, "fan clicks on and off")
    assert out[0] == ("stage", {"name": "searching"})
    reply = "".join(d["text"] for e, d in out if e == "token")
    assert reply == fake_llm.reply
    user, assistant = fake_store.messages(session["id"])
    assert (user["role"], user["content"]) == ("user", "fan clicks on and off")
    assert (assistant["role"], assistant["content"]) == ("assistant", reply)
    assert out[-2] == ("state", fake_llm.state.model_dump())
    assert out[-1] == ("done", {"message_id": assistant["id"]})
    for m in (user, assistant):
        assert len(m["id"]) == 32 and int(m["id"], 16) >= 0
        assert datetime.fromisoformat(m["created_at"]).utcoffset().total_seconds() == 0
        assert m["media"] == [] and m["observations"] is None


async def test_user_message_saved_before_llm_called(fake_llm, fake_store, session, car, monkeypatch):
    seen = []
    inner = llm.stream_reply

    async def spy(messages):
        seen.append([m["role"] for m in fake_store.messages(session["id"])])
        async for t in inner(messages):
            yield t

    monkeypatch.setattr(llm, "stream_reply", spy)
    await _run(session, car, "fan clicks")
    assert seen == [["user"]]


async def test_llm_failure_saves_user_message_only(fake_llm, fake_store, session, car):
    fake_llm.fail_stream_after = 1
    events = [e async for e, _ in run_turn(session, car, "fan clicks", [])]
    assert events[-1] == "error"
    user, failed = fake_store.messages(session["id"])
    assert user["role"] == "user" and (failed["role"], failed["status"], failed["retry_of"]) == ("assistant", "failed", user["id"])


async def test_llm_failure_keeps_state_and_title(fake_llm, fake_store, session, car):
    fake_llm.fail_stream_after = 2
    out = await _run(session, car, "fan clicks")
    assert out[-1][0] == "error" and out[-1][1]["message"] == LLM_ERROR
    assert [e for e, _ in out].count("token") == 2
    row = fake_store.rows[session["id"]]
    assert row["state"] == {} and row["title"] == "New diagnosis"
    assert fake_llm.calls == 1  # no state call after a failed reply


async def test_stream_failing_before_first_token_is_an_error(fake_llm, fake_store, session, car):
    fake_llm.fail_stream_after = 0
    out = await _run(session, car, "fan clicks")
    assert [e for e, _ in out] == ["stage", "error"] and out[-1][1]["message"] == LLM_ERROR
    assert [m["role"] for m in fake_store.messages(session["id"])] == ["user", "assistant"]  # the second is the failed attempt


async def test_utf8_round_trip(fake_llm, fake_store, session, car):
    fake_llm.reply = "Opens at 82 °C — check it. Überprüfen."
    _ = [x async for x in run_turn(session, car, "Temperatur?", [])]
    assert fake_store.messages(session["id"])[-1]["content"] == "Opens at 82 °C — check it. Überprüfen."


async def test_title_updated_from_state(fake_llm, fake_store, session, car):
    await _run(session, car, "fan clicks")
    row = fake_store.rows[session["id"]]
    assert row["title"] == "Fan cycling on and off"
    assert row["state"] == fake_llm.state.model_dump()


async def test_long_title_is_capped_before_it_is_saved(fake_llm, fake_store, session, car):
    fake_llm.state = DiagnosisState(title="  " + "Fan relay clicking " * 10)
    out = await _run(session, car, "fan clicks")
    title = fake_store.rows[session["id"]]["title"]
    assert title == ("Fan relay clicking " * 10)[:60].rstrip() and len(title) <= 60
    assert out[-2][1]["title"] == title


def test_state_from_llm_json_caps_title():
    assert len(DiagnosisState.model_validate({"title": "x" * 500}).title) == 60


async def test_unchanged_state_still_completes_and_keeps_title(fake_llm, fake_store, car):
    previous = DiagnosisState(safety_flags=["stop if overheating"], title="")
    session = fake_store.add("u", car["id"], title="Overheating", state=previous.model_dump())
    fake_llm.state = None  # extract_state falls back to the previous state
    out = await _run(session, car, "still hot")
    assert out[-2] == ("state", previous.model_dump()) and out[-1][0] == "done"
    row = fake_store.rows[session["id"]]
    assert row["title"] == "Overheating" and [m["role"] for m in row["messages"]] == ["user", "assistant"]


async def test_kb_failure_does_not_kill_the_turn(fake_llm, fake_kb, fake_store, session, car):
    fake_kb.error = RuntimeError("relation kb_passages does not exist")
    out = await _run(session, car, "fan clicks")
    assert out[-1][0] == "done"
    assert "none found" in fake_llm.prompt[-1]["content"][0]["text"]


async def test_kb_searched_with_text_and_car_and_passages_reach_prompt(fake_llm, fake_kb, fake_store, session, car):
    fake_kb.passages = [{"source_type": "manual", "title": "Echo manual", "page": 212, "content": "Coolant: Toyota SLLC."}]
    await _run(session, car, "coolant type")
    assert fake_kb.queries == [("coolant type", car["id"], get_settings().KB_TOP_K)]
    assert "[Echo manual, p. 212]" in fake_llm.prompt[-1]["content"][0]["text"]


async def test_state_call_gets_history_new_message_and_reply(fake_llm, fake_store, car):
    history = [
        {"id": "a", "role": "user", "content": "fan clicks", "media": [], "observations": None, "created_at": "x"},
        {"id": "b", "role": "assistant", "content": "check relay", "media": [], "observations": None, "created_at": "y"},
    ]
    session = fake_store.add("u", car["id"], messages=history)
    await _run(session, car, "relay is fine")
    assert fake_llm.conversation == [
        {"role": "user", "content": "fan clicks"},
        {"role": "assistant", "content": "check relay"},
        {"role": "user", "content": "relay is fine"},
        {"role": "assistant", "content": fake_llm.reply},
    ]
    assert [m["role"] for m in fake_llm.prompt] == ["system", "user", "assistant", "user"]


def test_sse_framing_keeps_utf8():
    assert sse("token", {"text": "°C — ü"}) == 'event: token\ndata: {"text": "°C — ü"}\n\n'


async def test_reported_facts_are_stored_on_the_user_message_and_feed_tests_done(fake_llm, fake_store, session, car):
    from app.schemas.session import Fact

    fake_llm.facts = [Fact(test="spun the bare hub, tire off", result="noise still there, once per revolution", part="rear left hub")]
    await _run(session, car, "tire off, hub only, noise still there, once per revolution")
    user = fake_store.messages(session["id"])[0]
    assert user["observations"]["facts"] == [fake_llm.facts[0].model_dump()]
    assert fake_llm.facts_input[0].startswith("tire off, hub only")
    state = fake_store.rows[session["id"]]["state"]
    assert state["tests_done"] == [{"test": "spun the bare hub, tire off", "result": "noise still there, once per revolution"}]


async def test_established_state_reaches_the_next_prompt(fake_llm, fake_store, session, car):
    from app.schemas.session import Fact

    fake_llm.facts = [Fact(test="spun the bare hub", result="noise once per revolution")]
    await _run(session, car, "spun the bare hub, noise once per revolution")
    fake_llm.state = None  # the state call keeps the previous state from here on
    await _run(fake_store.rows[session["id"]], car, "what next?")  # the route re-reads the session each turn
    text = fake_llm.prompt[-1]["content"][0]["text"]
    assert "test done: spun the bare hub -> noise once per revolution" in text


async def test_a_turn_without_facts_leaves_observations_null(fake_llm, fake_store, session, car):
    await _run(session, car, "hello")
    assert fake_store.messages(session["id"])[0]["observations"] is None


async def test_stored_reply_has_clean_citations(fake_llm, fake_store, session, car):
    fake_llm.reply = "Brakes still work (<Toyota Platz 2001 manual (s3.amazonaws.com), p. 140>)."
    await _run(session, car, "abs light")
    assert fake_store.messages(session["id"])[-1]["content"] == "Brakes still work (Toyota Platz 2001 manual, p. 140)."


async def test_timeout_saves_a_failed_attempt_and_retry_answers_the_same_message(fake_llm, fake_store, session, car, monkeypatch):
    from app.services import llm

    async def timing_out(messages):
        raise TimeoutError("model did not answer")
        yield ""  # pragma: no cover (makes this an async generator)

    good = llm.stream_reply
    monkeypatch.setattr(llm, "stream_reply", timing_out)
    out = await _run(session, car, "ABS light flickers on bumps")
    assert out[-1][0] == "error"
    user_id = out[-1][1]["retry_message_id"]
    assert [(m["role"], m["status"]) for m in fake_store.messages(session["id"])] == [("user", "ok"), ("assistant", "failed")]

    monkeypatch.setattr(llm, "stream_reply", good)  # the fake is back
    row = fake_store.rows[session["id"]]
    retried = [x async for x in run_turn(row, car, None, [], retry_of=user_id)]
    assert retried[-1][0] == "done"
    msgs = fake_store.messages(session["id"])
    assert [(m["role"], m["status"]) for m in msgs] == [("user", "ok"), ("assistant", "failed"), ("assistant", "ok")]
    assert [m["role"] for m in msgs].count("user") == 1  # nothing synthetic was added as a user message
    sent = fake_llm.prompt[-1]["content"][0]["text"]
    assert "ABS light flickers on bumps" in sent  # the retry answers the original text
    assert [m["role"] for m in fake_llm.prompt[1:-1]] == []  # and the failed attempt is not history


def test_retry_only_applies_to_a_failed_last_message():
    import pytest

    from app.services.diagnosis import retry_target
    from app.utils.exceptions import BusinessLogicException, RecordNotFoundException

    u = {"id": "u1", "role": "user", "content": "hi", "media": []}
    failed = {"id": "a1", "role": "assistant", "content": "", "status": "failed"}
    ok = {"id": "a2", "role": "assistant", "content": "hello", "status": "ok"}
    assert retry_target([u, failed], "u1") is u
    for messages, mid, err in [([u], "u1", BusinessLogicException), ([u, ok], "u1", BusinessLogicException),
                               ([u, failed], "nope", RecordNotFoundException),
                               ([{**u, "media": [{"kind": "image", "path": "p"}]}, failed], "u1", BusinessLogicException)]:
        with pytest.raises(err):
            retry_target(messages, mid)
