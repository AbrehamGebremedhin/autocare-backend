import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.schemas.session import DiagnosisState
from app.services.llm import extract_state, ping, stream_reply


def _chunk(content=None, reasoning=None, choices=True):
    if not choices:
        return NS(choices=[])
    delta = NS(content=content, reasoning_content=reasoning)
    return NS(choices=[NS(delta=delta)])


@pytest.fixture
def fake_client(mocker):
    client = MagicMock()
    client.chat.completions.create = AsyncMock()
    mocker.patch("app.services.llm.get_client", return_value=client)
    return client


@pytest.fixture
def fake_stream(fake_client):
    async def gen():
        for c in [
            _chunk(reasoning="thinking"),
            _chunk(content="Check "),
            _chunk(content=None),
            _chunk(content=""),
            _chunk(choices=False),
            _chunk(content="coolant."),
        ]:
            yield c

    fake_client.chat.completions.create.return_value = gen()
    return fake_client


@pytest.fixture
def fake_completion(fake_client):
    def _set(text):
        fake_client.chat.completions.create.return_value = NS(
            choices=[NS(message=NS(content=text))]
        )
        return fake_client

    return _set


VALID = '{"suspected_causes":[{"cause":"Fan relay","likelihood":0.6,"evidence":"on/off bursts","next_check":"swap relay"}],"safety_flags":[],"title":"Fan cycling"}'


async def test_stream_reply_yields_only_content(fake_stream):
    out = [t async for t in stream_reply([{"role": "user", "content": "hi"}])]
    assert out == ["Check ", "coolant."]


async def test_stream_reply_passes_stream_and_model(fake_stream):
    msgs = [{"role": "user", "content": "hi"}]
    _ = [t async for t in stream_reply(msgs)]
    kwargs = fake_stream.chat.completions.create.call_args.kwargs
    assert kwargs["stream"] is True
    assert kwargs["messages"] == msgs
    from app.core.config import get_settings

    assert kwargs["model"] == get_settings().DEEPSEEK_MODEL


async def test_extract_state_keeps_previous_on_bad_json(fake_completion):
    fake_completion("not json at all")
    prev = DiagnosisState(suspected_causes=[], safety_flags=["hot"], title="Overheating")
    assert await extract_state([], prev) == prev


async def test_extract_state_parses_valid_json(fake_completion):
    fake_completion(VALID)
    s = await extract_state([], DiagnosisState(suspected_causes=[], safety_flags=[], title=""))
    assert s.suspected_causes[0].cause == "Fan relay"


async def test_extract_state_handles_fences_and_prose(fake_completion):
    fake_completion("Here you go:\n```json\n" + VALID + "\n```\nDone.")
    s = await extract_state([], DiagnosisState())
    assert s.title == "Fan cycling"


async def test_extract_state_rejects_out_of_range_likelihood(fake_completion):
    fake_completion(VALID.replace("0.6", "1.7"))
    prev = DiagnosisState(title="keep")
    assert await extract_state([], prev) == prev


async def test_extract_state_keeps_previous_on_api_error(fake_client):
    fake_client.chat.completions.create.side_effect = RuntimeError("boom")
    prev = DiagnosisState(title="keep")
    assert await extract_state([], prev) == prev


async def test_extract_state_sends_last_messages_and_previous(fake_completion):
    client = fake_completion(VALID)
    convo = [{"role": "user", "content": f"m{i} é"} for i in range(25)]
    prev = DiagnosisState(title="Überhitzung")
    await extract_state(convo, prev)
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs.get("stream") is not True
    body = json.dumps(kwargs["messages"], ensure_ascii=False)
    assert "Überhitzung" in body and "m24" in body and "m0 " not in body


@pytest.fixture
def fresh_ping(monkeypatch, fake_client):
    from app.services import llm

    monkeypatch.setattr(llm, "_ping_cache", None)
    return fake_client


async def test_ping_true(fresh_ping):
    fresh_ping.models.list = AsyncMock(return_value=[])
    assert await ping() is True


async def test_ping_false_on_exception(fresh_ping):
    fresh_ping.models.list = AsyncMock(side_effect=RuntimeError("down"))
    assert await ping() is False


async def test_ping_result_is_cached_for_ttl(fresh_ping, mocker):
    from app.services import llm

    clock = mocker.patch("app.services.llm.time.monotonic", return_value=1000.0)
    fresh_ping.models.list = AsyncMock(side_effect=RuntimeError("down"))
    assert await ping() is False
    fresh_ping.models.list = AsyncMock(return_value=[])
    clock.return_value = 1000.0 + llm._PING_TTL - 1
    assert await ping() is False  # still the cached result, no new call
    fresh_ping.models.list.assert_not_called()
    clock.return_value = 1000.0 + llm._PING_TTL + 1
    assert await ping() is True  # expired: pinged again
    fresh_ping.models.list.assert_called_once()
