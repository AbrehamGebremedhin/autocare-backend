import importlib.util
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("e2e", ROOT / "scripts" / "e2e.py")
e2e = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e2e)

GOOD = [
    "event: stage", 'data: {"stage": "retrieve"}', "",
    "event: token", 'data: {"text": "Hel"}', "",
    "event: token", 'data: {"text": "lo"}', "",
    "event: state", 'data: {"title": "t"}', "",
    "event: done", "data: {}", "",
]


def test_parse_sse_yields_named_events_in_order():
    events = list(e2e.parse_sse(GOOD))
    assert [e for e, _ in events] == ["stage", "token", "token", "state", "done"]
    assert events[1][1] == {"text": "Hel"}


def test_parse_sse_handles_unicode_and_missing_trailing_blank_line():
    events = list(e2e.parse_sse(["event: token", 'data: {"text": "café"}']))
    assert events == [("token", {"text": "café"})]


def test_good_turn_has_no_problems():
    assert e2e.check_events(list(e2e.parse_sse(GOOD))) == []


def test_no_token_is_a_failure():
    events = [("stage", {}), ("state", {}), ("done", {})]
    assert "no token event" in e2e.check_events(events)


def test_state_must_appear_exactly_once():
    assert any("got 0" in p for p in e2e.check_events([("token", {})]))
    assert any("got 2" in p for p in e2e.check_events([("token", {}), ("state", {}), ("state", {})]))


def test_error_event_is_a_failure_and_reported():
    problems = e2e.check_events([("token", {}), ("state", {}), ("error", {"message": "boom"})])
    assert problems == ["error event: boom"]


def test_result_line_shows_timings_and_status():
    ok = e2e.Result("text", [], 1.234, 5.0).line()
    assert ok.startswith("OK") and "1.2s" in ok and "5.0s" in ok
    bad = e2e.Result("video", ["no token event"], None, 2.0).line()
    assert bad.startswith("FAIL") and "n/a" in bad and "no token event" in bad


def _client(routes: dict, calls: list) -> httpx.Client:
    """routes: path -> (status, json body); every request path is recorded in `calls`."""
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        status, body = routes[request.url.path]
        return httpx.Response(status, json=body)

    return httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler))


TOKENS = {"access_token": "acc", "refresh_token": "ref", "token_type": "bearer", "expires_in": 1800, "user_id": "u"}
LOGIN, REGISTER = "/api/v1/auth/login", "/api/v1/auth/register"


def test_sign_in_logs_in_without_registering():
    calls = []
    client = _client({LOGIN: (200, TOKENS)}, calls)
    assert e2e.sign_in(client, "a@b.co", "pw", register=True) == "acc"
    assert calls == [LOGIN]


def test_sign_in_registers_after_401_only_when_asked():
    calls = []
    client = _client({LOGIN: (401, {}), REGISTER: (200, TOKENS)}, calls)
    assert e2e.sign_in(client, "a@b.co", "pw", register=True) == "acc"
    assert calls == [LOGIN, REGISTER]

    calls.clear()
    assert e2e.sign_in(client, "a@b.co", "pw", register=False) is None
    assert calls == [LOGIN]


def test_sign_in_does_not_register_on_other_errors():
    calls = []
    client = _client({LOGIN: (429, {}), REGISTER: (200, TOKENS)}, calls)
    assert e2e.sign_in(client, "a@b.co", "pw", register=True) is None
    assert calls == [LOGIN]


def test_sign_in_reports_failed_registration():
    client = _client({LOGIN: (401, {}), REGISTER: (409, {})}, [])
    assert e2e.sign_in(client, "a@b.co", "pw", register=True) is None


def test_health_problems_need_database_and_deepseek():
    assert e2e.health_problems({"status": "ok", "database": True, "deepseek": True}) == []
    assert e2e.health_problems({"status": "ok", "database": False, "deepseek": True}) == ["database down"]
    assert e2e.health_problems({"status": "ok", "database": True, "deepseek": False}) == ["deepseek down"]
