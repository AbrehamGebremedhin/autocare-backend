import importlib.util
from pathlib import Path

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
