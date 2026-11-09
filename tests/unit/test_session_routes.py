import asyncio
import gc
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import httpx
import pytest

from app.api.v1 import session_route
from app.CRUD.chat_session_crud import ChatSessionCRUD
from app.services import diagnosis
from app.services.diagnosis import Lease
from app.utils.limiter import limiter

BUSY = "This session is busy: a reply is still being generated, or it is being deleted."


@pytest.fixture(autouse=True)
def _clean():
    limiter.reset()


def _url(session_id):
    return f"/api/v1/sessions/{session_id}/messages"


def _parse(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.split("\n\n"):
        if block:
            event, data = block.split("\n")
            events.append((event.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return events


async def _until(predicate):
    while not predicate():
        await asyncio.sleep(0.01)


# --- session CRUD routes ------------------------------------------------------

def test_create_session_for_car_in_profile(client, auth_headers, fake_store, fake_garage, car, user_id):
    r = client.post("/api/v1/sessions", json={"car_id": car["id"]}, headers=auth_headers)
    assert r.status_code == 200
    body = r.json()
    assert {"id", "car_id", "title", "messages", "state", "created_at", "updated_at"} <= body.keys()
    assert body["car_id"] == car["id"] and body["messages"] == []
    assert fake_store.rows[body["id"]]["user_id"] == user_id


@pytest.mark.parametrize("car_id", ["toyota\x00echo", "x" * 201, ""])
def test_create_session_rejects_bad_car_ids_with_422(client, auth_headers, fake_store, fake_garage, car_id):
    r = client.post("/api/v1/sessions", json={"car_id": car_id}, headers=auth_headers)
    assert r.status_code == 422
    assert fake_store.rows == {}


def test_create_session_with_car_not_in_profile_is_404(client, auth_headers_other_user, fake_store, fake_garage, car):
    r = client.post("/api/v1/sessions", json={"car_id": car["id"]}, headers=auth_headers_other_user)
    assert r.status_code == 404
    assert fake_store.rows == {}


def test_list_sessions_shows_only_own_summaries(client, auth_headers, existing_session, fake_store, other_user_id, car):
    fake_store.add(other_user_id, car["id"])
    r = client.get("/api/v1/sessions", headers=auth_headers)
    assert r.status_code == 200
    assert r.json() == [{
        "id": existing_session["id"], "car_id": car["id"], "title": "New diagnosis",
        "updated_at": existing_session["updated_at"], "message_count": 0,
    }]


def test_get_own_session(client, auth_headers, existing_session):
    r = client.get(f"/api/v1/sessions/{existing_session['id']}", headers=auth_headers)
    assert r.status_code == 200
    assert r.json()["id"] == existing_session["id"]
    assert r.headers["cache-control"].startswith("no-store")


def test_get_session_upgrades_a_legacy_state_and_returns_no_float_likelihood(
    client, auth_headers, existing_session, fake_store
):
    fake_store.rows[existing_session["id"]]["state"] = {
        "title": "Rear noise",
        "suspected_causes": [{"cause": "Wheel bearing", "likelihood": 0.85}],
        "safety_flags": ["check the hub nut torque before driving"],
    }
    state = client.get(f"/api/v1/sessions/{existing_session['id']}", headers=auth_headers).json()["state"]
    assert state["suspected_causes"][0]["rank"] == "most_likely" and "likelihood" not in state["suspected_causes"][0]
    assert state["safety_flags"] == [{"text": "check the hub nut torque before driving", "severity": "critical", "resolved": False}]
    assert state["title"] == "Rear noise" and state["current_focus"] == "Rear noise"


def test_other_users_session_is_404(client, auth_headers_other_user, existing_session):
    assert client.get(f"/api/v1/sessions/{existing_session['id']}", headers=auth_headers_other_user).status_code == 404


def test_delete_own_session(client, auth_headers, existing_session, fake_store):
    r = client.delete(f"/api/v1/sessions/{existing_session['id']}", headers=auth_headers)
    assert r.status_code == 200 and r.json() == {"deleted": existing_session["id"]}
    assert existing_session["id"] not in fake_store.rows


def test_delete_other_users_session_is_404(client, auth_headers_other_user, existing_session, fake_store, media_root, user_id):
    keep = _put(media_root, user_id, existing_session["id"], "m1/0.jpg")
    r = client.delete(f"/api/v1/sessions/{existing_session['id']}", headers=auth_headers_other_user)
    assert r.status_code == 404
    assert existing_session["id"] in fake_store.rows and keep.exists()


def _put(root, user_id, sid, rel):
    f = root / user_id / sid / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(b"x")
    return f


def test_delete_removes_the_sessions_media(client, auth_headers, existing_session, media_root, user_id):
    sid = existing_session["id"]
    mine = [_put(media_root, user_id, sid, "m1/0.jpg"), _put(media_root, user_id, sid, "m2/0.mp4")]
    other = _put(media_root, user_id, "other-session", "m1/0.jpg")
    r = client.delete(f"/api/v1/sessions/{sid}", headers=auth_headers)
    assert r.status_code == 200
    assert not any(f.exists() for f in mine) and not (media_root / user_id / sid).exists()
    assert other.exists()


def test_delete_succeeds_when_storage_fails(client, auth_headers, existing_session, fake_store, media_root, monkeypatch):
    def boom(*a, **k):
        raise OSError("disk gone")

    monkeypatch.setattr("app.services.media_store.shutil.rmtree", boom)
    _put(media_root, existing_session["user_id"], existing_session["id"], "m1/0.jpg")
    r = client.delete(f"/api/v1/sessions/{existing_session['id']}", headers=auth_headers)
    assert r.status_code == 200 and existing_session["id"] not in fake_store.rows


def test_delete_while_a_turn_is_in_flight_is_409(client, auth_headers, existing_session, fake_store, media_root, user_id):
    keep = _put(media_root, user_id, existing_session["id"], "m1/0.jpg")
    fake_store.hold(existing_session["id"])  # a turn on any worker holds the lease
    r = client.delete(f"/api/v1/sessions/{existing_session['id']}", headers=auth_headers)
    assert r.status_code == 409 and r.json()["message"] == BUSY
    assert existing_session["id"] in fake_store.rows and keep.exists()


def test_delete_after_an_expired_lease_succeeds(client, auth_headers, existing_session, fake_store):
    fake_store.hold(existing_session["id"], seconds=-1)  # its worker crashed mid-turn
    r = client.delete(f"/api/v1/sessions/{existing_session['id']}", headers=auth_headers)
    assert r.status_code == 200 and existing_session["id"] not in fake_store.rows


def test_delete_releases_the_lease_when_the_row_delete_fails(client, auth_headers, existing_session, fake_store, media_root, user_id):
    keep = _put(media_root, user_id, existing_session["id"], "m1/0.jpg")

    async def broken(filters):
        raise RuntimeError("db broke")

    fake_store.delete = broken
    with pytest.raises(RuntimeError):
        client.delete(f"/api/v1/sessions/{existing_session['id']}", headers=auth_headers)
    assert not fake_store.busy(existing_session["id"])
    assert keep.exists()  # the row comes first: a session that stays keeps its media


def test_delete_removes_the_row_before_the_media(client, auth_headers, existing_session, fake_store, media_root, monkeypatch):
    from app.services import diagnosis as diag

    order = []
    real_delete, real_prefix = fake_store.delete, diag.media_store.delete_prefix

    async def delete(filters):
        order.append("row")
        return await real_delete(filters)

    async def delete_prefix(*args):
        order.append("media")
        return await real_prefix(*args)

    fake_store.delete = delete
    monkeypatch.setattr(diag.media_store, "delete_prefix", delete_prefix)
    assert client.delete(f"/api/v1/sessions/{existing_session['id']}", headers=auth_headers).status_code == 200
    assert order == ["row", "media"]


@pytest.mark.parametrize("method", ["send", "delete"])
def test_a_session_deleted_before_the_claim_is_404_not_409(client, auth_headers, existing_session, fake_store, fake_llm, method):
    sid = existing_session["id"]
    real = fake_store.claim

    async def deleted_meanwhile(session_id, user_id, seconds=300):
        fake_store.rows.pop(session_id)  # another request's DELETE finished after our ownership check
        return await real(session_id, user_id, seconds)

    fake_store.claim = deleted_meanwhile
    if method == "send":
        r = client.post(_url(sid), data={"text": "fan clicks"}, headers=auth_headers)
    else:
        r = client.delete(f"/api/v1/sessions/{sid}", headers=auth_headers)
    assert r.status_code == 404 and r.json()["message"] == "Session not found"


async def test_send_while_delete_is_running_is_409(auth_headers, existing_session, fake_store, media_root, fake_llm):
    import main

    sid = existing_session["id"]
    gate, entered = asyncio.Event(), asyncio.Event()
    real_delete = fake_store.delete

    async def slow_delete(filters):
        entered.set()
        await gate.wait()
        return await real_delete(filters)

    fake_store.delete = slow_delete
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as ac:
        deleting = asyncio.create_task(ac.delete(f"/api/v1/sessions/{sid}", headers=auth_headers))
        await asyncio.wait_for(entered.wait(), 5)
        r = await ac.post(_url(sid), data={"text": "fan clicks"}, headers=auth_headers)
        assert r.status_code == 409 and r.json()["message"] == BUSY
        again = await ac.delete(f"/api/v1/sessions/{sid}", headers=auth_headers)
        assert again.status_code == 409 and again.json()["message"] == BUSY
        gate.set()
        assert (await asyncio.wait_for(deleting, 5)).status_code == 200
    assert fake_llm.calls == 0 and sid not in fake_store.rows
    assert not fake_store.busy(sid)


# --- send ---------------------------------------------------------------------

def test_send_strips_nul_characters_from_the_text(client, auth_headers, existing_session, fake_llm, fake_store):
    r = client.post(_url(existing_session["id"]), data={"text": "fan" + chr(0) + " clicks"}, headers=auth_headers)
    assert r.status_code == 200
    assert fake_store.messages(existing_session["id"])[0]["content"] == "fan clicks"


def test_send_streams_sse(client, auth_headers, existing_session, fake_llm, fake_store):
    r = client.post(_url(existing_session["id"]), data={"text": "fan clicks"}, headers=auth_headers)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["cache-control"] == "no-cache, no-store" and r.headers["x-accel-buffering"] == "no"
    events = _parse(r.text)
    assert events[0] == ("stage", {"name": "searching"}) and [e for e, _ in events[-2:]] == ["state", "done"]
    assert [m["role"] for m in fake_store.messages(existing_session["id"])] == ["user", "assistant"]
    assert not fake_store.busy(existing_session["id"])


def test_send_to_other_users_session_is_404(client, auth_headers_other_user, existing_session, fake_llm):
    r = client.post(_url(existing_session["id"]), data={"text": "hi"}, headers=auth_headers_other_user)
    assert r.status_code == 404 and fake_llm.calls == 0


def test_send_requires_text_or_file(client, auth_headers, existing_session):
    r = client.post(f"/api/v1/sessions/{existing_session['id']}/messages", data={}, headers=auth_headers)
    assert r.status_code == 422


def test_send_whitespace_only_is_422(client, auth_headers, existing_session, fake_llm):
    r = client.post(_url(existing_session["id"]), data={"text": "   "}, headers=auth_headers)
    assert r.status_code == 422 and fake_llm.calls == 0


def test_send_text_over_5000_chars_is_422(client, auth_headers, existing_session, fake_llm, fake_store):
    r = client.post(_url(existing_session["id"]), data={"text": "x" * 5001}, headers=auth_headers)
    assert r.status_code == 422 and fake_llm.calls == 0
    assert fake_store.messages(existing_session["id"]) == []


def test_turn_uses_the_session_as_read_after_taking_the_lock(client, auth_headers, existing_session, fake_llm, fake_store):
    sid = existing_session["id"]
    fake_store.rows[sid]["messages"] = [
        {"id": "a", "role": "user", "content": "fan clicks", "media": [], "observations": None, "created_at": "x"},
        {"id": "b", "role": "assistant", "content": "check relay", "media": [], "observations": None, "created_at": "y"},
    ]
    real = fake_store.get_for_user
    reads = []

    async def stale_first(session_id, user_id):
        row = await real(session_id, user_id)
        reads.append(session_id)
        if len(reads) == 1:
            row["messages"] = []  # snapshot taken before a previous turn finished writing
        return row

    fake_store.get_for_user = stale_first
    r = client.post(_url(sid), data={"text": "relay is fine"}, headers=auth_headers)
    assert _parse(r.text)[-1][0] == "done" and len(reads) == 2
    assert [m["role"] for m in fake_llm.prompt] == ["system", "user", "assistant", "user"]


def test_send_is_rate_limited(client, auth_headers, fake_store):
    codes = [client.post(_url("nope"), data={"text": "hi"}, headers=auth_headers).status_code for _ in range(21)]
    assert codes[:20] == [404] * 20 and codes[20] == 429


# --- daily message cap ----------------------------------------------------------

@pytest.fixture
def daily_cap(monkeypatch):
    """A cap of 2 messages a day, on a clock the test moves: the clock's value is a one-item list."""
    from app.core.config import get_settings
    from app.services import usage_limits

    monkeypatch.setattr(get_settings(), "DAILY_MESSAGE_LIMIT", 2)
    now = [datetime(2026, 3, 1, 23, 0, 0, tzinfo=timezone.utc)]
    monkeypatch.setattr(usage_limits, "_utc_now", lambda: now[0])
    return now


def _send(client, headers, sid, text="fan clicks"):
    return client.post(_url(sid), data={"text": text}, headers=headers)


def test_messages_over_the_daily_cap_are_429_with_retry_after(client, auth_headers, existing_session, fake_llm, daily_cap):
    sid = existing_session["id"]
    assert [_send(client, auth_headers, sid).status_code for _ in range(2)] == [200, 200]
    calls = fake_llm.calls
    r = _send(client, auth_headers, sid)
    assert r.status_code == 429
    assert r.headers["retry-after"] == "3600"  # 23:00 UTC -> midnight
    body = r.json()
    assert body["error_code"] == "EXT_003" and "daily message limit" in body["message"].lower()
    assert fake_llm.calls == calls


def test_a_new_utc_day_starts_a_fresh_count(client, auth_headers, existing_session, fake_llm, daily_cap):
    sid = existing_session["id"]
    for _ in range(2):
        _send(client, auth_headers, sid)
    assert _send(client, auth_headers, sid).status_code == 429
    daily_cap[0] += timedelta(hours=1)
    assert _send(client, auth_headers, sid).status_code == 200


def test_the_cap_is_per_account(client, auth_headers, auth_headers_other_user, existing_session, fake_store, fake_llm, daily_cap, other_user_id):
    theirs = fake_store.add(other_user_id, existing_session["car_id"])
    for _ in range(2):
        _send(client, auth_headers, existing_session["id"])
    assert _send(client, auth_headers, existing_session["id"]).status_code == 429
    assert _send(client, auth_headers_other_user, theirs["id"]).status_code == 200


def test_old_days_are_pruned(client, auth_headers, existing_session, fake_llm, daily_cap, user_id):
    from app.services import usage_limits

    _send(client, auth_headers, existing_session["id"])
    daily_cap[0] += timedelta(days=2)
    _send(client, auth_headers, existing_session["id"])
    assert list(usage_limits._counts) == [(user_id, daily_cap[0].date())]


def test_a_cap_of_zero_is_no_cap(client, auth_headers, existing_session, fake_llm, daily_cap, monkeypatch):
    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "DAILY_MESSAGE_LIMIT", 0)
    assert [_send(client, auth_headers, existing_session["id"]).status_code for _ in range(5)] == [200] * 5


def test_refused_requests_do_not_use_up_the_cap(client, auth_headers, auth_headers_other_user, existing_session, fake_store, fake_llm, daily_cap):
    sid = existing_session["id"]
    assert _send(client, auth_headers_other_user, sid).status_code == 404  # not their session
    assert _send(client, auth_headers, "nope").status_code == 404
    assert client.post(_url(sid), data={"text": "  "}, headers=auth_headers).status_code == 422
    assert [_send(client, auth_headers, sid).status_code for _ in range(2)] == [200, 200]


def test_a_refusal_by_the_cap_frees_the_lease_and_does_no_model_work(client, auth_headers, existing_session, fake_store, fake_llm, daily_cap):
    sid = existing_session["id"]
    for _ in range(2):
        _send(client, auth_headers, sid)
    calls, messages = fake_llm.calls, len(fake_store.messages(sid))
    assert _send(client, auth_headers, sid).status_code == 429
    assert not fake_store.busy(sid) and fake_llm.calls == calls and len(fake_store.messages(sid)) == messages
    assert _send(client, auth_headers, sid).status_code == 429  # and again: the first refusal left no lease behind


def _counted(user_id) -> int:
    from app.services import usage_limits

    return sum(n for (uid, _), n in usage_limits._counts.items() if uid == user_id)


def test_a_session_busy_with_another_turn_does_not_use_up_the_cap(client, auth_headers, existing_session, fake_store, fake_llm, daily_cap, user_id):
    fake_store.hold(existing_session["id"])
    assert _send(client, auth_headers, existing_session["id"]).status_code == 409
    assert _counted(user_id) == 0


def test_a_database_outage_at_the_claim_does_not_use_up_the_cap(client, auth_headers, existing_session, fake_store, fake_llm, daily_cap, user_id):
    from app.db.pool import DatabaseUnavailable

    async def down(*args):
        raise DatabaseUnavailable("down")

    fake_store.claim = down
    assert _send(client, auth_headers, existing_session["id"]).status_code == 503
    assert _counted(user_id) == 0


def test_a_rejected_upload_does_not_use_up_the_cap(client, auth_headers, existing_session, fake_llm, daily_cap, user_id):
    r = client.post(_url(existing_session["id"]), files={"files": ("x.bin", b"not media", "application/octet-stream")}, headers=auth_headers)
    assert r.status_code == 415 and _counted(user_id) == 0


def test_an_accepted_message_counts_once(client, auth_headers, existing_session, fake_llm, daily_cap, user_id):
    assert _send(client, auth_headers, existing_session["id"]).status_code == 200
    assert _counted(user_id) == 1


# --- session lease ----------------------------------------------------------------

async def test_concurrent_send_gets_409(auth_headers, existing_session, slow_llm, fake_store):
    import main

    sid = existing_session["id"]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as ac:
        first = asyncio.create_task(ac.post(_url(sid), data={"text": "fan clicks"}, headers=auth_headers))
        await asyncio.wait_for(_until(lambda: fake_store.busy(sid)), 5)
        second = await ac.post(_url(sid), data={"text": "again"}, headers=auth_headers)
        assert second.status_code == 409 and second.json()["message"] == BUSY
        slow_llm.gate.set()
        r1 = await asyncio.wait_for(first, 5)
    assert r1.status_code == 200 and _parse(r1.text)[-1][0] == "done"
    assert not fake_store.busy(sid)
    assert [m["content"] for m in fake_store.messages(sid) if m["role"] == "user"] == ["fan clicks"]


def test_send_while_another_worker_holds_the_lease_is_409(client, auth_headers, existing_session, fake_store, fake_llm):
    sid = existing_session["id"]
    fake_store.hold(sid)
    r = client.post(_url(sid), data={"text": "fan clicks"}, headers=auth_headers)
    assert r.status_code == 409 and r.json()["message"] == BUSY
    assert fake_llm.calls == 0 and fake_store.messages(sid) == []


def test_send_takes_over_an_expired_lease(client, auth_headers, existing_session, fake_store, fake_llm):
    sid = existing_session["id"]
    fake_store.hold(sid, seconds=-1)  # left by a crashed worker
    r = client.post(_url(sid), data={"text": "fan clicks"}, headers=auth_headers)
    assert r.status_code == 200 and _parse(r.text)[-1][0] == "done"
    assert not fake_store.busy(sid)


async def _claim(store, session) -> Lease:
    holder = await store.claim(session["id"], session["user_id"])
    assert holder is not None
    return Lease(session["id"], holder)


async def test_heartbeat_keeps_a_turn_longer_than_the_lease_alive(existing_session, car, slow_llm, fake_store, monkeypatch):
    monkeypatch.setattr(diagnosis, "LEASE_SECONDS", 0.05)  # a 300 s lease, scaled down
    monkeypatch.setattr(diagnosis, "RENEW_EVERY_SECONDS", 0.01)
    sid = existing_session["id"]
    lease = await _claim(fake_store, existing_session)
    stream, _ = session_route._events(lease, existing_session, car, "fan clicks")
    assert (await anext(stream)).startswith("event: stage")
    rest = asyncio.ensure_future(_drain(stream))  # the model keeps it waiting for 6 lease lengths
    for _ in range(6):
        await asyncio.sleep(0.05)
        assert fake_store.busy(sid) and await fake_store.claim(sid, existing_session["user_id"]) is None
    slow_llm.gate.set()
    events = await asyncio.wait_for(rest, 2)
    assert events[-1][0] == "done" and fake_store.renewals >= 5
    assert [m["role"] for m in fake_store.messages(sid)] == ["user", "assistant"]
    assert not fake_store.busy(sid)


async def test_a_turn_that_lost_its_lease_stops_and_saves_nothing_more(existing_session, car, slow_llm, fake_store, monkeypatch):
    monkeypatch.setattr(diagnosis, "RENEW_EVERY_SECONDS", 0.01)
    sid = existing_session["id"]
    lease = await _claim(fake_store, existing_session)
    stream, _ = session_route._events(lease, existing_session, car, "fan clicks")
    assert (await anext(stream)).startswith("event: stage")
    rest = asyncio.ensure_future(_drain(stream))
    await asyncio.wait_for(_until(lambda: slow_llm.calls == 1), 2)
    newcomer = fake_store.hold(sid)  # A's lease expired meanwhile and B claimed the session
    await asyncio.wait_for(_until(lambda: lease.lost), 2)
    slow_llm.gate.set()
    events = await asyncio.wait_for(rest, 2)
    assert events == [("error", {"message": diagnosis.LEASE_LOST})]  # not even the first token
    assert [m["role"] for m in fake_store.messages(sid)] == ["user"] and fake_store.rows[sid]["state"] == {}
    assert fake_store.holder(sid) == newcomer  # B's lease is untouched by A's release


async def test_lease_lost_during_the_state_update_saves_no_reply(existing_session, car, fake_llm, fake_store, monkeypatch):
    sid = existing_session["id"]
    lease = await _claim(fake_store, existing_session)
    real = fake_llm.extract_state

    async def slow_state(conversation, previous):
        lease.lost = True  # what the heartbeat does when a renew finds someone else's lease
        return await real(conversation, previous)

    monkeypatch.setattr(diagnosis.llm, "extract_state", slow_state)
    stream, _ = session_route._events(lease, existing_session, car, "fan clicks")
    events = await _drain(stream)
    assert events[-1] == ("error", {"message": diagnosis.LEASE_LOST})
    assert [m["role"] for m in fake_store.messages(sid)] == ["user"] and fake_store.rows[sid]["state"] == {}


async def test_a_failed_renew_is_retried_not_treated_as_lost(existing_session, car, slow_llm, fake_store, monkeypatch, caplog):
    monkeypatch.setattr(diagnosis, "RENEW_EVERY_SECONDS", 0.01)

    async def down(*args):
        raise ConnectionError("db down")

    lease = await _claim(fake_store, existing_session)
    monkeypatch.setattr(fake_store, "renew", down)
    stream, _ = session_route._events(lease, existing_session, car, "fan clicks")
    rest = asyncio.ensure_future(_drain(stream))
    await asyncio.sleep(0.05)
    slow_llm.gate.set()
    events = await asyncio.wait_for(rest, 2)
    assert events[-1][0] == "done" and not lease.lost
    assert "session lease renew failed: ConnectionError" in caplog.text


async def test_renewals_that_keep_failing_end_the_turn_before_the_lease_can_expire(
    existing_session, car, slow_llm, fake_store, monkeypatch
):
    monkeypatch.setattr(diagnosis, "LEASE_SECONDS", 0.05)  # a 300 s lease, scaled down
    monkeypatch.setattr(diagnosis, "RENEW_EVERY_SECONDS", 0.01)

    async def down(*args):
        raise ConnectionError("db down")

    sid = existing_session["id"]
    lease = await _claim(fake_store, existing_session)
    monkeypatch.setattr(fake_store, "renew", down)
    stream, _ = session_route._events(lease, existing_session, car, "fan clicks")
    assert (await anext(stream)).startswith("event: stage")
    rest = asyncio.ensure_future(_drain(stream))
    await asyncio.wait_for(_until(lambda: lease.lost), 2)  # after about LEASE_SECONDS - RENEW_EVERY_SECONDS
    slow_llm.gate.set()
    events = await asyncio.wait_for(rest, 2)
    assert events == [("error", {"message": diagnosis.LEASE_LOST})]
    assert [m["role"] for m in fake_store.messages(sid)] == ["user"] and fake_store.rows[sid]["state"] == {}


async def test_heartbeat_stops_with_the_turn(existing_session, car, fake_llm, fake_store, monkeypatch):
    monkeypatch.setattr(diagnosis, "RENEW_EVERY_SECONDS", 0.01)
    lease = await _claim(fake_store, existing_session)
    stream, _ = session_route._events(lease, existing_session, car, "fan clicks")
    await _drain(stream)
    renewals = fake_store.renewals
    await asyncio.sleep(0.05)
    assert fake_store.renewals == renewals


async def _drain(stream) -> list[tuple[str, dict]]:
    return _parse("".join([chunk async for chunk in stream]))


async def test_send_claims_only_after_the_ownership_check(auth_headers_other_user, existing_session, fake_store, fake_llm):
    import main

    claims = []
    real = fake_store.claim

    async def spy(session_id, user_id, seconds=300):
        claims.append(user_id)
        return await real(session_id, user_id, seconds)

    fake_store.claim = spy
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as ac:
        r = await ac.post(_url(existing_session["id"]), data={"text": "hi"}, headers=auth_headers_other_user)
    assert r.status_code == 404 and claims == [] and not fake_store.busy(existing_session["id"])


def test_lock_released_after_llm_failure(client, auth_headers, existing_session, fake_llm):
    fake_llm.fail_stream_after = 1
    r = client.post(_url(existing_session["id"]), data={"text": "fan clicks"}, headers=auth_headers)
    assert _parse(r.text)[-1] == ("error", {"message": "The diagnosis service failed, please retry."})
    fake_llm.fail_stream_after = None
    r = client.post(_url(existing_session["id"]), data={"text": "fan clicks"}, headers=auth_headers)
    assert r.status_code == 200 and _parse(r.text)[-1][0] == "done"


def test_lock_released_if_building_response_fails(client, auth_headers, existing_session, fake_llm, fake_store, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(session_route, "StreamingResponse", broken)
    with pytest.raises(RuntimeError):
        client.post(_url(existing_session["id"]), data={"text": "fan clicks"}, headers=auth_headers)
    assert not fake_store.busy(existing_session["id"])


async def test_lease_is_free_before_the_done_event_arrives(existing_session, car, fake_llm, fake_store):
    sid = existing_session["id"]
    lease = await _claim(fake_store, existing_session)
    stream, _ = session_route._events(lease, existing_session, car, "fan clicks")
    async for chunk in stream:
        if chunk.startswith("event: done"):
            assert not fake_store.busy(sid)  # a client may send its next message right now
            break
    else:
        pytest.fail("no done event")
    await stream.aclose()


async def test_lease_is_free_before_an_error_event_arrives(existing_session, car, fake_llm, fake_store):
    fake_llm.fail_stream_after = 1
    sid = existing_session["id"]
    lease = await _claim(fake_store, existing_session)
    stream, _ = session_route._events(lease, existing_session, car, "fan clicks")
    async for chunk in stream:
        if chunk.startswith("event: error"):
            assert not fake_store.busy(sid)
            break
    await stream.aclose()


async def test_lock_released_when_client_disconnects(existing_session, car, fake_llm, fake_store):
    sid = existing_session["id"]
    lease = await _claim(fake_store, existing_session)
    stream, _ = session_route._events(lease, existing_session, car, "fan clicks")
    assert (await anext(stream)).startswith("event: stage")
    assert (await anext(stream)).startswith("event: token")
    await stream.aclose()  # what the server does when the client goes away
    await asyncio.wait_for(_until(lambda: not fake_store.busy(sid)), 2)
    assert [m["role"] for m in fake_store.messages(sid)] == ["user"]  # no partial reply saved


async def test_lock_released_when_the_stream_is_cancelled(existing_session, car, slow_llm, fake_store):
    # Starlette streams inside an anyio task group and cancels its scope on a disconnect. anyio keeps
    # cancelling every await inside that scope, so an awaited release in the stream's cleanup would never
    # reach the database; the release runs in a task of its own.
    import anyio

    sid = existing_session["id"]
    lease = await _claim(fake_store, existing_session)
    stream, _ = session_route._events(lease, existing_session, car, "fan clicks")

    async def consume():
        async for _ in stream:
            pass

    async with anyio.create_task_group() as tg:
        tg.start_soon(consume)
        await asyncio.wait_for(_until(lambda: slow_llm.calls == 1), 2)  # now waiting on the model
        tg.cancel_scope.cancel()
    await asyncio.wait_for(_until(lambda: not fake_store.busy(sid)), 2)


async def test_lock_released_if_stream_never_starts(existing_session, car, fake_store):
    sid = existing_session["id"]
    lease = await _claim(fake_store, existing_session)
    stream, _ = session_route._events(lease, existing_session, car, "fan clicks")
    del stream
    gc.collect()  # the finalizer cannot await: it schedules the release on the loop
    await asyncio.wait_for(_until(lambda: not fake_store.busy(sid)), 2)


async def test_finished_stream_does_not_release_a_later_turn(existing_session, car, fake_llm, fake_store):
    sid, uid = existing_session["id"], existing_session["user_id"]
    lease = await _claim(fake_store, existing_session)
    stream, _ = session_route._events(lease, existing_session, car, "fan clicks")
    _ = [chunk async for chunk in stream]
    assert not fake_store.busy(sid)  # released before the body ended, so the next message is accepted
    assert await fake_store.claim(sid, uid)  # the next turn takes the lease
    del stream
    gc.collect()
    await asyncio.sleep(0.05)
    assert fake_store.busy(sid)


async def test_release_from_another_thread_is_handed_to_the_loop(existing_session, fake_store):
    sid = existing_session["id"]
    lease = await _claim(fake_store, existing_session)
    loop = asyncio.get_running_loop()
    assert await asyncio.to_thread(session_route._release, loop, lease, None) is None  # a GC in a worker thread
    await asyncio.wait_for(_until(lambda: not fake_store.busy(sid)), 2)


def test_release_with_the_loop_gone_leaves_the_lease_to_expire(tmp_path):
    loop = asyncio.new_event_loop()
    loop.close()
    tmpdir = tmp_path / "turn"
    tmpdir.mkdir()
    assert session_route._release(loop, Lease("sid", "holder"), str(tmpdir)) is None  # no error, nothing scheduled
    assert not tmpdir.exists()


async def test_a_failed_release_is_logged_and_left_to_expire(existing_session, fake_store, caplog):
    async def down(session_id, holder):
        raise ConnectionError("db down")

    fake_store.release = down
    await session_route._free_lease(Lease(existing_session["id"], "holder"))
    assert "lease release failed" in caplog.text and "ConnectionError" in caplog.text


async def test_a_stale_holder_can_neither_free_nor_extend_the_next_lease(existing_session, fake_store):
    sid, uid = existing_session["id"], existing_session["user_id"]
    stale = await fake_store.claim(sid, uid, seconds=-1)  # A's lease, already expired
    newcomer = await fake_store.claim(sid, uid)  # B takes it over
    assert stale and newcomer and newcomer != stale
    assert await fake_store.renew(sid, stale) is False
    await fake_store.release(sid, stale)
    assert fake_store.holder(sid) == newcomer and fake_store.busy(sid)


# --- ChatSessionCRUD ------------------------------------------------------------

async def test_list_for_user_counts_messages_in_sql_without_loading_them(fake_pool):
    fake_pool.rows = [{"id": "s", "car_id": "c", "title": "t", "updated_at": "now", "message_count": 2}]
    assert await ChatSessionCRUD().list_for_user("u1") == fake_pool.rows
    [(method, sql, args)] = fake_pool.calls
    assert "jsonb_array_length(messages) AS message_count" in sql and "messages," not in sql
    assert "WHERE user_id = $1" in sql and "ORDER BY updated_at DESC" in sql and args == ("u1",)


async def test_append_messages_is_one_atomic_concat_statement(fake_pool):
    crud = ChatSessionCRUD()
    await crud.append_messages("sid", [{"id": "a"}])
    await crud.append_messages("sid", [{"id": "b"}])
    # Two appends, two statements, no read in between: neither can overwrite the other.
    assert [c[0] for c in fake_pool.calls] == ["execute", "execute"]
    for _, sql, _ in fake_pool.calls:
        assert sql == 'UPDATE "Chatsession" SET messages = messages || $2::jsonb, updated_at = now() WHERE id = $1'
    assert [c[2] for c in fake_pool.calls] == [("sid", [{"id": "a"}]), ("sid", [{"id": "b"}])]


async def test_get_for_user_filters_by_owner_and_rejects_bad_ids(monkeypatch):
    crud = ChatSessionCRUD()
    sid = "33333333-3333-4333-8333-333333333333"
    monkeypatch.setattr(crud, "read", AsyncMock(return_value=[{"id": sid}]))
    assert await crud.get_for_user(sid, "u1") == {"id": sid}
    crud.read.assert_awaited_once_with({"id": sid, "user_id": "u1"})
    assert await crud.get_for_user("not-a-uuid", "u1") is None
    assert await crud.get_for_user("33333333-3333-4333-8333-33333333333", "u1") is None  # one digit short
    crud.read.return_value = []
    assert await crud.get_for_user(sid, "u2") is None


async def test_get_for_user_queries_the_canonical_uuid(monkeypatch):
    crud = ChatSessionCRUD()
    monkeypatch.setattr(crud, "read", AsyncMock(return_value=[]))
    await crud.get_for_user("{33333333333343338333333333333333}", "u1")
    crud.read.assert_awaited_once_with({"id": "33333333-3333-4333-8333-333333333333", "user_id": "u1"})


async def test_malformed_session_id_is_404_not_500(client, auth_headers, fake_pool):
    response = client.get("/api/v1/sessions/not-a-uuid", headers=auth_headers)
    assert response.status_code == 404
    assert fake_pool.calls == []


async def test_save_state_writes_state_title_and_updated_at(monkeypatch):
    crud = ChatSessionCRUD()
    monkeypatch.setattr(crud, "update", AsyncMock())
    await crud.save_state("sid", {"title": "Fan"}, "Fan")
    filters, data = crud.update.call_args.args
    assert filters == {"id": "sid"}
    assert data["state"] == {"title": "Fan"} and data["title"] == "Fan"
    assert isinstance(data["updated_at"], datetime) and data["updated_at"].utcoffset() == timedelta(0)


async def test_update_message_patches_one_element_by_id_in_one_statement(fake_pool):
    fake_pool.status = "UPDATE 1"
    patch = {"media": [{"kind": "image", "path": "p"}]}
    assert await ChatSessionCRUD().update_message("sid", "a", patch) is True
    [(method, sql, args)] = fake_pool.calls
    assert method == "execute" and args == ("sid", "a", patch)
    assert "jsonb_set(s.messages" in sql and "WITH ORDINALITY" in sql
    assert "e.elem ->> 'id' = $2" in sql and "|| $3::jsonb" in sql and "updated_at = now()" in sql
    assert "'a'" not in sql and "sid" not in sql  # every value is a bind parameter


async def test_update_message_for_unknown_message_or_session_writes_nothing(fake_pool):
    fake_pool.status = "UPDATE 0"
    assert await ChatSessionCRUD().update_message("sid", "missing", {"media": []}) is False


async def test_claim_is_one_conditional_update_scoped_to_the_owner(fake_pool):
    sid = "33333333-3333-4333-8333-333333333333"
    fake_pool.value = "holder-token"  # RETURNING busy_by
    assert await ChatSessionCRUD().claim(sid, "u1", seconds=60) == "holder-token"
    [(method, sql, args)] = fake_pool.calls
    assert method == "fetchval" and args[:3] == (sid, "u1", 60.0)
    assert len(args[3]) == 36  # a fresh uuid4 holder token per claim
    assert "SET busy_until = now() + make_interval(secs => $3), busy_by = $4" in sql
    assert "WHERE id = $1 AND user_id = $2 AND (busy_until IS NULL OR busy_until < now()) RETURNING busy_by" in sql
    fake_pool.value = None  # no row matched: held, missing or another user's
    assert await ChatSessionCRUD().claim(sid, "u1") is None
    assert fake_pool.calls[0][2][3] != fake_pool.calls[1][2][3]


async def test_claim_of_a_malformed_id_is_none_without_a_query(fake_pool):
    assert await ChatSessionCRUD().claim("not-a-uuid", "u1") is None
    assert fake_pool.calls == []


async def test_renew_and_release_act_only_for_the_holder(fake_pool):
    fake_pool.value = True
    assert await ChatSessionCRUD().renew("sid", "h1") is True
    await ChatSessionCRUD().release("sid", "h1")
    (_, renew_sql, renew_args), (method, release_sql, release_args) = fake_pool.calls
    assert "make_interval(secs => $3)" in renew_sql and "WHERE id = $1 AND busy_by = $2" in renew_sql
    assert renew_args == ("sid", "h1", 300.0)
    assert method == "execute" and release_sql == (
        'UPDATE "Chatsession" SET busy_until = NULL, busy_by = NULL WHERE id = $1 AND busy_by = $2'
    )
    assert release_args == ("sid", "h1")
    fake_pool.value = None
    assert await ChatSessionCRUD().renew("sid", "someone-else") is False
