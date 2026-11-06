import asyncio
import gc
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import httpx
import pytest

from app.api.v1 import session_route
from app.CRUD.chat_session_crud import ChatSessionCRUD
from app.utils.limiter import limiter

BUSY = "A reply is still being generated for this session."
DELETING = "This session is being deleted."


@pytest.fixture(autouse=True)
def _clean():
    limiter.reset()
    session_route._in_flight.clear()
    session_route._deleting.clear()
    yield
    session_route._in_flight.clear()
    session_route._deleting.clear()


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
    session_route._in_flight.add(existing_session["id"])
    r = client.delete(f"/api/v1/sessions/{existing_session['id']}", headers=auth_headers)
    assert r.status_code == 409 and r.json()["message"] == BUSY
    assert existing_session["id"] in fake_store.rows and keep.exists()


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
        assert r.status_code == 409 and r.json()["message"] == DELETING
        again = await ac.delete(f"/api/v1/sessions/{sid}", headers=auth_headers)
        assert again.status_code == 409 and again.json()["message"] == DELETING
        gate.set()
        assert (await asyncio.wait_for(deleting, 5)).status_code == 200
    assert fake_llm.calls == 0 and sid not in fake_store.rows
    assert sid not in session_route._in_flight and sid not in session_route._deleting


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
    assert existing_session["id"] not in session_route._in_flight


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


# --- in-flight lock -------------------------------------------------------------

async def test_concurrent_send_gets_409(auth_headers, existing_session, slow_llm, fake_store):
    import main

    sid = existing_session["id"]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as ac:
        first = asyncio.create_task(ac.post(_url(sid), data={"text": "fan clicks"}, headers=auth_headers))
        await asyncio.wait_for(_until(lambda: sid in session_route._in_flight), 5)
        second = await ac.post(_url(sid), data={"text": "again"}, headers=auth_headers)
        assert second.status_code == 409 and second.json()["message"] == BUSY
        slow_llm.gate.set()
        r1 = await asyncio.wait_for(first, 5)
    assert r1.status_code == 200 and _parse(r1.text)[-1][0] == "done"
    assert sid not in session_route._in_flight
    assert [m["content"] for m in fake_store.messages(sid) if m["role"] == "user"] == ["fan clicks"]


def test_lock_released_after_llm_failure(client, auth_headers, existing_session, fake_llm):
    fake_llm.fail_stream_after = 1
    r = client.post(_url(existing_session["id"]), data={"text": "fan clicks"}, headers=auth_headers)
    assert _parse(r.text)[-1] == ("error", {"message": "The diagnosis service failed, please retry."})
    fake_llm.fail_stream_after = None
    r = client.post(_url(existing_session["id"]), data={"text": "fan clicks"}, headers=auth_headers)
    assert r.status_code == 200 and _parse(r.text)[-1][0] == "done"


def test_lock_released_if_building_response_fails(client, auth_headers, existing_session, fake_llm, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(session_route, "StreamingResponse", broken)
    with pytest.raises(RuntimeError):
        client.post(_url(existing_session["id"]), data={"text": "fan clicks"}, headers=auth_headers)
    assert existing_session["id"] not in session_route._in_flight


async def test_lock_released_when_client_disconnects(existing_session, car, fake_llm, fake_store):
    sid = existing_session["id"]
    session_route._in_flight.add(sid)
    stream, _ = session_route._events(existing_session, car, "fan clicks")
    assert (await anext(stream)).startswith("event: stage")
    assert (await anext(stream)).startswith("event: token")
    await stream.aclose()  # what the server does when the client goes away
    assert sid not in session_route._in_flight
    assert [m["role"] for m in fake_store.messages(sid)] == ["user"]  # no partial reply saved


def test_lock_released_if_stream_never_starts(existing_session, car):
    sid = existing_session["id"]
    session_route._in_flight.add(sid)
    stream, _ = session_route._events(existing_session, car, "fan clicks")
    del stream
    gc.collect()
    assert sid not in session_route._in_flight


async def test_finished_stream_does_not_release_a_later_turn(existing_session, car, fake_llm, fake_store):
    sid = existing_session["id"]
    session_route._in_flight.add(sid)
    stream, _ = session_route._events(existing_session, car, "fan clicks")
    _ = [chunk async for chunk in stream]
    assert sid not in session_route._in_flight
    session_route._in_flight.add(sid)  # the next turn takes the lock
    del stream
    gc.collect()
    assert sid in session_route._in_flight


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
