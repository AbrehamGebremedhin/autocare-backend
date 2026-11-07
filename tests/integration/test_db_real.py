"""Real Postgres check of schema.sql, the pool and BaseCRUD.

Runs only with `-m db` and TEST_DATABASE_URL pointing at a throwaway database
(the schema is applied to it). Every other run deselects or skips it.
"""
import asyncio
import os
from pathlib import Path
from uuid import uuid4

import asyncpg
import pytest

from app.CRUD.chat_session_crud import ChatSessionCRUD
from app.CRUD.user_crud import UserCRUD
from app.db import pool
from app.db.crud import BaseCRUD
from app.services import kb
from app.services.user_profile_service import UserProfileService

pytestmark = [
    pytest.mark.db,
    pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL is not set"),
]

SCHEMA = Path(__file__).resolve().parents[2] / "app" / "db" / "sql" / "schema.sql"


@pytest.fixture
async def real_pool():
    pool.set_pool_for_tests(None)  # drop the autouse fake
    await pool.init_pool(os.environ["TEST_DATABASE_URL"])
    try:
        yield
    finally:
        await pool.close_pool()


async def test_schema_is_idempotent_and_crud_and_search_work(real_pool):
    sql = SCHEMA.read_text(encoding="utf-8")
    await pool.execute(sql)
    await pool.execute(sql)  # second run is a no-op

    tag = uuid4().hex[:8]
    car_id = f"test-make-model-{tag}"
    cars = BaseCRUD("Car")
    try:
        created = await cars.create({
            "id": car_id, "make": "Test", "model": "Model", "year": 2001,
            "car_guide_links": [{"link": "https://example.com", "summary": "ጎማ"}],
        })
        assert created["kb_status"] == "pending" and created["car_guide_links"][0]["summary"] == "ጎማ"
        assert [r["id"] for r in await cars.read({"id": car_id})] == [car_id]
        assert (await cars.update({"id": car_id}, {"kb_status": "ready"}))[0]["kb_status"] == "ready"

        await pool.execute(
            "insert into kb_passages (source_type, car_id, title, page, content) values ('manual', $1, $2, 1, $3)",
            car_id, f"manual {tag}", f"The radiator fan relay {tag} clicks when the engine overheats.",
        )
        hits = await pool.fetch("select * from search_kb($1, $2, $3)", f"radiator & {tag}", car_id, 5)
        assert [h["title"] for h in hits] == [f"manual {tag}"]
        assert set(hits[0]) == {"source_type", "car_id", "title", "page", "content", "rank"}
    finally:
        await pool.execute("delete from kb_passages where car_id = $1", car_id)
        await cars.delete({"id": car_id})


async def test_chatsession_round_trip_keeps_str_ids_and_utf8(real_pool):
    await pool.execute(SCHEMA.read_text(encoding="utf-8"))
    sessions = ChatSessionCRUD()
    tag = uuid4().hex[:8]
    user_id = (await UserCRUD().create_google_user(f"session-{tag}@example.com", f"sub-{tag}"))["id"]
    try:
        created = await sessions.create({"user_id": user_id, "car_id": "test-make-model-2001"})
        sid = created["id"]
        assert isinstance(sid, str) and created["user_id"] == user_id
        assert (await sessions.get_for_user(sid, user_id))["id"] == sid
        assert await sessions.get_for_user(sid, str(uuid4())) is None

        await sessions.append_messages(sid, [{"id": "m1", "role": "user", "content": "ጎማ"}])
        await sessions.save_state(sid, {"title": "ጎማ"}, "ጎማ")
        row = await sessions.get_for_user(sid, user_id)
        assert row["messages"] == [{"id": "m1", "role": "user", "content": "ጎማ"}]
        assert row["state"] == {"title": "ጎማ"} and row["title"] == "ጎማ"

        [listed] = await sessions.list_for_user(user_id)
        assert listed["id"] == sid and listed["message_count"] == 1

        raw = await pool.fetchval('select messages::text from "Chatsession" where id = $1', sid)
        assert "ጎማ" in raw and "\\u" not in raw
    finally:
        await pool.execute("delete from users where id = $1", user_id)  # cascades to the sessions
    assert await pool.fetchval('select count(*) from "Chatsession" where user_id = $1', user_id) == 0


async def test_users_unique_email_and_google_link(real_pool):
    await pool.execute(SCHEMA.read_text(encoding="utf-8"))
    users = UserCRUD()
    tag = uuid4().hex[:8]
    email = f"driver-{tag}@example.com"
    try:
        created = await users.create_password_user(email, "$argon2id$placeholder")
        assert isinstance(created["id"], str)
        assert await users.exists(created["id"], await pool.fetchval("select extract(epoch from now())::float8"))
        with pytest.raises(asyncpg.UniqueViolationError):
            await users.create_google_user(email.upper(), f"sub-{tag}")  # unique on lower(email)
        before_link = await pool.fetchval("select extract(epoch from clock_timestamp())::float8")
        assert await users.exists(created["id"], before_link) is True
        linked = await users.link_google(email, f"sub-{tag}")
        assert linked == created
        bump = await pool.fetchval(
            "select extract(epoch from tokens_valid_after)::float8 from users where id = $1", created["id"]
        )
        assert await users.exists(created["id"], before_link) is False  # tokens issued before the link
        assert await users.exists(created["id"], bump - 0.000001) is False
        assert await users.exists(created["id"], bump) is True
        row = await users.get_by_google_sub(f"sub-{tag}")
        assert row["password_hash"] is None and row["email"] == email
        assert await users.link_google(email, f"other-{tag}") is None  # already linked
    finally:
        await pool.execute("delete from users where lower(email) = $1", email)


async def _make_user(tag: str) -> str:
    return (await UserCRUD().create_google_user(f"u-{tag}@example.com", f"sub-{tag}"))["id"]


async def test_update_message_patches_exactly_one_element(real_pool):
    await pool.execute(SCHEMA.read_text(encoding="utf-8"))
    sessions = ChatSessionCRUD()
    user_id = await _make_user(uuid4().hex[:8])
    try:
        sid = (await sessions.create({"user_id": user_id, "car_id": "c"}))["id"]
        await sessions.append_messages(sid, [
            {"id": f"m{i}", "role": "user", "media": [], "observations": None} for i in (1, 2, 3)
        ])
        patch = {"media": [{"kind": "image", "path": "p"}], "observations": {"transcript": "ጎማ", "sounds": []}}
        assert await sessions.update_message(sid, "m2", patch) is True
        msgs = (await sessions.get_for_user(sid, user_id))["messages"]
        assert [m["id"] for m in msgs] == ["m1", "m2", "m3"]  # no shift, no extra element
        assert msgs[1] == {"id": "m2", "role": "user", **patch}  # merged: role kept, two keys replaced
        assert msgs[0]["media"] == [] and msgs[2]["observations"] is None  # neighbours untouched
        assert await sessions.update_message(sid, "nope", {"media": []}) is False
        assert await sessions.update_message(str(uuid4()), "m1", {"media": []}) is False
        assert (await sessions.get_for_user(sid, user_id))["messages"] == msgs
        assert await sessions.update_message(sid, "m3", {"media": [1]}) is True  # last element too
        assert [m["id"] for m in (await sessions.get_for_user(sid, user_id))["messages"]] == ["m1", "m2", "m3"]
    finally:
        await pool.execute("delete from users where id = $1", user_id)


async def test_concurrent_appends_all_survive(real_pool):
    await pool.execute(SCHEMA.read_text(encoding="utf-8"))
    sessions = ChatSessionCRUD()
    user_id = await _make_user(uuid4().hex[:8])
    try:
        sid = (await sessions.create({"user_id": user_id, "car_id": "c"}))["id"]
        await asyncio.gather(*(sessions.append_messages(sid, [{"id": f"c{i}"}]) for i in range(20)))
        ids = [m["id"] for m in (await sessions.get_for_user(sid, user_id))["messages"]]
        assert sorted(ids) == sorted(f"c{i}" for i in range(20))
    finally:
        await pool.execute("delete from users where id = $1", user_id)


async def test_profile_cars_array_append_and_remove(real_pool):
    await pool.execute(SCHEMA.read_text(encoding="utf-8"))
    tag = uuid4().hex[:8]
    user_id = await _make_user(tag)
    service = UserProfileService()
    try:
        assert await service._get_user_profile(user_id) is None  # no profile row yet
        first = await service.add_car_to_user(user_id, "Test", f"Add{tag}", 2001)  # creates the profile too
        assert first["success"]
        car_id = first["car_id"]
        await asyncio.gather(*(service.add_car_to_user(user_id, "Test", f"Add{tag}", 2001) for _ in range(5)))
        assert (await service._get_user_profile(user_id))["cars"] == [car_id]  # five more adds, still one entry
        second = await service.add_car_to_user(user_id, "Test", f"Other{tag}", 2002)
        assert (await service._get_user_profile(user_id))["cars"] == [car_id, second["car_id"]]
        assert (await service.remove_car_from_user(user_id, car_id))["success"] is True
        assert (await service.remove_car_from_user(user_id, car_id))["success"] is False
        assert (await service._get_user_profile(user_id))["cars"] == [second["car_id"]]
        assert (await service.remove_car_from_user(str(uuid4()), car_id))["success"] is False  # no profile
    finally:
        await pool.execute("delete from users where id = $1", user_id)
        await pool.execute('delete from "Car" where lower(model) like $1', f"%{tag}%")


async def test_create_or_get_car_survives_concurrent_inserts(real_pool):
    from app.CRUD.car_crud import CarCRUD

    await pool.execute(SCHEMA.read_text(encoding="utf-8"))
    model = f"Race{uuid4().hex[:8]}"
    try:
        results = await asyncio.gather(
            *(CarCRUD().create_or_get_car({"make": "Test", "model": model, "year": 2001}) for _ in range(8))
        )
        assert len({r[0]["id"] for r in results}) == 1
    finally:
        await pool.execute('delete from "Car" where model = $1', model)


def _pdf(mocker, pages):
    reader = mocker.patch("app.services.kb.PdfReader")
    reader.return_value.pages = [mocker.Mock(extract_text=mocker.Mock(return_value=t)) for t in pages]


async def _count(title: str) -> int:
    return await pool.fetchval("select count(*) from kb_passages where title = $1", title)


async def test_ingest_pdf_replaces_rolls_back_and_finds_ranked_passages(real_pool, mocker, tmp_path):
    await pool.execute(SCHEMA.read_text(encoding="utf-8"))
    tag = uuid4().hex[:8]
    title, car_id = f"Manual {tag}", f"kb-car-{tag}"
    try:
        _pdf(mocker, [f"The radiator fan relay {tag} clicks. Check the fan relay first.", f"Brake pads {tag} wear."])
        assert await kb.ingest_pdf(tmp_path / "a.pdf", source_type="manual", title=title, car_id=car_id) == 2
        hits = await kb.search(f"radiator fan {tag}", car_id, 5)
        assert hits[0]["page"] == 1 and hits[0]["title"] == title and len(hits) == 2

        # Re-ingest replaces (no duplicates).
        assert await kb.ingest_pdf(tmp_path / "a.pdf", source_type="manual", title=title, car_id=car_id) == 2
        assert await _count(title) == 2

        # A failing second batch rolls back the delete and batch 1: the old rows stay.
        mocker.patch("app.services.kb._INSERT_BATCH", 1)
        _pdf(mocker, ["new one", "new two", "new three"])
        real = pool.Connection.executemany
        calls = []

        async def flaky(self, sql, args_list):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("batch failed")
            await real(self, sql, args_list)

        mocker.patch.object(pool.Connection, "executemany", flaky)
        with pytest.raises(RuntimeError):
            await kb.ingest_pdf(tmp_path / "a.pdf", source_type="manual", title=title, car_id=car_id)
        rows = await pool.fetch("select content from kb_passages where title = $1 order by page", title)
        assert len(rows) == 2 and all("new" not in r["content"] for r in rows)
    finally:
        await pool.execute("delete from kb_passages where title = $1", title)


async def test_concurrent_ingests_of_one_document_leave_no_duplicates(real_pool, mocker, tmp_path):
    await pool.execute(SCHEMA.read_text(encoding="utf-8"))
    title = f"Race {uuid4().hex[:8]}"
    try:
        _pdf(mocker, [f"page {i}" for i in range(30)])
        await asyncio.gather(
            *(kb.ingest_pdf(tmp_path / "r.pdf", source_type="book", title=title) for _ in range(6))
        )
        assert await _count(title) == 30
    finally:
        await pool.execute("delete from kb_passages where title = $1", title)


async def test_deleting_a_user_cascades_to_sessions_and_profile(real_pool):
    await pool.execute(SCHEMA.read_text(encoding="utf-8"))
    user_id = await _make_user(uuid4().hex[:8])
    sid = (await ChatSessionCRUD().create({"user_id": user_id, "car_id": "c"}))["id"]
    await pool.execute("insert into user_profiles (user_id) values ($1)", user_id)
    assert isinstance(sid, str) and isinstance(user_id, str)
    await pool.execute("delete from users where id = $1", user_id)
    assert await pool.fetchval('select count(*) from "Chatsession" where id = $1', sid) == 0
    assert await pool.fetchval("select count(*) from user_profiles where user_id = $1", user_id) == 0


async def test_revoke_tokens_cutoff_is_inclusive_and_never_moves_back(real_pool):
    await pool.execute(SCHEMA.read_text(encoding="utf-8"))
    users = UserCRUD()
    user_id = await _make_user(uuid4().hex[:8])
    try:
        cutoff = await pool.fetchval("select extract(epoch from now())::float8") + 60
        await users.revoke_tokens(user_id, cutoff)
        assert await users.exists(user_id, cutoff) is True  # minted in the same instant: valid
        assert await users.exists(user_id, cutoff - 0.00001) is False
        await users.revoke_tokens(user_id, cutoff - 30)  # an older cutoff does not reopen anything
        assert await users.exists(user_id, cutoff - 0.00001) is False
    finally:
        await pool.execute("delete from users where id = $1", user_id)


async def test_logout_revokes_access_and_refresh_tokens_and_a_new_login_works(real_pool):
    import httpx

    import main

    await pool.execute(SCHEMA.read_text(encoding="utf-8"))
    email = f"logout-{uuid4().hex[:8]}@example.com"
    creds = {"email": email, "password": "s3cret-pass"}

    def bearer(tokens):
        return {"Authorization": f"Bearer {tokens['access_token']}"}

    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as ac:
            assert (await ac.post("/api/v1/auth/register", json=creds)).status_code == 200
            old = (await ac.post("/api/v1/auth/login", json=creds)).json()
            assert (await ac.get("/api/v1/cars", headers=bearer(old))).status_code == 200
            assert (await ac.post("/api/v1/auth/logout", headers=bearer(old))).status_code == 200

            assert (await ac.get("/api/v1/cars", headers=bearer(old))).status_code == 401
            refresh = await ac.post("/api/v1/auth/refresh", json={"refresh_token": old["refresh_token"]})
            assert refresh.status_code == 401

            fresh = await ac.post("/api/v1/auth/login", json=creds)
            assert fresh.status_code == 200
            assert (await ac.get("/api/v1/cars", headers=bearer(fresh.json()))).status_code == 200
            renewed = await ac.post("/api/v1/auth/refresh", json={"refresh_token": fresh.json()["refresh_token"]})
            assert renewed.status_code == 200
    finally:
        await pool.execute("delete from users where lower(email) = $1", email)


async def test_session_lease_claim_renew_release_and_expiry(real_pool):
    await pool.execute(SCHEMA.read_text(encoding="utf-8"))
    sessions = ChatSessionCRUD()
    tag = uuid4().hex[:8]
    owner, other = await _make_user(tag), await _make_user(f"{tag}-other")
    try:
        sid = (await sessions.create({"user_id": owner, "car_id": "c"}))["id"]

        async def lease():
            return await pool.fetchval(
                'select extract(epoch from busy_until - now())::float8 from "Chatsession" where id = $1', sid
            )

        assert await sessions.claim(sid, other) is None  # another user's id never takes it
        assert await lease() is None
        wins = await asyncio.gather(*(sessions.claim(sid, owner) for _ in range(8)))
        holders = [w for w in wins if w is not None]
        assert len(holders) == 1  # concurrent claims: exactly one wins
        holder = holders[0]
        assert 290 < await lease() <= 300
        assert await sessions.claim(sid, owner) is None  # held

        await pool.execute("""update "Chatsession" set busy_until = now() + interval '5 seconds' where id = $1""", sid)
        assert await sessions.renew(sid, holder, 120) is True
        assert 110 < await lease() <= 120

        await sessions.release(sid, holder)
        assert await lease() is None
        assert await sessions.renew(sid, holder) is False  # nothing held, nothing renewed
        holder = await sessions.claim(sid, owner)
        assert holder is not None  # free again

        await pool.execute("""update "Chatsession" set busy_until = now() - interval '1 second' where id = $1""", sid)
        assert await sessions.claim(sid, other) is None  # an expired lease is still owner-only
        assert await sessions.claim(sid, owner) is not None  # a crashed worker's lease expires on its own

        assert await sessions.claim(str(uuid4()), owner) is None  # no such session
        assert await sessions.claim("not-a-uuid", owner) is None
        await sessions.delete({"id": sid, "user_id": owner})  # the lease dies with the row
        assert await sessions.claim(sid, owner) is None
    finally:
        await pool.execute("delete from users where id = any($1::uuid[])", [owner, other])


async def test_a_stale_lease_holder_cannot_touch_the_next_holders_lease(real_pool):
    await pool.execute(SCHEMA.read_text(encoding="utf-8"))
    sessions = ChatSessionCRUD()
    owner = await _make_user(uuid4().hex[:8])
    try:
        sid = (await sessions.create({"user_id": owner, "car_id": "c"}))["id"]
        a = await sessions.claim(sid, owner, 0.2)
        assert isinstance(a, str)
        await asyncio.sleep(0.3)  # A's lease expires while its turn is still running
        b = await sessions.claim(sid, owner)
        assert isinstance(b, str) and b != a
        assert await sessions.renew(sid, a) is False  # A's late renew finds the lease gone
        await sessions.release(sid, a)  # A's late release leaves B's lease alone
        row = await pool.fetchrow(
            'select busy_by::text as holder, extract(epoch from busy_until - now())::float8 as left'
            ' from "Chatsession" where id = $1', sid,
        )
        assert row["holder"] == b and 290 < row["left"] <= 300
        assert await sessions.claim(sid, owner) is None  # still B's
        assert await sessions.renew(sid, b, 120) is True
        await sessions.release(sid, b)
        row = await pool.fetchrow('select busy_by, busy_until from "Chatsession" where id = $1', sid)
        assert row == {"busy_by": None, "busy_until": None}
    finally:
        await pool.execute("delete from users where id = $1", owner)


async def test_heartbeat_keeps_a_lease_alive_past_its_length(real_pool, monkeypatch):
    from app.services import diagnosis

    await pool.execute(SCHEMA.read_text(encoding="utf-8"))
    monkeypatch.setattr(diagnosis, "sessions", ChatSessionCRUD())
    monkeypatch.setattr(diagnosis, "LEASE_SECONDS", 0.3)  # a 300 s lease, scaled down
    monkeypatch.setattr(diagnosis, "RENEW_EVERY_SECONDS", 0.05)
    owner = await _make_user(uuid4().hex[:8])
    try:
        sid = (await diagnosis.sessions.create({"user_id": owner, "car_id": "c"}))["id"]
        lease = diagnosis.Lease(sid, await diagnosis.sessions.claim(sid, owner, 0.3))
        beat = asyncio.create_task(diagnosis._heartbeat(lease))
        for _ in range(12):  # 1.2 s: four lease lengths
            await asyncio.sleep(0.1)
            assert await diagnosis.sessions.claim(sid, owner) is None
        beat.cancel()
        assert not lease.lost
        await asyncio.sleep(0.4)  # no more beats: the lease runs out on its own
        assert await diagnosis.sessions.claim(sid, owner) is not None
    finally:
        await pool.execute("delete from users where id = $1", owner)
