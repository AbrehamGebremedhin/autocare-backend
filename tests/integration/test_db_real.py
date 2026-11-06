"""Real Postgres check of schema.sql, the pool and BaseCRUD.

Runs only with `-m db` and TEST_DATABASE_URL pointing at a throwaway database
(the schema is applied to it). Every other run deselects or skips it.
"""
import os
from pathlib import Path
from uuid import uuid4

import asyncpg
import pytest

from app.CRUD.chat_session_crud import ChatSessionCRUD
from app.CRUD.user_crud import UserCRUD
from app.db import pool
from app.db.crud import BaseCRUD

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
        assert isinstance(created["id"], str) and await users.exists(created["id"])
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
