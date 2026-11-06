"""Real Postgres check of schema.sql, the pool and BaseCRUD.

Runs only with `-m db` and TEST_DATABASE_URL pointing at a throwaway database
(the schema is applied to it). Every other run deselects or skips it.
"""
import os
from pathlib import Path
from uuid import uuid4

import pytest

from app.CRUD.chat_session_crud import ChatSessionCRUD
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
    user_id = str(uuid4())
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
        await pool.execute('delete from "Chatsession" where user_id = $1', user_id)
