"""Real Postgres check of the prune strip update. Runs only with `-m db` and TEST_DATABASE_URL."""
import importlib.util
import os
from pathlib import Path
from uuid import uuid4

import pytest

from app.CRUD.chat_session_crud import ChatSessionCRUD
from app.CRUD.user_crud import UserCRUD
from app.db import pool

pytestmark = [
    pytest.mark.db,
    pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL is not set"),
]

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = ROOT / "app" / "db" / "sql" / "schema.sql"
_spec = importlib.util.spec_from_file_location("prune_media", ROOT / "scripts" / "prune_media.py")
prune_media = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prune_media)


@pytest.fixture
async def real_pool():
    pool.set_pool_for_tests(None)
    await pool.init_pool(os.environ["TEST_DATABASE_URL"])
    try:
        yield
    finally:
        await pool.close_pool()


async def test_strip_refs_rebuilds_media_and_keeps_everything_else(real_pool):
    await pool.execute(SCHEMA.read_text(encoding="utf-8"))
    sessions = ChatSessionCRUD()
    tag = uuid4().hex[:8]
    user_id = (await UserCRUD().create_google_user(f"prune-{tag}@example.com", f"sub-{tag}"))["id"]
    try:
        sid = (await sessions.create({"user_id": user_id, "car_id": "c"}))["id"]
        other = (await sessions.create({"user_id": user_id, "car_id": "c"}))["id"]
        a, b, c = (f"{user_id}/{sid}/m{i}/0.jpg" for i in (1, 2, 3))
        msgs = [
            {"id": "m1", "role": "user", "content": "ጎማ", "media": [{"kind": "image", "path": a}, {"kind": "image", "path": c}]},
            {"id": "m2", "role": "user", "content": "keep", "media": [{"kind": "image", "path": b}], "observations": None},
            {"id": "m3", "role": "assistant", "content": "no media key"},
            {"id": "m4", "role": "user", "content": "gone", "media": [{"kind": "image", "path": c}]},
        ]
        await sessions.append_messages(sid, msgs)
        await sessions.append_messages(other, [{"id": "m1", "media": [{"kind": "image", "path": a}]}])

        assert await prune_media.strip_refs(sid, [a, c]) == 2
        got = (await sessions.get_for_user(sid, user_id))["messages"]
        assert [m["id"] for m in got] == ["m1", "m2", "m3", "m4"]
        assert got[0] == {**msgs[0], "media": []}
        assert got[1] == msgs[1] and got[2] == msgs[2]
        assert got[3] == {**msgs[3], "media": []}
        untouched = (await sessions.get_for_user(other, user_id))["messages"]
        assert untouched[0]["media"] == [{"kind": "image", "path": a}]  # other sessions are not affected

        assert await prune_media.strip_refs(sid, [a, c]) == 0  # idempotent
        assert await prune_media.strip_refs(sid, []) == 0
    finally:
        await pool.execute("delete from users where id = $1", user_id)
