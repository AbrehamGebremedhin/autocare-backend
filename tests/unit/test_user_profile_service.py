"""user_profiles on the asyncpg pool: one bound statement per step, no read-then-insert race on creation."""
from unittest.mock import MagicMock

from app.services.user_profile_service import UserProfileService

USER = "11111111-1111-4111-8111-111111111111"


def _service() -> UserProfileService:
    return UserProfileService(car_crud=MagicMock())


async def test_profile_read_is_one_bound_select(fake_pool):
    fake_pool.rows = [{"user_id": USER, "cars": ["c1"], "preferences": {}}]
    assert (await _service()._get_user_profile(USER))["cars"] == ["c1"]
    [(method, sql, args)] = fake_pool.calls
    assert method == "fetch" and sql.startswith('SELECT * FROM "user_profiles"') and args == (USER,)


async def test_missing_profile_is_none(fake_pool):
    assert await _service()._get_user_profile(USER) is None


async def test_get_or_create_inserts_on_conflict_do_nothing_then_reads(fake_pool):
    fake_pool.status = "INSERT 0 1"
    fake_pool.rows = [{"user_id": USER, "cars": [], "preferences": {}}]
    profile = await _service()._get_or_create_user_profile(USER)
    assert profile == {"user_id": USER, "cars": [], "preferences": {}}
    (insert_method, insert_sql, insert_args), (read_method, _, read_args) = fake_pool.calls
    assert insert_method == "execute" and "ON CONFLICT (user_id) DO NOTHING" in insert_sql
    assert insert_args == (USER,) and read_method == "fetch" and read_args == (USER,)


async def test_update_returns_whether_a_row_changed(fake_pool):
    fake_pool.rows = [{"user_id": USER, "cars": ["c1"]}]
    assert await _service()._update_user_profile(USER, {"cars": ["c1"]}) is True
    _, sql, args = fake_pool.calls[0]
    assert sql.startswith('UPDATE "user_profiles" SET "cars" = $1, "updated_at" = $2 WHERE "user_id" = $3')
    assert sql.endswith("RETURNING *")
    assert args[0] == ["c1"] and args[2] == USER

    fake_pool.rows = []
    assert await _service()._update_user_profile(USER, {"cars": []}) is False
