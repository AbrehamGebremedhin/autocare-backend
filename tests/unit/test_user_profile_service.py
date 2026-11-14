"""user_profiles on the asyncpg pool: one bound statement per step, no read-then-insert race on creation."""
from unittest.mock import AsyncMock, MagicMock

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


async def test_add_car_is_one_guarded_array_append(fake_pool):
    service = UserProfileService(car_crud=MagicMock(create_or_get_car=AsyncMock(return_value=[{"id": "c1"}])))
    fake_pool.status = "UPDATE 1"
    result = await service.add_car_to_user(USER, "Toyota", "Echo", 2001)
    assert result["success"] and result["car_id"] == "c1"
    [(method, sql, args)] = [c for c in fake_pool.calls if "array_append" in c[1]]  # the profile is never read-modified
    assert method == "execute" and args == (USER, "c1")
    assert "SET cars = array_append(cars, $2::text)" in sql
    assert "WHERE user_id = $1 AND NOT ($2::text = ANY(cars))" in sql


async def test_add_car_already_present_is_still_success(fake_pool):
    service = UserProfileService(car_crud=MagicMock(create_or_get_car=AsyncMock(return_value=[{"id": "c1"}])))
    fake_pool.status = "UPDATE 0"
    assert (await service.add_car_to_user(USER, "Toyota", "Echo", 2001))["success"] is True


async def test_remove_car_is_one_guarded_array_remove(fake_pool):
    fake_pool.status = "UPDATE 1"
    assert (await _service().remove_car_from_user(USER, "c1"))["success"] is True
    (method, sql, args), (detail_method, detail_sql, detail_args), (find_method, find_sql, find_args) = fake_pool.calls
    assert method == "execute" and args == (USER, "c1")
    assert "SET cars = array_remove(cars, $2::text)" in sql and "AND $2::text = ANY(cars)" in sql
    assert detail_method == "execute" and detail_args == (USER, "c1") and "DELETE FROM user_car_details" in detail_sql
    assert find_method == "execute" and find_args == (USER, "c1") and "DELETE FROM car_findings" in find_sql


async def test_remove_car_not_in_garage(fake_pool):
    fake_pool.status = "UPDATE 0"
    result = await _service().remove_car_from_user(USER, "c1")
    assert result["success"] is False


async def test_an_outage_is_never_an_empty_garage(fake_pool):
    import pytest

    from app.db.pool import DatabaseUnavailable
    from app.utils.exceptions import DatabaseException

    service = UserProfileService()
    fake_pool.error = ConnectionRefusedError()
    for call in (service.get_user_cars(USER), service.add_car_to_user(USER, "Toyota", "Echo", 2001),
                 service.remove_car_from_user(USER, "toyota-echo-2001")):
        with pytest.raises(DatabaseUnavailable):
            await call
    fake_pool.error = RuntimeError("bug")
    with pytest.raises(DatabaseException):  # any other failure is a 500, not [] either
        await service.get_user_cars(USER)


async def test_set_transmission_only_for_a_car_in_the_garage(fake_pool):
    fake_pool.status = "INSERT 0 1"
    assert await _service().set_transmission(USER, "c1", "manual") is True
    [(method, sql, args)] = fake_pool.calls
    assert method == "execute" and args == (USER, "c1", "manual")
    assert "ANY(cars)" in sql and "ON CONFLICT (user_id, car_id) DO UPDATE" in sql
    fake_pool.status = "INSERT 0 0"
    assert await _service().set_transmission(USER, "someone-elses", "manual") is False


async def test_add_car_stores_the_transmission(fake_pool):
    service = UserProfileService(car_crud=MagicMock(create_or_get_car=AsyncMock(return_value=[{"id": "c1"}])))
    fake_pool.status = "UPDATE 1"
    result = await service.add_car_to_user(USER, "Toyota", "Corolla", 2006, "automatic")
    assert result["car_info"]["transmission"] == "automatic"
    assert any("INSERT INTO user_car_details" in c[1] and c[2] == (USER, "c1", "automatic") for c in fake_pool.calls)


async def test_listing_cars_is_three_statements_whatever_the_garage_size(fake_pool, monkeypatch):
    ids = [f"make-model-{2000 + i}" for i in range(5)]
    answers = iter([
        [{"user_id": USER, "cars": ids, "preferences": {}}],
        [],
        [{"id": i, "make": "make", "model": "model", "year": 2000} for i in ids],
    ])

    async def fetch(sql, *args):
        fake_pool.calls.append(("fetch", sql, args))
        return next(answers)

    monkeypatch.setattr(fake_pool, "fetch", fetch)
    cars = await UserProfileService().get_user_cars(USER)
    assert [c["id"] for c in cars] == ids and len(fake_pool.calls) == 3


async def test_get_many_keeps_the_garage_order_and_leaves_out_unknown_ids(fake_pool):
    from app.CRUD.car_crud import CarCRUD

    fake_pool.rows = [{"id": "b", "make": "m", "model": "m", "year": 1}, {"id": "a", "make": "m", "model": "m", "year": 1}]
    cars = await CarCRUD().get_many(["a", "gone", "b"])
    assert [c["id"] for c in cars] == ["a", "b"]
    [(_, sql, args)] = fake_pool.calls
    assert "ANY($1::text[])" in sql and args == (["a", "gone", "b"],)
