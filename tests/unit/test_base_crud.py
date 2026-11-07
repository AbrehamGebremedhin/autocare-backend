"""BaseCRUD builds parameterised SQL with quoted, checked identifiers. Fake pool; no database."""
import pytest

from app.db.crud import BaseCRUD, _ident


async def test_read_with_filter_is_parameterised(fake_pool):
    fake_pool.rows = [{"id": "x"}]
    assert await BaseCRUD("Car").read({"id": "x"}) == [{"id": "x"}]
    assert fake_pool.calls == [("fetch", 'SELECT * FROM "Car" WHERE "id" = $1', ("x",))]


async def test_read_without_filters_selects_everything(fake_pool):
    await BaseCRUD("Car").read()
    assert fake_pool.calls == [("fetch", 'SELECT * FROM "Car"', ())]


async def test_read_with_several_filters_numbers_placeholders_in_order(fake_pool):
    await BaseCRUD("Chatsession").read({"id": "s", "user_id": "u"})
    assert fake_pool.calls == [("fetch", 'SELECT * FROM "Chatsession" WHERE "id" = $1 AND "user_id" = $2', ("s", "u"))]


async def test_create_is_a_parameterised_insert_returning_the_row(fake_pool):
    fake_pool.rows = [{"id": "toyota-echo-2001", "year": 2001}]
    row = await BaseCRUD("Car").create({"id": "toyota-echo-2001", "year": 2001, "car_guide_links": []})
    assert row == {"id": "toyota-echo-2001", "year": 2001}
    assert fake_pool.calls == [(
        "fetchrow",
        'INSERT INTO "Car" ("id", "year", "car_guide_links") VALUES ($1, $2, $3) RETURNING *',
        ("toyota-echo-2001", 2001, []),
    )]


async def test_update_sets_then_filters_and_returns_rows(fake_pool):
    fake_pool.rows = [{"id": "c"}]
    rows = await BaseCRUD("Car").update({"id": "c"}, {"kb_status": "ready", "kb_passage_count": 3})
    assert rows == [{"id": "c"}]
    assert fake_pool.calls == [(
        "fetch",
        'UPDATE "Car" SET "kb_status" = $1, "kb_passage_count" = $2 WHERE "id" = $3 RETURNING *',
        ("ready", 3, "c"),
    )]


async def test_delete_filters_and_returns_rows(fake_pool):
    fake_pool.rows = [{"id": "s"}]
    assert await BaseCRUD("Chatsession").delete({"id": "s", "user_id": "u"}) == [{"id": "s"}]
    assert fake_pool.calls == [(
        "fetch", 'DELETE FROM "Chatsession" WHERE "id" = $1 AND "user_id" = $2 RETURNING *', ("s", "u"),
    )]


BAD_NAMES = ['id"; drop table x; --', "Car; drop", "", "1abc", "a-b", "a b", 'a"b', "Car\n", "ጎማ"]


@pytest.mark.parametrize("name", BAD_NAMES)
def test_ident_rejects_anything_but_plain_identifiers(name):
    with pytest.raises(ValueError):
        _ident(name)


def test_ident_quotes_plain_identifiers():
    assert _ident("Car") == '"Car"' and _ident("_x9") == '"_x9"'


def test_bad_table_name_raises_without_touching_the_pool(fake_pool):
    with pytest.raises(ValueError):
        BaseCRUD("Car; drop")
    assert fake_pool.calls == []


@pytest.mark.parametrize("call", [
    lambda c: c.read({'id"; drop table x; --': 1}),
    lambda c: c.create({'id"; drop table x; --': 1}),
    lambda c: c.update({"id": "c"}, {'id"; drop table x; --': 1}),
    lambda c: c.update({'id"; drop table x; --': 1}, {"make": "x"}),
    lambda c: c.delete({'id"; drop table x; --': 1}),
])
async def test_bad_column_name_raises_without_touching_the_pool(fake_pool, call):
    with pytest.raises(ValueError):
        await call(BaseCRUD("Car"))
    assert fake_pool.calls == []


@pytest.mark.parametrize("call", [
    lambda c: c.update({}, {"make": "x"}),
    lambda c: c.update({"id": "c"}, {}),
    lambda c: c.delete({}),
    lambda c: c.create({}),
])
async def test_unfiltered_writes_and_empty_data_are_refused(fake_pool, call):
    with pytest.raises(ValueError):
        await call(BaseCRUD("Car"))
    assert fake_pool.calls == []
