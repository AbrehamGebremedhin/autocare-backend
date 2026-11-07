"""app.db.pool against a fake asyncpg pool: no database."""
import asyncio

import asyncpg
import pytest

from app.db import pool


async def test_fetch_passes_sql_and_args_and_returns_plain_dicts(fake_pool):
    fake_pool.rows = [{"id": "a"}, {"id": "b"}]
    rows = await pool.fetch('SELECT * FROM "Car" WHERE "id" = $1', "a")
    assert fake_pool.calls == [("fetch", 'SELECT * FROM "Car" WHERE "id" = $1', ("a",))]
    assert rows == [{"id": "a"}, {"id": "b"}] and all(type(r) is dict for r in rows)


async def test_fetchrow_returns_dict_or_none(fake_pool):
    fake_pool.rows = [{"id": "a"}]
    assert await pool.fetchrow("SELECT 1 WHERE $1", True) == {"id": "a"}
    fake_pool.rows = []
    assert await pool.fetchrow("SELECT 1 WHERE $1", False) is None
    assert [c[2] for c in fake_pool.calls] == [(True,), (False,)]


async def test_fetchval_and_execute_pass_through(fake_pool):
    fake_pool.value = 1
    assert await pool.fetchval("select 1") == 1
    assert await pool.execute('DELETE FROM "Car" WHERE "id" = $1', "x") == "OK"
    assert fake_pool.calls == [("fetchval", "select 1", ()), ("execute", 'DELETE FROM "Car" WHERE "id" = $1', ("x",))]


async def test_transaction_commits_on_success(fake_pool):
    fake_pool.rows = [{"n": 1}]
    async with pool.transaction() as tx:
        await tx.execute("INSERT INTO t VALUES ($1)", 1)
        assert await tx.fetch("SELECT $1", 2) == [{"n": 1}]
        assert await tx.fetchrow("SELECT $1", 3) == {"n": 1}
        await tx.fetchval("SELECT $1", 4)
    assert fake_pool.transactions == ["commit"]
    assert [c[0] for c in fake_pool.calls] == ["execute", "fetch", "fetchrow", "fetchval"]
    assert fake_pool.acquired == 0  # the connection went back to the pool


async def test_transaction_rolls_back_on_exception(fake_pool):
    with pytest.raises(RuntimeError):
        async with pool.transaction() as tx:
            await tx.execute("INSERT INTO t VALUES ($1)", 1)
            raise RuntimeError("batch 2 failed")
    assert fake_pool.transactions == ["rollback"]
    assert fake_pool.acquired == 0


async def test_helpers_raise_when_pool_not_initialised():
    pool.set_pool_for_tests(None)
    with pytest.raises(RuntimeError):
        await pool.fetchval("select 1")


class _FakeConnection:
    def __init__(self):
        self.codecs = {}

    async def set_type_codec(self, typename, *, encoder, decoder, schema, format="text"):
        self.codecs[typename] = (encoder, decoder, schema, format)


async def test_connection_init_registers_json_codecs_without_ascii_escapes():
    conn = _FakeConnection()
    await pool._init_connection(conn)
    for typename in ("jsonb", "json"):
        encoder, decoder, schema, fmt = conn.codecs[typename]
        assert (schema, fmt) == ("pg_catalog", "text")
        encoded = encoder({"t": "ጎማ"})
        assert "ጎማ" in encoded and "\\u" not in encoded
        assert decoder(encoded) == {"t": "ጎማ"}


async def test_connection_init_maps_uuid_to_str():
    conn = _FakeConnection()
    await pool._init_connection(conn)
    encoder, decoder, schema, fmt = conn.codecs["uuid"]
    assert (schema, fmt) == ("pg_catalog", "text")
    sid = "33333333-3333-4333-8333-333333333333"
    assert encoder(sid) == sid and decoder(sid) == sid


@pytest.fixture
def created(monkeypatch, fake_pool):
    """No pool installed; asyncpg.create_pool is replaced and records its kwargs."""
    pool.set_pool_for_tests(None)
    calls = []

    async def create_pool(**kwargs):
        calls.append(kwargs)
        return fake_pool

    monkeypatch.setattr(pool.asyncpg, "create_pool", create_pool)
    return calls


async def test_init_pool_sizes_and_codecs_and_is_idempotent(created, fake_pool):
    await pool.init_pool("postgresql://u:p@db.invalid/x")
    await pool.init_pool("postgresql://u:p@db.invalid/x")
    assert len(created) == 1
    kwargs = created[0]
    assert kwargs["dsn"] == "postgresql://u:p@db.invalid/x"
    assert (kwargs["min_size"], kwargs["max_size"]) == (1, 10)
    assert kwargs["init"] is pool._init_connection
    fake_pool.value = 1
    assert await pool.fetchval("select 1") == 1


async def test_concurrent_init_pool_calls_open_one_pool(monkeypatch, fake_pool):
    pool.set_pool_for_tests(None)
    calls = []

    async def slow_create_pool(**kwargs):
        calls.append(kwargs)
        await asyncio.sleep(0.01)  # both callers are waiting here without the lock
        return fake_pool

    monkeypatch.setattr(pool.asyncpg, "create_pool", slow_create_pool)
    await asyncio.gather(*(pool.init_pool("postgresql://u:p@db.invalid/x") for _ in range(3)))
    assert len(calls) == 1


async def test_init_pool_defaults_to_settings_database_url(created):
    from app.core.config import get_settings

    await pool.init_pool()
    assert created[0]["dsn"] == get_settings().DATABASE_URL


async def test_close_pool_closes_and_forgets_the_pool(fake_pool):
    await pool.close_pool()
    assert fake_pool.closed
    await pool.close_pool()  # a second close is a no-op
    with pytest.raises(RuntimeError):
        await pool.fetch("select 1")


@pytest.mark.parametrize(
    "error",
    [
        ConnectionRefusedError("connect refused"),
        TimeoutError(),
        OSError("no route to host"),
        asyncpg.ConnectionDoesNotExistError("connection was closed in the middle of operation"),
        asyncpg.CannotConnectNowError("the database system is starting up"),
        asyncpg.AdminShutdownError("terminating connection due to administrator command"),
        asyncpg.TooManyConnectionsError("too many clients already"),
        asyncpg.InternalClientError("PoolConnectionHolder.connect() called while another connection already exists"),
    ],
)
async def test_connectivity_failures_become_database_unavailable(fake_pool, error):
    fake_pool.error = error
    for call in (pool.fetch, pool.fetchrow, pool.fetchval, pool.execute):
        with pytest.raises(pool.DatabaseUnavailable) as raised:
            await call("select 1")
        assert raised.value.__cause__ is error and str(raised.value) == type(error).__name__  # no driver text


@pytest.mark.parametrize(
    "error", [asyncpg.UniqueViolationError("duplicate key"), asyncpg.DataError("out of range"), RuntimeError("bug")]
)
async def test_query_errors_are_not_reported_as_an_outage(fake_pool, error):
    fake_pool.error = error
    with pytest.raises(type(error)):
        await pool.fetch("select 1")
