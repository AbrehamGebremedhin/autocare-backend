"""One asyncpg pool per process, opened and closed by the FastAPI lifespan.

Every helper acquires a connection for one statement and gives it straight back;
`transaction()` holds one connection only for the length of its block. Getting a
connection waits at most ACQUIRE_TIMEOUT_SECONDS and a statement runs at most
COMMAND_TIMEOUT_SECONDS, so an unreachable database answers 503 quickly. Rows come
back as plain dicts. jsonb/json travel as Python objects (UTF-8, no \\u escapes)
and uuid as str, so ids compare and format as plain strings everywhere.
"""
import asyncio
import json
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncIterator, Iterator

import asyncpg

from app.core.config import get_settings

_pool = None
_init_lock = asyncio.Lock()

CONNECT_TIMEOUT_SECONDS = 5.0  # opening one connection
ACQUIRE_TIMEOUT_SECONDS = 5.0  # waiting for a free pooled connection, or opening a new one
COMMAND_TIMEOUT_SECONDS = 30.0  # one statement; a database that stops answering (no RST) times out here


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


async def _init_connection(conn) -> None:
    for typename in ("jsonb", "json"):
        await conn.set_type_codec(typename, encoder=_dumps, decoder=json.loads, schema="pg_catalog", format="text")
    await conn.set_type_codec("uuid", encoder=str, decoder=str, schema="pg_catalog", format="text")


async def init_pool(dsn: str | None = None) -> None:
    """Open the pool; a second call, or one racing the first, does nothing."""
    global _pool
    async with _init_lock:
        if _pool is None:
            _pool = await asyncpg.create_pool(
                dsn=dsn or get_settings().DATABASE_URL,
                min_size=1,
                max_size=10,
                init=_init_connection,
                timeout=CONNECT_TIMEOUT_SECONDS,
                command_timeout=COMMAND_TIMEOUT_SECONDS,
            )


async def close_pool() -> None:
    global _pool
    pool, _pool = _pool, None
    if pool is not None:
        await pool.close()


def set_pool_for_tests(fake) -> None:
    global _pool, _init_lock
    _pool = fake
    _init_lock = asyncio.Lock()  # each test has its own event loop; a used lock is bound to the old one


def _require():
    if _pool is None:
        raise RuntimeError("database pool is not initialised")
    return _pool


class DatabaseUnavailable(Exception):
    """Postgres cannot be reached or cannot serve: refused, unreachable, timed out, connection lost, shutting
    down or out of connections. The API answers 503 for it; it is never turned into "not found" or an empty
    list. Errors of the query itself (bad data, constraint violations, bugs) are not this."""


_UNAVAILABLE = (
    OSError,  # refused, unreachable, DNS, and TimeoutError (asyncio's included)
    asyncpg.PostgresConnectionError,  # SQLSTATE 08: no connection, or it dropped mid-statement
    asyncpg.CannotConnectNowError,  # 57P03: starting up or in recovery
    asyncpg.AdminShutdownError,
    asyncpg.CrashShutdownError,
    asyncpg.TooManyConnectionsError,
    # The pool's own state errors, seen live when its connections die. It also covers ProtocolError and
    # OutdatedSchemaCacheError (a client bug, a schema changed under a cached statement); those read as an
    # outage too, which is acceptable: the request can be retried.
    asyncpg.InternalClientError,
    # Only the InterfaceErrors of _CLOSED_MESSAGES (see _is_outage); any other is client misuse, a 500.
    asyncpg.InterfaceError,
)
# InterfaceErrors raised during a shutdown or restart; "another operation is in progress", a wrong argument
# count and the like are bugs and keep their own error.
_CLOSED_MESSAGES = ("pool is closing", "pool is closed", "connection is closed")


def _is_outage(exc: BaseException) -> bool:
    if isinstance(exc, asyncpg.InterfaceError):
        return str(exc).startswith(_CLOSED_MESSAGES)
    return isinstance(exc, _UNAVAILABLE)


@contextmanager
def _availability() -> Iterator[None]:
    try:
        yield
    except _UNAVAILABLE as exc:
        if not _is_outage(exc):
            raise
        raise DatabaseUnavailable(type(exc).__name__) from exc


class Connection:
    """The four query helpers bound to one connection (or to the pool, one connection per statement).
    Connectivity failures come out as DatabaseUnavailable."""

    def __init__(self, target):
        self._target = target

    async def fetch(self, sql: str, *args) -> list[dict]:
        with _availability():
            return [dict(r) for r in await self._target.fetch(sql, *args)]

    async def fetchrow(self, sql: str, *args) -> dict | None:
        with _availability():
            row = await self._target.fetchrow(sql, *args)
        return None if row is None else dict(row)

    async def fetchval(self, sql: str, *args) -> Any:
        with _availability():
            return await self._target.fetchval(sql, *args)

    async def execute(self, sql: str, *args) -> str:
        with _availability():
            return await self._target.execute(sql, *args)

    async def executemany(self, sql: str, args_list: list) -> None:
        with _availability():
            await self._target.executemany(sql, args_list)


@asynccontextmanager
async def _connection() -> AsyncIterator[Connection]:
    """One pooled connection for one statement, waiting at most ACQUIRE_TIMEOUT_SECONDS for it."""
    with _availability():
        async with _require().acquire(timeout=ACQUIRE_TIMEOUT_SECONDS) as conn:
            yield Connection(conn)


async def fetch(sql: str, *args) -> list[dict]:
    async with _connection() as conn:
        return await conn.fetch(sql, *args)


async def fetchrow(sql: str, *args) -> dict | None:
    async with _connection() as conn:
        return await conn.fetchrow(sql, *args)


async def fetchval(sql: str, *args) -> Any:
    async with _connection() as conn:
        return await conn.fetchval(sql, *args)


async def execute(sql: str, *args) -> str:
    async with _connection() as conn:
        return await conn.execute(sql, *args)


@asynccontextmanager
async def transaction() -> AsyncIterator[Connection]:
    """Commits when the block finishes, rolls back when it raises. Failing to get the connection, to begin,
    commit or roll back is DatabaseUnavailable like any statement; the block's own exceptions pass through."""
    raised = None
    try:
        async with _require().acquire(timeout=ACQUIRE_TIMEOUT_SECONDS) as conn:
            async with conn.transaction():
                try:
                    yield Connection(conn)
                except BaseException as exc:
                    raised = exc
                    raise
    except _UNAVAILABLE as exc:
        if exc is raised or not _is_outage(exc):
            raise  # the block's own error (say an OSError reading a file), or a client bug: not an outage
        raise DatabaseUnavailable(type(exc).__name__) from exc
