"""One asyncpg pool per process, opened and closed by the FastAPI lifespan.

Every helper acquires a connection for one statement and gives it straight back;
`transaction()` holds one connection only for the length of its block. Rows come
back as plain dicts. jsonb/json travel as Python objects (UTF-8, no \\u escapes)
and uuid as str, so ids compare and format as plain strings everywhere.
"""
import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import asyncpg

from app.core.config import get_settings

_pool = None
_init_lock = asyncio.Lock()


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
                dsn=dsn or get_settings().DATABASE_URL, min_size=1, max_size=10, init=_init_connection
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


class Connection:
    """The four query helpers bound to one connection (or to the pool, one connection per statement)."""

    def __init__(self, target):
        self._target = target

    async def fetch(self, sql: str, *args) -> list[dict]:
        return [dict(r) for r in await self._target.fetch(sql, *args)]

    async def fetchrow(self, sql: str, *args) -> dict | None:
        row = await self._target.fetchrow(sql, *args)
        return None if row is None else dict(row)

    async def fetchval(self, sql: str, *args) -> Any:
        return await self._target.fetchval(sql, *args)

    async def execute(self, sql: str, *args) -> str:
        return await self._target.execute(sql, *args)

    async def executemany(self, sql: str, args_list: list) -> None:
        await self._target.executemany(sql, args_list)


async def fetch(sql: str, *args) -> list[dict]:
    return await Connection(_require()).fetch(sql, *args)


async def fetchrow(sql: str, *args) -> dict | None:
    return await Connection(_require()).fetchrow(sql, *args)


async def fetchval(sql: str, *args) -> Any:
    return await Connection(_require()).fetchval(sql, *args)


async def execute(sql: str, *args) -> str:
    return await Connection(_require()).execute(sql, *args)


@asynccontextmanager
async def transaction() -> AsyncIterator[Connection]:
    """Commits when the block finishes, rolls back when it raises."""
    async with _require().acquire() as conn:
        async with conn.transaction():
            yield Connection(conn)
