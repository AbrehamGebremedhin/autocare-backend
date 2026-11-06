"""Dict-filter CRUD over the asyncpg pool.

Table and column names are checked against a strict identifier pattern and
double-quoted; every value is a bind parameter. Filters are ANDed equalities.
"""
import re
from typing import Any, Dict, List, Optional

from app.db import pool

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _ident(name: str) -> str:
    """The double-quoted identifier; ValueError for anything that is not a plain name."""
    if not isinstance(name, str) or not _IDENTIFIER.fullmatch(name):
        raise ValueError("invalid SQL identifier")
    return f'"{name}"'


def _equals(columns: Dict[str, Any], first: int = 1) -> list[str]:
    """'"col" = $n' for each key, numbering placeholders from `first`."""
    return [f"{_ident(key)} = ${n}" for n, key in enumerate(columns, first)]


class BaseCRUD:
    def __init__(self, table_name: str):
        self.table_name = table_name
        self._table = _ident(table_name)

    async def create(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not data:
            raise ValueError("nothing to insert")
        columns = ", ".join(_ident(key) for key in data)
        params = ", ".join(f"${n}" for n in range(1, len(data) + 1))
        sql = f"INSERT INTO {self._table} ({columns}) VALUES ({params}) RETURNING *"
        return await pool.fetchrow(sql, *data.values())

    async def read(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if not filters:
            return await pool.fetch(f"SELECT * FROM {self._table}")
        where = " AND ".join(_equals(filters))
        return await pool.fetch(f"SELECT * FROM {self._table} WHERE {where}", *filters.values())

    async def update(self, filters: Dict[str, Any], update_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not filters or not update_data:
            raise ValueError("update needs filters and data")
        sets = ", ".join(_equals(update_data))
        where = " AND ".join(_equals(filters, len(update_data) + 1))
        sql = f"UPDATE {self._table} SET {sets} WHERE {where} RETURNING *"
        return await pool.fetch(sql, *update_data.values(), *filters.values())

    async def delete(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not filters:
            raise ValueError("delete needs filters")
        where = " AND ".join(_equals(filters))
        return await pool.fetch(f"DELETE FROM {self._table} WHERE {where} RETURNING *", *filters.values())
