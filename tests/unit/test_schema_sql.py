"""app/db/sql/schema.sql defines every column the code reads or writes. Parsed offline; no database."""
import re
from pathlib import Path

import pytest

from app.schemas.Car import CarBase
from app.services import kb

SCHEMA = Path(__file__).resolve().parents[2] / "app" / "db" / "sql" / "schema.sql"

# Columns the code touches outside the pydantic models.
CHATSESSION_COLUMNS = {"id", "user_id", "car_id", "title", "messages", "state", "created_at", "updated_at"}  # chat_session_crud.py, session_route.py
KB_PASSAGE_COLUMNS = {"source_type", "car_id", "title", "page", "content"}  # kb.ingest_pdf rows and its delete filter
PROFILE_COLUMNS_WRITTEN = {"user_id", "cars", "preferences", "created_at", "updated_at"}  # user_profile_service
USER_COLUMNS = {"id", "email", "password_hash", "google_sub", "created_at", "tokens_valid_after"}  # accounts (email+password, Google sign-in)
TABLES = {"users", "user_profiles", "Car", "Chatsession", "kb_passages"}

_CONSTRAINT_WORDS = {"constraint", "primary", "unique", "check", "foreign", "exclude"}
_IDENT = r'("[^"]+"|\w+)'


def _sql() -> str:
    return re.sub(r"--[^\n]*", "", SCHEMA.read_text(encoding="utf-8"))


def _name(raw: str) -> str:
    return raw.strip('"')


def _paren_body(text: str, start: int) -> str:
    """Text inside the parenthesis that opens at text[start]."""
    depth = 0
    for i in range(start, len(text)):
        depth += {"(": 1, ")": -1}.get(text[i], 0)
        if depth == 0:
            return text[start + 1:i]
    raise ValueError("unbalanced parentheses")


def _split_top_level(body: str) -> list[str]:
    parts, depth, current = [], 0, ""
    for ch in body:
        depth += {"(": 1, ")": -1}.get(ch, 0)
        if ch == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += ch
    return [p.strip() for p in [*parts, current] if p.strip()]


def _columns(body: str) -> set[str]:
    return {_name(item.split()[0]) for item in _split_top_level(body) if item.split()[0].lower() not in _CONSTRAINT_WORDS}


def _table_bodies(sql: str) -> dict[str, str]:
    bodies = {}
    for m in re.finditer(rf"create\s+table\s+(?:if\s+not\s+exists\s+)?{_IDENT}\s*\(", sql, re.IGNORECASE):
        bodies[_name(m.group(1))] = _paren_body(sql, m.end() - 1)
    return bodies


@pytest.fixture(scope="module")
def schema() -> dict[str, set[str]]:
    """Replay the create table statements: table -> columns."""
    return {table: _columns(body) for table, body in _table_bodies(_sql()).items()}


def test_creates_exactly_the_app_tables(schema):
    assert set(schema) == TABLES


def test_file_is_one_transaction_with_lf_endings():
    raw = SCHEMA.read_bytes()
    assert b"\r" not in raw
    statements = [s.strip().lower() for s in _sql().split(";") if s.strip()]
    assert statements[0] == "begin" and statements[-1] == "commit"
    assert [s for s in statements if s in ("begin", "commit")] == ["begin", "commit"]


def test_every_statement_is_guarded_and_nothing_is_dropped():
    sql = _sql()
    assert not re.search(r"\bcreate\s+table\b(?!\s+if\s+not\s+exists)", sql, re.IGNORECASE)
    assert not re.search(r"\bcreate\s+(?:unique\s+)?index\b(?!\s+if\s+not\s+exists)", sql, re.IGNORECASE)
    assert not re.search(r"\bcreate\s+function\b", sql, re.IGNORECASE)  # only create or replace
    assert re.search(r"\bcreate\s+or\s+replace\s+function\s+search_kb\b", sql, re.IGNORECASE)
    assert not re.search(r"\b(?:drop|alter|truncate)\b", sql, re.IGNORECASE)
    assert not re.search(r"\bdelete\s+from\b|\binsert\s+into\b|\bupdate\s+\S+\s+set\b", sql, re.IGNORECASE)


def test_no_rls_or_auth_schema_leftovers():
    sql = _sql().lower()
    assert "row level security" not in sql and "create policy" not in sql
    assert "auth." not in sql and "to_regclass" not in sql


def test_users_table(schema):
    assert USER_COLUMNS <= schema["users"]
    body = " ".join(_table_bodies(_sql())["users"].split()).lower()
    assert "id uuid primary key default gen_random_uuid()" in body
    assert "email text not null" in body
    assert "tokens_valid_after timestamptz not null default now()" in body
    assert re.search(r"check \(password_hash is not null or google_sub is not null\)", body)
    for column in ("password_hash", "google_sub"):
        assert not re.search(rf"\b{column} text not null", body)


def test_users_unique_indexes():
    sql = " ".join(_sql().split()).lower()
    assert re.search(r"create unique index if not exists \w+ on users \(lower\(email\)\);", sql)
    assert re.search(r"create unique index if not exists \w+ on users \(google_sub\) where google_sub is not null;", sql)


def test_user_profiles_reference_users(schema):
    assert PROFILE_COLUMNS_WRITTEN <= schema["user_profiles"]
    body = " ".join(_table_bodies(_sql())["user_profiles"].split()).lower()
    assert "user_id uuid primary key references users (id) on delete cascade" in body


def test_car_has_every_model_field(schema):
    assert set(CarBase.model_fields) | {"created_at"} <= schema["Car"]
    assert {"is_vectorized", "vector_chunk_count"}.isdisjoint(schema["Car"])
    body = " ".join(_table_bodies(_sql())["Car"].split()).lower()
    assert "check (kb_status in ('pending', 'ready', 'failed', 'no_manual'))" in body


def test_chatsession_columns(schema):
    assert CHATSESSION_COLUMNS <= schema["Chatsession"]
    body = " ".join(_table_bodies(_sql())["Chatsession"].split()).lower()
    assert "user_id uuid not null references users (id) on delete cascade" in body


def test_kb_passages_columns(schema):
    assert KB_PASSAGE_COLUMNS <= schema["kb_passages"]
    sql = " ".join(_sql().split()).lower()
    assert "create index if not exists kb_passages_tsv_idx on kb_passages using gin (tsv);" in sql


def _search_kb_signature() -> tuple[list[str], set[str]]:
    sql = _sql()
    m = re.search(r"function\s+search_kb\s*\(", sql, re.IGNORECASE)
    args = [a.split()[0] for a in _split_top_level(_paren_body(sql, m.end() - 1))]
    ret = re.search(r"returns\s+table\s*\(", sql[m.end():], re.IGNORECASE)
    returned = _columns(_paren_body(sql, m.end() + ret.end() - 1))
    return args, returned


def test_search_kb_signature_unchanged():
    args, returned = _search_kb_signature()
    assert args == ["tsq", "p_car_id", "k"]
    assert returned == {"source_type", "car_id", "title", "page", "content", "rank"}


async def test_search_kb_args_match_kb_search_params(fake_pool):
    await kb.search("engine overheating", "toyota-echo-2001", 5)
    [(_, sql, params)] = fake_pool.calls
    args, _ = _search_kb_signature()
    assert sql == "SELECT * FROM search_kb($1, $2, $3)"
    assert len(params) == len(args) == 3


def test_search_kb_returns_every_passage_field():
    _, returned = _search_kb_signature()
    assert set(kb.Passage.__annotations__) <= returned
