"""The SQL files in app/db/sql define every column the code reads or writes. Parsed offline; no database."""
import re
from pathlib import Path

import pytest

from app.schemas.Car import CarBase
from app.schemas.User import UserProfile
from app.services import kb

SQL_DIR = Path(__file__).resolve().parents[2] / "app" / "db" / "sql"
FILES = sorted(SQL_DIR.glob("*.sql"))

# Columns the code touches outside the pydantic models.
CHATSESSION_COLUMNS = {"id", "user_id", "car_id", "title", "messages", "state", "updated_at"}  # chat_session_crud.py, session_route.py
KB_PASSAGE_COLUMNS = {"source_type", "car_id", "title", "page", "content"}  # kb.ingest_pdf rows, kb._delete_existing filters
PROFILE_COLUMNS_WRITTEN = {"user_id", "cars", "preferences", "created_at", "updated_at"}  # user_service, user_profile_service

_CONSTRAINT_WORDS = {"constraint", "primary", "unique", "check", "foreign", "exclude"}


def _sql(path: Path) -> str:
    return re.sub(r"--[^\n]*", "", path.read_text(encoding="utf-8"))


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
    cols = set()
    for item in _split_top_level(body):
        first = item.split()[0]
        if first.lower() not in _CONSTRAINT_WORDS:
            cols.add(_name(first))
    return cols


_IDENT = r'("[^"]+"|\w+)'
_STATEMENT = re.compile(
    rf"create\s+table\s+(?:if\s+not\s+exists\s+)?{_IDENT}\s*\("
    rf"|drop\s+table\s+(?:if\s+exists\s+)?{_IDENT}"
    rf"|alter\s+table\s+{_IDENT}\s+(add|drop)\s+column\s+(?:if\s+(?:not\s+)?exists\s+)?{_IDENT}",
    re.IGNORECASE,
)


def apply(paths, tables=None) -> dict[str, set[str]]:
    """Replay create/drop table and add/drop column statements, in file order."""
    tables = {} if tables is None else tables
    for path in paths:
        sql = _sql(path)
        for m in _STATEMENT.finditer(sql):
            created, dropped, altered, action, column = m.groups()
            if created:
                tables[_name(created)] = _columns(_paren_body(sql, m.end() - 1))
            elif dropped:
                tables.pop(_name(dropped), None)
            else:
                assert _name(altered) in tables, f"{path.name} alters {altered} before any file creates it"
                cols = tables[_name(altered)]
                (cols.add if action.lower() == "add" else cols.discard)(_name(column))
    return tables


@pytest.fixture(scope="module")
def schema():
    return apply(FILES)


def test_base_schema_runs_first():
    assert [p.name for p in FILES][:2] == ["000_base_schema.sql", "001_v2_schema.sql"]


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_file_is_one_transaction_with_lf_endings(path):
    raw = path.read_bytes()
    assert b"\r" not in raw
    statements = [s.strip().lower() for s in _sql(path).split(";") if s.strip()]
    assert statements[0] == "begin" and statements[-1] == "commit"


def test_car_has_every_model_field(schema):
    assert set(CarBase.model_fields) <= schema["Car"]


def test_user_profiles_has_every_model_field(schema):
    assert set(UserProfile.model_fields) | PROFILE_COLUMNS_WRITTEN <= schema["user_profiles"]


def test_chatsession_columns(schema):
    assert CHATSESSION_COLUMNS <= schema["Chatsession"]


def test_kb_passages_columns(schema):
    assert KB_PASSAGE_COLUMNS <= schema["kb_passages"]


def test_base_schema_alone_has_what_v2_adds_to_car():
    """001 only no-ops on "Car" when 000 already has its kb columns."""
    base = apply(FILES[:1])
    assert {"kb_status", "kb_passage_count"} <= base["Car"]
    assert {"is_vectorized", "vector_chunk_count"}.isdisjoint(base["Car"])


def _kb_status_checks(sql: str) -> list[str]:
    found = re.findall(r"constraint\s+car_kb_status_check\s+check\s*\((.*?)\)\s*\)", sql, re.IGNORECASE | re.DOTALL)
    return [" ".join(c.split()) for c in found]


def test_kb_status_check_matches_between_files():
    base, v2 = (_kb_status_checks(_sql(p)) for p in FILES[:2])
    assert len(base) == 1 and base == v2


def test_v2_is_idempotent_on_top_of_base():
    sql = _sql(FILES[1])
    assert not re.search(r"\badd\s+column\b(?!\s+if\s+not\s+exists)", sql, re.IGNORECASE)
    assert not re.search(r"\bdrop\s+(?:column|table|constraint)\b(?!\s+if\s+exists)", sql, re.IGNORECASE)
    assert not re.search(r"\bcreate\s+index\b(?!\s+if\s+not\s+exists)", sql, re.IGNORECASE)
    for name in re.findall(r"add\s+constraint\s+(\w+)", sql, re.IGNORECASE):
        assert re.search(rf"drop\s+constraint\s+if\s+exists\s+{name}\b", sql, re.IGNORECASE)
    # A plain create table is only allowed after a drop table if exists of the same table.
    for name in re.findall(rf"\bcreate\s+table\b(?!\s+if\s+not\s+exists)\s+{_IDENT}", sql, re.IGNORECASE):
        assert re.search(rf"drop\s+table\s+if\s+exists\s+{re.escape(name)}", sql, re.IGNORECASE)


def test_base_schema_creates_only_if_missing_and_enables_rls():
    sql = _sql(FILES[0])
    created = re.findall(rf"create\s+table\s+(if\s+not\s+exists\s+)?{_IDENT}", sql, re.IGNORECASE)
    assert {_name(t) for _, t in created} == {"Car", "user_profiles"}
    assert all(guard for guard, _ in created)
    for table in ("Car", "user_profiles"):
        assert re.search(rf'alter\s+table\s+"?{table}"?\s+enable\s+row\s+level\s+security', sql, re.IGNORECASE)
    assert not re.search(r"create\s+policy", sql, re.IGNORECASE)
    assert re.search(r"user_id\s+uuid\s+primary\s+key\s+references\s+auth\.users\s*\(\s*id\s*\)\s+on\s+delete\s+cascade", sql, re.IGNORECASE)


def _search_kb_signature() -> tuple[list[str], set[str]]:
    sql = _sql(FILES[1])
    m = re.search(r"function\s+search_kb\s*\(", sql, re.IGNORECASE)
    args = [a.split()[0] for a in _split_top_level(_paren_body(sql, m.end() - 1))]
    ret = re.search(r"returns\s+table\s*\(", sql[m.end():], re.IGNORECASE)
    returned = _columns(_paren_body(sql, m.end() + ret.end() - 1))
    return args, returned


async def test_search_kb_args_match_kb_rpc_params(mocker):
    rpc = mocker.patch("app.services.kb._rpc", return_value=[])
    await kb.search("engine overheating", "toyota-echo-2001", 5)
    name, params = rpc.call_args.args
    args, _ = _search_kb_signature()
    assert name == "search_kb"
    assert set(params) == set(args)


def test_search_kb_returns_every_passage_field():
    _, returned = _search_kb_signature()
    assert set(kb.Passage.__annotations__) <= returned
