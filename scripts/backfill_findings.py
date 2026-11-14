"""Fill car_findings from the diagnosis state of existing sessions.

Usage: uv run python scripts/backfill_findings.py [--dry-run]
Sessions are only read. Safe to run twice: every row carries a backfill_key (session, kind, text) and a
conflict is skipped, so a second run adds nothing, and a row the owner edited or retracted is left as it is.
Only cars still in the owner's garage are filled. Prints counts, never content.

What each part of a session's state becomes:
- tests_done -> test_done, user_confirmed when a user message of that session carries the extracted fact for
  that test (the whole message is kept as the owner's words), else model_suggested.
- confirmed_findings -> confirmed_finding, always model_suggested: the state model wrote them, and nothing says
  the user did. The owner confirms one in the app (PATCH .../findings/{id} with confirm: true).
- suspected_causes -> suspected_cause and ruled_out -> ruled_out, model_suggested.
A row's date is its session's last update (the true date of the finding is not stored).
One limit: a backfilled row the owner deleted outright (not retracted) is added again by a later run.
"""
import argparse
import asyncio
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import pool  # noqa: E402
from app.schemas.session import DiagnosisState, _norm  # noqa: E402
from app.services.findings import MODEL, USER, clean  # noqa: E402

_SESSIONS_SQL = """
SELECT s.id, s.user_id, s.car_id, s.state, coalesce(s.updated_at, s.created_at, now()) AS at,
       (SELECT coalesce(jsonb_agg(jsonb_build_object('content', e ->> 'content', 'facts', e -> 'observations' -> 'facts')), '[]'::jsonb)
          FROM jsonb_array_elements(s.messages) e
         WHERE e ->> 'role' = 'user' AND jsonb_typeof(e -> 'observations' -> 'facts') = 'array') AS user_facts
FROM "Chatsession" s
JOIN user_profiles p ON p.user_id = s.user_id AND s.car_id = ANY(p.cars)
ORDER BY s.updated_at, s.id
"""
_INSERT_SQL = (
    "INSERT INTO car_findings (user_id, car_id, kind, text, norm, source, evidence_quote, session_id, created_at, updated_at, backfill_key)"
    " VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8::uuid, $9, $9, $10) ON CONFLICT DO NOTHING"
)


def _owner_words(user_facts: list[dict], test: str) -> str:
    """The user's message in which the extractor found this test, or "" when there is none."""
    for m in user_facts:
        if any(isinstance(f, dict) and _norm(f.get("test") or "") == _norm(test) for f in m.get("facts") or []):
            return clean(m.get("content") or "")
    return ""


def rows_for(session: dict) -> list[dict]:
    """The findings one session's state stands for, as rows of car_findings (without ids)."""
    state = DiagnosisState.model_validate(session.get("state") or {})  # legacy states are upgraded on the way in
    user_facts = session.get("user_facts") or []
    wanted: list[tuple[str, str, str, str, str]] = []  # kind, text, key, source, quote
    for t in state.tests_done:
        quote = _owner_words(user_facts, t.test)
        wanted.append(("test_done", clean(f"{t.test}: {t.result}" if t.result else t.test), t.test, USER if quote else MODEL, quote))
    wanted += [("confirmed_finding", clean(f), f, MODEL, "") for f in state.confirmed_findings]
    wanted += [("suspected_cause", clean(c.cause), c.cause, MODEL, "") for c in state.suspected_causes]
    wanted += [("ruled_out", clean(f"{r.cause} ({r.reason})" if r.reason else r.cause), r.cause, MODEL, "") for r in state.ruled_out]
    rows = []
    for kind, text, key, source, quote in wanted:
        norm = _norm(key)
        if text and norm:
            rows.append({
                "user_id": session["user_id"], "car_id": session["car_id"], "kind": kind, "text": text, "norm": norm,
                "source": source, "evidence_quote": quote or None, "session_id": session["id"], "created_at": session["at"],
                "backfill_key": hashlib.sha1(f"{session['id']}|{kind}|{norm}".encode()).hexdigest(),
            })
    return rows


def plan(sessions: list[dict]) -> list[dict]:
    """All rows, the owner's own first, so a cause two sessions share keeps its strongest source."""
    return sorted((r for s in sessions for r in rows_for(s)), key=lambda r: (r["source"] != USER, r["created_at"]))


async def backfill(dry_run: bool) -> tuple[int, int]:
    """(rows the sessions stand for, rows added; 0 added on a dry run)."""
    rows = plan(await pool.fetch(_SESSIONS_SQL))
    added = 0
    if not dry_run:
        async with pool.transaction() as conn:
            for r in rows:
                status = await conn.execute(
                    _INSERT_SQL, r["user_id"], r["car_id"], r["kind"], r["text"], r["norm"], r["source"], r["evidence_quote"],
                    r["session_id"], r["created_at"], r["backfill_key"],
                )
                added += status == "INSERT 0 1"
    return len(rows), added


async def main(dry_run: bool) -> None:
    await pool.init_pool()
    try:
        total, added = await backfill(dry_run)
    finally:
        await pool.close_pool()
    print(f"{'would consider' if dry_run else 'considered'} {total} findings, added {added}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="count only; write nothing")
    asyncio.run(main(parser.parse_args().dry_run))
