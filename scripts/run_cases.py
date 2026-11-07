"""Run the diagnosis quality cases through run_turn and write a Markdown report.

Uses the real DeepSeek client and KB (needs the .env settings and the applied schema), with an in-memory
session store and no Storage uploads. Checkboxes are pre-marked by case-insensitive substring and meant
to be confirmed by a person: [x] means satisfied (a must_mention is present, a must_not is absent).
Media cases are skipped, with a notice, while their files are missing.
Usage: uv run scripts/run_cases.py [--only id,id] [--out reports/cases-<timestamp>.md]
"""
import argparse
import asyncio
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from app.db import pool  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CASES_FILE = ROOT / "tests" / "cases" / "diagnosis_cases.yaml"
REQUIRED_FIELDS = ("id", "car_id", "must_mention", "must_not", "expects_citation")
CARS = {"toyota-echo-2001": {"id": "toyota-echo-2001", "make": "Toyota", "model": "Echo", "year": 2001}}


def load_cases(path: Path = CASES_FILE) -> list[dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def select_cases(cases: list[dict], only: str | None) -> list[dict]:
    if not only:
        return cases
    wanted = [i.strip() for i in only.split(",") if i.strip()]
    unknown = set(wanted) - {c["id"] for c in cases}
    if unknown:
        raise SystemExit(f"unknown case id(s): {', '.join(sorted(unknown))}")
    return [c for c in cases if c["id"] in wanted]


def missing_media(case: dict, root: Path = ROOT) -> list[str]:
    return [m for m in case.get("media") or [] if not (root / m).is_file()]


def checklist(case: dict, reply: str) -> list[str]:
    low = reply.lower()
    lines = [f"- [{'x' if s.lower() in low else ' '}] mentions: {s}" for s in case["must_mention"]]
    lines += [f"- [{' ' if s.lower() in low else 'x'}] avoids: {s}" for s in case["must_not"]]
    return lines


def render_case(case: dict, reply: str, state: dict | None, error: str | None = None) -> str:
    out = [f"## {case['id']}", ""]
    if case.get("text"):
        out += [f"> {case['text']}", ""]
    if case.get("media"):
        out += [f"Media: {', '.join(case['media'])}", ""]
    out += ["### Reply", "", reply or "(empty)", ""]
    if error:
        out += [f"**Error event:** {error}", ""]
    out += ["### Final state", "", "```json", json.dumps(state, indent=2, ensure_ascii=False), "```", ""]
    out += ["### Checklist", "", *checklist(case, reply)]
    if case["expects_citation"]:
        out.append("- [ ] cites a KB passage (confirm by hand)")
    return "\n".join(out) + "\n"


def render_skipped(case: dict, missing: list[str]) -> str:
    return f"## {case['id']}\n\nSKIPPED: missing media file(s): {', '.join(missing)}\n"


class MemorySessions:
    """The ChatSessionCRUD methods run_turn calls, over a dict."""

    def __init__(self, row: dict):
        self.row = row

    async def append_messages(self, session_id: str, messages: list[dict]) -> None:
        self.row["messages"] = [*self.row["messages"], *messages]

    async def update_message(self, session_id: str, message_id: str, patch: dict) -> bool:
        found = any(m["id"] == message_id for m in self.row["messages"])
        self.row["messages"] = [{**m, **patch} if m["id"] == message_id else m for m in self.row["messages"]]
        return found

    async def save_state(self, session_id: str, state: dict, title: str) -> None:
        self.row["state"], self.row["title"] = state, title


class NoStore:
    async def upload(self, user_id, session_id, message_id, n, kind, ext, src):
        return {"kind": kind, "path": f"{user_id}/{session_id}/{message_id}/{n}{ext}"}

    async def delete_prefix(self, *args, **kwargs):
        return 0


async def run_case(case: dict) -> tuple[str, dict | None, str | None]:
    """Run one case through the real pipeline; returns (reply, final state, error message)."""
    from starlette.datastructures import UploadFile

    from app.services import diagnosis
    from app.services.media import validate_uploads

    row = {"id": str(uuid4()), "user_id": "cases", "title": "", "messages": [], "state": {}}
    diagnosis.sessions, diagnosis.media_store = MemorySessions(row), NoStore()
    car = CARS.get(case["car_id"], {"id": case["car_id"]})
    tokens: list[str] = []
    state = error = None
    with tempfile.TemporaryDirectory() as tmp:
        handles = [(ROOT / m).open("rb") for m in case.get("media") or []]
        try:
            files = [UploadFile(file=h, filename=Path(h.name).name) for h in handles]
            uploads = await validate_uploads(files, Path(tmp))
            async for event, data in diagnosis.run_turn(dict(row), car, case.get("text"), uploads):
                print(f"  [{case['id']}] {event}", flush=True)
                if event == "token":
                    tokens.append(data["text"])
                elif event == "state":
                    state = data
                elif event == "error":
                    error = data["message"]
        finally:
            for h in handles:
                h.close()
    return "".join(tokens), state, error


async def main(only: str | None, out: Path | None) -> Path:
    cases = select_cases(load_cases(), only)
    sections = []
    await pool.init_pool()  # kb.search runs on the pool; without it every case would answer ungrounded
    try:
        for case in cases:
            missing = missing_media(case)
            if missing:
                print(f"SKIP {case['id']}: missing media {', '.join(missing)}")
                sections.append(render_skipped(case, missing))
                continue
            print(f"RUN {case['id']}", flush=True)
            reply, state, error = await run_case(case)
            sections.append(render_case(case, reply, state, error))
    finally:
        await pool.close_pool()
    if out is None:
        out = ROOT / "reports" / f"cases-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    header = f"# Diagnosis cases, {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n\n"
    out.write_text(header + "\n".join(sections), encoding="utf-8")
    print(f"Report: {out}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="comma-separated case ids")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    asyncio.run(main(args.only, args.out))
