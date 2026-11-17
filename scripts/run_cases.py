"""Run the diagnosis quality cases through run_turn and write a Markdown report.

Uses the real DeepSeek client and KB (needs the .env settings and the applied schema), with an in-memory
session store, no Storage uploads, and the findings a case's optional `history:` seeds (never read from or
written to the database). Checkboxes are pre-marked by case-insensitive substring and meant
to be confirmed by a person: [x] means satisfied (a must_mention is present, a must_not is absent).
Media cases are skipped, with a notice, while their files are missing.
Clarify-gate cases need no real media: `seeded_media` (video|audio) sends a placeholder file through the real turn with
the case's `observations` (transcript, sounds, signal) standing in for hearing and `frames` (names from FRAMES, drawn
with Pillow) standing in for video frames; `messages` seeds earlier turns and `pending_clarification: true` makes the
last seeded assistant question the session's pending one.
Usage: uv run scripts/run_cases.py [--only id,id] [--out reports/cases-<timestamp>.md]
"""
import argparse
import asyncio
import json
import re
import sys
import io
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from app.db import pool  # noqa: E402
from app.schemas.session import DiagnosisState, Message, Observations, PendingClarification, _norm  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CASES_FILE = ROOT / "tests" / "cases" / "diagnosis_cases.yaml"
# `judge: <name>` on a case: one temperature-0 call to the same DeepSeek model with the question below and the reply;
# the case passes only if the answer is NO. Substring patterns cannot tell "the hub fixed it" from "the history does not say
# whether it fixed it".
JUDGES = {
    "hub_outcome": "Does this reply state or imply that the hub replacement changed the original noise "
    "(fixed, reduced, removed, or did not change it)? Answer YES or NO.",
}
JUDGE_FIXTURES = ROOT / "tests" / "cases" / "judge_fixtures.yaml"
# Three more judge questions for the hub-replaced-new-noise case. They also see the seeded history block and the final
# state JSON (JUDGE_SEES), so "a fact not in the history" and "the state treats it as resolved" can be answered.
JUDGES |= {
    "hub_q1": "Does the reply or state treat the original noise as resolved, or the repaired corner as excluded or "
    "probably fine? Answer YES or NO.",
    "hub_q2": "Does the reply or state say, as a fact, when the new noise began relative to the repair? Answer YES or NO.",
    "hub_q3": "Does the reply or state assert a fact about the car's history that is not in the history block? "
    "Answer YES or NO.",
}
JUDGE_SEES = {"hub_q1", "hub_q2", "hub_q3"}
HUB_FIXTURES = ROOT / "tests" / "cases" / "judge_fixtures_hub.yaml"
HUB_CASE = "hub-replaced-new-noise"
REQUIRED_FIELDS = ("id", "car_id", "must_mention", "must_not", "expects_citation")
HISTORY_FIELDS = ("kind", "text", "source", "date")  # a case's optional `history:` seeds the car's earlier findings
CARS = {"toyota-echo-2001": {"id": "toyota-echo-2001", "make": "Toyota", "model": "Echo", "year": 2001}}


def _door_seam() -> bytes:
    """A drawing, not a photo: a door panel with a visible seam and a label. Stands in for frames of an unrelated object."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (640, 480), (176, 180, 188))
    draw = ImageDraw.Draw(img)
    draw.rectangle((40, 40, 600, 440), outline=(60, 60, 70), width=4)
    draw.rectangle((318, 40, 326, 440), fill=(40, 40, 46))  # the seam
    font = ImageFont.load_default(size=26)
    draw.text((60, 60), "CAR DOOR PANEL", fill=(20, 20, 30), font=font)
    draw.text((60, 400), "close-up of the seam between two panels", fill=(20, 20, 30), font=font)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return buf.getvalue()


FRAMES = {"door-seam": _door_seam}  # names a case's `frames:` may use


def seed_row(case: dict) -> dict:
    """The session row a case starts from: its `messages` (earlier turns) and, with `pending_clarification: true`,
    the last of them (an assistant clarifying question) as the state's pending question."""
    messages = [
        Message(
            id=f"seed{i}", role=m["role"], content=m["content"], created_at="2026-11-14T10:00:00+00:00",
            media=m.get("media") or [], observations=m.get("observations"), clarify=m.get("clarify"),
        ).model_dump(mode="json")
        for i, m in enumerate(case.get("messages") or [])
    ]
    state: dict = {}
    if case.get("pending_clarification"):
        asked, media_msg = messages[-1], messages[-2]
        state = DiagnosisState(pending_clarification=PendingClarification(
            message_id=asked["id"], media_message_id=media_msg["id"], question_id=asked["clarify"]["question_id"],
            question=asked["clarify"]["question"], options=asked["clarify"]["options"],
        )).model_dump()
    return {"id": str(uuid4()), "user_id": "cases", "title": "", "messages": messages, "state": state}


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


def without_questions(reply: str) -> str:
    """The reply minus its sentences that end in "?": asking whether something happened is not claiming it did."""
    return re.sub(r"[^.!?\n]*\?", "", reply)


def parse_judge(answer: str) -> bool:
    """True for YES, False for NO; anything else raises: an unreadable answer must never pass a case."""
    m = re.match(r"\W*(YES|NO)\b", answer.strip().upper())
    if not m:
        raise ValueError(f"judge answered neither YES nor NO: {answer[:80]!r}")
    return m.group(1) == "YES"


JUDGE_VOTES = 3  # a temperature-0 call still flips on borderline text, so a verdict is the majority of three


def judge_prompt(name: str, reply: str, history: str | None = None, state: dict | None = None) -> str:
    """The question, then (for the JUDGE_SEES questions) the history block the reply was written against, the reply and
    the final state JSON; the older questions get the reply alone."""
    q = '"""'
    parts = [JUDGES[name]]
    if name in JUDGE_SEES:
        parts.append(f"Car history block (what the assistant was shown):\n{q}\n{history or '(none)'}\n{q}")
    parts.append(f"Reply:\n{q}\n{reply}\n{q}")
    if name in JUDGE_SEES:
        state_json = json.dumps(state, indent=2, ensure_ascii=False) if state is not None else "(none)"
        parts.append(f"Final state (JSON):\n{q}\n{state_json}\n{q}")
    return "\n\n".join(parts)


async def judge_votes(name: str, reply: str, history: str | None = None, state: dict | None = None) -> list[bool]:
    """The judge's JUDGE_VOTES answers for `reply`, True meaning YES."""
    from app.core.config import get_settings
    from app.services.llm import get_client

    async def ask() -> bool:
        resp = await get_client().chat.completions.create(
            model=get_settings().DEEPSEEK_MODEL, temperature=0,
            messages=[{"role": "user", "content": f'{JUDGES[name]}\n\nReply:\n"""\n{reply}\n"""'}],
        )
        return parse_judge(resp.choices[0].message.content or "")

    return list(await asyncio.gather(*(ask() for _ in range(JUDGE_VOTES))))


def majority(votes: list[bool]) -> bool:
    return sum(votes) * 2 > len(votes)


async def judge(name: str, reply: str, history: str | None = None, state: dict | None = None) -> bool:
    """The judge's verdict for `reply`: True means YES (the majority of the votes)."""
    return majority(await judge_votes(name, reply, history, state))


def checklist(
    case: dict, reply: str, clarify: list[dict] | None = None, state: dict | None = None, judged: bool | None = None
) -> list[str]:
    low = reply.lower()
    asserted = without_questions(reply).lower()  # `must_not` patterns are checked against statements only
    lines = [f"- [{'x' if s.lower() in low else ' '}] mentions: {s}" for s in case["must_mention"]]
    lines += [f"- [{' ' if s.lower() in asserted else 'x'}] avoids: {s}" for s in case["must_not"]]
    lines += [f"- [{' ' if s.lower() in low else 'x'}] does not name: {s}" for s in case.get("must_not_name") or []]
    causes = (state or {}).get("suspected_causes") or []
    top = causes[0]["cause"].lower() if causes else ""
    lines += [f"- [{' ' if s.lower() in top else 'x'}] first-ranked cause does not name: {s}" for s in case.get("top_cause_must_not_name") or []]
    ruled_out = " ".join(f"{r.get('cause', '')} {r.get('reason', '')}" for r in (state or {}).get("ruled_out") or []).lower()
    lines += [f"- [{' ' if s.lower() in ruled_out else 'x'}] ruled_out does not name: {s}" for s in case.get("ruled_out_must_not_name") or []]
    if case.get("judge"):
        answer = "not run" if judged is None else "YES" if judged else "NO"
        lines.append(f"- [{'x' if judged is False else ' '}] judge (lower bound) {case['judge']}: answered {answer} (must be NO)")
    if "expects_clarify" in case:
        asked = bool(clarify)
        lines.append(f"- [{'x' if asked == case['expects_clarify'] else ' '}] {'asks' if case['expects_clarify'] else 'does not ask'} a clarifying question")
    if "max_questions" in case:
        lines.append(f"- [{'x' if len(clarify or []) <= case['max_questions'] else ' '}] at most {case['max_questions']} clarifying question(s)")
    return lines


def render_case(
    case: dict, reply: str, state: dict | None, error: str | None = None, clarify: list[dict] | None = None,
    judged: bool | None = None,
) -> str:
    out = [f"## {case['id']}", ""]
    if case.get("text"):
        out += [f"> {case['text']}", ""]
    if case.get("media"):
        out += [f"Media: {', '.join(case['media'])}", ""]
    if case.get("seeded_media"):
        out += [f"Seeded {case['seeded_media']} (synthetic): frames {case.get('frames') or []}, observations {json.dumps(case.get('observations'))}", ""]
    out += ["### Reply", "", reply or "(empty)", ""]
    if clarify:
        out += ["### Clarify event", "", "```json", json.dumps(clarify, indent=2, ensure_ascii=False), "```", ""]
    if error:
        out += [f"**Error event:** {error}", ""]
    out += ["### Final state", "", "```json", json.dumps(state, indent=2, ensure_ascii=False), "```", ""]
    out += ["### Checklist", "", *checklist(case, reply, clarify, state, judged)]
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


def history_rows(case: dict) -> list[dict]:
    """The case's `history:` entries as car_findings rows, as if an earlier session had stored them."""
    rows = []
    for i, h in enumerate(case.get("history") or []):
        text = " ".join(h["text"].split())
        rows.append({
            "id": f"seed-{i}", "kind": h["kind"], "text": text, "norm": _norm(text), "status": "active", "source": h["source"],
            "evidence_quote": text if h["source"] == "user_confirmed" else None,
            "created_at": datetime.fromisoformat(str(h["date"])).replace(tzinfo=timezone.utc),
        })
    return rows


class MemoryFindings:
    """The FindingCRUD methods run_turn calls, over the case's seed; what the turn writes is dropped."""

    def __init__(self, rows: list[dict]):
        self.rows = rows

    async def in_garage(self, user_id, car_id):
        return True  # the case's car is the seed's car

    async def history(self, user_id, car_id, exclude_session):
        return self.rows

    async def for_car(self, user_id, car_id):
        return []

    async def apply(self, user_id, car_id, session_id, ops):
        pass


class NoStore:
    async def upload(self, user_id, session_id, message_id, n, kind, ext, src):
        return {"kind": kind, "path": f"{user_id}/{session_id}/{message_id}/{n}{ext}"}

    async def delete_prefix(self, *args, **kwargs):
        return 0


async def run_case(case: dict) -> tuple[str, dict | None, str | None, list[dict]]:
    """Run one case through the real pipeline; returns (reply, final state, error message, clarify events)."""
    from starlette.datastructures import UploadFile

    from app.services import diagnosis, findings
    from app.services.media import Upload, validate_uploads

    row = seed_row(case)
    diagnosis.sessions, diagnosis.media_store = MemorySessions(row), NoStore()
    findings.crud = MemoryFindings(history_rows(case))
    car = CARS.get(case["car_id"], {"id": case["car_id"]})
    tokens: list[str] = []
    state = error = None
    clarify: list[dict] = []
    real_observe = diagnosis._observe
    with tempfile.TemporaryDirectory() as tmp:
        handles = [(ROOT / m).open("rb") for m in case.get("media") or []]
        try:
            files = [UploadFile(file=h, filename=Path(h.name).name) for h in handles]
            uploads = await validate_uploads(files, Path(tmp))
            if case.get("seeded_media"):  # a placeholder file; the case's frames and observations stand in for the pipeline
                placeholder = Path(tmp) / "seeded.bin"
                placeholder.write_bytes(b"synthetic placeholder, never decoded")
                uploads = [Upload(case["seeded_media"], placeholder, 8.0, True)]
                frames = [FRAMES[n]() for n in case.get("frames") or []]
                heard = Observations.model_validate(case["observations"]) if case.get("observations") else None

                async def seeded_observe(_uploads):
                    return frames, heard

                diagnosis._observe = seeded_observe
            async for event, data in diagnosis.run_turn(dict(row), car, case.get("text"), uploads):
                print(f"  [{case['id']}] {event}", flush=True)
                if event == "token":
                    tokens.append(data["text"])
                elif event == "state":
                    state = data
                elif event == "clarify":
                    clarify.append(data)
                elif event == "error":
                    error = data["message"]
        finally:
            diagnosis._observe = real_observe
            for h in handles:
                h.close()
    return "".join(tokens), state, error, clarify


async def check_judge(name: str = "hub_outcome", rounds: int = 3) -> int:
    """Run the judge over the fixtures (known NO and known YES replies) `rounds` times; returns the number of wrong answers."""
    fixtures = yaml.safe_load(JUDGE_FIXTURES.read_text(encoding="utf-8"))
    wrong = total = 0
    for rnd in range(1, rounds + 1):
        for fx in fixtures:
            votes = await judge_votes(name, fx["reply"])
            said_yes = majority(votes)
            right = said_yes == (fx["expected"] == "YES")
            wrong += not right
            total += 1
            shown = "/".join("Y" if v else "N" for v in votes)
            print(f"round {rnd} {fx['id']:<14} expected {fx['expected']:<3} verdict (lower bound) {'YES' if said_yes else 'NO':<3} votes {shown} {'ok' if right else 'WRONG'}", flush=True)
    print(f"{total - wrong}/{total} right")
    return wrong


def case_history_block(case_id: str = HUB_CASE) -> str:
    """The Car history block a case's seeded history renders to: what the model (and the judge) sees."""
    from app.services.findings import render_history

    return render_history(history_rows(next(c for c in load_cases() if c["id"] == case_id)))


def hub_fixtures() -> list[dict]:
    """(id, reply, state, labels) for --check-hub-judges: D1-D10 with all three questions, and the earlier ten outcome
    fixtures with their current YES/NO labels as the Q1 label (no state: they were reply-only)."""
    d = yaml.safe_load(HUB_FIXTURES.read_text(encoding="utf-8"))
    old = yaml.safe_load(JUDGE_FIXTURES.read_text(encoding="utf-8"))
    return d + [{"id": f["id"], "reply": f["reply"], "state": None, "labels": {"q1": f["expected"]}} for f in old]


async def check_hub_judges(rounds: int = 3) -> int:
    """Run Q1-Q3 over hub_fixtures() `rounds` times, print every verdict; returns the wrong non-borderline verdicts."""
    history, fixtures = case_history_block(), hub_fixtures()
    wrong = total = borderline_yes = borderline = 0
    for rnd in range(1, rounds + 1):
        jobs = [(fx, q, label) for fx in fixtures for q, label in fx["labels"].items()]
        results = await asyncio.gather(*(judge_votes(f"hub_{q}", fx["reply"], history, fx["state"]) for fx, q, _ in jobs))
        for (fx, q, label), votes in zip(jobs, results, strict=True):
            said_yes = majority(votes)
            shown = "/".join("Y" if v else "N" for v in votes)
            verdict = "YES" if said_yes else "NO"
            if label == "BORDERLINE":
                borderline += 1
                borderline_yes += said_yes
                mark = "borderline (reported, not in the bar)"
            else:
                total += 1
                wrong += verdict != label
                mark = "ok" if verdict == label else "WRONG"
            print(f"round {rnd} {fx['id']:<10} {q.upper()} label {label:<10} judge (lower bound) {verdict:<3} votes {shown} {mark}", flush=True)
    print(f"{total - wrong}/{total} non-borderline right; {borderline} borderline verdicts, {borderline_yes} of them YES")
    return wrong


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
            reply, state, error, clarify = await run_case(case)
            judged = await judge(case["judge"], reply) if case.get("judge") and reply else None
            sections.append(render_case(case, reply, state, error, clarify, judged))
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
    parser.add_argument("--check-hub-judges", action="store_true", help="validate the Q1-Q3 judges on the hub fixtures, 3 rounds")
    parser.add_argument("--check-judge", action="store_true", help="run the judge over tests/cases/judge_fixtures.yaml, 3 rounds")
    args = parser.parse_args()
    if args.check_hub_judges:
        sys.exit(1 if asyncio.run(check_hub_judges()) else 0)
    if args.check_judge:
        sys.exit(1 if asyncio.run(check_judge()) else 0)
    asyncio.run(main(args.only, args.out))
