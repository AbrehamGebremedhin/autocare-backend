"""Run the extract_facts cases against the real extractor (DeepSeek, needs the .env settings) and write a Markdown report.

Not part of pytest: it costs one model call per case. The cases are tests/cases/extractor_cases.yaml (invented messages).
Each case's context goes through prompts.history_turn, as the turn path does with the two messages before the latest one.
Usage: uv run scripts/run_extractor_cases.py [--only id,id] [--out reports/extractor-<timestamp>.md]
"""
import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from app.schemas.session import Fact, _norm  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CASES_FILE = ROOT / "tests" / "cases" / "extractor_cases.yaml"
EXPECTATIONS = ("expect_test", "expect_no_test", "expect_finding", "expect_repair", "expect_no_facts", "mention", "must_not_mention")


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


def _text(facts: list[Fact]) -> str:
    return " ".join(f"{f.test} {f.result} {f.part} {f.finding} {f.repair} {f.symptom_change}" for f in facts).lower()


def evaluate(case: dict, facts: list[Fact]) -> list[tuple[bool, str]]:
    """(passed, description) for each thing the case expects, plus the quote check every fact must pass."""
    checks: list[tuple[bool, str]] = []
    message = _norm(case["message"])
    for f in facts:
        if f.quote:
            checks.append((_norm(f.quote) in message, f"quote is a real substring of the message: {f.quote!r}"))
    if case.get("expect_test"):
        checks.append((any(f.test and f.result for f in facts), "a fact has a test and a result"))
    if case.get("expect_no_test"):
        checks.append((not any(f.test for f in facts), "no fact has a test"))
    if case.get("expect_finding"):
        checks.append((any(f.finding and f.basis == "stated" for f in facts), 'a fact states a finding (basis "stated")'))
    if case.get("expect_repair"):
        checks.append((any(f.repair for f in facts), "a fact has a repair"))
    if case.get("expect_no_facts"):
        checks.append((not facts, "no facts"))
    text = _text(facts)
    checks += [(w.lower() in text, f"facts mention: {w}") for w in case.get("mention") or []]
    checks += [(w.lower() not in text, f"facts do not mention: {w}") for w in case.get("must_not_mention") or []]
    return checks


def render_case(case: dict, facts: list[Fact], error: str | None = None) -> str:
    out = [f"## {case['id']}", "", f"> {' '.join(case['message'].split())}", ""]
    if error:
        return "\n".join(out + [f"**Error:** {error}", ""]) + "\n"
    out += [f"### Facts ({len(facts)})", ""]
    for f in facts:
        out.append("- " + ", ".join(f"{k}={v!r}" for k, v in f.model_dump().items() if v))
    out += ["", "### Checks", ""]
    out += [f"- [{'x' if ok else ' '}] {what}" for ok, what in evaluate(case, facts)]
    return "\n".join(out) + "\n"


async def run_case(case: dict) -> tuple[list[Fact], str | None]:
    from app.services import llm
    from app.services.prompts import history_turn

    context = [history_turn({"role": m["role"], "content": m["content"], "media": [], "observations": None}) for m in case["context"]]
    facts = await llm.extract_facts(" ".join(case["message"].split()), context[-2:])
    return facts, None


async def main(only: str | None, out: Path | None) -> Path:
    sections = []
    for case in select_cases(load_cases(), only):
        print(f"RUN {case['id']}", flush=True)
        facts, error = await run_case(case)
        sections.append(render_case(case, facts, error))
    out = out or ROOT / "reports" / f"extractor-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"# Extractor cases, {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n\n" + "\n".join(sections), encoding="utf-8")
    print(f"Report: {out}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="comma-separated case ids")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    asyncio.run(main(args.only, args.out))
