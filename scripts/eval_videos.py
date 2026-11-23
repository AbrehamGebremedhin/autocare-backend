"""Run the video-suggestion pipeline live on synthetic (car, job) pairs and write a Markdown table for a person to rate.

Calls the real YouTube API (needs YOUTUBE_API_KEY, uses search quota: up to two search.list calls per pair) and the real
DeepSeek model, through the same functions a turn uses, minus the cache and the database. No automated scoring.
For each pair: the top 3 candidates that pass the filters (before the relevance check) and the top 3 the check keeps.
Usage: uv run scripts/eval_videos.py [--cases tests/cases/video_eval_cases.yaml] [--out <path outside the repo>]
"""
import argparse
import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from app.services import videos  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CASES = ROOT / "tests" / "cases" / "video_eval_cases.yaml"
DEFAULT_OUT = Path(tempfile.gettempdir()) / "autocare-video-eval.md"


def _cell(text) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _row(stage: str, c: dict, verdict: str) -> str:
    minutes = f"{c['duration_seconds'] // 60}:{c['duration_seconds'] % 60:02d}"
    url = f"https://www.youtube.com/watch?v={c['video_id']}"
    return f"| {stage} | {_cell(c['title'])} | {_cell(c['channel'])} | {minutes} | {url} | {verdict} |"


async def _one(case: dict) -> str:
    car = {**case["car"], "id": "eval"}
    cause = videos.clean(case["job"])
    lines = [f"## {car['year']} {car['make']} {car['model']}: {case['job']}", ""]
    cands = await videos.candidates(car, cause)
    if cands is None:
        return "\n".join(lines + ["No candidates: no key, quota used up, or the API failed.", ""])
    verdicts = await videos.judge(car, cause, cands) if cands else []
    lines += ["| Stage | Title | Channel | Duration | URL | Verdict |", "|---|---|---|---|---|---|"]
    lines += [_row("before", c, "-") for c in cands[:3]]
    kept = videos.survivors(cands, verdicts or []) if verdicts is not None else []
    by_id = {c["video_id"]: c for c in cands}
    lines += [_row("after", by_id[k["video_id"]], k["verdict"]) for k in kept]
    if verdicts is None:
        lines.append("| after | (relevance check failed: no verdicts) | | | | |")
    elif not kept:
        lines.append("| after | (nothing kept: the app would show the search link) | | | | |")
    return "\n".join(lines + [""])


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    cases = yaml.safe_load(args.cases.read_text(encoding="utf-8"))
    sections = ["# Video suggestion eval", "", "Rate each row by hand; nothing here is scored.", ""]
    for case in cases:
        sections.append(await _one(case))
    args.out.write_text("\n".join(sections), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
