import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("run_cases", ROOT / "scripts" / "run_cases.py")
run_cases = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_cases)


def test_yaml_loads_and_every_case_has_required_fields():
    cases = run_cases.load_cases()
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)) and len(cases) >= 9
    for c in cases:
        assert all(f in c for f in run_cases.REQUIRED_FIELDS), c["id"]
        assert c["must_mention"] and c["must_not"] and isinstance(c["expects_citation"], bool)
        assert c.get("text") or c.get("media"), c["id"]
        assert all(m.startswith("tests/fixtures/e2e/") for m in c.get("media") or [])
    assert [c["id"] for c in cases if c["expects_citation"]] == ["overheat-fan-cycling", "reservoir-steam"]


def test_checklist_marks_by_case_insensitive_substring():
    case = {"must_mention": ["Radiator Cap", "head gasket"], "must_not": ["Refer to your manual", "other"]}
    lines = run_cases.checklist(case, "Check the RADIATOR CAP. Please refer to your manual.")
    assert lines == [
        "- [x] mentions: Radiator Cap",
        "- [ ] mentions: head gasket",
        "- [ ] avoids: Refer to your manual",
        "- [x] avoids: other",
    ]


def test_media_case_is_skipped_when_file_missing(tmp_path):
    case = {"id": "m", "media": ["tests/fixtures/e2e/a.jpg", "tests/fixtures/e2e/b.jpg"]}
    (tmp_path / "tests/fixtures/e2e").mkdir(parents=True)
    (tmp_path / "tests/fixtures/e2e/a.jpg").write_bytes(b"x")
    assert run_cases.missing_media(case, tmp_path) == ["tests/fixtures/e2e/b.jpg"]
    assert run_cases.missing_media({"id": "t"}, tmp_path) == []
    assert "SKIPPED" in run_cases.render_skipped(case, ["tests/fixtures/e2e/b.jpg"])


@pytest.mark.asyncio
async def test_main_skips_missing_media_and_writes_report(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_cases, "ROOT", tmp_path)
    cases = [{"id": "m", "car_id": "c", "media": ["tests/fixtures/e2e/none.jpg"], "must_mention": ["x"],
              "must_not": ["y"], "expects_citation": False}]
    monkeypatch.setattr(run_cases, "load_cases", lambda: cases)
    out = tmp_path / "reports" / "r.md"
    assert await run_cases.main(None, out) == out
    assert "SKIPPED" in out.read_text(encoding="utf-8")
    assert "SKIP m" in capsys.readouterr().out


def test_only_filters_and_rejects_unknown():
    cases = run_cases.load_cases()
    assert [c["id"] for c in run_cases.select_cases(cases, "rough-idle, brake-grinding")] == ["brake-grinding", "rough-idle"]
    assert run_cases.select_cases(cases, None) == cases
    with pytest.raises(SystemExit):
        run_cases.select_cases(cases, "nope")


@pytest.mark.asyncio
async def test_memory_sessions_mirror_the_crud_calls():
    row = {"messages": [], "state": {}, "title": ""}
    s = run_cases.MemorySessions(row)
    await s.append_messages("s", [{"id": "1", "content": "a"}])
    await s.update_message("s", "1", {"media": []})
    await s.save_state("s", {"k": 1}, "T")
    assert row == {"messages": [{"id": "1", "content": "a", "media": []}], "state": {"k": 1}, "title": "T"}
