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
        assert c.get("text") or c.get("media") or c.get("seeded_media"), c["id"]
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


@pytest.mark.asyncio
async def test_main_opens_the_pool_for_the_cases_and_closes_it(tmp_path, monkeypatch, mocker, fake_pool):
    """kb.search needs the pool; without it every case would answer ungrounded."""
    from app.db import pool

    init = mocker.patch.object(pool, "init_pool", wraps=pool.init_pool)
    seen = []

    async def run_case(case):
        seen.append(init.await_count == 1 and not fake_pool.closed)
        return "reply", {}, None, []

    monkeypatch.setattr(run_cases, "run_case", run_case)
    monkeypatch.setattr(run_cases, "load_cases", lambda: [{"id": "t", "car_id": "c", "must_mention": [],
                                                           "must_not": [], "expects_citation": False}])
    await run_cases.main(None, tmp_path / "r.md")
    assert seen == [True]
    assert fake_pool.closed


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


def test_history_case_is_synthetic_and_seeds_findings_the_prompt_can_render():
    from app.services.findings import render_history

    [case] = [c for c in run_cases.load_cases() if c["id"] == "abs-after-new-hub"]
    assert all(set(h) == set(run_cases.HISTORY_FIELDS) for h in case["history"])
    block = render_history(run_cases.history_rows(case))
    assert "2026-11-05 (repair): new rear-left hub fitted" in block and "2026-11-02: rear-left wheel bearing worn" in block
    assert "rear-left" not in case["text"] and "bearing" not in case["text"]  # the check needs the history to pass
    assert [c["id"] for c in run_cases.load_cases() if c.get("history")] == ["abs-after-new-hub"]


@pytest.mark.asyncio
async def test_memory_findings_serve_the_seed_and_drop_writes():
    rows = run_cases.history_rows({"history": [{"kind": "repair_done", "source": "user_confirmed", "date": "2026-11-05", "text": "new hub"}]})
    store = run_cases.MemoryFindings(rows)
    assert await store.history("u", "c", "s") == rows and await store.for_car("u", "c") == []
    assert await store.apply("u", "c", "s", [("insert", {})]) is None


def _case(cid):
    return next(c for c in run_cases.load_cases() if c["id"] == cid)


def test_clarify_cases_are_synthetic_and_need_no_media_files():
    ids = ["clarify-noise-unrelated-frames", "clarify-clear-text-no-media", "clarify-clear-speech-in-media",
           "clarify-abs-light-unclear-audio", "clarify-answer-dont-know", "clarify-answer-with-speed"]
    cases = [_case(i) for i in ids]
    assert all(not c.get("media") and run_cases.missing_media(c) == [] for c in cases)
    assert [c["expects_clarify"] for c in cases] == [True, False, False, True, False, False]
    assert all(n in run_cases.FRAMES for c in cases for n in c.get("frames") or [])


def test_a_drawn_frame_is_a_decodable_jpeg_made_at_run_time():
    import io

    from PIL import Image

    with Image.open(io.BytesIO(run_cases.FRAMES["door-seam"]())) as im:
        assert im.format == "JPEG" and im.size == (640, 480)


def test_seed_row_makes_the_last_assistant_question_the_pending_one():
    from app.schemas.session import DiagnosisState, Message

    row = run_cases.seed_row(_case("clarify-answer-with-speed"))
    assert [m["role"] for m in row["messages"]] == ["user", "assistant"] and all(Message.model_validate(m) for m in row["messages"])
    pending = DiagnosisState.model_validate(row["state"]).pending_clarification
    assert (pending.message_id, pending.media_message_id, pending.question_id) == ("seed1", "seed0", "moving")
    assert [o.id for o in pending.options] == ["stationary", "slow", "town", "fast"]
    assert run_cases.seed_row({"id": "x"})["state"] == {} and run_cases.seed_row({"id": "x"})["messages"] == []


def test_checklist_reports_the_clarify_expectations():
    case = {"must_mention": [], "must_not": [], "must_not_name": ["door seam"], "expects_clarify": True, "max_questions": 1}
    asked = [{"question": "When?"}]
    assert run_cases.checklist(case, "When does it happen?", asked) == [
        "- [x] does not name: door seam", "- [x] asks a clarifying question", "- [x] at most 1 clarifying question(s)"]
    assert run_cases.checklist(case, "It is the door seam.", []) == [
        "- [ ] does not name: door seam", "- [ ] asks a clarifying question", "- [x] at most 1 clarifying question(s)"]
    assert run_cases.checklist({**case, "expects_clarify": False}, "x", [])[1] == "- [x] does not ask a clarifying question"


async def test_a_seeded_case_runs_the_real_turn_and_reports_the_clarify_event(fake_llm, monkeypatch):
    import json

    from app.services import diagnosis, findings

    for obj, name in ((diagnosis, "sessions"), (diagnosis, "media_store"), (diagnosis, "_observe"), (findings, "crud")):
        monkeypatch.setattr(obj, name, getattr(obj, name))  # run_case replaces these; the fixture puts them back
    seen = {}

    async def spy(messages):
        seen["images"] = [b for b in messages[-1]["content"] if b["type"] == "image_url"]
        seen["text"] = messages[-1]["content"][0]["text"]
        yield "CLARIFY " + json.dumps({"reason": "frames_unrelated_to_noise", "question_id": "when", "safety_note": "", "seen": "x"})

    monkeypatch.setattr(diagnosis.llm, "stream_reply", spy)
    reply, state, error, clarify = await run_cases.run_case(_case("clarify-noise-unrelated-frames"))
    assert error is None and clarify[0]["options"][0] == {"id": "accelerating", "label": "When accelerating"}
    assert len(seen["images"]) == 1 and "engine knocking (0.25)" in seen["text"] and "## Clarify check" in seen["text"]
    assert reply.startswith("When does it happen?") and state["pending_clarification"]["question_id"] == "when"


async def test_an_answer_case_starts_with_the_question_pending(fake_llm, monkeypatch):
    from app.services import diagnosis, findings

    for obj, name in ((diagnosis, "sessions"), (diagnosis, "media_store"), (diagnosis, "_observe"), (findings, "crud")):
        monkeypatch.setattr(obj, name, getattr(obj, name))
    reply, state, error, clarify = await run_cases.run_case(_case("clarify-answer-with-speed"))
    prompt = fake_llm.prompt[-1]["content"][0]["text"]
    assert clarify == [] and state["pending_clarification"] is None and "## Pending clarification" in prompt and "about 90 km/h" in prompt


def test_the_first_ranked_cause_check_reads_the_final_state():
    case = {"must_mention": [], "must_not": [], "top_cause_must_not_name": ["door"]}
    bad = {"suspected_causes": [{"cause": "Door seal wind noise"}, {"cause": "Wheel bearing"}]}
    good = {"suspected_causes": [{"cause": "Wheel bearing"}, {"cause": "Door seal wind noise"}]}  # lower down is allowed
    assert run_cases.checklist(case, "x", [], bad) == ["- [ ] first-ranked cause does not name: door"]
    assert run_cases.checklist(case, "x", [], good) == ["- [x] first-ranked cause does not name: door"]
    assert run_cases.checklist(case, "x", [], None) == ["- [x] first-ranked cause does not name: door"]
    answered = [c for c in run_cases.load_cases() if c.get("top_cause_must_not_name")]
    assert [c["id"] for c in answered] == ["clarify-answer-dont-know", "clarify-answer-with-speed"]
