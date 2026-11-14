"""The extractor eval's shape and its checks. The live run (real DeepSeek) is scripts/run_extractor_cases.py; nothing here calls it."""
import importlib.util
from pathlib import Path

import pytest

from app.schemas.session import Fact

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("run_extractor_cases", ROOT / "scripts" / "run_extractor_cases.py")
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)

IDS = ["tyre-spin-test-with-result", "bare-hub-spin-once-per-turn", "hub-removed-test-and-stated-finding", "new-hub-repair-answer",
       "symptom-report-is-not-a-test", "sending-a-clip-is-not-a-test", "bare-yes-answers-a-repair-question",
       "bare-yes-agreeing-with-a-diagnosis-confirms-nothing", "deliberate-condition-change-is-a-test"]


def _case(cid):
    return next(c for c in run.load_cases() if c["id"] == cid)


def test_the_cases_have_the_shape_the_runner_reads():
    cases = run.load_cases()
    assert [c["id"] for c in cases] == IDS
    for c in cases:
        assert c["message"].strip() and len(c["context"]) == 2 and [m["role"] for m in c["context"]] == ["user", "assistant"]
        assert any(k in c for k in run.EXPECTATIONS), c["id"]
        assert set(c) <= {"id", "message", "context", *run.EXPECTATIONS}, c["id"]
        assert isinstance(c["message"], str)  # a bare yes or off in YAML is a boolean
        assert all(isinstance(w, str) for w in [*c.get("mention", []), *c.get("must_not_mention", [])]), c["id"]


def test_each_kind_of_expectation_has_a_case():
    cases = run.load_cases()
    for key in ("expect_test", "expect_no_test", "expect_finding", "expect_repair", "repair_basis", "expect_no_confirmed_finding"):
        assert any(c.get(key) for c in cases), key


def test_the_continuation_message_is_not_a_case_any_more():
    """Old-client data only: the current client resumes through the retry route and never sends that text."""
    assert not any("carry on" in c["message"].lower() or "continue" in c["message"].lower() for c in run.load_cases())


def test_a_good_extraction_passes_every_check():
    c = _case("hub-removed-test-and-stated-finding")
    facts = [Fact(test="separated the hub from the drum", result="noise stayed", part="hub", quote="I separated the hub from the drum", basis="stated"),
             Fact(finding="the bearing was the culprit", part="bearing", quote="The bearing was the culprit.", basis="stated")]
    checks = run.evaluate(c, facts)
    assert all(ok for ok, _ in checks) and any("finding" in what for _, what in checks)


def test_a_quote_that_is_not_in_the_message_fails():
    c = _case("new-hub-repair-answer")
    checks = run.evaluate(c, [Fact(repair="new hub fitted", part="hub", quote="I fitted a new hub last week", basis="answered_question")])
    assert [ok for ok, what in checks if what.startswith("quote")] == [False]


def test_a_test_in_a_symptom_report_and_an_inferred_word_both_fail():
    c = _case("symptom-report-is-not-a-test")
    bad = [Fact(test="driving it", result="light flickers after the repair", quote="the warning light flickers on", basis="stated")]
    failed = [what for ok, what in run.evaluate(c, bad) if not ok]
    assert "no fact has a test" in failed and "facts do not mention: after" in failed and "facts do not mention: repair" in failed
    good = [Fact(result="the warning light flickers on, then goes off again", quote="the warning light flickers on", basis="stated")]
    assert all(ok for ok, _ in run.evaluate(c, good))


def test_the_no_facts_check_still_works_for_a_case_that_asks_for_it():
    case = {"message": "carry on", "expect_no_facts": True}
    assert run.evaluate(case, []) == [(True, "no facts")]
    assert run.evaluate(case, [Fact(test="spun the wheels", result="thump")]) == [(False, "no facts")]


def test_a_bare_yes_must_give_a_repair_with_basis_answered_question():
    c = _case("bare-yes-answers-a-repair-question")
    good = [Fact(repair="replaced the hub", part="hub", quote="yes", basis="answered_question")]
    assert all(ok for ok, _ in run.evaluate(c, good))
    for bad in ([], [Fact(repair="replaced the hub", part="hub", quote="yes", basis="stated")], [Fact(repair="replaced the hub", part="hub", quote="yes", basis="")]):
        assert not all(ok for ok, _ in run.evaluate(c, bad))


def test_a_bare_yes_to_a_diagnosis_may_give_agreed_diagnosis_or_nothing_never_a_stated_finding():
    c = _case("bare-yes-agreeing-with-a-diagnosis-confirms-nothing")
    assert all(ok for ok, _ in run.evaluate(c, []))
    assert all(ok for ok, _ in run.evaluate(c, [Fact(finding="it is the bearing", quote="yes", basis="agreed_diagnosis")]))
    for basis in ("stated", "answered_question", ""):
        assert not all(ok for ok, _ in run.evaluate(c, [Fact(finding="it is the bearing", quote="yes", basis=basis)]))


def test_a_deliberate_condition_change_must_give_a_test_with_a_result():
    c = _case("deliberate-condition-change-is-a-test")
    assert all(ok for ok, _ in run.evaluate(c, [Fact(test="turned the AC off", result="the noise stayed", quote="I turned the AC off and the noise stayed", basis="stated")]))
    assert not all(ok for ok, _ in run.evaluate(c, [Fact(result="the noise stayed with the AC off", quote="the noise stayed", basis="stated")]))


def test_select_cases_filters_and_rejects_unknown():
    assert [c["id"] for c in run.select_cases(run.load_cases(), "new-hub-repair-answer")] == ["new-hub-repair-answer"]
    with pytest.raises(SystemExit):
        run.select_cases(run.load_cases(), "nope")


def test_the_report_lists_facts_and_checks_and_never_the_context():
    c = _case("new-hub-repair-answer")
    text = run.render_case(c, [Fact(repair="new hub fitted", quote="a brand new hub assembly is fitted", basis="answered_question")])
    assert "### Facts (1)" in text and "- [x] a fact has a repair" in text and c["context"][0]["content"] not in text
