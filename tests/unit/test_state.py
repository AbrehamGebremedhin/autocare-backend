from app.schemas.session import DiagnosisState, RuledOut, SafetyFlag, SuspectedCause, TestDone, merge_state


def test_legacy_state_is_upgraded_without_losing_anything():
    old = {
        "title": "Bumpy-road ABS light",
        "suspected_causes": [
            {"cause": "a", "likelihood": 0.4, "evidence": "e", "next_check": "n"},
            {"cause": "b", "likelihood": 0.85},
            {"cause": "c", "likelihood": 0.75},
            {"cause": "d", "likelihood": 0.65},
        ],
        "safety_flags": ["do not drive until the hub nut is checked", "keep hands clear"],
    }
    s = DiagnosisState.model_validate(old)
    assert [(c.cause, c.rank) for c in s.suspected_causes] == [
        ("b", "most_likely"), ("c", "likely"), ("d", "possible"), ("a", "possible")]
    assert "likelihood" not in s.model_dump_json()
    assert s.suspected_causes[3].evidence == "e"
    assert [(f.text, f.severity) for f in s.safety_flags] == [
        ("do not drive until the hub nut is checked", "critical"), ("keep hands clear", "important")]
    assert s.title == "Bumpy-road ABS light" and s.current_focus == s.title and s.topic_history == [s.title]


def test_flags_are_deduplicated_sorted_and_capped():
    flags = [SafetyFlag(text="Keep hands clear.", severity="info"), SafetyFlag(text="keep hands clear", severity="info")]
    flags += [SafetyFlag(text=f"important {i}", severity="important") for i in range(8)]
    flags.append(SafetyFlag(text="hub nut", severity="critical"))
    out = DiagnosisState(safety_flags=flags).safety_flags
    assert len(out) == 6 and out[0].text == "hub nut" and all(f.severity != "info" for f in out)


def test_merge_never_drops_what_is_established_and_keeps_the_title():
    prev = DiagnosisState(
        title="Rear noise", current_focus="rear noise", topic_history=["rear noise"],
        confirmed_findings=["tire is fine"], tests_done=[TestDone(test="spun hub", result="noise")],
        ruled_out=[RuledOut(cause="Tire", reason="noise with tire off")],
        safety_flags=[SafetyFlag(text="check hub nut", severity="critical")],
    )
    new = DiagnosisState(
        title="ABS light", current_focus="ABS light", confirmed_findings=["new hub fitted"],
        suspected_causes=[SuspectedCause(cause="tire", rank="likely"), SuspectedCause(cause="connector", rank="most_likely")],
    )
    m = merge_state(prev, new, [TestDone(test="reseated connector", result="light still flickers")])
    assert m.title == "Rear noise"
    assert m.current_focus == "ABS light" and m.topic_history == ["rear noise", "ABS light"]
    assert m.confirmed_findings == ["tire is fine", "new hub fitted"]
    assert [t.test for t in m.tests_done] == ["spun hub", "reseated connector"]
    assert [c.cause for c in m.suspected_causes] == ["connector"]  # a ruled-out cause is not suggested again
    assert m.safety_flags[0].text == "check hub nut"
