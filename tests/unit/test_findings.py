"""The findings write rules and the Car history block. Synthetic data only."""
from datetime import datetime, timezone

import pytest

from app.CRUD.finding_crud import FindingCRUD
from app.schemas.session import DiagnosisState, Fact, RuledOut, SuspectedCause
from app.services import findings
from app.services.findings import MODEL, USER, Proposal, plan_writes, render_history

T1 = datetime(2026, 11, 2, 9, tzinfo=timezone.utc)
T2 = datetime(2026, 11, 5, 9, tzinfo=timezone.utc)
T3 = datetime(2026, 11, 8, 9, tzinfo=timezone.utc)


def row(id, kind, text, source=MODEL, status="active", created=T1, **extra):
    return {"id": id, "kind": kind, "text": text, "norm": findings._norm(text), "status": status, "source": source,
            "evidence_quote": "q" if source == USER else None, "created_at": created, **extra}


def fact(**kw):
    return Fact(**{"basis": "stated", **kw})


def ops_of(existing, proposals, kind):
    return [o for o in plan_writes(existing, proposals) if o[0] == kind]


# --- who may confirm -------------------------------------------------------------------------------------

def test_the_model_path_never_produces_a_user_confirmed_proposal():
    state = DiagnosisState(
        suspected_causes=[SuspectedCause(cause="Worn wheel bearing")],
        ruled_out=[RuledOut(cause="Loose wheel nut", reason="torque checked")],
        confirmed_findings=["rear-left bearing is worn"],
    )
    assert {p.source for p in findings.model_proposals(state)} == {MODEL}
    inserted = ops_of([], findings.model_proposals(state), "insert")
    assert len(inserted) == 3 and all(o[1]["source"] == MODEL and o[1]["evidence_quote"] is None for o in inserted)


def test_a_model_proposal_cannot_promote_or_touch_a_confirmed_row():
    existing = [row("a", "confirmed_finding", "Rear-left bearing worn", source=USER)]
    ops = plan_writes(existing, [Proposal("confirmed_finding", "rear-left bearing worn!", MODEL)])
    assert ops == []


def test_a_model_proposal_cannot_supersede_a_confirmed_row():
    existing = [row("a", "suspected_cause", "Loose wheel nut", source=USER)]
    ops = plan_writes(existing, [Proposal("ruled_out", "Loose wheel nut (fine)", MODEL, key="Loose wheel nut")])
    assert [o[0] for o in ops] == ["insert"]


def test_a_user_statement_promotes_a_matching_suggestion_with_its_quote():
    existing = [row("a", "confirmed_finding", "Rear-left bearing worn")]
    proposals = findings.user_proposals(
        [fact(finding="rear-left bearing worn", quote="the bearing spins rough")], "I took the wheel off and the bearing spins rough."
    )
    assert ops_of(existing, proposals, "promote") == [("promote", "a", "the bearing spins rough")]


def test_user_facts_are_confirmed_only_with_a_real_quote_and_an_accepted_basis():
    said = "I replaced the rear-left hub yesterday"
    good = findings.user_proposals([fact(repair="new rear-left hub fitted", quote="replaced the rear-left hub")], said)
    assert [(p.kind, p.source, p.quote) for p in good] == [("repair_done", USER, "replaced the rear-left hub")]
    for bad in (
        fact(repair="new hub fitted", quote="I fitted a Bosch hub"),  # the quote is not in the message
        fact(repair="new hub fitted", quote=""),
        fact(repair="new hub fitted", quote="replaced the rear-left hub", basis="agreed_diagnosis"),
        fact(repair="new hub fitted", quote="replaced the rear-left hub", basis=""),
    ):
        [p] = findings.user_proposals([bad], said)
        assert p.source == MODEL and p.quote == ""


def test_a_bare_yes_counts_only_for_a_question_about_an_observation_or_repair():
    [seen] = findings.user_proposals([fact(finding="the hub spins rough", quote="Yes", basis="answered_question")], "Yes")
    [agreed] = findings.user_proposals([fact(finding="it is the bearing", quote="Yes", basis="agreed_diagnosis")], "Yes")
    assert (seen.source, seen.quote) == (USER, "Yes") and agreed.source == MODEL


def test_a_test_fact_becomes_a_test_row_keyed_by_the_test():
    [p] = findings.user_proposals([fact(test="spun the hub", result="rough", quote="spun the hub")], "I spun the hub by hand")
    assert (p.kind, p.text, p.norm) == ("test_done", "spun the hub: rough", "spun the hub")


# --- supersede, never delete -----------------------------------------------------------------------------

def test_ruled_out_supersedes_the_cause_it_names_and_links_it():
    existing = [row("c", "suspected_cause", "Loose wheel nut")]
    ops = plan_writes(existing, [Proposal("ruled_out", "Loose wheel nut (torqued)", MODEL, key="loose wheel nut")])
    [(_, new)] = [o for o in ops if o[0] == "insert"]
    assert ("supersede", "c", new["id"]) in ops and not [o for o in ops if o[0] == "delete"]


def test_the_owners_finding_supersedes_a_ruled_out_it_contradicts():
    existing = [row("r", "ruled_out", "Worn wheel bearing")]
    ops = plan_writes(existing, [Proposal("confirmed_finding", "Worn wheel bearing", USER, quote="it is worn")])
    [(_, new)] = [o for o in ops if o[0] == "insert"]
    assert ops_of(existing, [], "supersede") == [] and ("supersede", "r", new["id"]) in ops


def test_the_model_does_not_bring_back_what_the_owner_retracted_or_newer_evidence_replaced():
    existing = [row("a", "suspected_cause", "Tone ring damaged", status="retracted"),
                row("b", "suspected_cause", "Loose wheel nut", status="superseded")]
    assert plan_writes(existing, [Proposal("suspected_cause", "tone ring damaged", MODEL),
                                  Proposal("suspected_cause", "Loose wheel nut", MODEL)]) == []
    # the owner saying it again is new evidence
    assert [o[0] for o in plan_writes(existing, [Proposal("suspected_cause", "Tone ring damaged", USER, quote="q")])] == ["insert"]


def test_suggestions_are_capped_oldest_first():
    existing = [row(f"{i:03}", "suspected_cause", f"cause {i}", created=datetime(2026, 1, 1, tzinfo=timezone.utc).replace(day=1 + i % 28, month=1 + i // 28)) for i in range(findings.MAX_MODEL_ROWS)]
    ops = plan_writes(existing, [Proposal("suspected_cause", "a brand new cause", MODEL)])
    assert ops_of(existing, [Proposal("suspected_cause", "a brand new cause", MODEL)], "delete") == [("delete", "000")]
    assert len(ops) == 2


# --- the block -------------------------------------------------------------------------------------------

def test_block_lists_confirmed_before_hypotheses_before_ruled_out_newest_first():
    rows = [
        row("1", "suspected_cause", "ABS tone ring damaged", created=T3),
        row("2", "confirmed_finding", "rear-left wheel bearing worn, confirmed by wheel-off spin test", source=USER, created=T1),
        row("3", "repair_done", "new rear-left hub fitted", source=USER, created=T2),
        row("4", "ruled_out", "loose wheel nut (torque checked)", created=T3),
        row("5", "suspected_cause", "old hypothesis", status="superseded"),
        row("6", "confirmed_finding", "retracted thing", source=USER, status="retracted"),
    ]
    block = render_history(rows)
    assert block.splitlines() == [
        "Confirmed by the owner:",
        "- 2026-11-05 (repair): new rear-left hub fitted",
        "- 2026-11-02: rear-left wheel bearing worn, confirmed by wheel-off spin test",
        "Unverified hypotheses (the model suggested these; nobody confirmed them):",
        "- 2026-11-08: ABS tone ring damaged",
        "Ruled out earlier (may change with new evidence):",
        "- 2026-11-08: loose wheel nut (torque checked)",
    ]


def test_block_is_the_same_text_whatever_order_the_rows_arrive_in():
    rows = [row(str(i), "suspected_cause", f"cause {i}", created=T1) for i in range(5)]
    assert render_history(rows) == render_history(reversed(rows)) == render_history(sorted(rows, key=lambda r: r["text"]))


def test_empty_history_gives_no_block():
    assert render_history([]) == ""


def test_over_the_cap_the_oldest_suggestions_go_first_then_ruled_out_then_confirmed():
    long = "x" * 290
    confirmed = [row(f"c{i}", "confirmed_finding", f"{long}{i}", source=USER, created=datetime(2026, 3, 1 + i, tzinfo=timezone.utc)) for i in range(5)]
    guesses = [row(f"g{i}", "suspected_cause", f"{long}{i}", created=datetime(2026, 2, 1 + i, tzinfo=timezone.utc)) for i in range(10)]
    out = [row(f"o{i}", "ruled_out", f"{long}{i}", created=datetime(2026, 2, 1 + i, tzinfo=timezone.utc)) for i in range(10)]
    block = render_history([*confirmed, *guesses, *out], limit=5500)
    assert len(block) <= 5500
    assert all(f"2026-03-0{1 + i}" in block for i in range(5))  # every confirmed item survives
    hypotheses = block.split("Unverified")[1].split("Ruled out")[0]
    assert "2026-02-10" in hypotheses and "2026-02-01" not in hypotheses  # the oldest suggestions went
    tiny = render_history([*confirmed, *guesses, *out], limit=1700)
    assert len(tiny) <= 1700 and "Unverified" not in tiny and "Ruled out" not in tiny and "2026-03-05" in tiny
    assert len(render_history([*confirmed, *guesses, *out])) <= findings.HISTORY_CHARS


def test_text_is_one_line_and_capped():
    assert findings.clean("a\nb\x00c " + "z" * 400) == ("a b c " + "z" * 400)[:300].rstrip()


# --- the SQL names the owner every time ------------------------------------------------------------------

async def test_every_statement_is_scoped_to_user_and_car(fake_pool):
    crud, uid, car = FindingCRUD(), "11111111-1111-1111-1111-111111111111", "toyota-echo-2001"
    fake_pool.rows = [{"id": "x"}]
    fake_pool.status = "DELETE 0"
    await crud.for_car(uid, car)
    await crud.history(uid, car, None)
    await crud.listing(uid, car, "active")
    await crud.get(uid, car, "22222222-2222-2222-2222-222222222222")
    await crud.edit(uid, car, "22222222-2222-2222-2222-222222222222", "t", "t", "q")
    await crud.retract(uid, car, "22222222-2222-2222-2222-222222222222")
    await crud.delete_one(uid, car, "22222222-2222-2222-2222-222222222222")
    await crud.delete_all(uid, car)
    await crud.apply(uid, car, None, [("insert", row("a", "ruled_out", "x")), ("promote", "a", "q"), ("supersede", "a", "b"), ("delete", "a")])
    assert len(fake_pool.calls) == 12
    for _, sql, args in fake_pool.calls:
        assert "user_id = $1" in sql or "user_id = $2::uuid" in sql or "VALUES ($1::uuid, $2::uuid" in sql, sql
        assert uid in args and car in args


@pytest.mark.parametrize("name", ["for_car", "history", "listing", "get"])
def test_no_read_takes_an_id_alone(name):
    import inspect

    params = list(inspect.signature(getattr(FindingCRUD, name)).parameters)
    assert params[1:3] == ["user_id", "car_id"]
