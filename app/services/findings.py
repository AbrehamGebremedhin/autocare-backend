"""What an owner has established about their car across sessions: the write rules and the prompt block.

The rules are pure (plan_writes, render_history) so they are tested without a database; record_turn applies
them through FindingCRUD. The one invariant: `user_confirmed` is created only from a user's own words (a fact
the extraction model read from the user's message, whose quote is checked against that message) or by the
owner through the API. What the model proposes is `model_suggested`, whatever it says about itself.
"""
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

from app.CRUD.finding_crud import FindingCRUD
from app.schemas.session import DiagnosisState, Fact, _norm

USER = "user_confirmed"
MODEL = "model_suggested"
TEXT_CHARS = 300
MAX_MODEL_ROWS = 50  # active model_suggested rows per (user, car): the oldest go first (T6: nothing grows unbounded)
HISTORY_CHARS = 6000  # about 1,500 tokens at 4 characters each; the provider's tokenizer is not available here
APP_CONFIRMED = "(confirmed by the owner in the app)"
_USER_BASES = ("stated", "answered_question")  # Fact.basis values that count; "agreed_diagnosis" does not

crud = FindingCRUD()


def clean(text: str) -> str:
    """One line, at most TEXT_CHARS: stored text goes into a prompt, so it carries no line breaks."""
    return " ".join(text.replace("\x00", " ").split())[:TEXT_CHARS].rstrip()


@dataclass(frozen=True)
class Proposal:
    kind: str
    text: str
    source: str
    quote: str = ""
    key: str = ""  # what identifies the finding when the text carries more (a ruled-out cause's reason)

    @property
    def norm(self) -> str:
        return _norm(self.key or self.text)


def user_proposals(facts: Iterable[Fact], said: str) -> list[Proposal]:
    """Findings from what the user said this turn. A fact counts as the user's only when it is a statement, or a
    short answer to the assistant's direct question about an observation or repair (never agreement with the
    assistant's diagnosis), and its quote is really in the user's message. Anything else is kept as a suggestion."""
    said_norm = _norm(said)
    out = []
    for f in facts:
        quote = clean(f.quote)
        confirmed = f.basis in _USER_BASES and bool(_norm(quote)) and _norm(quote) in said_norm
        source = USER if confirmed else MODEL
        for kind, text, key in (
            ("test_done", clean(f"{f.test}: {f.result}" if f.result else f.test), f.test),
            ("confirmed_finding", clean(f.finding), ""),
            ("repair_done", clean(f.repair), ""),
        ):
            if text:
                out.append(Proposal(kind, text, source, quote if confirmed else "", key))
    return out


def model_proposals(state: DiagnosisState) -> list[Proposal]:
    """The state model's view as hypotheses. Its confirmed_findings stay hypotheses too: it cannot know the user said them."""
    out = [Proposal("suspected_cause", clean(c.cause), MODEL) for c in state.suspected_causes]
    out += [Proposal("ruled_out", clean(f"{r.cause} ({r.reason})" if r.reason else r.cause), MODEL, key=r.cause) for r in state.ruled_out]
    out += [Proposal("confirmed_finding", clean(t), MODEL) for t in state.confirmed_findings]
    return [p for p in out if p.text]


def plan_writes(existing: Sequence[Mapping], proposals: Iterable[Proposal], now: datetime | None = None) -> list[tuple]:
    """The writes for `proposals` against the owner's existing rows (all statuses), as ops for FindingCRUD.apply:
    ("insert", row), ("promote", id, quote), ("supersede", old_id, new_id), ("delete", id).

    - An identical active row is left alone. A user proposal promotes a matching suggestion; a model proposal
      never changes `source`, in either direction.
    - A model proposal for something the owner retracted, or that was superseded, is dropped.
    - A new ruled_out supersedes the suspected_cause it names; the owner's new finding or repair supersedes a
      ruled_out it contradicts. A model proposal can supersede only model_suggested rows.
    - More than MAX_MODEL_ROWS active suggestions: the oldest are deleted.
    """
    now = now or datetime.now(timezone.utc)
    rows = [dict(r) for r in existing]
    ops: list[tuple] = []
    for p in proposals:
        same = [r for r in rows if r["kind"] == p.kind and r["norm"] == p.norm]
        active = next((r for r in same if r["status"] == "active"), None)
        if active is not None:
            if p.source == USER and active["source"] != USER:
                active["source"] = USER
                ops.append(("promote", active["id"], p.quote))
            continue
        if p.source == MODEL and same:
            continue  # retracted by the owner, or superseded by newer evidence
        row = {
            "id": str(uuid4()), "kind": p.kind, "text": p.text, "norm": p.norm, "status": "active", "source": p.source,
            "evidence_quote": p.quote or None, "created_at": now,
        }
        rows.append(row)
        ops.append(("insert", row))
        if p.kind == "ruled_out":
            target = "suspected_cause"
        elif p.kind in ("confirmed_finding", "repair_done") and p.source == USER:
            target = "ruled_out"
        else:
            continue
        for old in rows:
            if old["kind"] == target and old["norm"] == p.norm and old["status"] == "active" and (p.source == USER or old["source"] == MODEL):
                old["status"] = "superseded"
                ops.append(("supersede", old["id"], row["id"]))
    suggestions = sorted(
        (r for r in rows if r["status"] == "active" and r["source"] == MODEL), key=lambda r: (r["created_at"], r["id"])
    )
    ops += [("delete", r["id"]) for r in suggestions[: max(0, len(suggestions) - MAX_MODEL_ROWS)]]
    return ops


async def record_turn(
    user_id: str, car_id: str, session_id: str, facts: Sequence[Fact], said: str, state: DiagnosisState
) -> None:
    """Store what a finished turn established. Raises on a database error: the caller must not let that fail the turn."""
    proposals = [*user_proposals(facts, said), *model_proposals(state)]
    ops = plan_writes(await crud.for_car(user_id, car_id), proposals)
    if ops:
        await crud.apply(user_id, car_id, session_id, ops)


async def car_history(user_id: str, car_id: str, session_id: str | None) -> list[dict]:
    return await crud.history(user_id, car_id, session_id)


# --- the prompt block -------------------------------------------------------------------------------

_HEADINGS = (
    "Confirmed by the owner:",
    "Unverified hypotheses (the model suggested these; nobody confirmed them):",
    "Ruled out earlier (may change with new evidence):",
)
_TAGS = {"repair_done": " (repair)", "test_done": " (test)"}


def _day(value) -> str:
    return value.date().isoformat() if isinstance(value, datetime) else str(value)[:10]


def _line(row: Mapping) -> str:
    return f"- {_day(row['created_at'])}{_TAGS.get(row['kind'], '')}: {row['text']}"


def _group(row: Mapping) -> int:
    if row["kind"] == "ruled_out":
        return 2
    return 0 if row["source"] == USER else 1


def render_history(rows: Iterable[Mapping], limit: int = HISTORY_CHARS) -> str:
    """The Car history block body, or "" when there is nothing. Confirmed items, then hypotheses, then ruled-out
    ones; newest first, ties by id, so the same rows always give the same text (prefix caching). Over `limit`
    characters the oldest hypotheses go first, then the oldest ruled-out items, then the oldest confirmed ones."""
    groups: list[list[Mapping]] = [[], [], []]
    for r in sorted(rows, key=lambda r: str(r["id"])):
        if r.get("status", "active") == "active":
            groups[_group(r)].append(r)
    for g in groups:
        g.sort(key=lambda r: r["created_at"], reverse=True)  # stable: equal dates stay in id order

    def render() -> str:
        return "\n".join(
            line for head, g in zip(_HEADINGS, groups, strict=True) if g for line in (head, *map(_line, g))
        )

    text = render()
    for g in (groups[1], groups[2], groups[0]):
        while len(text) > limit and g:
            g.pop()
            text = render()
    return text
