import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

TITLE_MAX_CHARS = 60  # the state prompt asks for at most 60; enforced here because the model may not comply
DEFAULT_TITLE = "New diagnosis"  # the "Chatsession".title column default
MAX_FLAGS = 6  # safety flags kept and shown: critical first, then important, then info
# The state goes to the state model and comes back whole every turn, so lists that only accumulate would end in
# truncated JSON (and a state that silently stops updating) in a long session.
MAX_ESTABLISHED = 30  # findings, tests, ruled-out causes, topics: the newest are kept
MAX_CAUSES = 8  # suspected causes: the model lists them most likely first

Rank = Literal["most_likely", "likely", "possible"]
Severity = Literal["critical", "important", "info"]
_RANKS: tuple[Rank, ...] = ("most_likely", "likely", "possible")
_SEVERITIES: tuple[Severity, ...] = ("critical", "important", "info")
# Old sessions stored flags as bare strings. Those that say stop driving or involve a torque figure
# become critical; everything else important. ponytail: keyword guess, only ever applied to legacy strings.
_LEGACY_CRITICAL = re.compile(r"do not (keep )?driv|stop driving|not drive|torque", re.IGNORECASE)


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


class SuspectedCause(BaseModel):
    cause: str
    rank: Rank = "possible"
    evidence: str = ""
    next_check: str = ""


class SafetyFlag(BaseModel):
    text: str
    severity: Severity = "important"
    resolved: bool = False


class TestDone(BaseModel):
    __test__ = False  # not a pytest class
    test: str
    result: str = ""


class RuledOut(BaseModel):
    cause: str
    reason: str = ""


class DiagnosisState(BaseModel):
    suspected_causes: list[SuspectedCause] = Field(default_factory=list)
    safety_flags: list[SafetyFlag] = Field(default_factory=list)
    confirmed_findings: list[str] = Field(default_factory=list)
    tests_done: list[TestDone] = Field(default_factory=list)
    ruled_out: list[RuledOut] = Field(default_factory=list)
    topic_history: list[str] = Field(default_factory=list)
    current_focus: str = ""
    title: str = ""  # stable: set once, never replaced by a later topic

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy(cls, data):
        """Sessions saved before ranks and severities: float likelihoods become ranks by order, string flags
        become objects, and the old (overwritten) title becomes the first topic."""
        if not isinstance(data, dict):
            return data
        data = dict(data)
        causes = [dict(c) if isinstance(c, dict) else c for c in data.get("suspected_causes") or []]
        if any(isinstance(c, dict) and "likelihood" in c and "rank" not in c for c in causes):
            causes.sort(key=lambda c: -float(c.get("likelihood") or 0) if isinstance(c, dict) else 0)
            for i, c in enumerate(causes):
                if isinstance(c, dict):
                    c.pop("likelihood", None)
                    c.setdefault("rank", _RANKS[min(i, 2)])
        data["suspected_causes"] = causes
        data["safety_flags"] = [
            {"text": f, "severity": "critical" if _LEGACY_CRITICAL.search(f) else "important"} if isinstance(f, str) else f
            for f in data.get("safety_flags") or []
        ]
        if data.get("title") and not data.get("current_focus"):
            data["current_focus"] = data["title"]
            data.setdefault("topic_history", [data["title"]])
        return data

    @field_validator("title")
    @classmethod
    def _cap_title(cls, title: str) -> str:
        return title.strip()[:TITLE_MAX_CHARS].rstrip()

    @field_validator("confirmed_findings", "tests_done", "ruled_out", "topic_history")
    @classmethod
    def _keep_the_newest(cls, items: list) -> list:
        return items[-MAX_ESTABLISHED:]

    @field_validator("suspected_causes")
    @classmethod
    def _keep_the_top_causes(cls, causes: list) -> list:
        return causes[:MAX_CAUSES]

    @field_validator("safety_flags")
    @classmethod
    def _dedupe_and_cap_flags(cls, flags: list[SafetyFlag]) -> list[SafetyFlag]:
        """One flag per wording (the later one wins), open before resolved, most severe first, at most MAX_FLAGS."""
        by_text = {_norm(f.text): f for f in flags if f.text.strip()}
        ordered = sorted(by_text.values(), key=lambda f: (f.resolved, _SEVERITIES.index(f.severity)))
        return ordered[:MAX_FLAGS]


def _union(old: list, new: list, key) -> list:
    seen: dict = {key(x): x for x in old}
    for x in new:
        seen[key(x)] = x
    return list(seen.values())


def merge_state(previous: DiagnosisState, new: DiagnosisState, tests: list[TestDone] = ()) -> DiagnosisState:
    """What a turn adds to the session state. Causes and the focus follow the model's latest view; everything
    established (findings, tests, ruled-out causes, flags, the title) only accumulates."""
    ruled_out = _union(previous.ruled_out, new.ruled_out, lambda r: _norm(r.cause))
    gone = {_norm(r.cause) for r in ruled_out}
    focus = new.current_focus or previous.current_focus
    history = list(previous.topic_history)
    if focus and focus not in history:
        history.append(focus)
    return DiagnosisState(
        suspected_causes=[c for c in new.suspected_causes if _norm(c.cause) not in gone],
        safety_flags=_union(previous.safety_flags, new.safety_flags, lambda f: _norm(f.text)),
        confirmed_findings=_union(previous.confirmed_findings, new.confirmed_findings, _norm),
        tests_done=_union(previous.tests_done, [*new.tests_done, *tests], lambda t: _norm(t.test)),
        ruled_out=ruled_out,
        topic_history=history,
        current_focus=focus,
        title=previous.title or new.title,
    )


class Fact(BaseModel):
    """One thing the user reported: a test they ran and what happened, the part involved, or how a symptom changed."""

    test: str = ""
    result: str = ""
    part: str = ""
    symptom_change: str = ""
    finding: str = ""  # a conclusion the user states as established ("the bearing spins rough")
    repair: str = ""  # a repair or part replacement the user says they did ("replaced the rear-left hub")
    quote: str = ""  # the user's own words for it, copied from their message
    basis: str = ""  # "stated", "answered_question" or "agreed_diagnosis" (see FACTS_PROMPT)


class SoundTag(BaseModel):
    label: str
    score: float


class Observations(BaseModel):
    transcript: str = ""
    sounds: list[SoundTag] = Field(default_factory=list)
    facts: list[Fact] = Field(default_factory=list)


class MediaRef(BaseModel):
    kind: Literal["image", "audio", "video"]
    path: str


class Message(BaseModel):
    id: str
    role: Literal["user", "assistant"]
    content: str
    media: list[MediaRef] = Field(default_factory=list)
    observations: Observations | None = None
    created_at: str
    status: Literal["ok", "failed"] = "ok"  # "failed": an assistant attempt that produced no reply
    retry_of: str | None = None  # for a failed attempt, the id of the user message it answers
