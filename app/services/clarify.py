"""The clarify-first gate: when an unclear media turn gets one question instead of a diagnosis.

Code computes signals (free) and enforces the limits; the model judges the rest inside the main call and asks by
starting its reply with `CLARIFY {json}`. Everything that reaches the user from that line is validated here: the
question id and option ids come from the fixed BANK, only the wording is the model's. A line that does not validate
is never an error: it is shown as a normal reply and logged. See docs/superpowers/specs/2026-11-14-clarify-gate-design.md.
"""
import json
import math
import re
import string
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from app.core.config import Settings
from app.schemas.session import Clarify, ClarifyOption, DiagnosisState, Observations, PendingClarification, Signal
from app.services.hearing import N_TAG_LABELS

SENTINEL = "CLARIFY"
MAX_LINE_CHARS = 4000  # how much of a CLARIFY reply is read before it is judged
QUESTION_CHARS, LABEL_CHARS, NOTE_CHARS, SEEN_CHARS = 200, 80, 600, 200

# question_id -> the English default text and option ids/labels. Ids are fixed and validated; the model may rephrase
# the wording (and translate it) but not the ids.
BANK: dict[str, dict] = {
    "moving": {
        "question": "Was the car moving when this happened, and how fast?",
        "options": {"stationary": "Parked or stopped", "slow": "Slowly (under 30 km/h)", "town": "Around town (30 to 70 km/h)", "fast": "Faster (over 70 km/h)"},
    },
    "when": {
        "question": "When does it happen?",
        "options": {"accelerating": "When accelerating", "braking": "When braking", "turning": "When turning", "bumps": "Over bumps"},
    },
    "side": {
        "question": "Which corner or side does it seem to come from?",
        "options": {"front_left": "Front left", "front_right": "Front right", "rear_left": "Rear left", "rear_right": "Rear right"},
    },
    "follows": {
        "question": "Does it follow the car's speed or the engine's revs?",
        "options": {"road_speed": "Changes with road speed", "engine_revs": "Changes with engine revs", "steady": "Stays the same"},
    },
}
REASONS = (
    "audio_short", "audio_quiet", "speech_only", "no_audio_track", "tags_weak",
    "tag_conflicts_text", "frames_unrelated_to_noise", "no_stated_focus",
)
OTHER = "other"
WHITESPACE_OR_BRACE = string.whitespace + "{:"
TAIL = "Or tell me in your own words."

# The reply must carry the safety note when the user's text or an open flag involves one of these. The model writes
# it (in the user's language); these fixed English lines are the backstop for when it leaves it out.
_SAFETY = {
    "brakes": (re.compile(r"\b(brak\w*|abs)\b", re.I), "Safety first: if the brakes feel soft or pull to one side, or the ABS or brake warning light is on, do not rely on them. Stop driving and have them checked before you continue."),
    "steering": (re.compile(r"\bsteer\w*", re.I), "Safety first: if the steering feels loose, heavy or unusually noisy, stop driving and have it checked."),
    "overheating": (re.compile(r"\b(overheat\w*|over-heat\w*|temperature gauge)\b", re.I), "Safety first: if the engine is overheating or steaming, stop driving and let it cool with the bonnet closed. Open the radiator or pressure cap only with the engine cold."),
    "smoke": (re.compile(r"\b(smok\w*|burning smell|burnt)\b", re.I), "Safety first: if you see smoke or smell burning, stop driving and switch the engine off."),
    "fuel": (re.compile(r"\b(fuel|petrol|gasoline)\b", re.I), "Safety first: if you smell fuel or see a leak, do not drive, and keep sparks and flames away."),
}


class ClarifyError(ValueError):
    """A CLARIFY line that cannot be used; `code` is a fixed word, safe to log."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class Decision:
    question: str
    options: list[ClarifyOption]
    reason: str
    question_id: str
    safety_note: str = ""
    seen: str = ""


# --- signals and limits (code, free) ----------------------------------------------------------------------------


def signal_codes(obs: Observations | None, has_silent_video: bool, s: Settings) -> list[str]:
    """What the clip's numbers say, as codes. Hints for the model, never a verdict."""
    codes = []
    if has_silent_video:
        codes.append("no_audio_track")
    sig = obs.signal if obs else None
    if sig is not None:
        if sig.seconds < s.GATE_MIN_SECONDS:
            codes.append("audio_short")
        if sig.rms_dbfs < s.GATE_MIN_RMS_DBFS:
            codes.append("audio_quiet")
        if sig.speech_only:
            codes.append("speech_only")
    if obs is not None and (not obs.sounds or obs.sounds[0].score < s.GATE_STRONG_TAG):
        codes.append("tags_weak")
    return codes


Mode = Literal["off", "open", "answering"]


def asked(history: Sequence[Mapping]) -> int:
    return sum(1 for m in history if m.get("clarify") and m.get("status") != "failed")


def gate_mode(history: Sequence[Mapping], previous: DiagnosisState, has_media: bool, cap: int) -> Mode:
    """"answering": the user's message answers a pending question (no new question this turn). "open": a question
    may be asked. "off": plain turn. One question per media item follows from the first rule: an asked question
    stays pending until the next message, whatever it holds, and a text-only turn is never "open"."""
    if previous.pending_clarification is not None:
        return "answering"
    return "open" if has_media and cap > 0 and asked(history) < cap else "off"


# --- prompt text -------------------------------------------------------------------------------------------------

_HINTS = {
    "audio_short": "the audio is too short to judge",
    "audio_quiet": "the audio is too quiet to judge",
    "no_audio_track": "the video has no audio track",
    "speech_only": "the clip has speech and no vehicle sound (if the speech clearly describes the problem, answer from it)",
    "tags_weak": "no sound tag is strong",
}


def gate_block(codes: Sequence[str]) -> str:
    signals = "\n".join(f"- {c}: {_HINTS[c]}" for c in codes) or "- none"
    bank = "; ".join(f"{q}: " + ", ".join(b["options"]) for q, b in BANK.items())
    return f"""Signals from the media (hints, not a verdict):
{signals}
Decide before replying whether to ask ONE clarifying question instead of diagnosing. Ask only when:
- all sound tags are weak, or the top tag conflicts with what the user said;
- the complaint is a sound (noise, squeal, knock, rattle, hum) and the photos or frames show something unrelated to it;
- media is attached but the user's text does not say what to look or listen for;
- or a signal above means the audio cannot be judged and the complaint is a sound.
Never ask when the user's text names a clear symptom and the media agrees with it, or when speech in the clip describes
the problem clearly. When the complaint is a noise, never diagnose a visible object unless the user asks you to look at it.
To ask, begin your reply with exactly CLARIFY and then one JSON object on one line, and write nothing after it:
CLARIFY {{"reason": "<code>", "question_id": "<id>", "question": "<optional rewording>", "options": [{{"id": "<option id>", "label": "<label>"}}], "safety_note": "<text>", "seen": "<text>"}}
- reason: one of {", ".join(REASONS)}.
- question_id and option ids come from this list; choose the question that best separates your top suspected causes, and 2 to 4 of its options: {bank}.
- question and labels: reword in the language of the User message if you like; never change an id. The question
  asks only what the bank question asks, as one question with one question mark: never add a second thing to it
  (for example a warning light, or a second symptom).
- safety_note: required when the user's text or an open safety flag involves brakes, steering, ABS, overheating, smoke or fuel: say in the user's language what to do now. Otherwise "".
- seen: one sentence on what the photos or frames show, or "" if none.
If you do not ask, reply as normal and do not mention this check."""


def answer_block(p: PendingClarification) -> str:
    options = "; ".join(f"{o.id}: {o.label}" for o in p.options)
    return f"""You asked the user a clarifying question about the media they sent earlier: "{p.question}" (options: {options}).
The User message below is their answer. Use it together with the earlier media, but do not treat an object the frames
showed as the noise source, or rank it as a cause, unless the user said it is. If it says they do not know, skips,
or does not answer, do not ask again: diagnose with what you have and say what remains unknown. Do not ask a clarifying question now."""


# --- the CLARIFY line ---------------------------------------------------------------------------------------------


def _line(value, limit: int) -> str:
    """One line of at most `limit` characters. Over the limit it ends at the last full sentence (else the last word, with
    an ellipsis), so a safety note is never cut mid-word."""
    if not isinstance(value, str):
        return ""
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(mark) for mark in (". ", "! ", "? "))
    if end >= limit // 2:
        return cut[: end + 1]
    return cut[: cut.rfind(" ")].rstrip(" ,;:") + "…" if " " in cut else cut


def parse(rest: str) -> Decision:
    """Validate what follows CLARIFY. Raises ClarifyError(code) for anything unusable."""
    start = rest.find("{")
    if start < 0:
        raise ClarifyError("no_json")
    try:
        obj, _ = json.JSONDecoder().raw_decode(rest[start:])
    except ValueError:
        raise ClarifyError("bad_json") from None
    bank = BANK.get(obj.get("question_id"))
    if bank is None:
        raise ClarifyError("unknown_question")
    raw = obj.get("options")
    if raw in (None, []):
        options = [ClarifyOption(id=i, label=label) for i, label in bank["options"].items()]
    else:
        if not isinstance(raw, list) or not 2 <= len(raw) <= 4 or not all(isinstance(o, dict) for o in raw):
            raise ClarifyError("bad_options")
        ids = [o.get("id") for o in raw]
        if len(set(ids)) != len(ids) or any(i not in bank["options"] for i in ids):
            raise ClarifyError("unknown_option")
        options = [ClarifyOption(id=o["id"], label=_line(o.get("label"), LABEL_CHARS) or bank["options"][o["id"]]) for o in raw]
    if len(options) < 2:
        raise ClarifyError("bad_options")
    reason = obj.get("reason") if obj.get("reason") in REASONS else OTHER
    question = _line(obj.get("question"), QUESTION_CHARS)
    if not question or sum(question.count(mark) for mark in "?？؟") > 1:  # none given, or two in one: the bank's wording
        question = bank["question"]
    return Decision(
        question=question,
        options=options,
        reason=reason,
        question_id=obj["question_id"],
        safety_note=_line(obj.get("safety_note"), NOTE_CHARS),
        seen=_line(obj.get("seen"), SEEN_CHARS),
    )


async def sniff(stream: AsyncIterator[str], enabled: bool) -> AsyncIterator[tuple[str, object]]:
    """The reply stream as ("text", str), or when it starts with CLARIFY: ("clarify", Decision), or on a line that does
    not validate ("bad", code) followed by the model's own text as ("text", ...). Never raises for a bad line."""
    if not enabled:
        async for token in stream:
            yield "text", token
        return
    buf = ""
    mode = None  # None until the first characters decide it, then "text" (pass through) or "clarify" (collect)
    async for token in stream:
        if mode == "text":
            yield "text", token
            continue
        buf += token
        if mode is None:
            head = buf.lstrip()
            if len(head) <= len(SENTINEL) and SENTINEL.startswith(head):
                continue  # too early to tell: the sentinel needs one more character after it
            if head.startswith(SENTINEL) and head[len(SENTINEL)] in WHITESPACE_OR_BRACE:
                mode, buf = "clarify", head
            else:
                mode = "text"
                yield "text", buf
                buf = ""
                continue
        if len(buf) > MAX_LINE_CHARS:
            break
    if mode == "clarify":
        rest = buf[len(SENTINEL):]
        try:
            yield "clarify", parse(rest)
        except ClarifyError as exc:
            yield "bad", exc.code
            yield "text", rest.lstrip(" :\t\r\n")
    elif mode is None and buf:
        yield "text", buf  # a reply shorter than the sentinel


# --- the reply the user sees ---------------------------------------------------------------------------------------


def safety_note(model_note: str, text: str | None, state: DiagnosisState) -> str:
    """The model's note when it wrote one. Otherwise, if the user's text or an open flag involves a safety topic (or
    an open flag is CRITICAL), a fixed English line, so a question never goes out without it."""
    if model_note:
        return model_note
    open_flags = [f for f in state.safety_flags if not f.resolved]
    haystack = " ".join([text or "", *(f.text for f in open_flags)])
    notes = [note for pattern, note in _SAFETY.values() if pattern.search(haystack)]
    if not notes:
        critical = [f.text for f in open_flags if f.severity == "critical"]
        return " ".join(critical)
    return notes[0]


def render(question: str, options: Sequence[ClarifyOption], note: str = "") -> str:
    lines = [f"{i}. {o.label}" for i, o in enumerate(options, 1)]
    body = "\n".join([question, *lines, TAIL])
    return f"{note}\n\n{body}" if note else body


def stored(d: Decision) -> Clarify:
    return Clarify(question=d.question, options=d.options, reason=d.reason, question_id=d.question_id)


def band_note() -> str:
    return f"scores are shares among {N_TAG_LABELS} labels, not confidence"


def audio_signal(pcm: np.ndarray, sr: int, tags: Sequence, transcript: str) -> Signal:
    """Numbers about a clip's audio for the gate and its log: length, loudness, how flat the tag scores were."""
    rms = float(np.sqrt(np.mean(np.square(pcm, dtype=np.float64)))) if pcm.size else 0.0
    top = [round(float(x), 4) for x in (getattr(tags, "top_scores", None) or [t.score for t in tags[:3]])]
    return Signal(
        seconds=round(pcm.size / sr, 2),
        rms_dbfs=round(max(-120.0, 20 * math.log10(rms)), 1) if rms > 0 else -120.0,
        speech_only=bool(transcript.strip()) and not tags,
        top_scores=top,
        margin=round(top[0] - top[1], 4) if len(top) > 1 else None,
    )
