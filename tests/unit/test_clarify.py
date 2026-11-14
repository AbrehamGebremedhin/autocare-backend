"""The clarify gate's pure parts: signals, limits, the CLARIFY line, the safety backstop, prompt text and state.
Synthetic data only."""
import json

import numpy as np
import pytest

from app.core.config import get_settings
from app.schemas.session import (
    ClarifyOption, DiagnosisState, Observations, PendingClarification, SafetyFlag, Signal, SoundTag, merge_state,
)
from app.services import clarify, llm
from app.services.hearing import Tags
from app.services.prompts import build_messages, history_turn

S = get_settings()
CAR = {"id": "toyota-echo-2001", "make": "Toyota", "model": "Echo", "year": 2001}
LINE = {"reason": "frames_unrelated_to_noise", "question_id": "when", "safety_note": "", "seen": "a door seam"}


def _obs(seconds=10.0, rms=-30.0, speech_only=False, sounds=(("belt squeal", 0.7),), transcript=""):
    return Observations(
        transcript=transcript, sounds=[SoundTag(label=l, score=s) for l, s in sounds],
        signal=Signal(seconds=seconds, rms_dbfs=rms, speech_only=speech_only),
    )


# --- signals ---------------------------------------------------------------------------------------------------

def test_a_clear_loud_clip_with_a_strong_tag_raises_no_signal():
    assert clarify.signal_codes(_obs(), False, S) == []


@pytest.mark.parametrize("obs,silent,expected", [
    (_obs(seconds=1.5), False, ["audio_short"]),
    (_obs(rms=-60.0), False, ["audio_quiet"]),
    (_obs(speech_only=True, sounds=(), transcript="it rattles"), False, ["speech_only", "tags_weak"]),
    (_obs(sounds=(("engine knocking", 0.25),)), False, ["tags_weak"]),
    (None, True, ["no_audio_track"]),
])
def test_signal_codes(obs, silent, expected):
    assert clarify.signal_codes(obs, silent, S) == expected


def test_the_thresholds_are_exclusive_at_the_edge():
    assert clarify.signal_codes(_obs(seconds=S.GATE_MIN_SECONDS, rms=S.GATE_MIN_RMS_DBFS), False, S) == []
    assert clarify.signal_codes(_obs(sounds=(("belt squeal", S.GATE_STRONG_TAG),)), False, S) == []


def test_audio_signal_is_numbers_only_with_the_margin_of_the_top_two():
    pcm = np.full(96000, 0.1, dtype=np.float32)  # 2 s at 48 kHz, RMS 0.1 = -20 dBFS
    tags = Tags([SoundTag(label="engine knocking", score=0.25)], top_scores=[0.25, 0.21, 0.04])
    sig = clarify.audio_signal(pcm, 48000, tags, "it knocks")
    assert (sig.seconds, sig.rms_dbfs, sig.top_scores, sig.margin) == (2.0, -20.0, [0.25, 0.21, 0.04], 0.04)
    assert sig.speech_only is False  # a tag cleared the floor
    assert clarify.audio_signal(pcm, 48000, [], "it knocks").speech_only is True
    assert set(sig.model_dump()) == {"seconds", "rms_dbfs", "speech_only", "top_scores", "margin"}


def test_silence_and_an_empty_clip_are_the_floor_not_an_error():
    assert clarify.audio_signal(np.zeros(48000, dtype=np.float32), 48000, [], "").rms_dbfs == -120.0
    assert clarify.audio_signal(np.zeros(0, dtype=np.float32), 48000, [], "").seconds == 0.0


# --- limits ----------------------------------------------------------------------------------------------------

def _asked(n=1, failed=0):
    msgs = [{"role": "assistant", "content": "q", "clarify": {"question": "q"}} for _ in range(n)]
    return msgs + [{"role": "assistant", "content": "", "status": "failed", "clarify": {"question": "q"}}] * failed


def test_a_media_turn_may_ask_a_text_turn_never():
    assert clarify.gate_mode([], DiagnosisState(), True, 2) == "open"
    assert clarify.gate_mode([], DiagnosisState(), False, 2) == "off"


def test_the_session_cap_and_zero_turn_it_off():
    assert clarify.gate_mode(_asked(1), DiagnosisState(), True, 2) == "open"
    assert clarify.gate_mode(_asked(2), DiagnosisState(), True, 2) == "off"
    assert clarify.gate_mode([], DiagnosisState(), True, 0) == "off"


def test_failed_attempts_do_not_count():
    assert clarify.gate_mode(_asked(1, failed=3), DiagnosisState(), True, 2) == "open"


def _pending():
    return PendingClarification(message_id="a", media_message_id="u", question_id="when", question="When?",
                                options=[ClarifyOption(id="bumps", label="Over bumps"), ClarifyOption(id="braking", label="Braking")])


def test_a_pending_question_means_the_next_message_answers_it_even_with_new_media():
    state = DiagnosisState(pending_clarification=_pending())
    assert clarify.gate_mode(_asked(1), state, False, 2) == "answering"
    assert clarify.gate_mode(_asked(1), state, True, 2) == "answering"  # one question per media item: no second one yet


def test_a_new_upload_after_the_answer_is_a_new_media_item_and_may_ask_again():
    assert clarify.gate_mode(_asked(1), DiagnosisState(), True, 2) == "open"


# --- the CLARIFY line ------------------------------------------------------------------------------------------

def _parse(**over):
    return clarify.parse(" " + json.dumps({**LINE, **over}))


def test_a_valid_line_takes_the_bank_options_when_none_are_given():
    d = _parse()
    assert d.question == clarify.BANK["when"]["question"] and [o.id for o in d.options] == ["accelerating", "braking", "turning", "bumps"]
    assert (d.reason, d.question_id, d.seen) == ("frames_unrelated_to_noise", "when", "a door seam")


def test_only_the_labels_are_the_models_to_reword():
    d = _parse(question="Wann passiert es?", options=[{"id": "braking", "label": "Beim Bremsen"}, {"id": "bumps", "label": "Über Unebenheiten"}])
    assert d.question == "Wann passiert es?" and [(o.id, o.label) for o in d.options] == [("braking", "Beim Bremsen"), ("bumps", "Über Unebenheiten")]


@pytest.mark.parametrize("over,code", [
    ({"question_id": "colour"}, "unknown_question"),
    ({"question_id": None}, "unknown_question"),
    ({"options": [{"id": "braking", "label": "x"}, {"id": "flying", "label": "y"}]}, "unknown_option"),
    ({"options": [{"id": "braking", "label": "x"}, {"id": "braking", "label": "y"}]}, "unknown_option"),
    ({"options": [{"id": "braking", "label": "x"}]}, "bad_options"),
    ({"options": "braking"}, "bad_options"),
    ({"options": ["braking", "bumps"]}, "bad_options"),
])
def test_a_line_that_breaks_the_bank_is_rejected(over, code):
    with pytest.raises(clarify.ClarifyError) as e:
        _parse(**over)
    assert e.value.code == code


@pytest.mark.parametrize("rest,code", [(' {"reason": "x", "question_id": "wh', "bad_json"), (" no json here", "no_json"), (" [1, 2]", "no_json")])
def test_partial_or_missing_json_is_rejected_with_a_code(rest, code):
    with pytest.raises(clarify.ClarifyError) as e:
        clarify.parse(rest)
    assert e.value.code == code


def test_an_unknown_reason_is_other_and_long_text_is_capped_to_one_line():
    d = _parse(reason="because", safety_note="a\nb " + "x" * 900, seen="s" * 900)
    assert d.reason == "other" and "\n" not in d.safety_note and len(d.safety_note) <= clarify.NOTE_CHARS and len(d.seen) <= clarify.SEEN_CHARS


def test_a_long_safety_note_ends_at_a_sentence_never_mid_word():
    note = "Drive gently and avoid hard stops. " * 30 + "If the red brake light comes on, stop driving and have it checked."
    d = _parse(safety_note=note)
    assert len(d.safety_note) <= clarify.NOTE_CHARS and d.safety_note.endswith("stops.")
    one_long_sentence = "stop driving and have the brakes checked " * 40
    d = _parse(safety_note=one_long_sentence)
    assert len(d.safety_note) <= clarify.NOTE_CHARS + 1 and d.safety_note.endswith("…")
    assert d.safety_note.removesuffix("…").split()[-1] in one_long_sentence.split()  # a whole word, not a stump
    assert _parse(safety_note="Short note.").safety_note == "Short note."


async def _stream(*chunks):
    for c in chunks:
        yield c


async def _sniff(chunks, enabled=True):
    return [x async for x in clarify.sniff(_stream(*chunks), enabled)]


async def test_a_normal_reply_streams_through_unchanged():
    out = await _sniff(["Likely ", "the fan ", "relay."])
    assert "".join(v for k, v in out if k == "text") == "Likely the fan relay." and {k for k, _ in out} == {"text"}


async def test_a_sentinel_split_across_chunks_is_still_found():
    line = "CLARIFY " + json.dumps(LINE)
    out = await _sniff([line[:3], line[3:9], line[9:20], line[20:]])
    assert [k for k, _ in out] == ["clarify"] and out[0][1].question_id == "when"


async def test_text_after_the_json_is_dropped_and_leading_space_is_fine():
    out = await _sniff(["  CLARIFY ", json.dumps(LINE), "\nHere is a diagnosis anyway."])
    assert [k for k, _ in out] == ["clarify"]


async def test_a_word_that_only_starts_like_the_sentinel_is_a_normal_reply():
    out = await _sniff(["CLARIFYING the cause: ", "the fan."])
    assert "".join(v for k, v in out) == "CLARIFYING the cause: the fan." and {k for k, _ in out} == {"text"}


async def test_a_short_reply_is_not_lost():
    assert await _sniff(["OK"]) == [("text", "OK")]
    assert await _sniff(["CLAR"]) == [("text", "CLAR")]


async def test_a_bad_line_is_a_code_and_the_raw_line_never_comes_out_as_text():
    out = await _sniff(['CLARIFY {"reason": "x", "question_id": "wh'])
    assert [k for k, _ in out] == ["bad"] and out[0][1][0] == "bad_json" and "question_id" in out[0][1][1]
    out = await _sniff(["CLARIFY ", json.dumps({**LINE, "question_id": "colour"})])
    assert [k for k, _ in out] == ["bad"] and out[0][1][0] == "unknown_question"


def test_recover_gives_the_bank_question_when_the_line_still_names_one():
    d = clarify.recover('{"reason": "tags_weak", "question_id": "side", "options": [{"id": "nope"')  # cut off mid-way
    assert d.question_id == "side" and d.question == clarify.BANK["side"]["question"] and d.reason == "other"
    assert [o.id for o in d.options] == ["front_left", "front_right", "rear_left", "rear_right"]
    d = clarify.recover(json.dumps({**LINE, "options": "braking", "safety_note": "Stop driving."}))  # valid json, bad options
    assert d.question_id == "when" and d.safety_note == "Stop driving." and len(d.options) == 4


@pytest.mark.parametrize("line", [
    '{"question_id": "colour"}', '{"question_id": ["when"]}', '{"question_id": {"a": 1}}', '{"reason": "x"', " no json at all", "",
])
def test_recover_gives_nothing_when_no_bank_question_can_be_read(line):
    assert clarify.recover(line) is None


async def test_a_disabled_gate_never_reads_a_sentinel():
    out = await _sniff(["CLARIFY ", json.dumps(LINE)], enabled=False)
    assert {k for k, _ in out} == {"text"}


# --- the safety backstop ---------------------------------------------------------------------------------------

def test_the_models_note_is_used_as_written_and_the_backstop_stays_quiet():
    assert clarify.safety_note("Stop driving now.", "ABS light is on", DiagnosisState()) == "Stop driving now."


@pytest.mark.parametrize("text,word", [
    ("ABS light is on", "abs"), ("a noise when I brake", "brakes"), ("the steering wheel shakes", "steering"),
    ("it is overheating", "overheating"), ("there is smoke", "smoke"), ("I smell fuel", "fuel"),
])
def test_the_backstop_adds_a_note_for_each_safety_topic_when_the_model_left_it_out(text, word):
    note = clarify.safety_note("", text, DiagnosisState())
    assert note.startswith("Safety first") and word in note.lower()


def test_the_backstop_reads_open_flags_and_a_critical_flag_without_a_keyword():
    state = DiagnosisState(safety_flags=[SafetyFlag(text="Hub nut may be loose: do not drive", severity="critical")])
    assert clarify.safety_note("", "a noise", state) == "Hub nut may be loose: do not drive"
    resolved = DiagnosisState(safety_flags=[SafetyFlag(text="brake fluid leak", severity="critical", resolved=True)])
    assert clarify.safety_note("", "a noise", resolved) == ""


def test_no_safety_topic_no_note():
    assert clarify.safety_note("", "a rattle from the back", DiagnosisState()) == ""


def test_the_plain_text_rendering():
    opts = [ClarifyOption(id="braking", label="Braking"), ClarifyOption(id="bumps", label="Bumps")]
    assert clarify.render("When?", opts) == "When?\n1. Braking\n2. Bumps\n" + clarify.TAIL
    assert clarify.render("When?", opts, "Stop driving.").startswith("Stop driving.\n\nWhen?")


# --- prompt assembly -------------------------------------------------------------------------------------------

def _prompt(obs=None, gate=None, **kw):
    msgs = build_messages(car=CAR, state=kw.pop("state", DiagnosisState()), history=kw.pop("history", []), text="a noise",
                          observations=obs, passages=[], images=[], gate=gate)
    return msgs[-1]["content"][0]["text"]


def test_a_plain_turn_carries_no_gate_text_at_all():
    assert "Clarify check" not in _prompt() and "CLARIFY" not in _prompt() and _prompt() == _prompt(gate=None)


def test_the_gate_section_lists_signals_the_bank_and_the_never_diagnose_rule():
    text = _prompt(_obs(sounds=(("engine knocking", 0.25),)), gate=("Clarify check", clarify.gate_block(["tags_weak", "audio_short"])))
    block = text.split("## Clarify check")[1]
    assert "- tags_weak:" in block and "- audio_short:" in block
    assert "never diagnose a visible object unless the user asks" in block and "ONE clarifying question" in block
    for qid, bank in clarify.BANK.items():
        assert qid in block and all(o in block for o in bank["options"])
    assert text.index("## Reference passages") < text.index("## Clarify check") < text.index("## User message")


def test_the_tags_are_banded_so_a_quarter_is_not_read_as_a_finding():
    text = _prompt(_obs(sounds=(("engine knocking", 0.25), ("belt squeal", 0.61))))
    assert "not confidence" in text and "- strong (0.50 or more): belt squeal (0.61)" in text
    assert "- weak (0.20 to 0.50): engine knocking (0.25)" in text
    assert "- strong (0.50 or more): none" in _prompt(_obs(sounds=(("engine knocking", 0.25),)))


def test_the_answering_block_names_the_question_and_closes_it():
    text = _prompt(gate=("Pending clarification", clarify.answer_block(_pending())))
    block = text.split("## Pending clarification")[1]
    assert '"When?"' in block and "bumps: Over bumps" in block and "do not ask again" in block
    assert "CLARIFY" not in block and "Clarify check" not in text


def test_seen_is_labelled_as_an_earlier_model_description_not_a_fact():
    msg = {"role": "user", "content": "a noise", "media": [{"kind": "video", "path": "x"}],
           "observations": {"transcript": "", "sounds": [], "seen": "a door seam"}}
    note = history_turn(msg)["content"]
    assert "earlier model-written description of the frames, not a fact: a door seam" in note


# --- the state: carried, never changed by the state model -------------------------------------------------------

def test_merge_state_neither_drops_nor_alters_the_pending_question():
    pending = _pending()
    previous = DiagnosisState(pending_clarification=pending)
    other = PendingClarification(message_id="zz", media_message_id="zz", question_id="side", question="Which?")
    merged = merge_state(previous, DiagnosisState(pending_clarification=other, title="t"))  # the state model "returned" its own
    assert merged.pending_clarification == pending and merged.pending_clarification is previous.pending_clarification
    assert merge_state(DiagnosisState(), DiagnosisState(pending_clarification=other)).pending_clarification is None


async def test_the_state_model_never_sees_the_pending_question_and_cannot_set_one(mocker):
    from unittest.mock import AsyncMock, MagicMock
    from types import SimpleNamespace as NS

    client = MagicMock()
    returned = {"title": "x", "pending_clarification": {"message_id": "m", "media_message_id": "m", "question_id": "side", "question": "Which?"}}
    client.chat.completions.create = AsyncMock(return_value=NS(choices=[NS(message=NS(content=json.dumps(returned)))]))
    mocker.patch("app.services.llm.get_client", return_value=client)
    previous = DiagnosisState(pending_clarification=_pending())
    sent = await llm.extract_state([], previous)
    body = json.dumps(client.chat.completions.create.call_args.kwargs["messages"])
    assert "pending_clarification" not in body and "When?" not in body
    assert merge_state(previous, sent).pending_clarification == _pending()  # whatever the model returned, the turn's stands


def test_a_reworded_question_with_two_questions_falls_back_to_the_bank_wording():
    two = "When do you hear the noise? And does the ABS light come on at the same moment?"
    assert _parse(question=two).question == clarify.BANK["when"]["question"]
    assert _parse(question="When do you hear the noise?").question == "When do you hear the noise?"
    assert _parse(question="Wann hören Sie das Geräusch").question == "Wann hören Sie das Geräusch"  # no mark at all is fine


def test_the_gate_text_asks_for_one_single_question():
    assert "one question mark" in clarify.gate_block([]) and "never add a second thing" in clarify.gate_block([])


def test_the_door_seam_rule_is_in_the_system_prompt_for_every_turn_with_earlier_frames():
    from app.services.prompts import SYSTEM_PROMPT

    for phrase in ("never named as the problem", "do not treat the filmed object as the source of a noise complaint unless the user says it is",
                   "do not\n  rank it as a cause because it was in frame", "model-written is\n  not a fact"):
        assert phrase in SYSTEM_PROMPT, phrase
    media = SYSTEM_PROMPT.split("Media:")[1].split("Style:")[0]
    assert "filmed object" in media  # in the Media section, not only the answering block


def test_the_rule_travels_with_a_seen_label_and_the_answering_block_repeats_it():
    messages = build_messages(
        car=CAR, state=DiagnosisState(pending_clarification=_pending()),
        history=[{"id": "u", "role": "user", "content": "a noise", "media": [{"kind": "video", "path": "x"}],
                  "observations": {"transcript": "", "sounds": [], "seen": "a door seam"}, "created_at": "t"}],
        text="over bumps", observations=None, passages=[], images=[], gate=("Pending clarification", clarify.answer_block(_pending())),
    )
    assert "do not treat the filmed object as the source" in messages[0]["content"]
    assert "earlier model-written description of the frames, not a fact: a door seam" in messages[1]["content"]
    assert "rank it as a cause, unless the user said it is" in messages[-1]["content"][0]["text"].replace("\n", " ")


@pytest.mark.parametrize("over,code", [
    ({"question_id": ["when"]}, "unknown_question"),
    ({"question_id": {"id": "when"}}, "unknown_question"),
    ({"question_id": 7}, "unknown_question"),
    ({"options": [{"id": {"x": 1}, "label": "a"}, {"id": {"y": 2}, "label": "b"}]}, "unknown_option"),
    ({"options": [{"id": ["braking"], "label": "a"}, {"id": "bumps", "label": "b"}]}, "unknown_option"),
    ({"options": [{"id": 1, "label": "a"}, {"id": 2, "label": "b"}]}, "unknown_option"),
    ({"options": "braking, bumps"}, "bad_options"),
    ({"options": {"id": "braking"}}, "bad_options"),
])
def test_parse_never_raises_anything_but_a_clarify_error_for_odd_types(over, code):
    with pytest.raises(clarify.ClarifyError) as e:
        _parse(**over)
    assert e.value.code == code


@pytest.mark.parametrize("over", [{"question_id": ["when"]}, {"options": "braking, bumps"}, {"options": [{"id": {"x": 1}}, {"id": {"y": 2}}]}])
async def test_sniff_turns_odd_types_into_a_bad_line_not_an_exception(over):
    out = await _sniff(["CLARIFY ", json.dumps({**LINE, **over})])
    assert out[0][0] == "bad"


async def test_sniff_survives_a_parser_that_raises_something_unexpected(monkeypatch):
    def boom(rest):
        raise TypeError("unhashable type: 'list'")

    monkeypatch.setattr(clarify, "parse", boom)
    out = await _sniff(["CLARIFY ", json.dumps(LINE)])
    assert [k for k, _ in out] == ["bad"] and out[0][1][0] == "invalid"
