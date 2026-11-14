# Clarify-first gate for unclear audio and video turns: design

Date: 2026-11-14. Written at commit `83a3b99` on branch `feat/car-findings`. Status: **draft, nothing implemented.**
Decisions marked "Open" wait for the owner.

## Intent

A user sent a video of a door seam while complaining about a noise. The app diagnosed a "panel gap" from the frames,
the sound tagger said "engine knocking" at 0.25 on road noise, and the real fault was a rear wheel bearing. When the
evidence is unclear, or the media does not match the complaint, the app should ask what to look or listen for instead
of diagnosing the object in frame. One question, 2 to 4 tappable options, then it diagnoses.

The rule that matters most: **a clear text complaint with no media never gets a question.** Over-asking is a failure
worse than a wrong first diagnosis, so every limit below is built to under-fire.

## What the repo says (and where it differs from the brief)

1. **No clarification logic exists** (`grep -i clarif` finds only the findings spec). Confirmed.
2. **`MediaStore` is write-only.** It has `upload` and `delete_prefix`; no read. Originals are stored at upload
   (images as the re-encoded JPEG, video and audio as the original bytes), but **video frames are never stored**; they
   are extracted per turn and sent once. For a later turn the history carries only `Observations` (transcript, sound
   tags) through `prompts._attachment_note`, marked "(no longer shown)". So "reason over the earlier media without a
   re-upload" already holds for audio evidence and **does not hold for what the frames showed.** Originals are pruned
   after 90 days by `scripts/prune_media.py`.
3. **The sound tags are already framed as weak evidence**: `_render_observations` prints "(automatic, weak evidence)"
   with scores, and the system prompt says never to name a cause from sound alone. The door-seam failure was not tag
   presentation; it was the frames. The prompt has no rule that a noise complaint must not be answered by diagnosing
   a visible object.
4. **CLAP scores are a softmax over 29 labels** (27 after the two dropped ones), not calibrated confidences. A flat
   distribution scores about 0.03 per label, so 0.25 is roughly 8x chance yet still a minority of the probability mass.
   `SOUND_MIN_SCORE = 0.2` is a floor on what is stored, not a trust level. Top-k is 3.
5. **`DiagnosisState` is round-tripped through the state model** (`extract_state` receives `previous.model_dump()`)
   and `merge_state` builds a fresh object, so a new field is dropped unless `merge_state` carries it, and it becomes
   part of that model's input.
6. **Every turn runs three model calls**: the streamed reply, `extract_facts` (concurrent) and `extract_state`
   (after). Existing tests assert the turn ends `["state", "done"]` (`tests/unit/test_turn.py:17`).
7. **The audio PCM is already decoded** in `diagnosis._sounds` (`media.audio_pcm` at 48 kHz) for CLAP, so duration and
   loudness cost one numpy pass over data already in memory. Whisper's `vad_filter` gives the transcript.
8. **The eval harness cannot express this.** `scripts/run_cases.py` is single-turn, takes media from
   `tests/fixtures/e2e/` (not in the repo; media cases skip when missing), captures only `token`, `state` and `error`,
   and has no way to seed earlier messages, a pending clarification, or observations. It needs three additions (see
   Evals).
9. **`Message` ignores unknown fields on read and `_message(role, content, **extra)` already takes extras**, so a
   `clarify` field on the assistant message is additive; stored sessions stay valid.
10. **Language.** The reply must be in the user's language (Style rules). A fixed question bank would be English only.

## Design

### Where the decision is made

Deterministic checks (1a) and the weak-tag test (1b, first half) run in code before the model call and produce
**signals**, not a verdict. The semantic judgments (1b conflict, 1c, 1d) are made **inside the main model call**.
The model sees the signals and either answers or asks. This is the recommendation; the alternatives are costed below.

| | Latency | Tokens | Notes |
| --- | --- | --- | --- |
| **In the main call (recommended)** | none extra; first token waits on about 12 characters to see whether the reply starts with the sentinel | about +300 prompt tokens, and only on turns that carry media | frames are sent once; text-only turns get no gate text at all, so they cannot be over-asked |
| Separate cheap step before the main call | +1 to 3 s on every media turn (a vision call) | frames sent twice on every media turn that does not ask, plus about 300 | cleaner separation, testable alone; costs most where the gate does not fire |
| Separate step for (d) only, text only | +0.5 to 1 s | about 300, no images | (d) needs no frames; (c) still needs them, so this does not remove the main-call judgment |

### Signals (code, free)

Computed in `_look_and_listen` from the PCM already decoded, added to `Observations` as an optional `signal`
(`seconds`, `rms_dbfs`, `speech_only`):

- **very short**: audio under `GATE_MIN_SECONDS` (proposed 2.0, the same floor `hearing` uses for a tail window).
- **very quiet**: RMS under `GATE_MIN_RMS_DBFS` (proposed -50).
- **speech only**: transcript present and no tag survives `SOUND_MIN_SCORE`.
- **no audio track** on a video: already `has_audio=False`.
- **weak tags**: the top tag is below `GATE_STRONG_TAG` (proposed 0.5), or there are no tags.

Thresholds are config values with these proposals; **none is tuned on the one wrong 0.25 tag**, and nothing is
tuned until the gate rate has been logged (see Observability). The classifier floor stays at 0.2.

### "Weak evidence" presentation

Tags stay stored as today. The prompt line changes from a flat score list to a banded one, so the model cannot read
0.25 as a finding:

```
Sound tags (automatic; scores are shares among 27 labels, not confidence):
- strong (0.50 or more): none
- weak (0.20 to 0.50): engine knocking (0.25)
Signals: audio is 14 s and loud enough; no tag is strong.
```

The band edge is `GATE_STRONG_TAG`. Whether the tag conflicts with the user's words is left to the model (a keyword
map from the 27 labels to parts would be a second, English-only source of truth).

### The model's decision

When the turn carries media, the user block gets a "Clarify check" section and the system prompt gets a short rule.
The reply either starts normally, or **begins with a single line** `CLARIFY {json}`:

```
CLARIFY {"reason": "frames_unrelated_to_noise", "question_id": "when", "safety_note": "...", "options": null}
```

- The server buffers the first characters of the stream. No sentinel: tokens stream as today. Sentinel: the line is
  parsed, validated, rendered, and the rest of the model output is discarded.
- The model is told to ask only when: (b) all tags are weak or the top tag conflicts with what the user said; (c) the
  complaint is a sound and the visible object is unrelated; (d) media is attached and the text does not say what to
  look or listen for; or a signal says the audio is unusable. It is told never to ask when the text names a clear
  symptom and the media agrees, and never to diagnose a visible object for a noise complaint unless asked.
- **Provider note.** The model is `deepseek-flash` through the OpenAI client. A sentinel needs no provider support.
  Function calling would stream cleaner but I cannot verify the provider supports it without a live call.

### The question bank

`question_id` picks from a fixed bank in code (ids, English text, 2 to 4 options each, "Other/don't know" is always
the free-text path, not an option):

`moving` (was the car moving, and how fast), `when` (accelerating, braking, turning, over bumps), `side` (which corner
or side), `follows` (follows road speed or engine revs). The model chooses the one that best separates its top suspected
causes and may rephrase the text and labels in the user's language, keeping each option's `id` from the bank.
Rationale: bank ids make the stored data, tests and the gate-rate log stable; rephrasing handles language.

### Limits

Counted from the stored messages, so a failed or retried turn cannot drift a counter:

- **One clarifying turn per media item**: a user message with media can be followed by at most one assistant message
  with `clarify`. A second upload is a new media item.
- **At most `MAX_CLARIFICATIONS_PER_SESSION`** (config, default 2, 0 turns the gate off) assistant `clarify` messages
  per session, failed attempts not counted.
- **Skip proceeds.** When the pending clarification is answered with "don't know", a skip, or text the model cannot
  use, the prompt carries a "Clarification closed: do not ask again; diagnose and state what remains unknown" block.
  The gate text is removed from that turn entirely, so a second question is impossible, not just discouraged.
- The limits are enforced twice: the gate text is only sent while a question is still allowed, **and** a `CLARIFY`
  that arrives anyway is ignored (the buffered line is dropped and the model is re-asked to answer) if no question is
  allowed.

### Safety is never withheld

The `safety_note` is part of the payload and is rendered above the question. It is required when the user's text or an
open safety flag involves brakes, steering, ABS, overheating, smoke or fuel. Backstop: code checks the user text for
those words and the state's open flags; if either is present and the note is empty, a fixed English note for that
category is added, so a safety line is never missing. A CRITICAL open flag is always rendered first.

### Contract

`done` stays terminal; `clarify` replaces the model's reply, and `state` stays before `done`, as every turn does today
(the existing tests and clients rely on `[..., "state", "done"]`):

```
event: stage   {"name": "processing_media"} ... {"name": "searching"}
event: token   {"text": "..."}            the plain-text rendering, so an older client shows a normal message
event: clarify {"question": str, "options": [{"id": str, "label": str}], "reason": str}
event: state   {...}
event: done    {"message_id": "..."}
```

Plain-text rendering (also the assistant message's `content`):

```
<safety note, if any>

<question>
1. <label>
2. <label>
Or tell me in your own words.
```

The assistant `Message` gets `clarify: {question, options, reason, question_id} | None`. `reason` is a code (below),
not prose. The question and options are therefore stored twice (in `content` as text, in `clarify` as data); the text
is what later turns and `extract_facts` see.

### The pending clarification and the next turn

- Session state gets `pending_clarification: {message_id, media_message_id, question_id, question, options} | None`.
  `merge_state` carries it from the previous state (the state model never sees or edits it: it is stripped from the
  `extract_state` input and re-attached), set when a turn asks, cleared by the next turn.
- The user's answer is an ordinary user message (a tapped option sends its label as text).
- The next turn detects `pending_clarification`, adds a "Pending clarification" block (question, options, "the user's
  message is the answer; use it with the earlier media") and the closed-clarification rule above.
- **Earlier media.** Audio: the transcript and tags already stay in the history note. Video and photos: only a
  **description of what the frames showed** survives, and today nothing writes one. Proposed: the clarify payload gets
  `seen` (one sentence, at most 200 characters: what the frames show), stored on the user message's observations and
  rendered in `_attachment_note`. No re-extraction and no `MediaStore` read. Alternative in Open decisions.

### Interplay with findings

No write rule changes. What it touches, read from `diagnosis.py` and `findings.py`:

- `extract_facts(said, recent[-2:])` already receives the clarifying question: the context is the last two stored
  messages, and the assistant message's `content` carries the question and options as text. A test pins this.
- Every bank question asks about conditions of the symptom, not a test or a repair, so an answer yields no
  `test_done` or `repair_done`. If the extractor labels one anyway it is `answered_question` and the existing rules
  already cap it at those two kinds, never `confirmed_finding`. Nothing here can create a `user_confirmed` finding.
- **A clarify turn still runs `extract_state` and `record_turn`**, unchanged. Reason: the safety flags live in the state,
  and a turn that withheld them would break "safety is never withheld" for the next turn. Cost: one call, as today.
  Consequence: the state model may record the asking turn's context as `model_suggested` causes, as it does for any turn.
  Skipping `model_proposals` on a clarify turn would be a change in `record_turn`'s inputs; not made (Open decision 6).
- `said = text or transcript` is unchanged; a media-only first turn has no text, so `extract_facts` reads the transcript.

### Observability

One log line per media turn through the existing logger: `clarify gate: fired=<bool> reason=<code> signals=<codes>
asked=<n>/<cap>`. Reason codes: `audio_short`, `audio_quiet`, `speech_only`, `tags_weak`, `tag_conflicts_text`,
`frames_unrelated_to_noise`, `no_stated_focus`. Counts only; no question text, option text, transcript or user text.
The gate rate is `fired / media turns`, computed from the log. (A DB column is not added; see "left out".)

## Test plan

Offline, in the existing suite (`uv run pytest`):

- **Media checks**: short, quiet, speech-only, no-tag and silent-track cases from synthetic numpy arrays; boundaries at
  the thresholds; a loud clip with a strong tag raises no signal.
- **Limits**: one per media item (a second turn on the same media does not ask; a second upload can); the session cap
  and `0` disables; skip/"don't know" produces the closed block and no gate text; an ignored late `CLARIFY`.
- **Stream contract**: order `token`, `clarify`, `state`, `done`; `done` last; a normal turn has no `clarify`; a text-only
  turn's prompt contains no gate text at all; sentinel split across chunks; a sentinel with bad JSON falls back to a normal reply.
- **Prompt assembly**: gate text only with media and only while a question is allowed; banded tags; pending block;
  closed block; `seen` in the history note; the rules appear once.
- **Storage and replay**: the assistant message stores `clarify`; `GET /sessions/{id}` returns it and `content`
  holds the plain text; an old message without it loads; `pending_clarification` survives `merge_state` and is cleared by
  the answering turn; the state model's input does not contain it.
- **Findings**: `extract_facts` receives the question; no `user_confirmed` row from an option answer.
- **Safety**: the backstop note is added for ABS, brake, steering, overheat, smoke and fuel text and for an open flag.
- Existing tests that pin `["state", "done"]` keep passing.

### Evals (live, manual; `scripts/run_cases.py` additions)

Harness additions: (1) `messages:` seeds earlier turns and `pending_clarification:` seeds the state; (2) `observations:`
seeds a media turn's transcript, tags and signals so no real audio is needed; (3) `frames:` lists synthetic images drawn
with Pillow at run time (a labelled drawing of a door seam), never committed media; and the runner records `clarify` events.
Checks beyond substrings: `expects_clarify: true|false`, `must_not_name: [...]`, `max_questions`.

| Case | Seed | Expect |
| --- | --- | --- |
| `noise-unrelated-frames` | text "there is a noise when I drive", frames of a door seam, weak tag | clarify; reply does not name the seam or panel |
| `clear-text-no-media` | "grinding from the front wheels when I brake, pulls to one side" | **no clarify** (the most important case) |
| `clear-speech-in-media` | transcript clearly describing the problem, text empty | answers, no clarify |
| `abs-light-unclear-audio` | "ABS light is on", quiet audio, weak tags | clarify and carries the safety note |
| `answer-dont-know` | pending clarification, user says "I don't know" | answers, no second question, states what is unknown |
| `answer-with-speed` | pending clarification, user says "about 90 km/h" | answers and uses 90 km/h |

After implementation I run only these, once each, and report every reply in full.

## Decisions

Open (need the owner):

1. **Thresholds.** Proposed: `GATE_MIN_SECONDS` 2.0, `GATE_MIN_RMS_DBFS` -50, `GATE_STRONG_TAG` 0.5, classifier floor
   unchanged at 0.2. Accept, or give values? I have no real distribution to justify them; they are a starting point that
   the gate-rate log is meant to correct.
2. **Where (b conflict), (c), (d) are judged.** Recommended: in the main call (table above). Alternative: a separate vision step.
3. **Question wording.** Recommended: bank ids, model rephrases into the user's language. Alternative: templated English only
   (no model needed for the deterministic cases, so those turns skip the main call, the KB search and about two thirds of the
   cost, but non-English users get English).
4. **Earlier video frames.** Recommended: the model writes `seen` once (a sentence). Alternative: a `MediaStore.read`
   plus re-extracting frames on the answering turn (a vision payload again, and it breaks after the 90-day prune).
5. **Pending state and counters.** Recommended: `pending_clarification` in the state (as you asked), counters derived
   from messages. Alternative: derive pending from the last assistant message too and add nothing to `DiagnosisState`.
6. **Do clarify turns write model proposals to findings?** Recommended: yes, unchanged. Alternative: skip `model_proposals` on a
   clarify turn (one added argument to `record_turn`).
7. **`state` before `done` on a clarify turn.** Recommended: yes, as every turn today. You wrote "followed by `done`"; this still ends with `done`.
8. **Eval frames.** A drawing of a door seam is a weak stand-in for a video. Accept, or add one real-looking synthetic
   generator, or run (c) only as a prompt-assembly test?
9. **Safety backstop language.** The fixed note is English. Accept, or leave it out and rely on the model plus a test.

Taken by me (change any):

- Gate text exists only on turns with media; text-only turns are untouched.
- Skip is a closed block, not a softer instruction.
- Commit trailer: the system note asks for a `Co-Authored-By` line, your CLAUDE.md says never. Your CLAUDE.md wins; none added.
- No schema.sql change (messages and state are jsonb).

## Known limitations

- A model may still diagnose a visible object for a noise complaint; the prompt rule lowers it, only the eval measures it.
- The sentinel needs the model to follow a format; a malformed line falls back to a normal reply, never an error.
- Keyword safety backstop is English; open flags are in the user's language.
- A clip that is quiet because the phone was in a pocket is indistinguishable from a quiet fault.
- The gate rate comes from logs, so it has no history before this ships.

## Goals

- An unclear media turn gets one useful question and the safety note; the next turn diagnoses.
- A clear text complaint is never asked a question, and a turn with no media never sees gate text.
- Hard ceilings: one per media item, two per session, no loops.
- Old clients see an ordinary message; old sessions load unchanged.

## Non-goals

- Rhythm analysis of audio, the mobile UI for options, YouTube links, video downscaling.
- Changing findings write rules, the classifier, or its labels.

## Anti-goals

- Over-asking. A gate that fires on most media turns has failed even if every question is good.
- Logging content. Safety text withheld from a question turn.

## Considered and left out

- **A DB table for gate events.** Logs are enough for a rate; add if the case study needs per-session joins.
- **Tuning the 0.2 floor.** One data point.
- **A keyword map from sound labels to parts** to detect conflicts in code: a second English-only source of truth.

## Left to the builder

Module layout (a `clarify.py` service holding the bank, signals and parsing is likely), exact prompt wording, the
sentinel buffering code, and fixtures.
