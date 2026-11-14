# Per-car findings: design

Date: 2026-11-14. Written at commit `8d0af59` on `main`; implemented on branch `feat/car-findings`, cut from
`chore/v1-hardening` (nothing is merged into `main`). Status: **implemented.** The owner's answers of
2026-11-14 are recorded under "Decisions" and in the sections they changed.

## Intent

Diagnostic state lives in `"Chatsession".state`, so a second session about the same car starts blind. A
user who confirmed a rear-left wheel bearing fault, then opens a new session about an intermittent ABS
light after fitting a new hub, gets no memory of the first. This adds a per-(user, car) findings store,
a "Car history" block in the prompt, routes to inspect and correct it, and a backfill from existing
sessions.

The central rule: **history is context, never ground truth.** Only the user can make a finding
`user_confirmed`; what the model proposes stays `model_suggested` and reaches the model as an unverified
hypothesis.

## What the repo says (and where it differs from the brief)

1. **The v1-hardening spec is not on `main`.** `docs/superpowers/specs/2026-11-09-v1-hardening-design.md`
   exists only on `chore/v1-hardening` (commit `6dfa38d`); I read it from there and followed its section
   layout. It declares "no schema change" and "no prompt changes" for that pass. Both are scoped to that
   pass, but this task does both, so expect a conflict in `schema.sql` and `prompts.py` when that branch
   merges. Its finding T6 (state grows without bound) is the same failure mode this store must avoid, hence
   the caps below.
2. **There is no clarification gate in the code** (`grep -i clarif` over `app/`, `scripts/`, `tests/`
   finds nothing). "Ask the user when the symptom contradicts the history" is therefore a prompt rule only.
3. **"Deleting a car" means removing it from the garage.** `"Car"` rows are shared catalogue entries
   (`<make>-<model>-<year>`), never deleted per user; `remove_car_from_user` only edits `user_profiles` and
   `user_car_details`. It also leaves the car's sessions in place (no FK from `"Chatsession".car_id`).
   Findings will be deleted at that point, as `user_car_details` already is.
4. **Known limitation: `car_id` identifies a model-year, not a physical vehicle.** A user cannot hold two
   cars with the same id (`array_append ... NOT ANY(cars)`), so `(user_id, car_id)` is effectively one
   vehicle. Two physical cars of the same model and year would share one history, and a car replaced by the
   same model keeps the old car's findings until the owner deletes them. Accepted by the owner; not solvable
   without a vehicle entity, which this change does not add.
5. **`confirmed_findings` in state has no provenance.** It is written by the state-extraction model
   (`STATE_PROMPT`), which sees the conversation including assistant text, so the code cannot tell
   "the user said it" from "the model concluded it". Only `Fact`s (from `extract_facts`) are derived from a
   user message. This drives the backfill decision below.
6. **The eval harness is single-turn, manual and not part of pytest.** `scripts/run_cases.py` calls the
   real DeepSeek client with an in-memory session and no findings store; it pre-marks substring checkboxes
   for a person to confirm. A history case needs a `history:` seed field and an injectable store. It cannot
   run in the offline suite; `tests/unit/test_run_cases.py` only validates the YAML's shape.
7. **A real session export was committed.** `tests/fixtures/platz_abs_session.json` was "a trimmed export of a
   real session (ids and user_id removed)", used by `tests/unit/test_session_replay.py`. It is replaced by
   `tests/fixtures/synthetic_abs_session.json` and removed from HEAD in its own commit; history is not
   rewritten, so the old blob stays reachable in earlier commits (and on any remote that has them).
8. **No tokenizer is available** for the DeepSeek model. The 1,500-token cap is enforced as 6,000
   characters (about 4 per token, an upper-ish estimate for English).
9. **`extract_facts` sees the two previous messages, assistant included**, so a user's "yes" to "did you
   spin the wheel?" can become a fact. That is wanted, but it means "user statement" is an LLM reading of the
   user's message, not a string check. See Decision 2.
10. **`schema.sql` does not migrate existing databases**, but a new `create table if not exists` is safe to
    re-run by hand: `psql -v ON_ERROR_STOP=1 -f app/db/sql/schema.sql "$DATABASE_URL"`. The docker initdb hook
    only runs on a fresh volume, so the production database needs that manual run before the new api starts.

## Design

### Schema (`app/db/sql/schema.sql`)

```sql
create table if not exists car_findings (
    id            uuid primary key default gen_random_uuid(),
    user_id       uuid not null references users (id) on delete cascade,
    car_id        text not null,
    kind          text not null check (kind in
                    ('confirmed_finding','suspected_cause','ruled_out','test_done','repair_done')),
    text          text not null check (length(text) between 1 and 300),
    norm          text not null,                       -- lower-cased, punctuation-free text, for de-duplication
    status        text not null default 'active' check (status in ('active','superseded','retracted')),
    source        text not null check (source in ('user_confirmed','model_suggested')),
    evidence_quote text,                               -- the owner's own words; required when user_confirmed
    session_id    uuid references "Chatsession" (id) on delete set null,   -- a deleted session keeps its findings
    created_at    timestamptz not null default now(),
    updated_at    timestamptz not null default now(),
    superseded_by uuid references car_findings (id) on delete set null,
    backfill_key  text unique,                          -- set only by the backfill script
    check (source = 'model_suggested' or evidence_quote is not null)
);
create index if not exists car_findings_owner_idx on car_findings (user_id, car_id, status);
create unique index if not exists car_findings_active_dedupe_idx
    on car_findings (user_id, car_id, kind, norm) where status = 'active';
```

All fields in the brief are present; `norm` and `backfill_key` are bookkeeping and `evidence_quote` is the
owner's addition. The exact DDL, with named constraints, is in `app/db/sql/schema.sql`. Every query filters on
`user_id` **and** `car_id` in SQL; there is no read path that takes only an id. Text is capped at 300
characters (longer input answers 422 on the routes and is truncated on the model path).

### Write rules (`app/services/findings.py`, new)

Two entry points, and the source is fixed by which one is called, never passed in by a caller:

- `user_proposals(facts, said)`: the only producer of `user_confirmed` besides the API. Called with the
  `Fact`s from `extract_facts`. `test`+`result` becomes `test_done`; `Fact.finding` ("a conclusion the user
  states as established") becomes `confirmed_finding`; `Fact.repair` becomes `repair_done`. The extractor also
  returns `quote` (the user's own words, copied from the latest message) and `basis`. A fact is
  `user_confirmed` only when `basis` is `stated`, or `answered_question` (a short answer such as "yes" to the
  assistant's direct question about a physical observation or a repair the user did; it can confirm only a
  `test_done` or `repair_done`, never a `confirmed_finding`, which needs `stated`), **and** its quote,
  normalised, is found in the user's message (checked in code, not trusted). `agreed_diagnosis` ("yes, that
  sounds right"), an empty or missing basis, or a quote that is not in the message leaves the finding
  `model_suggested` with no quote. The quote is stored as `evidence_quote`. The two-message context is kept,
  so a "yes" can be read against the question. The new `Fact` fields default to `""`; stored
  `Observations.facts` stay valid.
- `model_proposals(state)`: `source = model_suggested`, always. Maps
  `suspected_causes` to `suspected_cause`, `ruled_out` to `ruled_out`, `confirmed_findings` to
  `confirmed_finding` (still a hypothesis to the next session). `tests_done` is not mirrored (user facts
  already cover it).

Rules (pure function `plan_writes(existing_rows, proposals) -> actions`, so they are unit-testable without a
database; SQL only applies the plan):

1. A proposal whose `(kind, norm)` matches an active row does nothing to `source`. A model proposal never
   changes a `user_confirmed` row, and never promotes or demotes anything. (No `updated_at` refresh: a no-op
   writes nothing.) A model proposal for something the owner retracted, or that was superseded, is dropped,
   so a retracted hypothesis does not come back next turn.
2. A user proposal matching an active `model_suggested` row promotes it in place (`source=user_confirmed`,
   `session_id` of the confirming session). This is the only promotion path besides the API.
3. **Supersede, don't delete.** A `ruled_out` proposal (from either source) with the same `norm` as an active
   `suspected_cause` supersedes it (`status=superseded`, `superseded_by=<new id>`). A `user_confirmed`
   `confirmed_finding`/`repair_done` whose `norm` equals an active `ruled_out` supersedes that. A model
   proposal can supersede only `model_suggested` rows. The model contradicting a `user_confirmed` row leaves
   it alone; the prompt rule tells the next turn to ask the user.
4. At most 50 active `model_suggested` rows per `(user, car)`; inserting beyond that deletes the oldest
   one. (The only hard delete outside the user's own request.)

Writes happen in `_turn` after the state is merged and before the `state` event, inside `try/except` that
logs the exception class and carries on: a findings failure must never fail a turn.

### Retrieval and the "Car history" block (`prompts.py`)

Fetched once per turn with one indexed query: active rows for `(user_id, car_id)` **excluding the current
session** (the current session's material is already in "Established so far"; including it would feed the
model's own suggestions back to it as history). Rendered between `## Car` and `## Safety flags`:

```
## Car history (from earlier sessions; context, not ground truth)
Confirmed by the owner:
- 2026-11-02: rear-left wheel bearing worn, confirmed by wheel-off spin test
- 2026-11-05 (repair): new rear-left hub fitted
Unverified hypotheses (the model suggested these; nobody confirmed them):
- 2026-11-02: ABS tone ring damaged
Ruled out earlier (may change with new evidence):
- 2026-11-02: loose wheel nut (torque checked)
```

Order: group 1 `source=user_confirmed` of kinds `confirmed_finding`, `repair_done`, `test_done`; group 2 every
`model_suggested` row that is not `ruled_out`; group 3 all `ruled_out` (either source). Newest first within
a group by `created_at`, ties broken by `id`, so the same rows render the same text whatever
order the database returns them in, and nothing in the block depends on the current time (dates are the
rows' own days). With unchanged findings the block is byte-identical turn to turn, so the prompt prefix
stays cacheable. A row's `created_at` never moves (promotion changes `source`, `evidence_quote` and
`session_id` only). Omitted when there is nothing. Text is single-lined, and the block is declared data,
not instructions, like the reference passages.

Cap: 6,000 characters (about 1,500 tokens). Over budget, drop the oldest group-2 rows, then the oldest
group-3 rows, then (last resort) the oldest group-1 rows; the first line of each group is kept while the
group has any row left. Superseded and retracted rows are never sent.

### Prompt wording (`SYSTEM_PROMPT`)

Added to the "Each user turn gives you" list: "the Car history from earlier sessions". New section:

```
Car history:
- The Car history is context from earlier sessions, not ground truth for the current complaint, and it
  is not "Established so far". Only items under "Confirmed by the owner" were stated by the user; the rest
  are unverified hypotheses or earlier exclusions that may no longer hold.
- If the current symptom contradicts the history (for example a fault the history says was repaired is back,
  or a ruled-out cause now fits), do not assume which is right: say what differs and ask the user, in one
  question.
- Use the confirmed repairs when you reason: a new hub fitted after a bearing fault is relevant to an ABS
  sensor or tone-ring fault, and a part fitted recently is a suspect for a problem that started after it.
- Never present a hypothesis from the history as a fact about the car.
```

The rule "Established so far is settled: do not ask the user for anything listed there" stays and applies
only to that block.

### API (`app/api/v1/finding_route.py`, new; router under `/cars/{car_id}/findings`)

All require auth. They are scoped by the caller and the car id in every query and do not consult the garage:
what is stored about a car stays listable and deletable after the car is removed from the garage. A car that is
not theirs has no rows (an empty list; 404 for a finding id).

| Route | Effect |
| --- | --- |
| `GET /cars/{car_id}/findings?status=active\|superseded\|retracted` | All the caller's rows for the car (all statuses when no filter), newest first, with every field except `user_id`. |
| `PATCH /cars/{car_id}/findings/{id}` body `{text?, confirm?}` | The owner's own word: editing the text (stored as `evidence_quote`) or `confirm: true` (quote "(confirmed by the owner in the app)") sets `source=user_confirmed`. This is how a model suggestion, including a backfilled one, is promoted. A text equal to another active row of the same kind is 409. |
| `POST /cars/{car_id}/findings/{id}/retract` | `status=retracted`; kept in the list, never sent to the model. |
| `POST /cars/{car_id}/findings/{id}/resolve` body `{what_fixed_it}` | Creates a `repair_done`/`user_confirmed` row with that text and supersedes the original (`superseded_by` link). |
| `DELETE /cars/{car_id}/findings/{id}` and `DELETE /cars/{car_id}/findings` | Hard delete, one or all. This is "delete everything stored about my car". |

Cascade: `users` FK deletes a user's rows; `remove_car_from_user` deletes `(user, car)` rows in the same
place it deletes `user_car_details`. Deleting a session keeps its findings (`session_id` becomes null).
Not added: a create route (the chat is the creation path), pagination (capped by rule 4 plus the user's own
confirmed rows).

### Backfill (`scripts/backfill_findings.py`, new)

Reads every `"Chatsession"`, maps its `state` through the same `DiagnosisState` validator (legacy rows keep
working) and inserts with `backfill_key = sha1(session_id|kind|norm)` and `ON CONFLICT (backfill_key) DO
NOTHING`, so a second run, or a run after the user retracted or edited a row, adds nothing. Mapping:
`tests_done` -> `test_done`, `user_confirmed` only when a user message of that session carries the extracted
fact for that test (that whole message, capped at 300 characters, becomes the `evidence_quote`), else
`model_suggested`; `suspected_causes`, `ruled_out` and `confirmed_findings` -> `model_suggested`
(Decision 1). Only sessions whose car is still in the owner's garage are read. A row the owner retracted or
edited is left alone on a rerun; one the owner hard-deleted comes back, so retract instead of deleting before
a second run. `created_at` is the session's
`updated_at` (the true date of the finding is not stored). `--dry-run` prints counts. Sessions are only read.

### Test plan

Offline, in the existing suite (`uv run pytest`):

- Write rules: the model path cannot create or promote `user_confirmed`; a model proposal cannot touch a
  `user_confirmed` row; user proposal promotes a matching suggestion; supersede links `superseded_by`;
  a model contradiction of a confirmed row leaves it; the 50-row cap.
- Retrieval: group order, newest first, dates rendered, empty block omitted, current session excluded.
- Size cap: oversized input drops oldest `model_suggested` first, then ruled-out, then confirmed; result is
  within 6,000 characters.
- Prompt assembly (synthetic fixture): session 1's confirmed "rear-left wheel bearing worn, confirmed by
  wheel-off spin test" and repair "new rear-left hub fitted", plus one suspected cause; a new session's
  prompt contains both with dates, both above the cause; system prompt contains the new rules.
- Routes (fake store, like `test_session_routes.py`): list/edit/retract/resolve/delete, 404 for another
  user's finding and for a car not in the garage, `user_id` absent from the body.
- `remove_car_from_user` deletes the findings.
- Backfill: run twice, row count unchanged; legacy float-likelihood state maps; dry run writes nothing.
- `test_run_cases.py`: the new case passes the shape checks.
- Per-user isolation at the SQL level, the unique, check and cascade constraints, the write rules end to end
  and the backfill twice are covered by `tests/integration/test_findings_real.py` (`-m db`, against
  `TEST_DATABASE_URL`; it passed against a throwaway Postgres 16). The offline suite also checks, through a
  recording fake pool, that every `FindingCRUD` statement carries the owner and the car.

Live (manual, costs DeepSeek calls, not run by me unless asked): new case `abs-after-new-hub` in
`tests/cases/diagnosis_cases.yaml`: `history:` seeds the two synthetic findings, `text` is "ABS light comes
on over bumps after the new hub", `must_mention: ["rear-left", "bearing", "sensor"]`. "hub" is deliberately
not used: the complaint itself contains it, so it would pass without any memory. `run_cases.py` gets a
`MemoryFindings` stand-in and passes the seed through.

## Decisions (the owner's, 2026-11-14)

1. **Backfill of `confirmed_findings`: `model_suggested`.** The confirm route promotes a suggestion to
   `user_confirmed` (the PATCH route above).
2. **Only `extract_facts` creates `user_confirmed`**, keeping the two-message context, with the user's words
   stored as `evidence_quote`. A bare "yes" counts only when it answers a direct question about a physical
   observation or repair, never when it agrees with the assistant's diagnosis; then the finding stays
   `model_suggested`.
3. **History is fetched per turn**, excluding the current session, with a deterministic block order so prefix
   caching holds while nothing changes.
4. **Branch from `chore/v1-hardening`; nothing is merged into `main`.**
5. **`car_id` as make-model-year is accepted** (finding 4 above).
6. **One live run** of the new replay case `abs-after-new-hub`; no other eval.
7. **The real session fixture is replaced** by a synthetic one in a separate commit; history is not rewritten.
8. **`schema.sql` stays idempotent**; the deploy steps are below.
9. No `Co-Authored-By` line in commits (the user's CLAUDE.md).

## Deploy

Run from the repository root on the host, with the compose project of the README. The schema comes before the
new api: with the table missing, the history read and the findings write only log a warning and the turn goes
on, but `DELETE /cars/{id}` and the findings routes fail.

1. Back up: `bash scripts/backup.sh`
2. Add the table (idempotent; running it twice changes nothing):
   `docker compose exec -T db psql -U autocare -d autocare -v ON_ERROR_STOP=1 < app/db/sql/schema.sql`
3. Deploy the api: `docker compose up -d --build api`
4. Count what the backfill would add (reads only):
   `docker compose run --rm api uv run --no-sync python scripts/backfill_findings.py --dry-run`
5. Backfill: `docker compose run --rm api uv run --no-sync python scripts/backfill_findings.py`
   It prints `considered N findings, added M`. A second run prints `added 0`.
6. Check: `docker compose exec -T db psql -U autocare -d autocare -c "select source, kind, count(*) from car_findings group by 1, 2 order by 1, 2"`

Rollback: redeploy the previous image. `car_findings` is additive and unused by the old code, so it can stay;
`drop table car_findings` is a manual decision. Between steps 3 and 5 new sessions simply do not see old
history. Existing sessions are never written by the backfill.

## Known limitations

- `car_id` is a model-year, not a vehicle (finding 4).
- **Supersede and insert race (accepted).** `plan_writes` reads the owner's rows, then `apply` writes them in one
  transaction. Two turns for the same owner and car at the same moment (two sessions open at once) can both plan
  the same insert; the partial unique index skips the second insert, and its `supersede` op then points at a row
  that was not inserted, so the transaction fails on the foreign key and that turn's findings are not saved
  (logged, the turn itself succeeds). The next turn plans again against the committed rows. No lock was added.
- A turn on an old session of a car that has left the garage still reads and writes findings for it (the turn
  path looks the car up by id, not by garage). Proposed, not done: skip both when the car is not in the garage.
- A backfilled row the owner hard-deletes (rather than retracts) returns on a rerun.

## Goals

- A new session on the same car sees earlier confirmed findings and repairs, dated, above hypotheses.
- The model can never turn its own suggestion into a user-confirmed fact, by any code path.
- The user can see, correct and delete everything stored, and removing the car or account removes it.
- The block never exceeds about 1,500 tokens and never displaces the Reference passages.
- Old sessions keep working unchanged; the backfill is safe to run twice.

## Non-goals

- Media handling, the sound-tag classifier and any clarification gate (none exists).
- Changing `DiagnosisState`, `merge_state` or what the state model is asked to produce.
- Cross-user sharing, vehicle identity beyond `car_id`, semantic or embedding retrieval.

## Anti-goals

- A findings failure failing or slowing a turn; no new LLM call is added (the two new `Fact` fields reuse the
  existing extraction call).
- Model suggestions read as facts, anywhere: prompt, API response, or backfill.
- Real session data, emails or VINs in the repo.

## Considered and left out

- **Storing the history on the session at start.** Staleness in a long session outweighs the saving of one
  indexed query.
- **Detecting contradictions with an LLM call.** Costs a call per turn and would be the model deciding what
  supersedes what. Exact-text supersede plus the prompt's "ask the user" rule is enough for now.
- **A create route and an undo for delete.** Add when a client asks.
- **Real token counting.** Add if the 4-chars-per-token estimate proves off for the provider's tokenizer.

## Left to the builder

Module layout beyond the files named, SQL statement shapes, and the fake store used in route tests.
