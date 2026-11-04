# AutoCare Backend: Refactor, Media Diagnosis, Finish

Date: 2026-11-04 · Status: draft, awaiting approval

## Intent

AutoCare is a chat-based car diagnosis backend. Its diagnoses were good, but one reply took up to **402 s**
(measured from a `/chat/send` call). The codebase also holds two parallel chat stacks and a lot of
"enterprise" scaffolding that nothing needs. The work has three parts:

1. Make diagnosis fast. The first words should stream back within seconds.
2. Let users diagnose from **photos, audio and video**, not only text.
3. Strip the codebase back to what's used, close the security holes, and ship it as a stack that
   `docker compose up` brings up.

There is **no client yet**, so the API can be redesigned freely.

## Why it's slow today (what the rebuild removes)

One "full diagnostic" turn runs these steps in sequence, with no streaming:

| Step | Where | LLM calls |
|---|---|---|
| Extract original issue, or check the message is on-topic | `ChatService.send_message` → `SessionContextAgent` | 1 |
| Classify simple vs full | `OrchestratorAgent.classify_message` | 1 |
| Symptom extraction (+ tree placement when the tree has >5 nodes) | `SymptomExtractorAgent`, `TreeManagerAgent` | 1–2 |
| Diagnosis: a ~60-field JSON object with torque specs, shop costs and so on | `DiagnosisAgent.diagnose` | 1 (very large output) |
| Rewrite that JSON into a chat reply | `UserInteractionAgent.process` | 1 |
| Session title | `generate_simple_session_title` / tree title | 0–1 |

Each call is a blocking `llm.invoke` pushed onto a thread executor. The knowledge-base search also runs
twice in parallel with the same query.

## Decisions (made by the user)

- **The LLM is DeepSeek.** It handles all reasoning and also reads images (`deepseek-flash` accepts
  JPEG/PNG/GIF/WebP in OpenAI-compatible `image_url` blocks).
  [Docs](https://api-docs.deepseek.com/guides/vision). `deepseek-chat`, used in config today, is no longer
  listed in DeepSeek's current docs.
- **The backend "hears" with local models.** Whisper transcribes speech, and CLAP tags sounds zero-shot.
  Both turn into text that DeepSeek reads. No DeepSeek or Claude model accepts audio input.
- **Video frames are read by `deepseek-flash` vision.**
- **Refactor depth: rebuild the core.** One pipeline with one main LLM call per turn, streamed. Merge
  the two chat stacks. Delete dead code and unused dependencies. Fix auth.
- **Knowledge search uses Postgres full-text search in Supabase.** Milvus (with etcd and minio) and
  Ollama embeddings are dropped. The old vector data is lost, so the index is rebuilt from the PDFs anyway.
- **Capture is record-then-upload.** No live streaming.
- **Done means:** `docker compose up` brings the stack up, the tests pass, the API is documented, and a
  scripted end-to-end check runs text, photo, audio and video diagnoses against real DeepSeek.

## Goals

- **One request path for a chat turn.** Load the session, process any media, search the knowledge
  base, make **one streaming DeepSeek call**, then save.
- **Streamed replies.** The client sees progress events and reply text as it's generated.
- **Media in a message:**
  - Up to 4 photos.
  - Or one audio clip of 60 s or less.
  - Or one video of 60 s or less. Its audio track and sampled frames are both used.
  - Text is optional alongside any of these.
- **Every route is authenticated.** The user identity comes from the verified Supabase JWT, never from
  the request body or query string.
- **Smaller codebase.** Small enough that one person can read the whole request path in under an hour.
- **Latency targets (my call, to be measured, not promised):**
  - First streamed text within ~10 s for a text-only message.
  - Media preprocessing within ~30 s for a 60 s video on a 4-CPU box with no GPU.

## Non-goals

- Live audio or video streaming.
- The "10,000 concurrent users" machinery: agent pool, distributed session manager, DB connection pool,
  horizontal-scaling flags.
- A frontend.
- Training or fine-tuning a car-sound model.
- Semantic (vector) search. Keyword search is the starting point.

## Anti-goals (failure even if it "works")

- **Diagnosis quality drops below today's**, as measured by the case set under *Tests* below.
  Today's two saved outputs were reviewed on 2026-11-04. They are safe and well ordered, but generic:
  - **Case 1** (`result.json`):
    - Names "faulty thermostat" as the main cause of overheating with on/off fan bursts, on weak
      reasoning.
    - Never mentions the coolant temperature sensor or fan relay.
    - Contains nothing specific to the Echo, and has a blank where the coolant type should be.
    - Twice sends the user to the owner's manual, against its own prompt.
  - **Case 2** (screenshot):
    - Leaks the model preamble "Okay, here's a message for you…".
    - Treats the unpressurized reservoir cap as the pressure cap.

  The new pipeline has to do better than this, not just match it.
- **The system overstates what it heard.** CLAP tags are approximate. The prompt must treat them as
  weak evidence ("sounds like a possible belt squeal"), and the reply must not name a cause from sound
  alone with high confidence.
- **Latency gets worse, or a turn makes more than 2 LLM calls.**

## Constraints

- Python 3.12 and FastAPI. Supabase for auth, Postgres and storage.
- **uv manages packages** (user's decision). `uv` 0.8.18 is installed locally.
  - **Dependencies:** they move into `[project]` / `[dependency-groups]` in `pyproject.toml`, and
    `uv.lock` is committed. `requirements.txt` is deleted. Pytest and other dev tools go in a `dev`
    group.
  - **CPU torch:** pinned through a `[[tool.uv.index]]` entry for the PyTorch CPU wheel index, with
    `explicit = true` and `[tool.uv.sources]` mapping `torch` to it. This keeps the multi-GB CUDA
    wheels out of the image.
  - **Commands:** everything is run with uv, e.g. `uv sync`, `uv run pytest`, `uv run uvicorn main:app`
    and `uv run scripts/e2e.py`. The README documents only these.
  - **Docker:** copies the `uv` binary from the official `ghcr.io/astral-sh/uv` image. It runs
    `uv sync --frozen --no-dev` in a cached layer before copying the app code. No pip.
- The existing Supabase project and tables: `Car`, `Chatsession`, `Groundknowledge`,
  `Diagnosis_history` and `user_profiles`.
- **Schema changes need explicit sign-off at the moment they're applied**, per the user's standing rule
  on DB migrations. Approving this design doesn't count as running them.
- CPU-only server by default. Whisper and CLAP must run acceptably on CPU.
- Secrets live in `.env`, which is never read or committed.
- **Existing Supabase data is test data and can be wiped** (user's decision). Old chat sessions, the
  `Groundknowledge` rows, and test users and cars don't need migrating. Tables are dropped or recreated
  instead, still with sign-off at the moment it happens.
- **The knowledge base is rebuilt from `car_data/` only** (user's decision): 7 repair books plus the
  Toyota Echo manual, and manuals downloaded per car as cars are added. The old "38,936 documents"
  claim is removed from the prompt.
- **Audience stays DIY owners**, as in the current diagnosis prompt: explain what to check and why, and
  flag when a shop is genuinely needed.

## Design

### Request flow (one chat turn)

```
POST /api/v1/sessions/{id}/messages   (multipart: text?, files[]?)  →  text/event-stream
  1. auth: verify JWT → user_id; load session (must belong to user_id)
  2. media (if any), in parallel:
       video  → ffmpeg: 16 kHz mono wav + up to 8 evenly spaced frames (JPEG)
       audio  → Whisper transcript + CLAP top-3 sound tags with scores
       frames/photos → passed straight to step 4 as image blocks
     emit SSE `stage` events as each finishes
  3. knowledge search: Postgres FTS over kb_passages
       (this car's manual + general repair books), top ~8 passages
  4. ONE streaming DeepSeek call (deepseek-flash):
       system prompt + car info + suspected-causes state + last N messages
       + observations (transcript, sound tags) + passages + images
     emit SSE `token` events
  5. after the reply finishes: one small non-streaming DeepSeek call (JSON output) that updates
     the session's suspected-causes list and title. Its result is sent as a final SSE `state`
     event, then saved.
```

- **Step 5 is the "cheap second call".** It happens after the user already has the reply, so it adds no
  waiting time.

**Prompt rules that fix the faults seen in the review:**
- **Grounding.**
  - When the reply uses a fact from a passage (a spec, fluid type or procedure), it cites the passage,
    e.g. "(Echo manual, p. 212)".
  - When no passage covers the fact, the reply says so plainly ("I couldn't find the Echo's coolant
    spec"). It never leaves a blank and never sends the user to "check your manual".
- **Reasoning.**
  - The top suspected cause must say which reported symptom it explains and how.
  - Causes that fit the symptoms better, such as sensor or relay faults for an on/off fan, can't be
    left out just because they're less common.
- **No preamble.** The reply speaks to the user directly; there's no "here's a message for you".
- **These removed steps have no replacement call:**
  - Classify: the model answers simple questions directly.
  - Off-topic gate: becomes one prompt instruction.
  - Symptom-extraction and tree-manager agents.
  - The user-interaction rewrite.
  - The session-context agent.

### Diagnosis state (replaces the diagnosis tree) — *my call*

The nested `DiagnosisTreeNode` tree becomes a flat list stored on the session:
`suspected_causes: [{cause, likelihood (0–1), evidence, next_check}]`, plus `safety_flags` and `title`.

- **Why:** the tree's hierarchy was assembled with keyword heuristics, and the diagnosis prompt flattened
  it back into a ranked list anyway.
- The existing column on `Chatsession` is reused or renamed during the schema step.

### Transport: SSE, no WebSocket — *my call*

- **Why:** One transport only, and it works through proxies and on mobile without any connection
  management. SSE covers the progress stages the WebSocket was used for.
- **What goes:** the four WebSocket modules (`websocket.py`, `concurrent_websocket.py`,
  `orchestrator_websocket.py`, `listen_socket.py`) and the `/ws` endpoint.
- **Event types:** `stage`, `token`, `state`, `error` and `done`.

### LLM client — *my call*

- Use the `openai` SDK (already installed) as `AsyncOpenAI(base_url="https://api.deepseek.com")`,
  with streaming.
- This drops LangChain, `langchain-deepseek` and the blocking-executor wrapper.
- Model name and base URL come from settings: `DEEPSEEK_MODEL=deepseek-flash`.
- **Builder to check:** DeepSeek's docs say thinking mode is on by default. Find out whether it can be
  turned off or reduced per request, measure first-token latency both ways, and pick what meets the
  target without hurting quality.

### Media processing

- **ffmpeg/ffprobe** is a system package in the Docker image.
  - Validate the duration, size and type of every upload before doing any work.
  - Reject anything outside the limits with a clear 4xx error.
- **Whisper:** `faster-whisper`, model `small`, int8 on CPU. Model size is configurable. Transcribes
  in whatever language was spoken.
- **CLAP:** `laion/clap-htsat-unfused` via `transformers`, CPU build of torch.
  - Zero-shot against a curated list of about 30 car-sound labels: knocking, ticking, belt squeal,
    grinding, hissing, whining, clunking, rattling, rough idle or misfire, normal idle, and so on.
  - Return the top 3 with their scores. Below a threshold, report "no distinctive sound detected".
  - The label list lives in one Python constant.
- **Frames:** up to 8 evenly spaced frames from the video, sent to DeepSeek as base64 image blocks.
  The count is configurable.
- **Model loading and storage:**
  - Load models once, lazily, and reuse them.
  - Run all CPU-heavy work through `asyncio.to_thread`.
- **Uploads:** store originals in a private Supabase Storage bucket `media` at
  `{user_id}/{session_id}/{message_id}/...`. The message JSON references them by path. Deleting a
  session deletes its media.

### Knowledge search (Postgres FTS)

- **New table** `kb_passages`:
  - Columns: `id`, `source_type` (`manual` | `book` | `guide`), `car_id` (null for books), `title`,
    `page`, `content`, and a generated `tsv tsvector`.
  - A GIN index on `tsv`.
- **SQL function** `search_kb(q text, car_id text, k int)`:
  - Uses `websearch_to_tsquery` + `ts_rank`.
  - Searches this car's manual passages plus all `book` passages.
  - Called from Python via `supabase.rpc`.
- **Ingestion:** `pypdf` extracts text per page, splits it into ~1,500-character passages and inserts
  them in batches.
  - A CLI script loads the books in `car_data/`.
  - Adding a car triggers the same path as a background task, using the manual PDF the existing
    `FetchCarDataService` downloads.
- **Online guide scraping** (`crawl4ai`) stays at ingestion time only and stores into `kb_passages` as
  `guide`. *Builder's call:* drop it if it's flaky or drags Playwright into the image.

### Auth and API surface

- Every route uses the real JWT dependency in `app/utils/auth_middleware.py`.
- **Deleted:**
  - `auth_middleware_mobile.py`, the "bypass" that returns a fake `mobile-user`.
  - `/auth/list-users`, `/auth/manual-confirm*` and `/auth/debug-confirm`.
- Routes drop `user_id` path/body params in favour of the token's user (`/cars`, `/sessions`, …).
- **Endpoint set:**
  - **Auth:** register, login, logout and confirm. These stay as they are.
  - **Cars:** list, add and remove.
  - **Sessions:** create, list, get (with messages and state), delete, and send message (the SSE
    endpoint above).
  - **Health:** one `/health` that checks Supabase and DeepSeek reachability.

### What gets deleted

**Infrastructure:**
- `ConcurrentChatService`, `ChatService`
- `agent_pool.py`, `global_agents.py`
- `session_manager.py`, `enhanced_connection_pool.py`, `redis_cache.py`
- All of `app/agents/`
- The four WebSocket modules
- `milvus_handler.py`, `embedding_service.py`, `car_vectorization_service.py`
- `core/celery.py`, `tasks.py`, `background_tasks.py`

**Middleware and utilities:**
- `validation_middleware.py`: regex "malicious input" checks that also reject legitimate text
- `monitoring.py`, `custom_docs.py`
- `diagnosis_tree*.py`, `tree_formatter.py`

**Main app, config and dependencies:**
- The large `main.py` health/metrics block
- About half of `Settings`
- From `requirements.txt`: `selenium`, `litellm`, `outlines`, `tortoise`, `aerich`, `sqlalchemy`,
  `pandas`, `datasets`, `nltk`, `rank-bm25`, `langchain*`, `pymilvus`, `celery`, `redis`, `ollama`,
  `googlesearch`.
- `torch` and `transformers` stay, but now as real dependencies of CLAP, with torch pinned to its CPU
  wheel.

**Containers:** Redis, Milvus, etcd and minio leave `docker-compose.yml`.

- **Rate limiting:** `slowapi` with in-memory storage. *Ceiling:* the limits are per process, so add
  Redis back when running more than one instance.
- **Audit logging and security headers:** stay if they're small and working. *Builder's call*, but
  they must not log request bodies or auth headers. Today the request middleware logs every header,
  JWTs included, and that has to stop.

### Tests

- Delete tests for deleted modules.
- **New focused tests:**
  - The turn pipeline with a fake LLM client: the event order, that state is saved, and that the
    reply stays under 2 LLM calls.
  - Upload validation: limits and bad types.
  - ffmpeg frame/audio extraction on a tiny fixture clip.
  - The CLAP/Whisper wrappers with the models stubbed.
  - Calling `search_kb` (RPC mocked).
  - Auth: every route returns 401 without a token, and one user can't read another's session.
- **`scripts/e2e.py`** runs against a live compose stack with real keys:
  - Register or log in, add a car, and open a session.
  - Send one text, one photo, one audio and one video message.
  - Assert each stream yields tokens and a `state` event, and print the timings.
  - Sample media lives in `tests/fixtures/`.
- **Quality case set:** `tests/cases/diagnosis_cases.yaml` holds 8–10 cases.
  - **What the cases are:**
    - The two historical ones above.
    - Common text symptoms: brake grinding, no-start that clicks, rough idle, a warning light.
    - At least one photo case (a leak or a dashboard light), one audio case (belt squeal or knock)
      and one video case.
  - **What each case records:** the car, the inputs, and a short checklist with three parts:
    - *must mention*, e.g. case 1: the coolant temperature sensor or fan relay as a suspect.
    - *must not*, e.g. "refer to your manual", a leaked preamble, or an unsupported top cause.
    - *uses car-specific facts with a citation where the KB has them*.
  - **How it's run:** `scripts/run_cases.py` sends every case through the pipeline against real
    DeepSeek and writes one Markdown report of the replies with each checklist next to them, for a
    person to read and tick.
  - **When it's run:** once at step 4, to compare with the old outputs, and again after any prompt
    change.
  - It's a script, not part of the automated test suite, because it spends API credit.
- **Encoding:** one test that a reply containing "°F", "—" and non-English text comes back intact
  from storage. Today's saved output shows � in their place.

## Rollout order

1. **Baseline.** Record the current latency of one text diagnosis. Note the current 66 failing unit tests.
2. **Delete and fix auth.** Switch to uv first: build `pyproject.toml` and `uv.lock` from only the
   dependencies that are actually imported, delete `requirements.txt` and the old `.venv`, then run
   `uv sync`. After that, remove dead code, wire real auth everywhere, slim
   `main.py` and config, and get the remaining tests green.
3. **Knowledge search.** Schema SQL (with sign-off), ingestion script, `search_kb`, and loading of
   `car_data/` books.
4. **Text pipeline.** New session endpoints, the streaming DeepSeek call, the state update and the new
   prompt. Write the case set and run `scripts/run_cases.py`. The new replies must beat the two
   reviewed old outputs on their checklists before moving on.
5. **Media.** Uploads and storage, ffmpeg, Whisper, CLAP and frames to vision.
6. **Ship.** Dockerfile (uv, ffmpeg, CPU torch from the uv-pinned index, model download at build time), compose, `scripts/e2e.py`,
   and a rewritten README and API docs (OpenAPI at `/docs`; the stale Postman collection is regenerated
   or dropped).

Each step leaves the app runnable, with small conventional commits per step.

## Left to the builder

- Prompt wording.
- Passage size and number of passages.
- FTS language config.
- CLAP label wording and threshold.
- Frame count.
- Whisper model size.
- Exact SSE payload fields.
- File layout of the new modules (keep it flat: a few service modules, not one class per concept).
- Whether `crawl4ai` guide scraping survives.
