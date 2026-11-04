# AutoCare Backend Finish — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the slow multi-agent chat backend into one streamed DeepSeek pipeline. Add
photo/audio/video diagnosis, strip dead code, and ship it as a `docker compose up` stack.

**Architecture:** FastAPI on Supabase (auth, Postgres, Storage).
- Each chat turn runs: validate the uploads → ffmpeg, Whisper and CLAP turn the media into
  observations → Postgres full-text search over `kb_passages` → **one** streaming DeepSeek call
  (`deepseek-flash`, images inline) → one small JSON call that updates the session state.
- Results stream to the client over server-sent events.

**Tech Stack:**
- **Runtime and API:** Python 3.12, uv, FastAPI 0.115, supabase-py.
- **AI:** `openai` SDK pointed at DeepSeek, faster-whisper, transformers CLAP, CPU torch.
- **Media and docs:** ffmpeg, pypdf.
- **Tests:** pytest.

**Spec:** `docs/superpowers/specs/2026-11-04-autocare-backend-finish-design.md`. Read it first; this
plan argues from it.

## Global Constraints

- Python 3.12. Every command runs through `uv`: `uv sync`, `uv run pytest`, `uv run uvicorn main:app`.
  No `pip`, no `requirements.txt`.
- **Dependency versions:**
  - Keep the currently pinned versions of the packages we retain. Ask the user before upgrading any
    of them.
  - New packages (faster-whisper, CPU torch) take the latest release compatible with the retained pins.
- torch comes from the PyTorch CPU index via `[[tool.uv.index]]` (`explicit = true`) plus
  `[tool.uv.sources]`.
- **LLM:** DeepSeek only. `DEEPSEEK_BASE_URL=https://api.deepseek.com`, `DEEPSEEK_MODEL=deepseek-flash`.
  Use the `openai` `AsyncOpenAI` client. No LangChain.
- **Call budget:** at most **2 LLM calls per chat turn**. That means 1 streaming reply call plus
  1 state call.
- **Media limits:**
  - Up to 4 images per message, each ≤ 10 MB.
  - **Or** one audio/video file of ≤ 60 s and ≤ 100 MB.
  - Text is optional, ≤ 5000 chars. A message needs text or at least one file.
- **Identity:** every route except `/api/v1/auth/*` and `/health` requires `Authorization: Bearer
  <app JWT>`. The user id always comes from the token and is never read from the body, query or path.
- **Logging:** never log request bodies, `Authorization` headers or tokens.
- **Text encoding:** UTF-8 end to end. JSON is written with `ensure_ascii=False`.
- **Database changes:** schema changes are SQL files that **the user runs** in the Supabase SQL
  editor, after explicit sign-off at that moment. The executor never runs DDL.
- **Commits:** small conventional commits (`feat:`, `fix:`, `chore:`, `refactor:`, `test:`) on branch
  `development`. **No `Co-Authored-By` line.** Never commit `.env*`.
- **Secrets:** never read `.env*`. Tests set settings through environment variables in
  `tests/conftest.py`.
- **Done per task:** `uv run pytest` passes before a task is marked done.

## Review Focus

1. **Long, chatty messages find zero passages.** `websearch_to_tsquery` ANDs every word. A person
   expects relevant manual pages for "my car is overheating and the fan keeps clicking on and off".
   → `build_tsquery` ORs keywords. Tested in Task 6.
2. **Uploads that lie or are odd:**
   - A `.mp4` that is really text → 415.
   - A video with no audio track → frames only, no crash.
   - An "audio" file over 60 s → 413 with a clear message.
   → Tested in Task 12.
3. **DeepSeek returns non-JSON or partial JSON in the state call.** The person still gets their
   reply, and the previous state is kept. → Tested in Task 8.
4. **The reply fails mid-stream** (API error or timeout). The person sees an `error` event. Their
   message is saved, and no half reply is saved. → Tested in Task 10.
5. **Double-tap send on the same session.** Two concurrent sends must not lose a message through the
   read-modify-write of `messages` jsonb. The second one gets 409 while the first is running.
   → Tested in Task 10.

---

## Baseline (already captured, no task)

- **Latency:** today a text diagnosis takes 402 s (screenshot `image.png`) and 11 min 42 s
  (`result.json` timestamps). The legacy stack can't be re-run: its Milvus data is gone. These two
  numbers are the latency baseline.
- **Tests:** 249 pass, 66 fail and 2 error (`uv run pytest tests/unit` after Task 1 must reproduce
  this before anything is deleted).

## Target file map

```
main.py                         app, middleware, routers, /health (rewrite, ~100 lines)
app/core/config.py              trimmed Settings (rewrite)
app/db/base.py, app/db/crud.py  keep; BaseCRUD runs .execute() via asyncio.to_thread
app/db/bucket_operations.py     keep (Storage)
app/db/sql/001_v2_schema.sql    NEW: kb_passages, search_kb(), Chatsession reset, Car columns
app/CRUD/car_crud.py            keep, vector code replaced by kb ingestion
app/CRUD/chat_session_crud.py   + get_for_user(), save_turn()
app/schemas/session.py          NEW pydantic: Message, Observations, SoundTag, SuspectedCause, DiagnosisState
app/schemas/User.py, Car.py     keep
app/api/v1/routes.py            auth, cars, sessions
app/api/v1/user_auth.py         keep register/login/logout/confirm only
app/api/v1/car_route.py         rewrite: /cars from token user
app/api/v1/session_route.py     NEW: /sessions CRUD + SSE send (replaces chat_route.py)
app/services/llm.py             NEW: DeepSeek client: stream_reply(), extract_state(), ping()
app/services/kb.py              NEW: build_tsquery(), chunk_pages(), ingest_pdf(), search()
app/services/prompts.py         NEW: SYSTEM_PROMPT, STATE_PROMPT, build_messages()
app/services/media.py           NEW: Upload, validate_upload(), probe(), audio_pcm(), extract_frames()
app/services/hearing.py         NEW: CAR_SOUND_LABELS, transcribe(), tag_sounds(), prefetch()
app/services/diagnosis.py       NEW: run_turn() (the pipeline), sse()
app/services/fetch_car_data_service.py, user_service.py, user_profile_service.py, base_service.py  keep, trimmed
app/utils/auth_middleware.py, logger.py, exceptions.py, limiter.py, security_middleware.py, audit_logging.py  keep, trimmed
scripts/ingest_kb.py, scripts/e2e.py, scripts/run_cases.py   NEW
tests/cases/diagnosis_cases.yaml                             NEW
```

---

## Phase 1 — Foundation

### Task 1: Switch to uv

**Files:**
- Modify: `pyproject.toml` (add `[project]`, `[dependency-groups]`, `[[tool.uv.index]]`, `[tool.uv.sources]`; keep the `[tool.pytest.*]`/coverage sections)
- Create: `uv.lock`
- Delete: `requirements.txt`, `.venv/` (local only, not tracked)

**Interfaces:** Produces the `uv run …` commands that every later task uses.

- [ ] **Step 1: Write `[project]` dependencies, retained pins only.** Keep the exact versions from
  `requirements.txt` for:
  - Web and data: `fastapi`, `uvicorn[standard]`, `pydantic`, `pydantic-settings`, `python-multipart`,
    `supabase`, `slowapi`.
  - Auth: `PyJWT`, `email-validator`.
  - LLM client: `openai`.
  - HTTP and scraping: `httpx`, `beautifulsoup4`, `aiofiles`, `tenacity`, `chardet`.
  - PDFs: `pypdf`.

  Their transitive deps come from the lock. The `dev` group gets `pytest`, `pytest-asyncio`,
  `pytest-mock` and `pytest-cov`.

  Don't add the media packages yet (Task 12/13 adds them).

  Temporarily also keep whatever the current code still imports (`langchain*`, `pymilvus`, `redis`,
  `celery`, `crawl4ai`, `numpy`, `websockets`) so that the baseline runs unchanged. Task 2 removes
  them.
- [ ] **Step 2: Lock and sync.** Run `uv lock && uv sync`. Expected: `.venv` is created and there are
  no resolver errors.
- [ ] **Step 3: Reproduce the baseline.** Run `uv run pytest tests/unit -q`.
  Expected: `66 failed, 249 passed, 2 errors`. If the counts differ, record the new numbers in the
  commit message. Don't fix anything yet.
- [ ] **Step 4: Delete `requirements.txt`** and update the Dockerfile `COPY requirements.txt` line to
  `COPY pyproject.toml uv.lock ./`. The full Dockerfile rewrite is Task 15. Until then the Docker build
  is knowingly broken; say so in the commit body.
- [ ] **Step 5: Commit** `chore: switch package management to uv`.

### Task 2: Delete the dead layers, slim `main.py` and config

**Files:**
- **Delete:**
  - `app/agents/` (all of it)
  - Services: `app/services/{chat_service,concurrent_chat_service,llm_service,embedding_service,car_vectorization_service,search_engine_service,scraper_service,parser_service,query_builder_service}.py`
  - Utilities: `app/utils/{concurrent_websocket,websocket,orchestrator_websocket,listen_socket,session_manager,redis_cache,monitoring,custom_docs,validation_middleware,auth_middleware_mobile,diagnosis_tree,diagnosis_tree_factory,tree_formatter,text_sanitizer,dataloader,json_utils,message_formatter,message_types,fast_responses,startup_checks}.py`
  - DB: `app/db/{enhanced_connection_pool,milvus_handler,migration}.py`, `app/db/migrations/`
  - Core and tasks: `app/core/{celery,interfaces,oauth_config}.py`, `app/tasks.py`
  - Routes: `app/api/v1/{background_tasks,chat_route,security_route,user_route,health}.py`
  - CRUD and schemas: `app/CRUD/{ground_knowledge_crud,diagnosis_history_crud}.py`, `app/schemas/Chat_Session.py`
  - Repo root: `dump.rdb`, `deploy.sh`
- **Delete tests:** every file under `tests/` that imports a deleted module. Use
  `uv run pytest --co` to find the import errors.
- **Rewrite:** `main.py`, `app/core/config.py`, `tests/conftest.py`.
- **Modify:**
  - `app/db/crud.py`: wrap each `.execute()` in `asyncio.to_thread`.
  - `app/db/base.py`, `app/services/base_service.py`: remove the websocket-manager parameters.
  - Any kept module that imports a deleted one.
- **Test:** `tests/unit/test_app.py`

**Interfaces:**
- Produces `get_settings() -> Settings`, where `Settings` has exactly these fields:
  - `ENV="development"`, `DEBUG=False`
  - `JWT_SECRET_KEY` (required, ≥ 32 chars), `ALLOWED_ORIGINS`
  - `SUPABASE_URL`, `SUPABASE_KEY`
  - `BASE_URL` (manual scraping source)
  - `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL="https://api.deepseek.com"`, `DEEPSEEK_MODEL="deepseek-flash"`
  - `WHISPER_MODEL="small"`, `CLAP_MODEL="laion/clap-htsat-unfused"`, `SOUND_MIN_SCORE=0.2`
  - `MAX_FRAMES=8`, `MAX_MEDIA_SECONDS=60`, `MAX_MEDIA_MB=100`, `MAX_IMAGES=4`, `MAX_IMAGE_MB=10`
  - `KB_TOP_K=8`, `HISTORY_MESSAGES=10`, `MEDIA_BUCKET="media"`

  Drop `SUPABASE_PASSWORD` unless a kept module reads it (grep for it).
- Produces `main.app` with:
  - CORS, `SecurityHeadersMiddleware`, slowapi (`app.state.limiter`).
  - One handler for `BaseAPIException` and one generic handler that returns
    `{"error_code","message"}` with no stack trace.
  - Router `/api/v1`, and `GET /health`.
  - No request-logging middleware. The current one logs every header, JWTs included.

- [ ] **Step 1: Write the failing test** in `tests/unit/test_app.py`:

```python
def test_app_imports_and_health_route_exists(client):
    paths = {r.path for r in client.app.routes}
    assert "/health" in paths
    assert not any(p.startswith("/ws") or "metrics" in p for p in paths)

def test_no_header_logging(client, caplog):
    client.get("/health", headers={"Authorization": "Bearer secret-token-xyz"})
    assert "secret-token-xyz" not in caplog.text
```

  `client` is a `TestClient(main.app)` fixture in the rewritten `tests/conftest.py` that sets the env
  vars. In this task `/health` returns `{"status": "ok"}` statically. Task 8 adds the real checks, and
  the fixture's monkeypatches of `llm.ping` / the Supabase health check to `True` along with them.
- [ ] **Step 2: Run `uv run pytest tests/unit/test_app.py -v`.** Expected: FAIL. The old `main.py`
  needs Redis and Milvus at startup.
- [ ] **Step 3: Delete the files listed above** and rewrite `main.py` and `config.py` to the
  interfaces above. Remove the stale dependencies from `pyproject.toml` (`langchain*`, `pymilvus`,
  `redis`, `celery`, `crawl4ai`, `websockets`, `numpy` unless still imported), then run `uv lock`.
  - Decision: no `crawl4ai`, because guide-page scraping is out of v1 (spec "Left to the builder").
    `fetch_car_data_service` keeps httpx and bs4 for finding manual links.
- [ ] **Step 4: Run the tests.** `uv run python -c "import main"` → no error. Then run
  `uv run pytest tests/unit/test_app.py -v` and expect PASS.
- [ ] **Step 5: Commit** `refactor: remove agent pipeline, websocket stack and unused infrastructure`.

### Task 3: Real auth on every route

**Files:**
- Modify:
  - `app/utils/auth_middleware.py`: the secret comes from `get_settings().JWT_SECRET_KEY` and no
    longer from an `os.getenv` fallback string.
  - `app/api/v1/user_auth.py`: delete `/auth/test`, `/auth/list-users`, `/auth/manual-confirm`,
    `/auth/manual-confirm-by-id` and `/auth/debug-confirm`.
  - `app/api/v1/car_route.py`, `app/api/v1/routes.py`.
- Test: `tests/unit/test_auth_routes.py`

**Interfaces:**
- Consumes `get_current_user(request, credentials) -> {"id", "email", "role", "permissions"}`. It
  already exists and stays as it is.
- Produces the routes:
  - `GET /api/v1/cars` → `list[Car]`
  - `POST /api/v1/cars` body `{make: str, model: str, year: int}` → `Car`
  - `DELETE /api/v1/cars/{car_id}` → `{"deleted": car_id}`

  These use `user_profile_service` with `current_user["id"]`.

- [ ] **Step 1: Write the failing tests.**

```python
PUBLIC = {"/health", "/docs", "/openapi.json", "/redoc", "/docs/oauth2-redirect"}

def test_every_non_auth_route_requires_token(client):
    for route in client.app.routes:
        path = getattr(route, "path", "")
        if path in PUBLIC or path.startswith("/api/v1/auth"):
            continue
        for method in route.methods - {"HEAD", "OPTIONS"}:
            r = client.request(method, path.replace("{", "").replace("}", ""))
            assert r.status_code == 401, (method, path)

def test_debug_auth_endpoints_gone(client):
    for p in ["/api/v1/auth/list-users", "/api/v1/auth/manual-confirm", "/api/v1/auth/debug-confirm"]:
        assert client.get(p).status_code in (404, 405)

def test_jwt_uses_settings_secret(monkeypatch):
    # a token signed with the old hard-coded fallback secret is rejected
    import jwt
    from app.utils.auth_middleware import jwt_handler, AuthenticationError
    bad = jwt.encode({"sub": "u", "type": "access"}, "your-super-secret-jwt-key-change-in-production", "HS256")
    with pytest.raises(AuthenticationError):
        jwt_handler.verify_token(bad)
```

- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.** In `car_route.py`, every route takes
  `current_user = Depends(get_current_user)`. `add_car` returns as soon as the car row exists;
  manual ingestion is wired in Task 7.
- [ ] **Step 4: Run them.** Expected: PASS.
- [ ] **Step 5: Commit** `fix: require JWT on all routes and remove unauthenticated debug endpoints`.

### Task 4: Green the surviving tests

**Files:** the failing tests left over from the baseline (`test_security_middleware.py`,
`test_models_car.py`, `test_rate_limiter.py`, `test_exceptions.py`, `test_auth_middleware.py`,
`test_models_user.py`, plus `tests/integration/*` whose modules survive), and the code they cover.

- [ ] **Step 1: Run `uv run pytest -q`** and list the failures.
- [ ] **Step 2: Decide fix-or-delete for each failing test.** Fix it (code or test) when it asserts
  current, desired behaviour of kept code. Delete it when it asserts deleted or obsolete behaviour.
  Record one line per deleted test in the commit body.
- [ ] **Step 3: Run `uv run pytest -q`.** Expected: 0 failed, 0 errors.
- [ ] **Step 4: Commit** `test: fix or remove stale tests after cleanup`.

---

## Phase 2 — Knowledge search

### Task 5: Schema SQL (user-run)

**Files:**
- Create: `app/db/sql/001_v2_schema.sql`

**Interfaces:** Produces these tables and functions:
- `kb_passages(id bigint identity pk, source_type text check in ('manual','book'), car_id text null, title text not null, page int, content text not null, tsv tsvector generated always as (to_tsvector('english', content)) stored)`.
  - GIN index on `tsv`, btree on `car_id`.
- `search_kb(tsq text, p_car_id text, k int default 8) returns table(source_type, car_id, title, page, content, rank real)`.
  - Matches `tsv @@ to_tsquery('english', tsq)` and `(source_type = 'book' or car_id = p_car_id)`.
  - Ordered by `ts_rank` desc, `limit k`.
  - Language `sql`, `stable`.
- **`Chatsession`**, dropped and recreated:
  - Columns: `id uuid pk default gen_random_uuid(), user_id uuid not null, car_id text not null, title text not null default 'New diagnosis', messages jsonb not null default '[]', state jsonb not null default '{}', created_at timestamptz default now(), updated_at timestamptz default now()`.
  - Index on `user_id`.
- **`Car`:** add `kb_status text not null default 'pending'` (`pending|ready|failed|no_manual`) and
  `kb_passage_count int not null default 0`. Drop `is_vectorized` and `vector_chunk_count`.
- **`Groundknowledge`:** `drop table if exists`.

Steps:
- [ ] **Step 1: Write the SQL file.**
- [ ] **Step 2: Gate.** Stop and ask the user to review the file and run it in the Supabase SQL
  editor. Wait for confirmation. The user approved wiping old data in the spec, but each DDL run still
  needs a yes at the moment it happens.
- [ ] **Step 3: Verify.** Run `uv run python -c "from app.db.base import SupabaseDBHandler as H; print(H().client.rpc('search_kb', {'tsq':'engine','p_car_id':'x','k':1}).execute().data)"`.
  Expected: `[]`, with no error.
- [ ] **Step 4: Commit** `feat: add v2 schema with kb_passages and search_kb`.

### Task 6: `kb.py` (query building, chunking, ingest, search)

**Files:**
- Create: `app/services/kb.py`
- Test: `tests/unit/test_kb.py`

**Interfaces:**
- `build_tsquery(text: str, max_terms: int = 12) -> str`
  - Lowercases, keeps alphanumeric tokens of 3+ chars, drops a small English stopword set
    (module constant), dedups in order and joins with `" | "`.
  - Returns `""` when nothing is left.
- `chunk_pages(pages: list[str], size: int = 1500) -> list[tuple[int, str]]`
  - Returns `(page_number_1_based, text)`.
  - Splits a page on blank lines and packs paragraphs up to `size` chars. A single paragraph over
    `size` is hard-split.
  - Skips whitespace-only chunks.
- `async ingest_pdf(path: Path, *, source_type: Literal["manual","book"], title: str, car_id: str | None = None) -> int`
  - Deletes the existing rows with the same `title` and `car_id`, extracts pages with `pypdf`, then
    inserts the chunks in batches of 500.
  - Returns the number of rows inserted (0 for scanned, text-less PDFs).
- `async search(query: str, car_id: str, k: int) -> list[Passage]`
  - `Passage` is a `TypedDict` with `source_type, title, page, content`.
  - Calls RPC `search_kb` with `tsq=build_tsquery(query)`, and returns `[]` without calling the RPC
    when the tsquery is empty.

- [ ] **Step 1: Write the failing tests.**

```python
def test_build_tsquery_ors_keywords_from_chatty_message():
    q = build_tsquery("My car is overheating and the fan keeps clicking on and off")
    assert q == "car | overheating | fan | keeps | clicking | off"

def test_build_tsquery_empty_for_stopwords_only():
    assert build_tsquery("is it on?") == ""

def test_chunk_pages_keeps_page_numbers_and_size():
    pages = ["a" * 1000 + "\n\n" + "b" * 1000, "", "c" * 4000]
    chunks = chunk_pages(pages, size=1500)
    assert [p for p, _ in chunks] == [1, 1, 3, 3, 3]
    assert all(len(t) <= 1500 for _, t in chunks)

async def test_search_skips_rpc_for_empty_query(mocker):
    rpc = mocker.patch("app.services.kb._rpc")
    assert await search("is it on?", "toyota-echo-2001", 8) == []
    rpc.assert_not_called()

async def test_search_passes_car_and_k(mocker):
    rpc = mocker.patch("app.services.kb._rpc", return_value=[{"source_type": "manual", "title": "Echo", "page": 3, "content": "x", "rank": 0.5}])
    out = await search("radiator fan", "toyota-echo-2001", 8)
    rpc.assert_called_once_with("search_kb", {"tsq": "radiator | fan", "p_car_id": "toyota-echo-2001", "k": 8})
    assert out[0]["page"] == 3
```

  `_rpc(name, params) -> list[dict]` is the module's one Supabase touchpoint. It runs the
  `client.rpc(...).execute()` call in `asyncio.to_thread`.
- [ ] **Step 2: Run `uv run pytest tests/unit/test_kb.py -v`.** Expected: FAIL (module missing).
- [ ] **Step 3: Implement** `app/services/kb.py` to the interfaces above.
- [ ] **Step 4: Run the same command.** Expected: PASS.
- [ ] **Step 5: Commit** `feat: postgres full-text knowledge search`.

### Task 7: Load the books, ingest a car's manual when it's added

**Files:**
- Create: `scripts/ingest_kb.py`
- Modify: `app/CRUD/car_crud.py`, replacing `_fetch_and_vectorize_manual` and every
  `vectorization_service` use with `ingest_car_manual`.
- Modify: `app/api/v1/car_route.py`, so `POST /cars` schedules `BackgroundTasks.add_task(ingest_car_manual, car)`.
- Test: `tests/unit/test_car_ingest.py`

**Interfaces:**
- `async ingest_car_manual(car: dict) -> None` in `car_crud.py`:
  - Uses `FetchCarDataService.build_url` / `scrape_links` / `download_pdf` to fetch the manual, then
    calls `kb.ingest_pdf(path, source_type="manual", title=f"{make} {model} {year} owner's manual", car_id=car["id"])`.
  - Sets `kb_status`: `ready` if rows > 0, `no_manual` if no link was found, `failed` on an exception
    (logged).
  - Updates `kb_passage_count`.
- `scripts/ingest_kb.py [--dir car_data]`:
  - Ingests every PDF in the directory **except** `*_manual.pdf` as `source_type="book"`, with the
    title taken from the filename (dashes become spaces, title-cased).
  - Files named `<car_id>_manual.pdf` ingest as `manual` for that `car_id`.
  - Prints `title: N passages` per file.

- [ ] **Step 1: Write the failing tests.**

```python
async def test_add_car_returns_before_ingest(client, auth_headers, mocker):
    ingest = mocker.patch("app.api.v1.car_route.ingest_car_manual")
    r = client.post("/api/v1/cars", json={"make": "Toyota", "model": "Echo", "year": 2001}, headers=auth_headers)
    assert r.status_code == 200 and r.json()["kb_status"] == "pending"
    ingest.assert_called_once()

async def test_ingest_marks_no_manual(mocker):
    mocker.patch("app.CRUD.car_crud.FetchCarDataService.scrape_links", return_value=None)
    update = mocker.patch("app.CRUD.car_crud.CarCRUD.update")
    await ingest_car_manual({"id": "x-y-2001", "make": "X", "model": "Y", "year": 2001})
    assert update.call_args.args[1]["kb_status"] == "no_manual"
```

  The `auth_headers` fixture in `conftest.py` signs a token with `jwt_handler.create_access_token`.
  The fixture also patches `validate_user_with_supabase` to return `True`.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them.** Expected: PASS.
- [ ] **Step 5: Real run, needs `.env` with Supabase.** Run `uv run scripts/ingest_kb.py`.
  - Expected: 8 lines, with `toyota-echo-2001_manual.pdf` reported as a manual. Every count should be
    > 0. A 0 means a scanned PDF; report it to the user and don't add OCR.
  - Then `kb.search("radiator fan overheating", "toyota-echo-2001", 8)` should return ≥ 1 manual
    passage.
- [ ] **Step 6: Commit** `feat: ingest books and car manuals into kb_passages`.

---

## Phase 3 — Text pipeline

### Task 8: DeepSeek client

**Files:**
- Create: `app/services/llm.py`
- Modify: `main.py` (`/health` checks `llm.ping()` and Supabase)
- Test: `tests/unit/test_llm.py`

**Interfaces:**
- `get_client() -> AsyncOpenAI`: lazily cached, `base_url=DEEPSEEK_BASE_URL`, `api_key=DEEPSEEK_API_KEY`.
- `async stream_reply(messages: list[dict]) -> AsyncIterator[str]`
  - Calls `chat.completions.create(model=DEEPSEEK_MODEL, messages=messages, stream=True)`.
  - Yields only non-empty `delta.content`, skipping reasoning deltas.
- `async extract_state(conversation: list[dict], previous: DiagnosisState) -> DiagnosisState`
  - Non-streaming. Sends `STATE_PROMPT` (Task 9) and asks for JSON.
  - On any parse or validation failure, returns `previous` unchanged and logs a warning.
  - Before using `response_format={"type": "json_object"}`, check DeepSeek's docs that it's
    supported. Otherwise parse the first `{…}` block.
- `async ping() -> bool`: `models.list()` succeeds within 5 s.
- `DiagnosisState` comes from `app/schemas/session.py` (created here):
  - `{suspected_causes: list[SuspectedCause], safety_flags: list[str], title: str}`.
  - `SuspectedCause` is `{cause: str, likelihood: float (0–1), evidence: str, next_check: str}`.

- [ ] **Step 1: Write the failing tests.** They use a fake client injected via
  `mocker.patch("app.services.llm.get_client")`:

```python
async def test_stream_reply_yields_only_content(fake_stream):
    # fake chunks: reasoning-only delta, "Check ", None, "coolant."
    assert [t async for t in stream_reply([{"role": "user", "content": "hi"}])] == ["Check ", "coolant."]

async def test_extract_state_keeps_previous_on_bad_json(fake_completion):
    fake_completion("not json at all")
    prev = DiagnosisState(suspected_causes=[], safety_flags=["hot"], title="Overheating")
    assert await extract_state([], prev) == prev

async def test_extract_state_parses_valid_json(fake_completion):
    fake_completion('{"suspected_causes":[{"cause":"Fan relay","likelihood":0.6,"evidence":"on/off bursts","next_check":"swap relay"}],"safety_flags":[],"title":"Fan cycling"}')
    s = await extract_state([], DiagnosisState(suspected_causes=[], safety_flags=[], title=""))
    assert s.suspected_causes[0].cause == "Fan relay"
```

- [ ] **Step 2: Run `uv run pytest tests/unit/test_llm.py -v`.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the same command.** Expected: PASS.
- [ ] **Step 5: Thinking-mode measurement.** This needs the real API key and is a one-off script
  that isn't committed.
  - Call `stream_reply` 3× on the case-1 prompt with DeepSeek's default settings. If the docs show a
    way to disable or reduce thinking, call it 3× that way too.
  - Record time-to-first-token and total time for each run in the commit body.
  - Keep the setting that gets first text in ≤ ~10 s, as long as the reply isn't visibly worse. Make
    it a `Settings` field if it's a request parameter.
- [ ] **Step 6: Commit** `feat: streaming DeepSeek client`.

### Task 9: Prompts and message assembly

**Files:**
- Create: `app/services/prompts.py`
- Test: `tests/unit/test_prompts.py`

**Interfaces:**
- `SYSTEM_PROMPT: str`. It must state the spec's prompt rules:
  - DIY-owner audience.
  - Cite as `(<title>, p. <page>)` when using a passage.
  - Otherwise say "I couldn't find …"; never say "check your manual" and never leave blanks.
  - The top cause names the symptom it explains.
  - Sound tags are weak evidence.
  - Safety first.
  - Reply in the user's language.
  - No preamble.
- `STATE_PROMPT: str`, which instructs the model to output the `DiagnosisState` JSON.
- `build_messages(*, car: dict, state: DiagnosisState, history: list[Message], text: str | None, observations: Observations | None, passages: list[Passage], images: list[bytes]) -> list[dict]`
  - Returns `[system, …last HISTORY_MESSAGES history turns as role/content…, user]`.
  - The final user content is a list of blocks:
    1. One text block with the sections "Car", "Current suspected causes", "Observations from media"
       (transcript plus sound tags with scores, or "none"), "Reference passages" (each labelled
       `[<title>, p. <page>]`) and "User message".
    2. Then one `{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,…"}}` per image.
- `Observations` and `SoundTag` are created in `app/schemas/session.py`:
  `Observations{transcript: str, sounds: list[SoundTag{label: str, score: float}]}`.

- [ ] **Step 1: Write the failing tests.**

```python
def test_system_prompt_rules_present():
    for phrase in ["p. ", "couldn't find", "weak evidence", "Do not tell the user to check"]:
        assert phrase in SYSTEM_PROMPT

def test_passages_are_labelled_for_citation(car, empty_state):
    msgs = build_messages(car=car, state=empty_state, history=[], text="fan clicks", observations=None,
                          passages=[{"source_type": "manual", "title": "Toyota Echo 2001 owner's manual", "page": 212, "content": "Coolant: Toyota Long Life"}], images=[])
    assert "[Toyota Echo 2001 owner's manual, p. 212]" in msgs[-1]["content"][0]["text"]

def test_history_truncated_and_images_appended(car, empty_state):
    history = [{"role": "user", "content": f"m{i}"} for i in range(30)]
    msgs = build_messages(car=car, state=empty_state, history=history, text=None, observations=None, passages=[], images=[b"\xff\xd8x"])
    assert len(msgs) == 1 + 10 + 1
    assert msgs[-1]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")

def test_sound_tags_rendered_with_scores(car, empty_state):
    obs = Observations(transcript="it squeals on cold start", sounds=[SoundTag(label="belt squeal", score=0.62)])
    text = build_messages(car=car, state=empty_state, history=[], text=None, observations=obs, passages=[], images=[])[-1]["content"][0]["text"]
    assert "belt squeal (0.62)" in text and "it squeals on cold start" in text
```

- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them.** Expected: PASS.
- [ ] **Step 5: Commit** `feat: diagnosis prompts and message assembly`.

### Task 10: The turn pipeline and session routes (text only)

**Files:**
- Create: `app/services/diagnosis.py`
- Create: `app/api/v1/session_route.py`
- Create: `app/services/media.py`, containing **only** the `Upload` dataclass as defined in Task 12's
  Interfaces. Task 12 adds the functions.
- Modify: `app/CRUD/chat_session_crud.py`, `app/api/v1/routes.py`
- Test: `tests/unit/test_turn.py`, `tests/unit/test_session_routes.py`

**Interfaces:**
- `sse(event: str, data: dict) -> str` returns `f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"`.
- `async run_turn(session: dict, car: dict, text: str | None, uploads: list[Upload]) -> AsyncIterator[tuple[str, dict]]`
  - Event sequence:
    1. `("stage", {"name": "searching"})`
    2. `("token", {"text": …})`, repeated
    3. `("state", DiagnosisState.model_dump())`
    4. `("done", {"message_id": …})`
  - On an LLM error: `("error", {"message": "The diagnosis service failed, please retry."})`, then stop.
  - `uploads` is always `[]` until Task 14.
  - Saves the user message **before** calling the LLM. Saves the assistant message, state and title
    only after the stream completes.
- `ChatSessionCRUD`:
  - `get_for_user(session_id: str, user_id: str) -> dict | None`
  - `append_messages(session_id: str, messages: list[dict]) -> None`, which reads, appends and
    writes `messages` and `updated_at`.
  - `save_state(session_id: str, state: dict, title: str) -> None`
- `Message` (in `schemas/session.py`): `{id: str, role: "user"|"assistant", content: str, media: list[{kind, path}], observations: Observations | None, created_at: str}`.
- Routes, all `Depends(get_current_user)`:
  - `POST /api/v1/sessions` body `{car_id}`. The car must be in the user's profile, else 404.
    Returns the session.
  - `GET /api/v1/sessions` returns `[{id, car_id, title, updated_at, message_count}]`.
  - `GET /api/v1/sessions/{id}` returns the full session. Another user's session is 404.
  - `DELETE /api/v1/sessions/{id}`. Media deletion is wired in Task 14.
  - `POST /api/v1/sessions/{id}/messages`: multipart `text: str | None = Form(None)`,
    `files: list[UploadFile] = File([])`. Returns
    `StreamingResponse(media_type="text/event-stream")`. Rate-limited `20/minute`.
- **In-flight lock:** a module-level `set[str]` of session ids. A second send while one is in flight
  gets 409 `{"message": "A reply is still being generated for this session."}`.
  `# ponytail: in-process, per-instance. Move to a DB advisory lock if we ever run >1 instance.`

- [ ] **Step 1: Write the failing tests.** They patch `llm.stream_reply`, `llm.extract_state` and
  `kb.search`, and use an in-memory fake `ChatSessionCRUD` fixture.

```python
async def test_event_order_and_two_llm_calls(fake_llm, fake_store, session, car):
    events = [e async for e, _ in run_turn(session, car, "fan clicks on and off", [])]
    assert events[0] == "stage" and events[-2:] == ["state", "done"] and "token" in events
    assert fake_llm.calls == 2

async def test_llm_failure_saves_user_message_only(fake_llm, fake_store, session, car):
    fake_llm.fail_stream_after = 1
    events = [e async for e, _ in run_turn(session, car, "fan clicks", [])]
    assert events[-1] == "error"
    assert [m["role"] for m in fake_store.messages(session["id"])] == ["user"]

async def test_utf8_round_trip(fake_llm, fake_store, session, car):
    fake_llm.reply = "Opens at 82 °C — check it. Überprüfen."
    _ = [x async for x in run_turn(session, car, "Temperatur?", [])]
    assert fake_store.messages(session["id"])[-1]["content"] == "Opens at 82 °C — check it. Überprüfen."

def test_other_users_session_is_404(client, auth_headers_other_user, existing_session):
    assert client.get(f"/api/v1/sessions/{existing_session['id']}", headers=auth_headers_other_user).status_code == 404

def test_concurrent_send_gets_409(client, auth_headers, existing_session, slow_llm):
    ...  # start one send in a thread; second send returns 409 with the message above

def test_send_requires_text_or_file(client, auth_headers, existing_session):
    r = client.post(f"/api/v1/sessions/{existing_session['id']}/messages", data={}, headers=auth_headers)
    assert r.status_code == 422
```

- [ ] **Step 2: Run `uv run pytest tests/unit/test_turn.py tests/unit/test_session_routes.py -v`.**
  Expected: FAIL.
- [ ] **Step 3: Implement.** `run_turn` calls `kb.search(text, car["id"], KB_TOP_K)`, then
  `build_messages`, then `stream_reply`, then `extract_state`.
- [ ] **Step 4: Run the same command.** Expected: PASS. Then run `uv run pytest -q` and expect all
  tests to pass.
- [ ] **Step 5: Manual smoke test.** With a real `.env`, start `uv run uvicorn main:app`, log in, add
  the Echo, create a session, and run
  `curl -N -F "text=My car is overheating and the radiator fan turns on in very short intervals" …/messages`.
  Expected: tokens stream, then a `state` event appears. Note the time to first token.
- [ ] **Step 6: Commit** `feat: streamed single-call diagnosis turn and session API`.

### Task 11: Quality case set (gate before media work)

**Files:**
- Create: `tests/cases/diagnosis_cases.yaml`
- Create: `scripts/run_cases.py`
- Add dependency: `pyyaml` (dev group)

**Interfaces:**
- Each case is `{id, car_id, text, media: [paths] (optional), must_mention: [str], must_not: [str], expects_citation: bool}`.
- `scripts/run_cases.py [--only id,…] [--out reports/cases-<timestamp>.md]`:
  - Calls `run_turn` in-process with real DeepSeek and KB, using an in-memory session store.
  - Writes Markdown: for each case, the reply, the final state, and a checklist of each
    `must_mention` / `must_not` item. The script pre-marks each item `[x]` / `[ ]` by
    case-insensitive substring, for a person to confirm.
  - Prints the `reports/` path.
  - `reports/` is gitignored.

- [ ] **Step 1: Write the YAML.** Text cases for this task (media cases are added in Task 14):
  - `overheat-fan-cycling` (case 1 from `result.json`). must_mention: `["temperature sensor", "relay", "coolant level"]`.
  - `reservoir-steam` (screenshot case 2). must_mention: `["radiator cap", "head gasket"]`.
  - `brake-grinding`
  - `no-start-clicking`
  - `rough-idle`
  - `check-engine-light`

  Every case gets `must_not: ["check your owner's manual", "refer to your manual", "here's a message"]`.
  Cases 1 and 2 get `expects_citation: true`.
- [ ] **Step 2: Implement the script.** Test it once with `--only overheat-fan-cycling`.
- [ ] **Step 3: Run all the cases.** Hand the report to the user.
  - **Gate:** the user confirms that the new replies beat the two reviewed old outputs.
  - If they don't, iterate on `prompts.py` only and re-run.
  - Record the before/after verdict in the commit body.
- [ ] **Step 4: Commit** `test: diagnosis quality case set and runner`.

---

## Phase 4 — Media

### Task 12: Upload validation and ffmpeg helpers

**Files:**
- Modify: `app/services/media.py` (the `Upload` dataclass exists from Task 10)
- Test: `tests/unit/test_media.py`
- Fixtures are generated at test time with ffmpeg `lavfi`, so no binary files are committed.

**Interfaces:**
- `Upload` (dataclass): `kind: Literal["image","audio","video"]`, `path: Path`, `mime: str`,
  `duration: float | None`, `has_audio: bool`.
- `async validate_uploads(files: list[UploadFile], tmpdir: Path) -> list[Upload]`
  - Streams each file to `tmpdir`, counting bytes; over the limit → 413.
  - Classifies by `ffprobe`, not by extension or the client's MIME:
    - Images go through Pillow `verify()` (Pillow is already a transitive dep; add it explicitly).
      Pillow-readable JPEG/PNG/WebP is `image`.
    - Video stream + duration > 0 is `video`.
    - Audio stream only is `audio`.
    - Anything else → 415.
  - Enforces the mix rule: ≤ 4 images, or exactly 1 audio/video. Mixed → 422.
  - Over 60 s → 413 `"Clips must be 60 seconds or shorter."`
- `async probe(path: Path) -> dict` with `{"duration": float, "has_audio": bool, "has_video": bool}`,
  from `ffprobe -v error -show_streams -show_format -of json`.
- `async audio_pcm(path: Path, sr: int = 48000) -> np.ndarray`, float32 mono, from
  `ffmpeg -i path -ac 1 -ar sr -f f32le -`.
- `async extract_frames(path: Path, duration: float, n: int) -> list[bytes]`
  - JPEGs at `t = duration * (i + 0.5) / n`, scaled to max 1024 px wide.
  - Fewer frames if the clip is short.
- All subprocess work goes through `asyncio.create_subprocess_exec`. A non-zero exit raises
  `MediaError(message)`, which the route maps to 422.

- [ ] **Step 1: Write the failing tests.** The `tmp_media` fixture makes files with
  `ffmpeg -f lavfi -i testsrc=duration=3` (+ `sine=frequency=440:duration=3`), a 61 s sine, and a
  text file renamed `fake.mp4`.

```python
async def test_video_with_audio_classified(tmp_media):     # testsrc+sine, 3s
    [u] = await validate_uploads([tmp_media.upload("clip.mp4")], tmp_media.dir)
    assert u.kind == "video" and u.has_audio and 2.5 < u.duration < 3.5

async def test_silent_video_ok(tmp_media):
    [u] = await validate_uploads([tmp_media.upload("silent.mp4")], tmp_media.dir)
    assert u.kind == "video" and not u.has_audio

async def test_fake_mp4_rejected(tmp_media):
    with pytest.raises(HTTPException) as e:
        await validate_uploads([tmp_media.upload("fake.mp4")], tmp_media.dir)
    assert e.value.status_code == 415

async def test_long_audio_rejected(tmp_media):
    with pytest.raises(HTTPException) as e:
        await validate_uploads([tmp_media.upload("long.wav")], tmp_media.dir)
    assert e.value.status_code == 413 and "60 seconds" in e.value.detail

async def test_frames_evenly_spaced_and_jpeg(tmp_media):
    frames = await extract_frames(tmp_media.dir / "clip.mp4", 3.0, 8)
    assert len(frames) == 8 and all(f[:2] == b"\xff\xd8" for f in frames)

async def test_audio_pcm_shape(tmp_media):
    pcm = await audio_pcm(tmp_media.dir / "clip.mp4")
    assert pcm.dtype == np.float32 and abs(len(pcm) / 48000 - 3.0) < 0.1
```

- [ ] **Step 2: Run `uv run pytest tests/unit/test_media.py -v`.** Expected: FAIL. ffmpeg must be on
  PATH locally; if it isn't, install it and say so to the user.
- [ ] **Step 3: Implement.** Add `numpy` and `pillow` to `[project]` explicitly, then run `uv lock`.
- [ ] **Step 4: Run the same command.** Expected: PASS.
- [ ] **Step 5: Commit** `feat: media upload validation and ffmpeg extraction`.

### Task 13: Hearing (Whisper and CLAP)

**Files:**
- Create: `app/services/hearing.py`
- Test: `tests/unit/test_hearing.py`
- Add deps: `faster-whisper`, `transformers` (keep the 4.53.0 pin unless it's incompatible; ask
  before changing it), `torch` from the CPU index.

**Interfaces:**
- `CAR_SOUND_LABELS: list[str]`, about 30 entries. Must include the fault sounds plus three
  catch-alls: `"normal engine idle"`, `"human speech"`, `"background noise"`.
  - Fault sounds: engine knocking, valve ticking, belt squeal, brake grinding, brake squeal,
    hissing (vacuum or coolant leak), whining (power steering or alternator), clunking over bumps,
    rattling heat shield, rough idle / misfire, exhaust leak puffing, starter clicking, CV joint
    clicking when turning, wheel bearing hum, and so on.
- `async transcribe(path: Path) -> str`
  - faster-whisper `WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")`, loaded lazily
    and cached in a module global.
  - Joins the segment texts. Returns `""` when nothing is spoken.
  - Runs in `asyncio.to_thread`.
- `async tag_sounds(pcm: np.ndarray, sr: int = 48000, top_k: int = 3) -> list[SoundTag]`
  - Uses the CLAP model and processor (`CLAP_MODEL`), loaded lazily.
  - Algorithm:
    1. Split into 10 s windows. The last partial window counts if it's ≥ 2 s; a clip under 2 s is
       one window.
    2. Compute `softmax(logits_per_audio)` per window.
    3. Take each label's max across windows.
    4. Drop `"human speech"` and `"background noise"`.
    5. Keep labels with score ≥ `SOUND_MIN_SCORE`, sorted desc, top `top_k`.
  - Check the installed transformers' `ClapProcessor` argument name (`audios=` vs `audio=`).
- `prefetch() -> None`: downloads both models into the HF/CT2 cache. Run with
  `uv run python -m app.services.hearing` (`__main__` calls `prefetch()`) for the Docker build.

- [ ] **Step 1: Write the failing tests.** Models are stubbed through `mocker.patch("app.services.hearing._clap")`,
  which returns a fake `(model, processor)`.

```python
async def test_tag_sounds_max_over_windows_and_threshold(fake_clap):
    # 25 s of audio → 3 windows; fake logits make "belt squeal" 0.7 in window 2 only, "background noise" 0.9 everywhere
    tags = await tag_sounds(np.zeros(48000 * 25, dtype=np.float32))
    assert tags[0].label == "belt squeal" and tags[0].score == pytest.approx(0.7)
    assert all(t.label not in ("human speech", "background noise") for t in tags)

async def test_tag_sounds_empty_when_nothing_clears_threshold(fake_clap_flat):
    assert await tag_sounds(np.zeros(48000 * 5, dtype=np.float32)) == []

@pytest.mark.slow
async def test_real_models_on_sine(tmp_media):   # opt-in: uv run pytest -m slow
    assert isinstance(await transcribe(tmp_media.dir / "clip.mp4"), str)
```

- [ ] **Step 2: Run `uv run pytest tests/unit/test_hearing.py -v -m "not slow"`.** Expected: FAIL.
- [ ] **Step 3: Implement.** Then run `uv lock && uv sync`, and confirm `uv tree | grep torch` shows
  the `+cpu` build.
- [ ] **Step 4: Run the same command.** Expected: PASS. Then run `uv run pytest -m slow` once and
  expect PASS. Record the CPU time for a 60 s clip (transcribe + tag) in the commit body against the
  ~30 s target.
- [ ] **Step 5: Commit** `feat: local speech transcription and car-sound tagging`.

### Task 14: Wire media into the turn

**Files:**
- Modify: `app/services/diagnosis.py`, `app/api/v1/session_route.py`
- Modify: `tests/cases/diagnosis_cases.yaml`, adding the media cases
- Test: `tests/unit/test_turn_media.py`

**Interfaces:**
- **In the send route:** run `validate_uploads` into a `TemporaryDirectory` before streaming starts,
  so validation errors are normal 4xx JSON and never mid-stream. Upload each original to Storage at
  `{MEDIA_BUCKET}/{user_id}/{session_id}/{message_id}/{n}{ext}`.
- **In `run_turn`**, before `searching`:
  - Emit `("stage", {"name": "processing_media"})`.
  - Run these in parallel with `asyncio.gather`:
    - images → bytes
    - video → `extract_frames(path, duration, MAX_FRAMES)`, plus, if `has_audio`, `audio_pcm` →
      `tag_sounds` and `transcribe`
    - audio → `audio_pcm` → `tag_sounds`, plus `transcribe`
  - The result is an `Observations` object, saved on the user `Message` along with the `media` paths.
  - The KB search query is `text + " " + transcript + " " + sound labels`.
- **`DELETE /sessions/{id}`** also removes `{MEDIA_BUCKET}/{user_id}/{session_id}/` from Storage.

- [ ] **Step 1: Write the failing tests.** Hearing, media extraction, the LLM and the store are all
  faked.

```python
async def test_video_turn_emits_media_stage_and_sends_frames(fake_all, session, car, video_upload):
    events = [(e, d) async for e, d in run_turn(session, car, None, [video_upload])]
    assert events[0] == ("stage", {"name": "processing_media"})
    sent = fake_all.llm.last_messages[-1]["content"]
    assert sum(b["type"] == "image_url" for b in sent) == 8
    assert "belt squeal (0.70)" in sent[0]["text"]

async def test_silent_video_skips_hearing(fake_all, session, car, silent_video_upload):
    _ = [x async for x in run_turn(session, car, "leak under engine", [silent_video_upload])]
    fake_all.hearing.transcribe.assert_not_called()

async def test_media_paths_saved_on_user_message(fake_all, session, car, video_upload):
    _ = [x async for x in run_turn(session, car, None, [video_upload])]
    user_msg = fake_all.store.messages(session["id"])[0]
    assert user_msg["media"][0]["kind"] == "video" and user_msg["observations"]["sounds"]

def test_validation_error_is_json_not_stream(client, auth_headers, existing_session, fake_mp4):
    r = client.post(f"/api/v1/sessions/{existing_session['id']}/messages", files={"files": fake_mp4}, headers=auth_headers)
    assert r.status_code == 415 and r.headers["content-type"].startswith("application/json")
```

- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them.** Expected: PASS. Then run `uv run pytest -q` and expect all tests to pass.
- [ ] **Step 5: Add the media cases** to the YAML.
  - Cases: `photo-leak` (photo of fluid under the car), `audio-belt-squeal`, `video-engine-knock`.
  - Their media files go in `tests/fixtures/e2e/`, which is gitignored if large. **Ask the user to
    supply real clips.** Use generated tones only as a pipeline check and label them that way.
  - Run `uv run scripts/run_cases.py --only photo-leak,audio-belt-squeal,video-engine-knock` and hand
    the report to the user.
- [ ] **Step 6: Commit** `feat: photo, audio and video diagnosis`.

---

## Phase 5 — Ship

### Task 15: Docker and compose

**Files:**
- Rewrite: `Dockerfile`, `docker-compose.yml`, `.dockerignore`

**Interfaces:**
- **Dockerfile:**
  1. Start `FROM python:3.12-slim`.
  2. `apt-get install --no-install-recommends ffmpeg`.
  3. `COPY --from=ghcr.io/astral-sh/uv:0.8.18 /uv /uvx /bin/`.
  4. Copy `pyproject.toml uv.lock`, then `RUN uv sync --frozen --no-dev --no-install-project` as a
     cached layer.
  5. Copy the app, then run `uv sync --frozen --no-dev`.
  6. `RUN uv run python -m app.services.hearing`, which bakes the models in.
  7. Run as a non-root user.
  8. `CMD ["uv", "run", "--no-sync", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]`.
- **Compose:** one service `api`, with `env_file: .env`, port `8000:8000`, and a healthcheck of
  `python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"`.
  No redis, milvus, etcd or minio.
- **`.dockerignore`:** `.venv`, `.env*`, `car_data/`, `tests/`, `reports/`, `*.png`, `result.json`, `__pycache__`.

- [ ] **Step 1: Build.** Run `docker compose build`. Expected: success. Note the image size; with CPU
  torch it should be < ~3 GB. If it's much bigger, the CUDA torch slipped in, so fix the uv index.
- [ ] **Step 2: Start it.** Run `docker compose up -d` and wait. `docker compose ps` should show
  `healthy`, and `curl localhost:8000/health` should return `{"status":"ok","supabase":true,"deepseek":true}`.
- [ ] **Step 3: Commit** `chore: uv-based Docker image and single-service compose`.

### Task 16: End-to-end script

**Files:**
- Create: `scripts/e2e.py`

**Interfaces:** `uv run scripts/e2e.py --base http://localhost:8000 --email … --password … [--media-dir tests/fixtures/e2e]` does the following:
1. Logs in.
2. Adds the Toyota Echo 2001 and polls `GET /cars` until `kb_status != "pending"`, up to 5 min.
3. Creates a session.
4. Sends 4 messages: text, photo, audio and video.
5. For each message, asserts ≥ 1 `token` event and exactly 1 `state` event, and prints
   time-to-first-token and the total time.
6. Deletes the session.
7. Exits non-zero on any failure.

- [ ] **Step 1: Implement** with `httpx` streaming.
- [ ] **Step 2: Run against the compose stack.** Expected: 4× OK lines with timings and exit 0. Report
  the timings to the user against the targets (≤ ~10 s to first text; ≤ ~30 s media preprocessing).
- [ ] **Step 3: Commit** `test: end-to-end check against running stack`.

### Task 17: Docs and final verification

**Files:**
- Rewrite: `README.md`
- Delete: `postman/`, `tests/README.md`, `tests/run_tests.py`
- Modify: `.gitignore` (add `reports/`, `tests/fixtures/e2e/`)

- [ ] **Step 1: Rewrite the README.** Cover, in this order:
  - What it does.
  - The `.env` variables, as the `Settings` field list with no values.
  - Supabase setup: run `app/db/sql/001_v2_schema.sql`.
  - `uv sync`, `uv run scripts/ingest_kb.py`, `uv run uvicorn main:app --reload` and
    `docker compose up`.
  - The API overview, with a link to `/docs` and the SSE event types `stage|token|state|error|done`
    with one example stream.
  - Media limits, `uv run pytest`, `scripts/run_cases.py` and `scripts/e2e.py`.

  Keep it short; OpenAPI at `/docs` is the reference.
- [ ] **Step 2: Final verification.**
  - `uv run pytest -q` passes.
  - `docker compose up` is healthy.
  - `scripts/e2e.py` exits 0.
  - `grep -rn "langchain\|milvus\|redis\|celery\|websocket" app main.py` finds nothing.
- [ ] **Step 3: Commit** `docs: rewrite README for v2`.
