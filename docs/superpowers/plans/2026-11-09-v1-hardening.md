# AutoCare Backend v1 Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tighten version 1 (security, stability, performance, hygiene) without adding features.

**Architecture:** Twelve independent tasks, most severe first. Each one changes a small set of files, lands with its own tests and README lines, and leaves the suite green, so the work can stop after any task and still be shippable. Nothing changes the database schema, and no runtime dependency is added.

**Tech Stack:** Python 3.12, FastAPI, asyncpg, uv, pytest, Docker Compose, Postgres 16.

**Spec:** `docs/superpowers/specs/2026-11-09-v1-hardening-design.md` (the review, the owner's decisions and what was left out). Finding ids such as S3 or T1 below refer to its tables.

## Global Constraints

- No new features, no schema changes, no new runtime dependencies. `ruff` is added to the `dev` group only.
- The deployment is one VPS and one uvicorn worker. In-memory counters stay in memory.
- transformers stays at `4.53.0`.
- Every error response keeps the body `{"error_code", "message", "correlation_id", "timestamp", ...}`.
- Logs carry class names, ids and fixed texts only: never request data, tokens, URLs with signatures or exception messages from argon2 or Google.
- No prompt text changes and no extra LLM call in a turn.
- Work on branch `chore/v1-hardening`. Small conventional commits (`fix:`, `chore:`, `test:`, `docs:`), no `Co-Authored-By` line.
- Baseline before starting: `uv run pytest -q` reports 1127 passed, 2 skipped, 49 deselected. Every task ends with the full offline suite green.
- `.env*` files are never read. The owner adds new setting names to `.env.example` (listed in Task 12).
- Stop rule: when two attempts at a step fail, stop and report what was tried.

## Review Focus

Failure modes the spec implies that no task would otherwise test. Each has a test in the task named.

1. **A full garage and a car already in it.** Re-adding a car the user already owns must still answer `200` with that car, not the "garage is full" `422`. (Task 3)
2. **An expired access token on an upload.** The early check must answer `401` with the message `Token has expired`, the same as the route does, so the client refreshes instead of signing in again. (Task 2)
3. **`Authorization: bearer <token>` in lower case.** The route accepts any case of the scheme today; the early check must too. (Task 2)
4. **A session saved before the caps, holding more than 30 findings.** `GET /sessions/{id}` must answer `200` with the newest 30, not a validation error. (Task 8)
5. **Google rotates its signing keys inside the cache window.** A token signed by a key missing from the cached set must trigger one fresh download, not an hour of `401`. (Task 6)

---

### Task 1: Dependency upgrades (S1)

**Files:**
- Modify: `pyproject.toml`, `uv.lock`
- Test: the existing suite

**Interfaces:**
- Produces: `pypdf==7.0.0`, `PyJWT==2.15.1`, `python-multipart==0.0.33`, `fastapi==0.162.2` (which brings `starlette` 1.10.0), `python-dotenv>=1.2.2` in the lock. These versions resolved together on 2026-11-09 with `slowapi==0.1.9` and `pydantic==2.11.4` unchanged.

- [ ] **Step 1: Create the branch**

Run: `git checkout -b chore/v1-hardening`

- [ ] **Step 2: Low-risk upgrades**

In `pyproject.toml` set `pypdf==7.0.0`, `PyJWT==2.15.1`, `python-multipart==0.0.33` and delete the `chardet==5.2.0` and `tenacity==9.1.2` lines (nothing imports them). Then:

Run: `uv lock --upgrade-package python-dotenv && uv sync`
Run: `uv run pytest -q`
Expected: 1127 passed, 2 skipped.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: upgrade pypdf, PyJWT, python-multipart and python-dotenv for known advisories; drop unused chardet and tenacity"
```

- [ ] **Step 4: FastAPI and starlette**

Set `fastapi==0.162.2` in `pyproject.toml`.

Run: `uv lock && uv sync && uv run pytest -q`
Expected: the same counts. Fix what the upgrade breaks in the app, never by loosening a test's assertion. Areas to expect: the 422 body built in `main.py:invalid_fields`, `File([])` on the chat route, `BaseHTTPMiddleware` behaviour around the SSE stream, `slowapi`'s use of starlette internals.

If the suite cannot be made green in two attempts, revert this step only (`git checkout pyproject.toml uv.lock && uv sync`), keep Step 3's commit and report the failures.

- [ ] **Step 5: Confirm the advisories are gone**

```bash
uv export --frozen --no-dev --no-hashes --no-emit-project | grep -v "^torch==" > /tmp/req.txt
uvx pip-audit -r /tmp/req.txt --no-deps --disable-pip
```
Expected: findings for `transformers` only.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: upgrade fastapi to 0.162.2 (starlette 1.10.0) for known advisories"
```

---

### Task 2: Check the bearer token before an upload is read (S3)

**Files:**
- Modify: `app/utils/auth_middleware.py`, `app/utils/body_limit.py`, `main.py`, `README.md` (Media section: bodies are capped and the token checked before the upload is read)
- Test: `tests/unit/test_body_limit.py`, `tests/unit/test_auth_middleware.py`

**Interfaces:**
- Produces: `access_claims(token: str | None) -> dict` in `app/utils/auth_middleware.py`. Returns the verified payload of an access token. Raises `AuthenticationError` with exactly the details the route uses today: `Missing authentication token` (None or empty), `Token has expired`, `Invalid token`, `Invalid token type`, `Invalid token payload` (no `sub`, or `iat` not a number). No database access.
- Produces: `BodySizeLimitMiddleware(app, max_bytes, upload_max_bytes, upload_path, upload_auth: Callable[[str | None], object] | None = None)`. When `upload_auth` is set and the path matches `upload_path`, the middleware calls it with the bearer token (or None) after the declared `Content-Length` check and before the first `receive()`. An `HTTPException` from it is answered at once as JSON `http_error_body(exc.status_code, exc.detail)` plus the correlation id and `exc.headers`.

- [ ] **Step 1: Write the failing tests** in `tests/unit/test_body_limit.py`, with a second app factory that passes `upload_auth=access_claims`, driven through the file's existing `_call` helper:

```python
async def test_upload_without_a_token_is_401_and_the_body_is_never_read():
    status, sent, receives = await _call(_auth_app([]), UPLOAD, [], [b"x" * 50])
    assert status == 401 and receives == 0
    assert _json(sent)["error_code"] == "AUTH_001"
    assert _json(sent)["message"] == "Missing authentication token"
    assert (b"www-authenticate", b"Bearer") in sent[0]["headers"]

async def test_upload_with_an_expired_token_says_so():        # Review Focus 2
    ...  # token minted with issued_at 2 hours ago -> 401, message "Token has expired", receives == 0

async def test_upload_with_a_refresh_token_is_401():           # message "Invalid token type"

async def test_upload_accepts_a_lower_case_bearer_scheme():    # Review Focus 3
    ...  # header b"bearer <valid access token>" -> 200 and the body is read

async def test_oversized_upload_is_still_413_without_a_token():  # declared length is checked first

async def test_json_routes_are_not_checked():                   # "/json" without a token still reaches the app
```

and in `tests/unit/test_auth_middleware.py` one test per `access_claims` message listed above.

- [ ] **Step 2: Run them and see them fail**

Run: `uv run pytest tests/unit/test_body_limit.py tests/unit/test_auth_middleware.py -q`
Expected: FAIL (`access_claims` and `upload_auth` do not exist).

- [ ] **Step 3: Implement `access_claims`** and make `get_current_user` call it in place of its inline verify, type and payload checks. Its audit logging and the `validate_user` database check stay in `get_current_user`.

- [ ] **Step 4: Implement `upload_auth` in the middleware.** Read the header from `scope["headers"]`, split once on a space, compare the scheme case-insensitively. Generalise `_reject` to take the exception.

- [ ] **Step 5: Wire it in `main.py`**: pass `upload_auth=access_claims` to `BodySizeLimitMiddleware`.

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest -q`
Expected: all pass. Chat-route tests that post files already send `auth_headers`.

- [ ] **Step 7: Commit**

```bash
git commit -am "fix: check the access token before a chat upload is read from the socket"
```

---

### Task 3: Limits on adding cars and opening sessions (S2)

**Files:**
- Modify: `app/core/config.py`, `app/services/usage_limits.py`, `app/services/user_profile_service.py`, `app/api/v1/car_route.py`, `app/api/v1/session_route.py`, `app/CRUD/car_crud.py`, `tests/conftest.py` (reset the new counter), `README.md` (Abuse protection, API)
- Test: `tests/unit/test_car_intake.py`, `tests/unit/test_car_ingest.py`, `tests/unit/test_session_routes.py`

**Interfaces:**
- Produces: settings `MAX_CARS_PER_USER: int = 20` and `DAILY_CAR_ADD_LIMIT: int = 20`; `0` turns either off.
- Produces: `usage_limits.count_daily_car_add(user_id: str) -> None`. Same behaviour as `count_daily_message` (429 `EXT_003`, `Retry-After` until midnight UTC, message `Daily limit of {limit} added cars reached. Try again after midnight UTC.`), counted in its own dict `_car_adds` with the same `(user_id, date)` keys. Both functions share one private helper; `_counts` keeps its shape.
- Produces: `UserProfileService.garage_ids(user_id: str) -> list[str]`, one query, `[]` without a profile.
- Produces: `car_crud._ingesting: set[str]`; `ingest_car_manual` returns at once when the car id is already in it.

Behaviour of `POST /cars`, in this order: per-address limit `10/minute`; garage cap, `422` `BIZ_001` `A garage holds at most {n} cars. Remove one to add another.`, skipped when the car is already in the garage; daily count; then today's flow. `POST /sessions` gets `30/minute`.

- [ ] **Step 1: Write the failing tests**

```python
def test_add_car_is_rate_limited_per_address(client, auth_headers, mocker):
    # 10 accepted, the 11th within a minute is 429 with error_code EXT_003

def test_full_garage_refuses_a_new_car(client, auth_headers, mocker, monkeypatch):
    # MAX_CARS_PER_USER=2, garage_ids -> ["a-b-2001", "c-d-2002"]; POST Toyota Echo 2001
    # -> 422, error_code BIZ_001, add_car_to_user never called

def test_full_garage_still_returns_a_car_already_in_it(...):      # Review Focus 1
    # garage_ids -> ["toyota-echo-2001", "c-d-2002"]; POST Toyota Echo 2001 -> 200

def test_daily_car_add_limit_is_429_with_retry_after(...):
    # DAILY_CAR_ADD_LIMIT=1; second POST -> 429, Retry-After header is seconds to midnight UTC

def test_daily_car_add_limit_zero_turns_it_off(...)

async def test_second_ingest_of_the_same_car_is_a_no_op_while_the_first_runs(mocker):
    # scrape_links blocks on an asyncio.Event; a second ingest_car_manual(car) returns at once;
    # scrape_links was awaited exactly once; after release "toyota-echo-2001" not in car_crud._ingesting

def test_create_session_is_rate_limited(...):   # 31st within a minute is 429
```

- [ ] **Step 2: Run them and see them fail**

Run: `uv run pytest tests/unit/test_car_intake.py tests/unit/test_car_ingest.py tests/unit/test_session_routes.py -q`

- [ ] **Step 3: Implement** in the order of the Interfaces block. The route computes the candidate id with `CarCRUD.unique_logic(make, model, year)`. `_ingesting` is filled and emptied in `try/finally` around the whole body of `ingest_car_manual`. Add `usage_limits._car_adds.clear()` to `conftest._reset_rate_limits`.

- [ ] **Step 4: Run the full suite**, then commit:

```bash
git commit -am "fix: rate limit, garage cap and daily cap on adding cars; one manual ingest per car at a time; rate limit on opening sessions"
```

---

### Task 4: Download the startmycar manual through the guarded fetcher (S4)

**Files:**
- Modify: `app/services/manual_search.py`, `app/CRUD/car_crud.py`, `app/services/fetch_car_data_service.py`, `app/services/user_profile_service.py`, `pyproject.toml`, `uv.lock`
- Delete: `app/services/base_service.py`
- Test: `tests/unit/test_manual_search.py`, `tests/unit/test_car_ingest.py`

**Interfaces:**
- Produces: `manual_search.fetch_pdf(url: str, dest: Path, transport: httpx.AsyncBaseTransport | None = None) -> bool`. True when `dest` holds a body that starts with `%PDF-`, was fetched from public addresses only (checked before every hop, at most `MAX_REDIRECTS`), answered `200` and stayed within `MANUAL_MAX_MB`. It is the first half of today's `download_manual`, which becomes `fetch_pdf(...)` followed by `_looks_right(...)`.
- Produces: `FetchCarDataService` reduced to `build_url`, `scrape_links(url)` (the owner-manual branch, returning the link or `""`), `_get_client` and `cleanup`. It no longer inherits from `BaseService`.

- [ ] **Step 1: Write the failing tests**

```python
async def test_fetch_pdf_refuses_a_private_address(tmp_path):         # http://10.0.0.5/x.pdf -> False, no request sent
async def test_fetch_pdf_refuses_a_body_that_is_not_a_pdf(tmp_path):  # 200 with b"<html>" -> False
async def test_fetch_pdf_stops_at_the_size_cap(tmp_path, monkeypatch) # MANUAL_MAX_MB=1, 2 MB body -> False
async def test_fetch_pdf_rechecks_the_address_after_a_redirect(tmp_path)  # public -> 302 to 127.0.0.1 -> False

async def test_startmycar_manual_is_fetched_with_fetch_pdf(mocker, update):
    # scrape_links -> "/files/echo.pdf"; fetch_pdf patched to return True
    # fetch_pdf awaited with "https://www.startmycar.com/files/echo.pdf"

async def test_startmycar_manual_that_fails_the_fetch_is_no_manual(mocker, update):
    # fetch_pdf -> False; final status "no_manual"; kb.ingest_pdf never called

def test_build_url_quotes_make_and_model():
    urls = FetchCarDataService().build_url("a/b?x", "c#d", 2001)
    assert urls["Owner_Manual"] == "https://www.startmycar.com/a%2Fb%3Fx/c%23d/info/manuals/2001"
```

Existing tests that patch `FetchCarDataService.download_pdf` (`test_car_ingest.py` lines 58, 70, 78) patch `app.CRUD.car_crud.manual_search.fetch_pdf` to return True instead.

- [ ] **Step 2: Run them and see them fail**

- [ ] **Step 3: Implement.** Split `download_manual`; call `fetch_pdf` from `_startmycar_manual` and return None when it is False; quote the two path segments in `build_url` with `urllib.parse.quote(segment, safe="")` after the existing lower-casing and space replacement; delete `download_pdf`, `scrape_links_with_retry`, the guides branch, `perform_action`, `_perform_action_impl`, the retry constructor arguments and the unused `settings` local; delete `base_service.py` and `UserProfileService.perform_action`; rename the ingest temp prefix to `autocare-manual-` (Task 7 sweeps it).

- [ ] **Step 4: Remove `aiofiles`** from `pyproject.toml`, then `uv lock && uv sync`.

- [ ] **Step 5: Run the full suite**, then commit:

```bash
git commit -am "fix: fetch the startmycar manual through the size-capped, public-address-only downloader; delete the unused scraping and cache code"
```

---

### Task 5: Logging off the thread pool, and a pool that media work cannot fill (T1)

**Files:**
- Modify: `app/utils/logger.py`, `main.py`
- Test: `tests/unit/test_app.py`

**Interfaces:**
- Produces: `Logger.info/warning/error/debug/critical` stay `async def` (about 150 call sites await them) and call the stdlib logger directly. `Logger._lock` is gone.
- Produces: `main.DEFAULT_EXECUTOR_THREADS = 32`; the lifespan installs a `ThreadPoolExecutor` of that size as the loop's default executor.

- [ ] **Step 1: Write the failing test**

```python
async def test_logging_never_waits_for_the_thread_pool(monkeypatch, caplog):
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(loop, "run_in_executor", lambda *a, **k: pytest.fail("logging used the executor"))
    with caplog.at_level(logging.INFO):
        await get_logger_instance("pool-test").info("hello")
    assert "hello" in caplog.text
```

- [ ] **Step 2: Run it and see it fail**

Run: `uv run pytest tests/unit/test_app.py::test_logging_never_waits_for_the_thread_pool -q`

- [ ] **Step 3: Implement.** Each logger method becomes one direct call with the same `extra={"log_type": ...}`. In the lifespan, before `init_pool`: `asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=DEFAULT_EXECUTOR_THREADS, thread_name_prefix="autocare"))`.

- [ ] **Step 4: Run the full suite**, then commit:

```bash
git commit -am "fix: log without a thread hop or a global lock; give the default executor 32 threads so media work cannot starve logins and DNS"
```

---

### Task 6: Cache Google's certificates and bound the fetch (T2)

**Files:**
- Modify: `app/services/auth_service.py`
- Test: `tests/unit/test_auth_service.py`

**Interfaces:**
- Produces: `auth_service._CertRequest`, a callable with google-auth's transport signature `(url, method="GET", **kwargs)`. It forwards to one `google.auth.transport.requests.Request` with `timeout=GOOGLE_HTTP_TIMEOUT_SECONDS` (10) and keeps the last response with status 200 per URL for `GOOGLE_CERTS_TTL_SECONDS` (3600) on the monotonic clock. `_CertRequest.clear() -> bool` empties the cache and says whether it held anything.
- Produces: `_verify_google_token` passes the module's single `_CertRequest`. When verification raises `ValueError` and `clear()` returns True, it verifies once more.

- [ ] **Step 1: Write the failing tests** against `_CertRequest` built around a counting fake transport:

```python
def test_certs_are_fetched_once_within_the_ttl(monkeypatch)          # two calls, one inner call
def test_certs_are_fetched_again_after_the_ttl(monkeypatch)          # monotonic advanced 3601 s -> two inner calls
def test_a_failed_cert_fetch_is_not_cached()                         # status 500 twice -> two inner calls
def test_the_cert_fetch_has_a_10_second_timeout()                    # inner call received timeout=10
def test_an_unknown_key_refetches_the_certs_once(mocker)             # Review Focus 5
    # verify_oauth2_token raises ValueError then returns claims, cache was warm
    # -> claims returned, verify called twice, cache was cleared between
def test_a_bad_token_with_a_cold_cache_is_not_retried(mocker)        # ValueError, clear() False -> one verify call
```

- [ ] **Step 2: Run them and see them fail**

Run: `uv run pytest tests/unit/test_auth_service.py -q`

- [ ] **Step 3: Implement**, then run the full suite and commit:

```bash
git commit -am "fix: cache Google's signing certificates for an hour and cap the fetch at 10 s"
```

---

### Task 7: Compose guard rails and a clean shutdown (T3, T4)

**Files:**
- Modify: `docker-compose.yml`, `Dockerfile`, `main.py`, `app/api/v1/session_route.py`, `README.md` (VPS setup: log rotation, shutdown behaviour, rollback note)
- Test: `tests/unit/test_compose.py`, `tests/unit/test_app.py`, `tests/unit/test_session_routes.py`

**Interfaces:**
- Produces in `docker-compose.yml`:
  - all three services: `logging: {driver: json-file, options: {max-size: "10m", max-file: "5"}}` through one `x-logging` anchor
  - `db`: `oom_score_adj: -500`
  - `api` and `visuals-worker`: `oom_score_adj: 500`, `security_opt: ["no-new-privileges:true"]`, `cap_drop: ["ALL"]`, `pids_limit: 512`
  - `api`: `stop_grace_period: 60s`
- Produces in `Dockerfile`: the CMD gains `"--timeout-graceful-shutdown", "45"`; the comment above it reads `One worker: rate limits, the daily caps and the media slots are counted in this process's memory (see README, Workers).`
- Produces: `session_route.drain_releases() -> None`, awaits every task in `_releasing` and swallows their errors. The lifespan awaits it before `close_pool()`.
- Produces: `main.sweep_stale_tmp() -> int`, removes directories named `autocare-turn-*` and `autocare-manual-*` in `tempfile.gettempdir()` and returns how many; called first in the lifespan.

- [ ] **Step 1: Write the failing tests**

```python
def test_every_service_rotates_its_logs(compose):
    for name in ("db", "api", "visuals-worker"):
        assert compose["services"][name]["logging"]["options"] == {"max-size": "10m", "max-file": "5"}

def test_the_database_outlives_the_api_under_memory_pressure(compose):
    s = compose["services"]
    assert s["db"]["oom_score_adj"] == -500 and s["api"]["oom_score_adj"] == 500

def test_app_containers_drop_capabilities(compose)       # cap_drop, security_opt, pids_limit on api and worker; db untouched
def test_api_has_time_to_finish_replies_on_stop(compose) # stop_grace_period == "60s"; Dockerfile CMD holds the flag and 45

def test_startup_removes_stale_turn_directories(monkeypatch, tmp_path):
    # tempfile.tempdir -> tmp_path; dirs autocare-turn-a, autocare-manual-b, other-c
    # sweep_stale_tmp() == 2; other-c still exists

async def test_shutdown_waits_for_pending_lease_releases(fake_store, session):
    # hold a lease, call _release(...), then drain_releases(); the store shows the lease free
```

`test_dockerfile_cmd_runs_one_uvicorn_worker_behind_proxy_headers` is updated for the longer CMD.

- [ ] **Step 2: Run them and see them fail**

- [ ] **Step 3: Implement.**

- [ ] **Step 4: Validate the compose file**

Run: `POSTGRES_PASSWORD=x docker compose config -q`
Expected: no output, exit 0.

- [ ] **Step 5: Run the full suite**, then commit:

```bash
git commit -am "fix: rotate container logs, protect Postgres from the OOM killer, drop capabilities, and let replies finish on shutdown"
```

Not verified by this task: the containers are not built or started here (the image bakes two models and is several GB). The README rollback note says which four lines to remove if a container fails to start with `cap_drop`.

---

### Task 8: Caps on data that grows without bound (T6, T7, P2, P3)

**Files:**
- Modify: `app/schemas/session.py`, `app/api/v1/session_route.py`, `app/CRUD/chat_session_crud.py`, `app/CRUD/car_crud.py`, `app/services/user_profile_service.py`, `README.md` (API: session list, session size)
- Test: `tests/unit/test_state.py`, `tests/unit/test_session_routes.py`, `tests/unit/test_user_profile_service.py`

**Interfaces:**
- Produces: `MAX_ESTABLISHED = 30` and `MAX_CAUSES = 8` in `app/schemas/session.py`. `DiagnosisState` keeps the newest 30 of each of `confirmed_findings`, `tests_done`, `ruled_out`, `topic_history` (the tail of the list) and the first 8 `suspected_causes`.
- Produces: `MAX_SESSION_MESSAGES = 400` in `session_route.py`. A new message to a session that already holds that many answers `422` `BIZ_001` `This session is full. Start a new session to continue.` A retry is not refused.
- Produces: `SESSION_LIST_LIMIT = 100` in `chat_session_crud.py`; `list_for_user` appends `LIMIT 100`.
- Produces: `CarCRUD.get_many(ids: list[str]) -> list[dict]`, one `WHERE id = ANY($1::text[])` query, rows returned in the order of `ids`, unknown ids left out. `get_user_cars` uses it.

- [ ] **Step 1: Write the failing tests**

```python
def test_established_lists_keep_the_newest_30():
    state = DiagnosisState(confirmed_findings=[f"f{i}" for i in range(50)])
    assert state.confirmed_findings == [f"f{i}" for i in range(20, 50)]

def test_merge_state_cannot_grow_past_the_caps()          # 30 previous + 5 new findings -> 30, the 5 new ones kept

def test_get_session_with_an_oversized_legacy_state_is_200(...)   # Review Focus 4: 50 stored findings -> 200, 30 returned

def test_a_full_session_refuses_a_new_message_and_frees_the_lease(...)
    # 400 stored messages; POST -> 422 BIZ_001; fake_store.busy(session_id) is False; nothing appended

def test_a_full_session_can_still_retry_its_last_message(...)

async def test_session_list_is_limited_to_100(fake_pool)  # "LIMIT 100" in the SQL

async def test_listing_cars_is_three_statements_whatever_the_garage_size(fake_pool):
    # profile with 5 car ids -> len(fake_pool.calls) == 3 (profile, details, cars)

async def test_get_many_keeps_the_garage_order(fake_pool) # rows returned b, a; ids [a, b] -> result a, b
```

- [ ] **Step 2: Run them and see them fail**

- [ ] **Step 3: Implement.** The session-size check sits in `_start_turn` after the re-read under the lease, inside the existing `try`, so the lease and the media slot are released like any other error.

- [ ] **Step 4: Run the full suite**, then commit:

```bash
git commit -am "fix: cap the diagnosis state lists, messages per session and the session list; load a garage's cars in one query"
```

---

### Task 9: Response hygiene (S5, S6, S7, P4)

**Files:**
- Modify: `app/api/v1/session_route.py`, `app/utils/security_middleware.py`, `app/core/config.py`, `README.md` (API: session fields)
- Test: `tests/unit/test_session_routes.py`, `tests/unit/test_security_headers.py`

**Interfaces:**
- Produces: `POST /sessions` and `GET /sessions/{id}` bodies without `user_id`, `busy_until` and `busy_by`.
- Produces: `SecurityHeadersMiddleware(app)` as a pure ASGI class (same pattern as `CorrelationIdMiddleware`). Headers on every response:
  - `X-Content-Type-Options: nosniff`, `Referrer-Policy: strict-origin-when-cross-origin`, `Permissions-Policy: geolocation=(), microphone=(), camera=()`
  - `X-Frame-Options: DENY` and `Content-Security-Policy: default-src 'none'; frame-ancestors 'none'`, both skipped for the docs paths when `ENABLE_DOCS` is on
  - `Strict-Transport-Security: max-age=31536000; includeSubDomains` when `get_settings().is_production()`
  - `Cache-Control: no-store` unless the response set its own
  
  Removed: `X-CSP-Nonce`, `X-XSS-Protection`, the `public, max-age=300` default.
- Produces: `Settings` refuses `ENV=production` with `DEBUG=true` (`ValueError` from a model validator).

- [ ] **Step 1: Write the failing tests**

```python
def test_session_json_has_no_internal_fields(...)   # create and get: keys do not include user_id, busy_until, busy_by

def test_health_is_never_cached(client):            # Cache-Control == "no-store"
def test_a_routes_own_cache_control_wins(...)       # SSE keeps "no-cache, no-store"; artwork keeps "private, max-age=..."
def test_no_nonce_or_xss_header(client)
def test_hsts_follows_the_settings_not_the_environment(client, monkeypatch)  # settings ENV=production -> header present
def test_production_refuses_debug(monkeypatch):     # Settings(ENV="production", DEBUG=True) raises ValidationError
```

`test_security_headers_on_health` and `test_csp_nonce_header_matches_csp` are rewritten to the new header set. Tests that assert the old long value on cars, sessions or auth responses (`no-store, no-cache, must-revalidate, proxy-revalidate`) now assert `no-store`.

- [ ] **Step 2: Run them and see them fail**

- [ ] **Step 3: Implement**, run the full suite, commit:

```bash
git commit -am "fix: keep internal columns out of session JSON; no-store by default, HSTS from settings, security headers as pure ASGI; refuse DEBUG in production"
```

---

### Task 10: Start-up work: interrupted ingests and model loading (T5, P1)

**Files:**
- Modify: `app/CRUD/car_crud.py`, `app/core/config.py`, `main.py`, `docker-compose.yml`, `README.md` (Configuration, API: `kb_status` after a restart)
- Test: `tests/unit/test_car_ingest.py`, `tests/unit/test_app.py`, `tests/unit/test_compose.py`

**Interfaces:**
- Consumes: `car_crud._ingesting` from Task 3.
- Produces: `car_crud.resume_pending_ingests() -> None`. One statement `UPDATE "Car" SET kb_status = 'failed' WHERE kb_status = 'pending' RETURNING *`, then `ingest_car_manual` for each row, one after another. Never raises (logs the class name). Marking the row `failed` first means a car whose ingest keeps killing the process is tried once per interruption, not on every start.
- Produces: setting `PRELOAD_MODELS: bool = False`; compose sets `PRELOAD_MODELS: "true"` on `api` only.
- Produces: the lifespan starts two background tasks after `init_pool` and cancels them on shutdown: `resume_pending_ingests()`, and when `PRELOAD_MODELS` is on, `asyncio.to_thread(hearing.prefetch)` wrapped so a failure is logged by class name and nothing else happens.

- [ ] **Step 1: Write the failing tests**

```python
async def test_resume_marks_pending_cars_failed_then_ingests_each(fake_pool, mocker):
    # fake_pool.rows = two cars; ingest_car_manual patched
    # first call's SQL is the UPDATE above; ingest awaited twice, in row order

async def test_resume_survives_a_database_error(fake_pool, mocker, caplog)   # fake_pool.error set -> returns None, logs "RuntimeError"
async def test_resume_continues_after_one_ingest_raises(...)

def test_models_are_not_loaded_at_startup_by_default(mocker, fake_pool)      # hearing.prefetch not called
def test_models_are_loaded_at_startup_when_enabled(mocker, monkeypatch, fake_pool)
def test_a_failed_preload_is_logged_and_the_app_still_starts(...)            # prefetch raises OSError -> /health answers
def test_compose_preloads_models_on_the_api_only(compose)
```

- [ ] **Step 2: Run them and see them fail**

- [ ] **Step 3: Implement.** The lifespan calls it as `main.resume_pending_ingests` (imported by name into `main`). The `client` fixture in `tests/conftest.py` patches `main.resume_pending_ingests` with an `AsyncMock`, next to its existing patches of `main.llm.ping` and `main.check_database`, so no test sees an extra statement on the fake pool or has its rows ingested as cars. Tests in `test_app.py` that start the app themselves patch it the same way.

- [ ] **Step 4: Run the full suite**, then commit:

```bash
git commit -am "fix: retry manual ingests interrupted by a restart once; load the hearing models at start-up in the container"
```

---

### Task 11: Back up the artwork volume (T8)

**Files:**
- Modify: `scripts/backup.sh`, `README.md` (Backup and restore), `docs/VEHICLE_VISUALS.md` (the line that says the volume is not backed up)
- Test: `tests/unit/test_compose.py`

**Interfaces:**
- Produces: a third archive `backups/YYYY-MM-DD-visuals.tar.gz` from the volume mounted at `/data/visuals` in the api container, written through `.partial` and rotated to the newest 14 like the other two. The media and visuals blocks share one shell function `backup_volume <mount> <suffix>`.

- [ ] **Step 1: Extend `test_backup_script_dumps_db_and_media_and_rotates`**

```python
    assert "/data/visuals" in script and "-visuals.tar.gz" in script
    assert script.count("backup_volume ") >= 3   # the definition and two calls
```

- [ ] **Step 2: Run it and see it fail**, implement, then check the script parses:

Run: `bash -n scripts/backup.sh`
Expected: no output.

- [ ] **Step 3: Add the restore command** for the visuals volume to the README, next to the media one, and commit:

```bash
git commit -am "fix: include the artwork volume in the backup"
```

---

### Task 12: Unused code, lint, and documentation drift

**Files:**
- Modify: `app/utils/auth_middleware.py`, `app/utils/exceptions.py`, `app/utils/audit_logging.py`, `app/CRUD/car_crud.py`, `app/CRUD/__init__.py`, `app/schemas/Car.py`, `app/services/hearing.py`, `pyproject.toml`, `uv.lock`, `README.md`
- Test: `tests/unit/test_auth_middleware.py`, `tests/unit/test_exceptions.py` (tests of deleted names are deleted with them)

**Interfaces:**
- Produces: `ruff==0.17.4` in the `dev` group and

```toml
[tool.ruff]
target-version = "py312"

[tool.ruff.lint]
select = ["E9", "F", "B"]
ignore = ["B008"]  # FastAPI's Depends() and File() defaults
```

- [ ] **Step 1: Delete, one commit per file group.** Before each deletion, `grep -rn <name> app scripts main.py tests` and stop if anything outside the name's own definition and its own tests uses it.
  - `auth_middleware.py`: `require_permissions`, `require_role`, `get_optional_user`, `get_current_active_user`, `AuthorizationError`.
  - `exceptions.py`: `InvalidTokenException`, `TokenExpiredException`, `SecurityViolationException`, `ExternalServiceException`, `SystemException`, `ConfigurationException`, `SessionException`, `SessionExpiredException`, `SessionNotFoundException`, `ServiceException`, `AgentException`, `ResourceException`, and `BaseAPIException._get_safe_message`.
  - `audit_logging.py`: `_handle_high_risk_event` and its call, the `session_id` entry of the record.
  - `car_crud.py`: `_normalize_car_id` and the second lookup in `get_car_by_id`, which becomes one `read` and a return.

- [ ] **Step 2: Add ruff and clean the 25 findings**

Run: `uv add --dev ruff==0.17.4 && uv run ruff check --fix .`
Then by hand: `raise ... from None` at the three B904 sites in `auth_middleware.py`, `strict=True` on the `zip` in `hearing.py:151`, an empty `app/CRUD/__init__.py`.

Run: `uv run ruff check .`
Expected: `All checks passed!`

- [ ] **Step 3: Clean the pytest configuration.** In `pyproject.toml` remove `timeout`, `timeout_method` and the markers nothing uses (`integration`, `unit`, `database`, `external`, `parametrize`, `smoke`, `regression`, `performance`); keep `slow`, `db`, `asyncio`.

Run: `uv run pytest -q`
Expected: all pass, and no `PytestConfigWarning` when run with `-W error::pytest.PytestConfigWarning`.

- [ ] **Step 4: Fix the README drift**
  - First paragraph: three LLM calls per turn (the streamed reply, fact extraction alongside it, the state update).
  - API table: add `POST /api/v1/sessions/{id}/messages/{message_id}/retry`; SSE table: the `error` event may carry `retry_message_id`.
  - Development: add `uv run ruff check .`.
  - Configuration: the new optional names `MAX_CARS_PER_USER`, `DAILY_CAR_ADD_LIMIT`, `PRELOAD_MODELS`.

- [ ] **Step 5: Commit**

```bash
git commit -am "chore: delete unused code, add a ruff configuration, fix README drift"
```

- [ ] **Step 6: Final verification of the branch**

```bash
uv run pytest -q
uv run ruff check .
uv export --frozen --no-dev --no-hashes --no-emit-project | grep -v "^torch==" > /tmp/req.txt && uvx pip-audit -r /tmp/req.txt --no-deps --disable-pip
POSTGRES_PASSWORD=x docker compose config -q
```
Expected: suite green, lint clean, advisories for transformers only, compose valid.

## Owner actions after the branch is merged

These are not done by the plan.

1. Add `MAX_CARS_PER_USER=20`, `DAILY_CAR_ADD_LIMIT=20` and `PRELOAD_MODELS=false` to `.env.example` (the implementer does not read `.env*` files).
2. Deploy with `docker compose up -d --build` and check `docker compose ps` shows all three services up. If `api` or `visuals-worker` fails to start, remove their `cap_drop`, `security_opt` and `pids_limit` lines and report it.
3. Run the real-database tests once against a throwaway Postgres (`uv run pytest -q -m db`, see README) and `scripts/e2e.py` against the deployed server, since the FastAPI upgrade is the largest change here.
4. Tell the client's developer: session JSON no longer has `user_id`, `busy_until`, `busy_by`; the session list returns the newest 100; `POST /cars` can answer `422` (garage full) and `429`; a session of 400 messages answers `422` on a new message.
