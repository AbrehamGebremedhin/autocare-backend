# AutoCare backend v1 hardening: review and design

Date: 2026-11-09. Reviewed at commit `8d0af59` on `main`.

## Intent

Version 1 gets no new features. This pass tightens what exists: security, stability, performance and
general hygiene. The review below is the evidence; the decisions section records what the owner chose;
the plan (`docs/superpowers/plans/2026-11-09-v1-hardening.md`) turns the findings into tasks.

How the review was done: every file under `app/`, `main.py`, both SQL files, the Dockerfile, the compose
file and the scripts were read in full. The offline suite was run (1127 passed, 2 skipped, 49 deselected).
`pip-audit` was run against the runtime packages exported from `uv.lock`. Library behaviour that a finding
depends on was checked in the installed packages, and each such finding says so.

## What the system does well

- **Upload handling is defensive.** Bodies are capped before parsing (`app/utils/body_limit.py`). Each file
  has a size cap, clips have a duration cap that is enforced again at decode time, images have a pixel cap,
  ffmpeg runs from an argument list with `-protocol_whitelist file` and a 30 s timeout, and the client's
  file name never reaches a path.
- **Auth is careful.** argon2id, one verify per login whatever the outcome, revocation through
  `users.tokens_valid_after` at microsecond precision, Google linking only when Google is authoritative
  for the address, and a database outage answers `503`, never `401`.
- **Concurrency is correct by construction.** The session lease is one atomic statement on the database
  clock with a holder token and a heartbeat. The visuals worker uses fencing tokens and a guarded publish.
- **Errors and logs are uniform.** One error body shape, a correlation id on every response and log line,
  class names instead of messages in logs, signed URLs redacted from the access log.
- **File stores resist path tricks.** Components are validated, symlinks refused, writes are atomic.
- **Operations basics exist.** Non-root image, database not published, api bound to loopback, backup and
  prune scripts, a health check that reports the database.
- **The test suite is large and fast** (about 1100 offline tests in 28 s) and includes real-database tests
  behind a marker.

## Findings

Severity is my judgement: High means an outsider or a single account can cost money, fill a disk or
degrade the service; Medium means a likely stability or performance problem under normal growth; Low is
hygiene.

### Security

| # | Severity | Finding | Evidence |
| --- | --- | --- | --- |
| S1 | High | Runtime dependencies with known advisories: pypdf 5.4.0 (67), PyJWT 2.10.1 (17), transformers 4.53.0 (13), starlette 0.46.2 (7), python-multipart 0.0.20 (6), python-dotenv 1.1.0 (1). pypdf parses PDFs fetched from web search results, so it is the most exposed. | `pip-audit` on `uv.lock`, 2026-11-09 |
| S2 | High | `POST /api/v1/cars` has no rate limit, no cap on cars per account and accepts any make/model text. Every new car starts a background ingest: one DeepSeek call, up to 5 Brave searches, a startmycar scrape and up to 3 downloads of up to 80 MB. Two requests for the same new car run two ingests. | `app/api/v1/car_route.py:38-50`, `app/CRUD/car_crud.py:192-244` |
| S3 | High | The chat upload is spooled to disk before the bearer token is checked. FastAPI reads the form before it resolves dependencies, so an anonymous client can make the server write 101 MB per request, 20 times a minute per address. | `fastapi/routing.py:247` (form) runs before `:291` (dependencies) in the installed 0.115.12 |
| S4 | Medium | The startmycar manual download has none of the protections the web-manual download has: no size cap, no public-address check, redirects followed freely, and the whole body is held in memory. The URL comes from a link scraped off a third-party page. Make and model are also placed in the URL path unquoted. | `app/services/fetch_car_data_service.py:78-79,181-209` versus `app/services/manual_search.py:138-161` |
| S5 | Low | Session JSON exposes internal columns: `user_id`, `busy_until` and the lease holder token `busy_by`. | `session_route.py:154,166` return the `SELECT *` row |
| S6 | Low | Security headers: `/health` and every non-listed path are served with `Cache-Control: public, max-age=300`; HSTS is decided from `os.getenv("ENV")`, not from settings, so it is missed when `ENV` comes from `.env` outside compose; the middleware stores the current request on the shared instance. | `app/utils/security_middleware.py:25,49,87` |
| S7 | Low | `ENV=production` together with `DEBUG=true` is accepted; FastAPI debug mode returns tracebacks. | `app/core/config.py`, `main.py:57` |

### Stability

| # | Severity | Finding | Evidence |
| --- | --- | --- | --- |
| T1 | High | Every log line is sent to the default thread pool under one global lock. The same pool runs Whisper, CLAP, argon2, pypdf, file copies and DNS lookups, and it has `min(32, cpus + 4)` threads. Two media turns on a 2-CPU host fill it, and then every request that logs, every login and every database reconnect waits for a transcription to finish. | `app/utils/logger.py:29-35`; `asyncio.to_thread` call sites |
| T2 | Medium | Google sign-in downloads Google's signing certificates on every call, with a 120 s timeout, in a pool thread. | `google/oauth2/id_token.py:135`, `google/auth/transport/requests.py:49` |
| T3 | Medium | Compose has no log rotation (Docker's json-file driver grows without bound) and nothing protects Postgres if the api exhausts memory. | `docker-compose.yml` |
| T4 | Medium | The api container gets Docker's default 10 s stop grace and uvicorn has no graceful-shutdown timeout. A deploy kills in-flight replies, and their session leases stay held for up to 5 minutes. Lease releases still pending at shutdown race the pool closing. Turn temp directories of a killed process stay in the container. | `docker-compose.yml`, `Dockerfile:42`, `main.py:45-49` |
| T5 | Medium | A car whose manual ingest is interrupted by a restart stays `pending` for ever; nothing picks it up. | `car_route.py:48-49` (FastAPI background task) |
| T6 | Medium | Diagnosis state only accumulates: findings, tests done, ruled-out causes and topics are never capped. The state is sent to and echoed by the state model each turn, so a long session ends with truncated JSON, after which the state silently stops updating. | `app/schemas/session.py:103-121`, `app/services/llm.py:55-78` |
| T7 | Medium | A session's messages are one jsonb array with no cap; each turn rewrites the whole value two to four times (user message, media refs, reply, extracted facts). | `app/CRUD/chat_session_crud.py:66-89` |
| T8 | Low | The `visuals` volume is not backed up. After a restore, rows say `ready` and the files are gone; provider-generated artwork costs money to regenerate. | `scripts/backup.sh`, `docs/VEHICLE_VISUALS.md:268` |

### Performance

| # | Severity | Finding | Evidence |
| --- | --- | --- | --- |
| P1 | Medium | Whisper and CLAP load on the first media turn after each restart, inside that user's request. | `app/services/hearing.py:65-83` |
| P2 | Medium | Listing sessions has no limit and reads every session's message array to count it. | `chat_session_crud.py:59-64` |
| P3 | Low | Listing cars runs one query per car (two when an id misses). | `app/services/user_profile_service.py:128-131`, `car_crud.py:117-144` |
| P4 | Low | The security-header middleware is a `BaseHTTPMiddleware`, so every SSE token passes through an extra task and memory stream. | `security_middleware.py:10` |

### Hygiene

- **Unused code**, about 500 lines: the cache and rate-limit layer in `BaseService` and both
  `perform_action` methods; guide scraping, retries and `download_pdf` in `FetchCarDataService`;
  `require_permissions`, `require_role`, `get_optional_user`, `get_current_active_user` and
  `AuthorizationError`; 12 exception classes that nothing raises; the high-risk branch and a dead
  `session_id` lookup in the audit logger; the make/model swap heuristic in `CarCRUD.get_car_by_id`.
- **Unused dependencies**: `chardet`, `tenacity`, and `aiofiles` once `download_pdf` is gone.
- **Documentation drift**: README says at most 2 LLM calls per turn, the code makes 3 (reply, fact
  extraction in parallel, state); the retry route and the `retry_message_id` field of the error event are
  not in the README; the Dockerfile comment says the session lock lives in process memory.
- **No linter is configured.** `ruff check --select E9,F,B` reports 25 findings outside the default
  ignores. `pyproject.toml` carries `timeout` settings for a plugin that is not installed and eight
  markers nothing uses.

## Decisions (the owner's, 2026-11-09)

1. **Deployment stays one VPS and one uvicorn worker.** In-memory counters stay in memory. No schema
   change is made by this pass.
2. **Upgrade everything with advisories except transformers.** pypdf, PyJWT, python-multipart,
   python-dotenv, and FastAPI together with starlette. transformers stays on 4.53.0: it loads one pinned
   model with fixed labels.
3. **Small client-visible changes are acceptable**: internal fields leave the session JSON, new caps and
   limits answer `422` or `429`, the session list returns the newest 100.
4. **Scope is everything above**, including deleting unused code, fixing the documentation drift and
   adding a ruff configuration.

## Goals

- `pip-audit` on the lockfile reports advisories for transformers only.
- No route lets an anonymous client make the server store data, and no single account can trigger
  unbounded paid work.
- A media turn cannot stall logins, logging or database reconnects.
- A deploy does not leave sessions locked or cars stuck in `pending`.
- Per-session and per-account data that used to grow without bound has a cap.
- The repository has no unused code paths named in this review, and lint runs clean.

## Non-goals

- New features of any kind, including account deletion, per-device logout and refresh-token rotation.
- Schema changes and moving counters into Postgres.
- Upgrading transformers to 5.x.

## Anti-goals

- Diagnosis quality or latency getting worse. No prompt changes, and no LLM call is added to a turn.
- A hardening change that locks out legitimate use, such as a memory limit guessed without knowing the
  host.
- Error bodies changing shape, or request data appearing in logs.

## Considered and left out (my calls)

- **Container memory limits and a read-only root filesystem.** The right numbers depend on the VPS, and a
  wrong one kills the api under normal load. The plan protects Postgres with `oom_score_adj` instead.
- **SSE keep-alive comments and an overall turn deadline.** No evidence yet of dropped idle streams; the
  release logic around the stream is delicate. Add when a real client reports stalls.
- **Swapping `SlowAPIMiddleware` for slowapi's ASGI middleware.** Its behaviour with streaming responses
  is not verified.
- **Pinning the resolved address for manual downloads** (DNS rebinding). Already marked in
  `manual_search.py`; unchanged.
- **Per-account login lockout.** The per-address limit and argon2 cost stand for v1.
- **Knowledge-base ranking speed.** Needs `EXPLAIN ANALYZE` on production data before any change.
- **A CI workflow.** Lint and tests run locally with two commands; hosting them is a separate decision.

## Left to the builder

File-level structure and test design within each task, as the plan specifies them.
