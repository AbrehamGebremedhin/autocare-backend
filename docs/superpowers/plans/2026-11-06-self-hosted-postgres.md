# Self-Hosted Postgres Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove Supabase entirely. The backend runs on a plain Postgres container on the user's VPS, with its own email+password accounts plus Google sign-in, and media on a Docker volume.

**Architecture:** One `asyncpg` pool is created in the FastAPI lifespan. `BaseCRUD` keeps its dict-filter API but runs parameterised SQL, so the CRUD subclasses barely change. Accounts live in a `users` table (argon2 hashes, optional `google_sub`) and the app keeps issuing its own JWTs. Google sign-in is client-side: the client sends a Google ID token to `POST /auth/google`, the server verifies it and returns the app's tokens. Media bytes go to `MEDIA_DIR` through a small store with the same method names as the old bucket manager. `docker compose up` starts `db` + `api`, and the schema is applied by Postgres' init-script mechanism on a fresh volume.

**Tech Stack:** asyncpg, argon2-cffi, google-auth (+ requests, its HTTP transport), postgres:16-alpine, existing FastAPI/DeepSeek/ffmpeg/Whisper/CLAP stack. Removed: `supabase`.

**Spec:** `docs/superpowers/specs/2026-11-04-autocare-backend-finish-design.md` (behaviour unchanged). This plan only replaces the data, auth and storage layer, per the user's decisions on 2026-11-06: plain Postgres on a VPS; email+password (no email confirmation) plus Google SSO via client-sent ID tokens; media on a Docker volume; replace Supabase directly on `development`, nothing was ever deployed to it.

## Global Constraints

All constraints of `2026-11-04-autocare-backend-finish.md` still apply, plus:

- uv only, no pip. New dependencies allowed: `asyncpg`, `argon2-cffi`, `google-auth`, `requests`. Do not upgrade any other pin.
- Commits are on `development`, conventional messages, **no `Co-Authored-By` line**, never push, never touch `main`.
- Never read `.env*` files. `.env.example` is edited by writing the whole file.
- Tests are offline: fakes only. The one real-database test is marked `db` and is skipped unless `TEST_DATABASE_URL` is set. The executor never runs SQL against a real database.
- User id always comes from the JWT. Never log request bodies, passwords, hashes, tokens or `DATABASE_URL`.
- Every SQL value is a bind parameter. Dynamic identifiers (table, column) must match `^[A-Za-z_][A-Za-z0-9_]*$` and are double-quoted.
- UTF-8 end to end; jsonb values are written with `ensure_ascii=False`.
- Preserve each existing file's line endings (many are CRLF); new files are LF.
- One DB connection is never held across an SSE stream or a model call; acquire per statement or per short transaction.
- Required settings after this plan: `JWT_SECRET_KEY` (≥32), `DATABASE_URL`, `DEEPSEEK_API_KEY`. Removed: `SUPABASE_URL`, `SUPABASE_KEY`, `MEDIA_BUCKET`. Added: `MEDIA_DIR` (default `/data/media`) and optional `GOOGLE_CLIENT_IDS` (comma-separated OAuth client ids; when empty, `POST /auth/google` returns 404 and Google sign-in is off).

## Review Focus

1. SQL injection through `BaseCRUD` filter keys or table names. A key like `id"; drop table x; --` must raise, never reach the database.
2. Path traversal in the media store. A `user_id`/`session_id` of `../x` must raise, and nothing is written outside `MEDIA_DIR`.
3. Account enumeration and timing. Unknown email and wrong password return the same 401 and both run one argon2 verify.
4. Concurrent registration of the same email. The unique violation maps to 409, not 500.
5. Lost updates. Two `append_messages` calls on one session must both survive (`messages || $new`, not read-modify-write).
6. Google ID token checks. A token with the wrong audience, wrong issuer, expired, `email_verified` false, or missing `sub` is rejected 401; the same token can't mint a session for another user.
7. Account takeover by pre-registration. An attacker registers `victim@gmail.com` with a password (emails are not verified); when the victim later signs in with Google, the account must NOT stay reachable with the attacker's password.
8. Pool exhaustion. A client that disconnects mid-stream must not leak a connection or leave the in-flight lock held.

---

## File Structure

- `app/core/config.py` — settings (modify).
- `app/db/pool.py` — pool lifecycle and `fetch`/`fetchrow`/`fetchval`/`execute`/`transaction` helpers (create).
- `app/db/crud.py` — `BaseCRUD` over the pool (rewrite). `app/db/base.py` (Supabase handler) deleted.
- `app/db/sql/schema.sql` — the single idempotent schema (create). `000_base_schema.sql` and `001_v2_schema.sql` deleted.
- `app/CRUD/user_crud.py` — users table (create).
- `app/services/auth_service.py` — hashing, register, login verification (create).
- `app/services/media_store.py` — disk store (create). `app/db/bucket_operations.py` deleted.
- `app/services/user_profile_service.py`, `app/CRUD/car_crud.py`, `app/CRUD/chat_session_crud.py`, `app/services/kb.py`, `app/api/v1/user_auth.py`, `app/utils/auth_middleware.py`, `app/services/diagnosis.py`, `app/api/v1/session_route.py`, `main.py` — move to the new layer (modify).
- `docker-compose.yml`, `Dockerfile`, `.env.example`, `README.md`, `scripts/backup.sh`, `scripts/e2e.py`, `pyproject.toml` — deployment (modify/create).

---

### Task 1: Database core — settings, pool, BaseCRUD, schema

**Files:**
- Modify: `app/core/config.py`, `pyproject.toml`/`uv.lock` (add `asyncpg`; keep `supabase` for now, Task 5 removes it)
- Create: `app/db/pool.py`, `app/db/sql/schema.sql`, `tests/unit/test_pool.py`, `tests/unit/test_base_crud.py`, `tests/unit/test_schema_sql.py` (rewrite), `tests/integration/test_db_real.py` (marker `db`)
- Rewrite: `app/db/crud.py`; delete `app/db/base.py`'s handler only when Task 5 lands (Tasks 2–4 still import it)
- Modify: `main.py` lifespan opens/closes the pool; `/health` reports `{"status","database","deepseek"}` via `select 1` (cached 60 s ping for DeepSeek stays).

**Interfaces:**
- Produces, in `app/db/pool.py`:
  - `async def init_pool(dsn: str | None = None) -> None` (min 1, max 10; registers jsonb/json codecs `json.dumps(ensure_ascii=False)` / `json.loads`; idempotent)
  - `async def close_pool() -> None`
  - `async def fetch(sql: str, *args) -> list[dict]`, `fetchrow(sql, *args) -> dict | None`, `fetchval(sql, *args)`, `execute(sql, *args) -> str`
  - `@asynccontextmanager async def transaction()` yielding an object with the same four methods bound to one connection
  - `def set_pool_for_tests(fake) -> None`
- Produces, in `app/db/crud.py`: `BaseCRUD(table_name)` with `create(data) -> dict`, `read(filters=None) -> list[dict]`, `update(filters, data) -> list[dict]`, `delete(filters) -> list[dict]`, same signatures as today but returning plain dicts. Identifier check helper `_ident(name) -> str` raises `ValueError`.
- Produces `schema.sql`: tables `users(id uuid pk default gen_random_uuid(), email text not null, password_hash text null, google_sub text null, created_at timestamptz default now(), check (password_hash is not null or google_sub is not null))` with a unique index on `google_sub` where not null with `unique index on lower(email)`; `user_profiles(user_id uuid pk references users(id) on delete cascade, cars text[] default '{}', preferences jsonb default '{}', created_at, updated_at)`; `"Car"` and `"Chatsession"` and `kb_passages` and `search_kb(...)` exactly as in the current 000/001 (same columns, same function signature and returned columns), minus row-level security and the Supabase guard. Everything `create ... if not exists`/`create or replace`; one transaction.

- [ ] **Step 1: Write failing tests.**
  - `test_pool.py`: with a fake asyncpg pool installed, `fetch/fetchrow/execute` pass SQL and args through; `transaction()` commits on success and rolls back on exception; jsonb codec round-trips `{"t": "ጎማ"}` without `\u` escapes.
  - `test_base_crud.py`: `read({"id": "x"})` issues `SELECT * FROM "Car" WHERE "id" = $1` with `["x"]`; `update` and `delete` build `RETURNING *`; `create` builds a parameterised insert; `read({'id"; drop table x; --': 1})` and `BaseCRUD('Car; drop')` raise `ValueError` without touching the pool.
  - `test_schema_sql.py`: replays the DDL and asserts every column read or written by the code (`CarBase`, `UserProfile`, `Chatsession` and `kb_passages` code paths, `users`) exists, the file is LF, one transaction, fully guarded with `if not exists`/`or replace`, and `search_kb` args/returned columns match `kb.search`.
  - `test_db_real.py` (skipped without `TEST_DATABASE_URL`): applies `schema.sql` twice (idempotent), inserts and reads a car, runs `search_kb` on a passage.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement** pool, `BaseCRUD`, `schema.sql`, config (`DATABASE_URL` required, `MEDIA_DIR`; keep old Supabase fields optional until Task 5), lifespan, `/health`.
- [ ] **Step 4: Run `uv run pytest -q`.** Expected: all green, old Supabase-based tests untouched still pass because nothing was removed yet.
- [ ] **Step 5: Commit** `feat: asyncpg pool, parameterised BaseCRUD and single schema file`.

---

### Task 2: Accounts — users table, argon2, Google sign-in, register/login/logout

**Files:**
- Create: `app/CRUD/user_crud.py`, `app/services/auth_service.py`, `tests/unit/test_auth_service.py`
- Modify: `app/api/v1/user_auth.py` (remove confirm routes and every Supabase call), `app/utils/auth_middleware.py`, `app/services/user_profile_service.py`, `app/schemas/User.py`, `tests/unit/test_auth_routes.py`, `tests/unit/test_auth_middleware.py`, `tests/unit/test_user_profile_service.py`, `tests/conftest.py` (replace the Supabase patches with a fake pool/fake user store)

**Interfaces:**
- Consumes: `app.db.pool`, `BaseCRUD` (Task 1).
- Produces, in `auth_service.py`:
  - `hash_password(pw: str) -> str` (argon2id defaults)
  - `verify_password(pw: str, hash: str | None) -> bool` — when `hash is None` it still verifies against a module-level dummy hash and returns `False`
  - `async register(email: str, password: str) -> dict` returns `{"id": str, "email": str}`; raises `EmailTaken` on unique violation (SQLSTATE 23505)
  - `async authenticate(email: str, password: str) -> dict | None`
  - `async user_exists(user_id: str) -> bool`
  - `async login_with_google(id_token: str) -> dict` returns `{"id", "email"}`. Verifies with `google.oauth2.id_token.verify_oauth2_token(token, Request(), audience=None)` run in `asyncio.to_thread`, then checks `aud in settings GOOGLE_CLIENT_IDS`, `iss in {"accounts.google.com","https://accounts.google.com"}`, `email_verified is True`, non-empty `sub` and `email`; any failure raises `GoogleAuthError` (never echoing its message). Resolution order: (1) user with this `google_sub` → return; (2) user with the lower-cased email and no `google_sub` → set `google_sub` AND set `password_hash = null` in one statement (this closes the pre-registration takeover; the owner of the verified Google email wins; document it); (3) otherwise insert a new user with `google_sub` and no password. Unique violations (race) retry the lookup once.
- Behaviour: emails are stored and compared lower-cased and stripped. `register` returns tokens like login (no confirmation step). Duplicate → 409 `{"message": "...", "error_code": ...}` in the existing error shape. Wrong password and unknown email → the same 401. `validate_user_with_supabase` is replaced by `validate_user(user_id)` calling `user_exists`; the middleware keeps its 401 behaviour for unknown users. A user with `password_hash is null` can never log in by password (same 401 as wrong password, one dummy verify). New route `POST /auth/google` with body `{"id_token": str}` (size-limited, rate-limited like login) returns the same token response as login; 404 when `GOOGLE_CLIENT_IDS` is empty; any `GoogleAuthError` → generic 401. Logout stays a 200 that tells the client to discard its tokens. `GET/POST /auth/confirm*` and `/auth/confirm-page` are deleted along with their tests and README mentions.

- [ ] **Step 1: Write failing tests** with a fake pool: hash/verify round trip; `verify_password(None-hash)` returns False and calls argon2 verify once; `register` lower-cases the email and maps a raised `asyncpg.UniqueViolationError` to `EmailTaken`; login returns the same body for unknown email and wrong password and never logs the password; middleware 401 for a token whose user is gone; concurrent double-register test (two `register` calls, second sees unique violation → 409). Google tests mock `verify_oauth2_token` only: valid token creates a user with no password; second sign-in with the same `sub` reuses it; email match on a password account sets `google_sub` and nulls `password_hash` (then password login → 401); wrong `aud`, wrong `iss`, `email_verified: False`, missing `sub`, expired (verify raises `ValueError`) → 401; empty `GOOGLE_CLIENT_IDS` → 404; the response never contains the Google token or error text.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement** the service, user CRUD, route changes, middleware, profile service on SQL (`insert ... on conflict do nothing`, `update ... returning`), conftest fixtures.
- [ ] **Step 4: Run `uv run pytest -q`.** Expected: green.
- [ ] **Step 5: Commit** `feat: self-hosted accounts: email+password and Google sign-in`.

---

### Task 3: Cars, sessions and knowledge base on SQL

**Files:**
- Modify: `app/CRUD/car_crud.py`, `app/CRUD/chat_session_crud.py`, `app/services/kb.py`, `app/services/user_profile_service.py` (car ops), `tests/unit/test_kb.py`, `tests/unit/test_car_ingest.py`, `tests/unit/test_session_routes.py`/`tests/conftest.py` (`FakeSessionStore` stays aligned), plus any test touching `.rpc`/`.table`

**Interfaces:**
- Consumes: Task 1 pool/`BaseCRUD`.
- Produces (signatures unchanged for callers): `kb.search(query, car_id, k) -> list[Passage]` runs `select * from search_kb($1,$2,$3)`; `kb.ingest_pdf(...)` deletes-then-inserts with `executemany` in batches of 500 inside one `transaction()` so a failed batch rolls everything back (the old compensating-delete code is removed); `ChatSessionCRUD.append_messages(session_id, messages)` uses `update "Chatsession" set messages = messages || $2::jsonb, updated_at = now()`; `update_message` patches one element by id in a single statement; `list_for_user(user_id)` returns `message_count` via `jsonb_array_length(messages)` and no message bodies; `save_state` unchanged.

- [ ] **Step 1: Write failing tests** (fake pool): `search` sends the exact SQL and params and maps rows to `Passage`; ingest rolls back when batch 2 raises (no partial rows remain, using the fake transaction); two sequential `append_messages` produce `||` SQL each (no read in between); `list_for_user` SQL contains `jsonb_array_length` and not `messages` selection; `update_message` targets by message id.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: `uv run pytest -q`.** Expected: green.
- [ ] **Step 5: Commit** `feat: cars, sessions and knowledge base on plain Postgres`.

---

### Task 4: Media on disk

**Files:**
- Create: `app/services/media_store.py`, `tests/unit/test_media_store.py`
- Modify: `app/services/diagnosis.py` (use the store), `app/api/v1/session_route.py` (delete path), `tests/unit/test_turn_media.py`, `tests/unit/test_session_routes.py`, `scripts/run_cases.py` (no-op store)
- Delete: `app/db/bucket_operations.py`

**Interfaces:**
- Produces: `class MediaStore` with the same method names and return shapes `SupabaseBucketManager` had (read `bucket_operations.py` first and mirror its upload and `delete_prefix` methods), writing under `settings.MEDIA_DIR/{user_id}/{session_id}/{message_id}/{n}{ext}` through `asyncio.to_thread`. A path component that is not `^[A-Za-z0-9_-]+$` (ids) or an extension outside `.jpg/.mp4/.m4a` raises `ValueError`. After joining, `resolve()` must stay inside `MEDIA_DIR`. `delete_prefix(user_id, session_id)` removes the session folder (best effort, never raises to the caller).
- A storage failure still must not kill the turn: that file is omitted from `media` exactly as today.

- [ ] **Step 1: Write failing tests** using `tmp_path`: save then file exists with the right bytes at the right path; `../x` and absolute-path ids raise and nothing is written; bad extension raises; `delete_prefix` removes the folder and a second call is a no-op; a write failure (read-only dir) is reported as failure to the caller and the turn tests still show `media=[]`.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement** and rewire.
- [ ] **Step 4: `uv run pytest -q`.** Expected: green.
- [ ] **Step 5: Commit** `feat: store uploaded media on a local volume`.

---

### Task 5: Deploy stack, remove Supabase, docs

**Files:**
- Modify: `docker-compose.yml`, `Dockerfile` (bake step placeholders: drop Supabase vars, add `DATABASE_URL`), `.env.example` (write whole file: required `JWT_SECRET_KEY`, `DATABASE_URL`, `DEEPSEEK_API_KEY`, `POSTGRES_PASSWORD`; optional `GOOGLE_CLIENT_IDS` and the rest, `MEDIA_DIR`), `README.md`, `scripts/e2e.py` (register instead of login-with-existing-account if `--register`; `/cars`, `database` health key), `pyproject.toml`/`uv.lock` (remove `supabase`, add `argon2-cffi`; confirm `uv.lock` diff touches no unrelated pin), `app/core/config.py` (delete the Supabase fields and `MEDIA_BUCKET`), `tests/unit/test_app.py`/`test_health.py`
- Create: `scripts/backup.sh` (`pg_dump` to a dated gzip plus `tar` of the media volume, prints cron line), `tests/unit/test_compose.py` (parses YAML)
- Delete: `app/db/base.py`, `app/db/enhanced_connection_pool.py`/`migration.py`/`migrations/` if present and unreferenced, `app/db/sql/000_base_schema.sql`, `app/db/sql/001_v2_schema.sql`, `.env.template`

**Interfaces:**
- Compose: services `db` (`postgres:16-alpine`, volume `pgdata`, `./app/db/sql/schema.sql` mounted to `/docker-entrypoint-initdb.d/01-schema.sql`, healthcheck `pg_isready`) and `api` (`depends_on: db: condition: service_healthy`, volume `media:/data/media`, env `DATABASE_URL=postgresql://autocare:${POSTGRES_PASSWORD}@db:5432/autocare`, `env_file: .env`). The `db` port is **not** published. The API healthcheck is unchanged (`/health`).
- README: setup on the VPS (`.env`, `docker compose up -d`), applying schema changes later by hand with `psql`, backup/restore, a Caddy reverse-proxy snippet with TLS, `flush_interval -1`/no response buffering for SSE and `request_body max_size 101MB`, a note that the in-flight session lock is per-process (one uvicorn worker), the removal of email confirmation (so email+password accounts are not verified), and a Google setup section: create OAuth client ids in Google Cloud Console (Web/Android/iOS as the client needs), put them comma-separated in `GOOGLE_CLIENT_IDS`, and what the client sends to `POST /auth/google`.

- [ ] **Step 1: Write failing tests:** `test_compose.py` asserts `db` has no published ports, `api` depends on a healthy `db`, a named volume backs both `pgdata` and media, the schema mount path exists; a test asserts `grep -ri supabase app main.py scripts` finds nothing (pure-Python walk).
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement** everything above.
- [ ] **Step 4: Verify.** `uv run pytest -q` green; `uv lock --check`; `docker compose config` exits 0 (no containers started).
- [ ] **Step 5: Commit** `chore: self-hosted postgres stack, drop supabase` (docs in a separate `docs:` commit).

---

## Execution notes

- Order is fixed: 1 → 2 → 3 → 4 → 5. Task 5 flips the app to `DATABASE_URL`-only; until then Supabase code still imports.
- Live checks the **user** runs on the VPS after Task 5: `docker compose up -d --build`, `curl /health` → `{"status":"ok","database":true,"deepseek":true}`, `docker compose run --rm -v ./car_data:/app/car_data:ro api uv run --no-sync python scripts/ingest_kb.py` (the image must therefore include `scripts/`; `car_data/` stays out of the image via `.dockerignore`), the 9 quality cases, `scripts/e2e.py`.
- Everything in the first plan's deferred live-checks list that mentions Supabase (bucket, key format, grants, RLS) disappears.
