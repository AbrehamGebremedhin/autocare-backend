# Vehicle Visuals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve one reviewed, shared workshop-manual illustration per car variant to the mobile Garage screen through
`GET /cars/{id}/visual` and `POST /cars/{id}/visual/prepare`, never blocking diagnosis, sessions or manual indexing.

**Architecture:** A variant catalog resolves a user's car to a variant. A `vehicle_visuals` row per (variant, style) holds the
state machine; a separate worker process (same image) claims due rows with a lease and fencing token, runs a generator
(curated files by default, a provider behind an off-by-default flag), normalizes the image, and parks it in `review`. An
operator CLI approves; publishing writes an immutable checksum-addressed file then atomically activates the row. Downloads
use self-authenticating HMAC-signed URLs served by the API from a private directory.

**Tech Stack:** FastAPI, asyncpg (existing `app.db.pool`), Pillow 13.0.0 (already pinned, has WebP), slowapi limiter (existing),
plain PostgreSQL 16. No new dependency.

**Spec:** `docs/superpowers/specs/2026-11-08-vehicle-visuals-brief.md` (read it first; the descriptor contract and client rules
there are exact).

## Decisions (made here so executors do not re-decide)

- Branch `feat/vehicle-visuals` is cut from **`main`**, not `development` (development lacks the hardening, brotli and
  auth fixes now on main). Never merge, push, deploy, or touch any real database or `.env*` file (never open `.env*`; never
  call `get_settings()` outside pytest).
- Storage: local filesystem under `VISUALS_DIR` (default `/data/visuals`), a new compose volume `visuals` shared by `api` and
  the new `visuals-worker` service. Layout: `drafts/{variant_key}/{style_version}/{job_token}.webp` (private),
  `objects/{sha256}.webp` (immutable, published). No S3: the repo has none and the deployment is one VPS.
- Signing: HMAC-SHA256 with a key derived from `JWT_SECRET_KEY` (`hmac(JWT_SECRET_KEY, b"visuals-url-v1")`), over
  `"{sha256}.{exp}"`. URL: `{PUBLIC_BASE_URL}/api/v1/visuals/files/{sha256}.webp?exp=<unix>&sig=<hex>`. No new secret to manage.
- Schema: a new idempotent file `app/db/sql/vehicle_visuals.sql` (not appended to `schema.sql`), mounted in compose as
  `02-vehicle_visuals.sql` for fresh volumes; for the existing VPS the owner applies it by hand. `schema.sql` stays untouched.
- `style_version` is the constant `"workshop-v1"`; `recipe_version` is `"r1"`; preset is `"garage_roll_v1"`.
- Per-user prepare throttle and the global job-creation cap are counted in memory / from the table respectively
  (single-worker API, like `usage_limits.py`); both return 429 with `Retry-After`.
- The two new routes live in a new `app/api/v1/visual_route.py` (router prefix `/cars`, registered in `routes.py`), plus a
  public download route in the same file under prefix `/visuals`. `car_route.py` is not touched.
- Any internal failure while resolving a descriptor (missing visuals tables, DB outage on those tables, missing object file)
  yields `state:"fallback"`, reason `temporarily_unavailable`, HTTP 200. Only auth and ownership errors are non-200.

## Global Constraints

- Descriptor: `schema_version` 1; `preset` `"garage_roll_v1"` or absent; ready artwork `format:"flat_webp"`, `width:1024`,
  `height:420`, `baseline_y:366`, 64-hex lowercase `sha256`, https `download_url`, ISO-8601 `expires_at`.
- States: queued/generating/review map to `pending`; failed/unknown map to `fallback`; reasons only
  `unknown_variant|choose_variant|temporarily_unavailable`. `retry_after_ms` only for pending, bounded 2000..60000.
- Download: 200, `Content-Type: image/webp`, `Content-Length` <= 409,600, `Cache-Control: private, max-age=<remaining expiry,
  <= URL TTL>`, `X-Content-Type-Options: nosniff`, no Authorization header, constant-time signature check, default TTL 900 s.
- 404 (never 403) for a car that is missing or not the caller's; errors use the existing envelope and exception classes.
- No signed URLs, tokens, user ids or raw provider payloads in logs or error bodies; correlation ids in logs.
- Image: canvas 1024x420 WebP, ground baseline y=366, all metadata stripped, <= 409,600 bytes, decoded size verified.
- Repo rules: TDD, small conventional commits, no `Co-Authored-By`, no `sed -i` on CRLF files (use Edit), baseline
  `uv run pytest -q` = 670 passed, 1 skipped, 21 deselected, 7 warnings; add no new warnings. DB tests (`-m db`) run only
  against a throwaway PostgreSQL 16 cluster on port 54329+ (`C:\Program Files\PostgreSQL\16\bin`), deleted afterwards.

## Review Focus

- Visuals tables not migrated yet (or DB error on them): both routes return 200 fallback `temporarily_unavailable`, never 500 (Task 7, 8).
- `car_id` with NUL, an over-long id, or odd characters: 422/404 like `delete_car`, never 500 (Task 8).
- `variant_key` that exists but belongs to another make/model or is not allowlisted: 422 (Task 3, 8).
- A decompression-bomb or huge-dimension image from a generator is rejected before it is fully decoded (Task 4).
- Signature of the wrong length, non-hex, non-ASCII, negative or huge `exp`, or `%2e%2e` in the name: refused with 4xx, never 500 (Task 2, 8).
- A worker killed mid-job: the lease expires and another claim succeeds; a stale token cannot publish (Task 6, 9).

---

### Task 1: Schema file, compose wiring

**Files:**
- Create: `app/db/sql/vehicle_visuals.sql`
- Modify: `docker-compose.yml` (db volume mount; new `visuals` volume; `visuals-worker` is added in Task 9)
- Test: `tests/unit/test_schema_sql.py` (extend), `tests/integration/test_visuals_db.py` (create, `db` marker)

**Interfaces:**
- Produces: tables `vehicle_visuals`, `user_car_visual_choices`, `visual_provider_calls`; functions
  `claim_due_visual_job(p_lease_seconds int) returns vehicle_visuals`,
  `finalize_visual_job(p_id uuid, p_token bigint, p_state text, p_draft_sha text, p_error text, p_next_attempt timestamptz) returns boolean`,
  `publish_visual_revision(p_id uuid, p_expected_revision int, p_sha text, p_width int, p_height int) returns boolean`.

- [ ] **Step 1: Failing tests.** In `test_schema_sql.py` add `test_visuals_sql_is_idempotent_text` (file exists, every `create table` uses `if not exists`, function bodies use `create or replace`, no `drop`). In `test_visuals_db.py` (copy the `real_pool` fixture and skip marks from `test_db_real.py`) add `test_visuals_schema_applies_twice_and_dedupes_variant`: apply `schema.sql` then `vehicle_visuals.sql` twice; insert the same (variant, style) twice with `on conflict do nothing`; assert one row.
- [ ] **Step 2: Run, expect FAIL** (`uv run pytest tests/unit/test_schema_sql.py -q`).
- [ ] **Step 3: Write `vehicle_visuals.sql`** inside `begin; ... commit;`:
  - `vehicle_visuals(id uuid pk default gen_random_uuid(), variant_key text not null, style_version text not null, state text not null check in ('queued','generating','review','ready','failed'), provenance text not null default 'curated' check in ('curated','provider'), recipe_version text not null default 'r1', sha256 text, draft_sha256 text, width int, height int, published_revision int not null default 0, attempt_count int not null default 0, next_attempt_at timestamptz not null default now(), lease_expires_at timestamptz, fencing_token bigint not null default 0, last_error_code text, created_at, updated_at, unique (variant_key, style_version))`; an index on `(state, next_attempt_at)`.
  - `user_car_visual_choices(user_id uuid references users(id) on delete cascade, car_id text not null, variant_key text not null, created_at, primary key (user_id, car_id))`.
  - `visual_provider_calls(id bigserial pk, variant_key text not null, called_at timestamptz not null default now())` plus an index on `called_at`.
  - `claim_due_visual_job`: one statement `select ... where state='queued' and next_attempt_at <= now() or (state='generating' and lease_expires_at < now()) order by next_attempt_at for update skip locked limit 1`, then update to `generating`, `fencing_token = fencing_token + 1`, `lease_expires_at = now() + p_lease_seconds`, `attempt_count = attempt_count + 1`; returns the updated row or null.
  - `finalize_visual_job`: updates only `where id = p_id and fencing_token = p_token and state = 'generating'`; sets state/draft_sha256/last_error_code/next_attempt_at, clears the lease; returns whether a row changed.
  - `publish_visual_revision`: updates only `where id = p_id and published_revision = p_expected_revision and state = 'review'`; sets `state='ready'`, `sha256 = p_sha`, width/height, `published_revision + 1`; returns whether a row changed.
- [ ] **Step 4: Compose.** Add the volume line `- ./app/db/sql/vehicle_visuals.sql:/docker-entrypoint-initdb.d/02-vehicle_visuals.sql:ro` to `db`, and the `visuals:` volume under `volumes:` (the `api` mount `visuals:/data/visuals` is added here too). Keep CRLF; use Edit. Extend `tests/unit/test_compose.py` with an assertion for each.
- [ ] **Step 5: Real DB.** Start a throwaway PG16 cluster, run `uv run pytest tests/integration/test_visuals_db.py -m db -q` with `TEST_DATABASE_URL`; expect pass; stop and delete the cluster.
- [ ] **Step 6: Commit** `feat: vehicle visuals tables and functions`.

### Task 2: Settings and URL signing

**Files:**
- Modify: `app/core/config.py` (CRLF; use Edit), `.env.example` (append with `>>` only; never read it)
- Create: `app/services/visual_signing.py`
- Test: `tests/unit/test_visual_signing.py`

**Interfaces:**
- Produces settings: `VISUALS_DIR: str = "/data/visuals"`, `PUBLIC_BASE_URL: str = ""`, `VISUAL_URL_TTL_SECONDS: int = 900`,
  `VISUAL_PREPARE_PER_HOUR: int = 20`, `VISUAL_JOBS_PER_HOUR: int = 60`, `VISUAL_POLL_SECONDS: int = 10`,
  `VISUAL_PROVIDER_ENABLED: bool = False`, `VISUAL_PROVIDER_DAILY_BUDGET: int = 20`, `VISUAL_CURATED_DIR: str = "/data/visuals/curated"`.
- Produces: `sign_url(sha256: str, now: float | None = None) -> tuple[str, str]` returns `(url, expires_at_iso)`;
  `verify(sha256: str, exp: str, sig: str, now: float | None = None) -> bool` (constant time, never raises).

- [ ] **Step 1: Failing tests.** `test_sign_then_verify_round_trips`; `test_expired_is_refused`; `test_tampered_sha_or_exp_is_refused`;
  `test_garbage_exp_and_sig_never_raise` (parametrize: "", "abc", "-1", "9"*30, non-ASCII, 3000-char strings, wrong-length hex);
  `test_url_is_https_and_contains_no_query_beyond_exp_sig`; `test_unset_public_base_url_yields_no_url` (returns `("", "")`).
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** with `hmac.compare_digest` on the hex digests, `exp` parsed with a strict `str.isdigit()` plus length cap, key derived as in Decisions.
- [ ] **Step 4: Run, expect PASS.** Also assert the settings defaults in an existing config test.
- [ ] **Step 5: Commit** `feat: visuals settings and signed download urls`.

### Task 3: Variant catalog and resolver

**Files:**
- Create: `app/services/vehicle_catalog.py`, `app/services/vehicle_catalog.json`
- Test: `tests/unit/test_vehicle_catalog.py`

**Interfaces:**
- Produces: `normalize(value: str) -> str`;
  `@dataclass(frozen=True) Resolution(kind: Literal["selected","choose","unknown"], variant_key: str | None, options: list[tuple[str, str]])`;
  `resolve(make: str, model: str, year: int, choice: str | None = None) -> Resolution`;
  `candidates(make: str, model: str, year: int) -> list[str]`; `is_allowed(variant_key: str) -> bool`;
  `variant_matches_car(variant_key: str, make: str, model: str) -> bool`; `label(variant_key: str) -> str`.
- The JSON holds `catalog_version`, and variants: `toyota-echo-2001-sedan` (Toyota, aliases `echo`/`yaris sedan`, years 2000-2005,
  unambiguous) and one reviewed Honda Civic demo variant `honda-civic-2008-sedan` (2006-2011).

- [ ] **Step 1: Failing tests.** `normalize(" Toyota  ECHO! ") == "toyota echo"`; Echo 2001 resolves `selected toyota-echo-2001-sedan`;
  Echo 1999 and an unknown make resolve `unknown`; a test-only catalog with two valid variants for one car resolves `choose` with
  labels only; a valid `choice` resolves to `selected`; a `choice` for another model is ignored by `resolve` and `variant_matches_car` is False;
  `is_allowed("../x")` False.
- [ ] **Step 2: Run, expect FAIL.** **Step 3: Implement** (load JSON once; `lru_cache`; ambiguity marker is `"ambiguous": true` per match entry). **Step 4: PASS.**
- [ ] **Step 5: Commit** `feat: vehicle variant catalog`.

### Task 4: Image normalization and validation

**Files:**
- Create: `app/services/visual_image.py`
- Test: `tests/unit/test_visual_image.py`

**Interfaces:**
- Produces: `CANVAS = (1024, 420)`, `BASELINE_Y = 366`, `MAX_BYTES = 409_600`, `MAX_SOURCE_PIXELS = 25_000_000`, `MAX_SOURCE_BYTES = 20_000_000`;
  `class InvalidArtwork(Exception)` with a stable `.code` (`too_large`, `not_image`, `bad_dimensions`, `bad_margins`, `too_many_bytes`);
  `normalize(data: bytes) -> tuple[bytes, str, int, int]` returns `(webp_bytes, sha256_hex, width, height)`;
  `validate_published(data: bytes) -> tuple[str, int, int]` returns `(sha256, w, h)` and raises `InvalidArtwork`.

- [ ] **Step 1: Failing tests** (build images in memory with Pillow). A 2000x900 PNG normalizes to a 1024x420 WebP of at most 409,600 bytes
  with no EXIF/ICC/XMP (`img.info` has none) and the artwork's lowest opaque row at `y=366`; a 5000x5000 image over the pixel cap raises `too_large`
  before the decode (assert `Image.open` + size check, not `load()`); a decompression bomb (a tiny file declaring huge dimensions) raises `too_large`;
  random bytes raise `not_image`; a WebP of the wrong size fails `validate_published` with `bad_dimensions`; artwork touching the canvas edge fails `bad_margins`;
  `sha256` equals `hashlib.sha256(webp_bytes)`.
- [ ] **Step 2: FAIL. Step 3: Implement** with Pillow: check `len(data)`, `Image.open` then `img.size` against the caps (set `Image.MAX_IMAGE_PIXELS = MAX_SOURCE_PIXELS` locally), convert RGBA, crop to the alpha bounding box, scale to fit with a margin, paste on a transparent 1024x420 canvas bottom-aligned so the lowest opaque row is 366, save WebP (lossless first, fall back to quality steps until under `MAX_BYTES`), re-open the result to verify. **Step 4: PASS.**
- [ ] **Step 5: Commit** `feat: artwork normalization and validation`.

### Task 5: Visuals file store

**Files:**
- Create: `app/services/visual_store.py`
- Test: `tests/unit/test_visual_store.py`

**Interfaces:**
- Produces `class VisualStore(root: str | None = None)` with:
  `write_draft(variant_key: str, style_version: str, token: int, data: bytes) -> str` (relative path);
  `read_draft(rel: str) -> bytes`; `put_object(sha256: str, data: bytes) -> None` (atomic temp file + `os.replace`, mode 0o600, idempotent, refuses if hash mismatch);
  `object_path(sha256: str) -> Path | None` (None if missing or the name is not 64 lowercase hex);
  `has_object(sha256: str) -> bool`; `sweep(older_than_seconds: int, keep_shas: set[str]) -> int` (removes stale drafts and unreferenced objects, never raises).
- Id components use the same `^[A-Za-z0-9_-]+$` `fullmatch` rule as `media_store.py`; symlinks refused.

- [ ] **Step 1: Failing tests:** put_object then object_path round trip; put_object with a wrong sha raises; traversal names (`..`, `a/b`, `%2e%2e`, uppercase hex, 63 chars) give None; a symlinked objects dir is refused; sweep keeps referenced objects and recent drafts, removes old ones; draft write is atomic (partial write leaves no file).
- [ ] **Step 2: FAIL. Step 3: Implement** modeled on `app/services/media_store.py` (read it first; reuse its helpers where they fit, do not copy-paste long blocks). **Step 4: PASS.**
- [ ] **Step 5: Commit** `feat: private visuals file store`.

### Task 6: Repository over the SQL functions

**Files:**
- Create: `app/CRUD/vehicle_visual_crud.py`
- Test: `tests/unit/test_vehicle_visual_crud.py` (FakePool), `tests/integration/test_visuals_db.py` (extend, `db`)

**Interfaces:**
- Produces module functions (all use `app.db.pool`; none hold a connection across a generator call):
  `get_visual(variant_key: str) -> dict | None`; `ensure_job(variant_key: str, provenance: str = "curated") -> tuple[dict, bool]` (`insert ... on conflict do nothing`, returns `(row, created)`);
  `jobs_created_last_hour() -> int`; `get_choice(user_id: str, car_id: str) -> str | None`; `set_choice(user_id: str, car_id: str, variant_key: str) -> None`; `clear_choice(user_id: str, car_id: str) -> None`;
  `claim_job(lease_seconds: int) -> dict | None`; `renew_lease(job_id: str, token: int, lease_seconds: int) -> bool`;
  `finalize_job(job_id: str, token: int, state: str, draft_sha: str | None, error: str | None, next_attempt_seconds: int | None) -> bool`;
  `publish(job_id: str, expected_revision: int, sha: str, width: int, height: int) -> bool`;
  `list_review() -> list[dict]`; `reject(variant_key: str) -> bool`; `requeue(variant_key: str) -> bool` (failed/review -> queued, attempts reset);
  `record_provider_call(variant_key: str) -> None`; `provider_calls_last_day() -> int`; `owned_car(user_id: str, car_id: str) -> dict | None`
  (`select c.* from user_profiles p join "Car" c on c.id = any(p.cars) where p.user_id = $1::uuid and c.id = $2`).

- [ ] **Step 1: Failing real-DB tests** (throwaway PG16): `test_concurrent_ensure_job_makes_one_row` (20 `asyncio.gather` calls, exactly one `created`);
  `test_claim_is_exclusive_and_lease_expiry_reclaims` (two claimers one row; set `lease_expires_at` into the past, second claim succeeds and the token is higher);
  `test_stale_token_cannot_finalize_or_renew`; `test_publish_requires_review_and_expected_revision` (stale revision False, then True; revision becomes 1);
  `test_choice_is_per_user_car`; `test_owned_car_is_none_for_other_user`; `test_requeue_only_from_failed_or_review`. Unit tests with FakePool assert parameterized SQL and that no f-string builds identifiers.
- [ ] **Step 2: FAIL. Step 3: Implement** thin wrappers; every value a bind parameter. **Step 4: Run unit then `-m db`; PASS.** Delete the cluster.
- [ ] **Step 5: Commit** `feat: vehicle visual repository`.

### Task 7: Descriptor service

**Files:**
- Create: `app/schemas/visual.py`, `app/services/vehicle_visuals.py`
- Test: `tests/unit/test_vehicle_visuals.py`, `tests/unit/test_visual_descriptor_contract.py`

**Interfaces:**
- Produces schemas: `Artwork`, `VisualOption(variant_key, label)`, `VisualDescriptor` with the exact fields of the spec (`model_config` forbids extras; `None` fields excluded on dump).
- Produces: `async def get_descriptor(user_id: str, car: dict) -> VisualDescriptor`;
  `async def prepare(user_id: str, car: dict, variant_key: str | None) -> tuple[VisualDescriptor, bool]` (bool = job newly created, drives 200 vs 202);
  raises `VariantMismatch` (-> 422 in Task 8) and `Throttled(retry_after: int)` (-> 429).
- Behaviour: `resolve()` first (with the stored/provided choice, re-validated against `candidates`); `selected` -> row lookup -> `ready` (only if `visual_store.has_object(sha)` and `sign_url` gave a URL) / `pending` (queued, generating, review, or no row yet) / `fallback temporarily_unavailable` (failed or missing object, logging an operator event without the sha path); `choose` -> `fallback choose_variant` with options; `unknown` -> `fallback unknown_variant`.
  Any exception from the visuals tables or store is caught, logged by class name, and returns `fallback temporarily_unavailable`.
  `prepare` throttles per user (`VISUAL_PREPARE_PER_HOUR`, in memory, same style as `usage_limits.py`) and refuses new jobs over `VISUAL_JOBS_PER_HOUR` with `Throttled`; reuse of an existing row never counts.

- [ ] **Step 1: Failing tests:** the contract test re-implements the client checklist (a `client_accepts(descriptor_dict) -> bool` helper in the test file) and asserts it for each state and rejects hand-broken ones (schema_version 2, http URL, 63-hex sha, width 1000); `ready` only when the object exists; pending `retry_after_ms` within 2000..60000; `choose_variant` carries labels only and no other state has `options`; tables missing (FakePool raising `asyncpg.UndefinedTableError`) gives `temporarily_unavailable`; wrong-model variant raises `VariantMismatch`; second prepare for the same variant creates no second job; the user throttle raises `Throttled` with a positive `retry_after`; a stored choice for a car the user no longer owns is cleared (not used); two users, one variant, same sha.
- [ ] **Step 2: FAIL. Step 3: Implement. Step 4: PASS.**
- [ ] **Step 5: Commit** `feat: vehicle visual descriptors`.

### Task 8: Routes and signed downloads

**Files:**
- Create: `app/api/v1/visual_route.py`
- Modify: `app/api/v1/routes.py` (include two routers), `main.py` only if the error handler needs the `Retry-After` header path (it already supports `headers`)
- Test: `tests/unit/test_visual_routes.py`

**Interfaces:**
- Consumes: Task 7 service, Task 2 `verify`, Task 5 `VisualStore`, `get_current_user`, `limiter`.
- Produces: `GET /api/v1/cars/{car_id}/visual` -> 200 descriptor; `POST /api/v1/cars/{car_id}/visual/prepare` (body `{}` or `{"variant_key": str}`) -> 200 or 202; `GET /api/v1/visuals/files/{name}` with query `exp`, `sig` (name = `<64 hex>.webp`).
- `car_id` path uses `Annotated[str, Path(max_length=MAX_CAR_ID_CHARS, pattern=NO_NUL)]` exactly like `delete_car`; ownership via `owned_car`, 404 `RecordNotFoundException("Car")` otherwise (also when the DB lookup of ownership fails with a non-outage error is not applicable: outage keeps the existing 503 mapping).
- Download route: no auth; verifies signature before touching disk; headers per Global Constraints; 404 for unknown/expired/tampered (one generic body, never reveal which); rate-limited with the existing `limiter` (`120/minute`).

- [ ] **Step 1: Failing tests:** 401 without a token on both car routes; other user's car 404 on both; own car 200; POST with `variant_key` of another model 422 and body lists field names only; POST throttled 429 with an integer `Retry-After` header; newly created job answers 202, reuse 200; `car_id` of 201 chars or containing NUL is 422 not 500; visuals tables missing gives 200 fallback; download: valid URL 200 `image/webp` with `Content-Length`, bytes hash to the descriptor sha, `Cache-Control` max-age <= TTL, `nosniff`, no auth header; expired, tampered, wrong-length sig, non-hex name, `..%2f` name, and uppercase-hex name each get 404 and never a 500; a descriptor URL fetched through the `client` fixture round-trips end to end.
- [ ] **Step 2: FAIL. Step 3: Implement. Step 4: PASS.** Confirm the existing `/openapi.json` stays disabled (no change) and CSP/security headers still apply.
- [ ] **Step 5: Commit** `feat: vehicle visual routes and signed downloads`.

### Task 9: Generators and the worker

**Files:**
- Create: `app/workers/__init__.py`, `app/workers/vehicle_visuals.py`, `app/services/visual_generators.py`
- Modify: `docker-compose.yml` (service `visuals-worker`: same `build: .`, `command: ["python", "-m", "app.workers.vehicle_visuals"]`, `env_file: .env`, the same `DATABASE_URL` override, volume `visuals:/data/visuals`, `depends_on` db healthy, `restart: unless-stopped`, no ports); `tests/unit/test_compose.py`
- Test: `tests/unit/test_visual_generators.py`, `tests/unit/test_visual_worker.py`, `tests/integration/test_visuals_db.py` (extend)

**Interfaces:**
- Produces: `class Generator(Protocol): async def generate(self, variant_key: str, style_version: str) -> bytes` raising `GeneratorError(code: str)`;
  `class CuratedGenerator(directory: str)` (reads `{directory}/{variant_key}.webp|png|jpg`, `GeneratorError("no_source")` if none);
  `class ProviderGenerator` (only constructed when `VISUAL_PROVIDER_ENABLED`; the actual HTTP call is one small method `_call(prompt: str) -> bytes` that this plan does not implement against a real service: it raises `GeneratorError("provider_not_configured")` unless the owner supplies it later; the prompt is built from the catalog label only, never user data);
  `build_generator() -> Generator`.
- Produces worker: `async def run_once(gen: Generator, store: VisualStore, now: Callable[[], float] = time.time) -> bool` (claim, run, post-process, park in `review` or schedule retry; True if a job was handled) and `async def main() -> None` (loop with `VISUAL_POLL_SECONDS`, graceful SIGTERM, sweeps the store hourly).
- Rules: claim in a short transaction, release before `generate`; renew the lease every `LEASE_SECONDS/3` while generating; at most 3 attempts then `failed` with `last_error_code`; backoff `min(2**attempt * 30, 900)` seconds plus up to 25% jitter (the jitter source is injectable for tests); before any provider call check `provider_calls_last_day() >= VISUAL_PROVIDER_DAILY_BUDGET` and, if hit, finalize `failed` with `budget_exhausted` and stop calling; a provider timeout still records the call (charged) and counts as an attempt; output goes through `normalize()` into `write_draft`, state `review`, `draft_sha256` set; an invalid image finalizes `failed`/`review` per code. Nothing is ever published here.

- [ ] **Step 1: Failing tests** (fake generator, fake clock): happy path ends in `review` with a draft file; generator error then retry then `failed` after 3 attempts with growing `next_attempt_at`; budget ceiling prevents `_call` (fake counts zero calls) and lands `failed budget_exhausted`; provider timeout is recorded as a call; invalid image (wrong bytes, bomb) lands `failed`, no draft; lease renewal happens during a slow generator; SIGTERM-style cancellation during a job leaves a lease that expires and is reclaimable (real-DB test: kill after claim, expire, second `run_once` handles it); stale-token finalize after reclaim returns False and changes nothing; curated generator reads a file and raises `no_source` when absent; `build_generator()` returns the curated one when the flag is off and never constructs the provider; no test reaches the network (assert `httpx`/`requests` are not called, via a socket-block fixture).
- [ ] **Step 2: FAIL. Step 3: Implement. Step 4: PASS (unit, then `-m db`).**
- [ ] **Step 5: Commit** `feat: vehicle visuals worker and generators`.

### Task 10: Operator CLI

**Files:**
- Create: `scripts/visuals_admin.py`
- Test: `tests/unit/test_visuals_admin.py`, `tests/integration/test_visuals_db.py` (extend)

**Interfaces:**
- Produces (argparse subcommands, each prints one line per result and never prints signed URLs or paths outside `VISUALS_DIR`):
  `list-review`; `approve VARIANT --revision N` (reads the draft, `validate_published`, `put_object`, then `publish(...)` conditioned on `--revision`; on a refused publish prints `refused: stale revision` and exits 2, leaving the previous revision active);
  `reject VARIANT`; `requeue VARIANT`;
  `seed VARIANT --image PATH [--approve]` (normalizes the image from the given path, puts the draft into `review`, creating the row if needed; `--approve` publishes it at the current revision). No default path is hardcoded; for the Echo, the owner passes the mobile repo's `assets/vehicles/toyota-echo.webp`.
- Opens the pool like `scripts/prune_media.py`/`ingest_kb.py` do (read them first; same `sys.path` and pool handling).

- [ ] **Step 1: Failing tests:** approve with a stale `--revision` exits 2 and serves nothing new; approve moves `review` to `ready` and the object exists under its sha; the previous object stays on disk until the new revision is active (publish order: object first, then activate); `seed` from a test image creates a row in `review`; `reject` clears the draft; `requeue` only from failed/review; `list-review` prints variant and revision only; the whole publish path leaves nothing half-done when `publish` returns False (object written but row unchanged is reconciled by `sweep`: assert `sweep` removes the unreferenced object after the grace period and keeps the active one).
- [ ] **Step 2: FAIL. Step 3: Implement. Step 4: PASS (unit, then `-m db`).**
- [ ] **Step 5: Commit** `feat: visuals operator cli`.

### Task 11: Docs, hardening pass, full run

**Files:**
- Create: `docs/VEHICLE_VISUALS.md`
- Modify: `README.md` (CRLF; Edit; one pointer line and the new env names in the optional list)
- Test: whole suite

- [ ] **Step 1:** `docs/VEHICLE_VISUALS.md` covers: architecture, storage and signing choice and why, env var names only (`VISUALS_DIR`, `PUBLIC_BASE_URL`, `VISUAL_URL_TTL_SECONDS`, `VISUAL_PREPARE_PER_HOUR`, `VISUAL_JOBS_PER_HOUR`, `VISUAL_POLL_SECONDS`, `VISUAL_PROVIDER_ENABLED`, `VISUAL_PROVIDER_DAILY_BUDGET`, `VISUAL_CURATED_DIR`), running the worker, seeding and publishing artwork (the exact `seed` + `approve` commands), operator commands, the descriptor contract, failure and fallback behaviour, and an owner deploy checklist: apply `app/db/sql/vehicle_visuals.sql` by hand with the documented `psql` line, set `PUBLIC_BASE_URL=https://api.192-255-184-148.sslip.io`, add the `visuals` volume and `visuals-worker` service (already in compose), then `docker compose up -d --build`; plus a **What could not be verified** section (the real provider call; the mobile client against the live API; the Docker build).
- [ ] **Step 2:** Run `uv run pytest -q` (expect baseline plus new tests, same 7 warnings), the `-m db` visuals tests on a throwaway PG16, then the linter/type checker the repo uses (look for `ruff`/`mypy` config; if none is configured say so in the report rather than adding one). Confirm no `.env*` file or secret was read or printed and no real DB touched; `git status` clean; stop and delete any throwaway cluster.
- [ ] **Step 3: Commit** `docs: vehicle visuals`.
