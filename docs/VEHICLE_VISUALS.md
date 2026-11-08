# Vehicle visuals

The Garage screen of the mobile app shows a workshop-manual illustration of the user's car. The backend works out which
car variant a user's car is, keeps one shared illustration per variant, lets an operator review and publish it, and
serves it through short-lived signed URLs. Artwork is optional by design: when it is missing the app draws its bundled
silhouette, so nothing here can block diagnosis, sessions or manual indexing.

## Architecture

```
 mobile app                       api (uvicorn)                          db (Postgres)
 ----------                       -------------                          -------------
 GET  /cars/{id}/visual   ------>  resolve variant from the catalog  --->  vehicle_visuals
 POST /cars/{id}/visual/prepare    (reads; prepare may insert 1 row)       user_car_visual_choices
 GET  download_url (no token) -->  verify HMAC, stream the object          visual_provider_calls
                                            |
                                            |  volume "visuals" (/data/visuals), shared by api and visuals-worker
                                            v
                                   objects/{sha256}.webp        published, immutable, checksum-addressed
                                   drafts/{variant}/workshop-v1/{token}.webp   unpublished, private
                                            ^
 visuals-worker  (same image)  ----------   |  claims a queued row, generates, normalizes, writes a draft,
   python -m app.workers.vehicle_visuals    |  parks the row in state "review". It never publishes.
                                            |
 operator (docker compose run ...)  --------+  scripts/visuals_admin.py: list-review / approve / reject /
                                               requeue / seed. Approve writes the object, then activates it.
```

Request path. `GET /api/v1/cars/{car_id}/visual` checks the car belongs to the caller (404 otherwise), maps make, model
and year to a variant with `app/services/vehicle_catalog.py` (data in `vehicle_catalog.json`, matching ignores case,
spaces and punctuation), reads the `vehicle_visuals` row and answers with a descriptor. A ready descriptor carries a
signed `download_url`. `POST .../visual/prepare` does the same and, if the variant has no row yet, inserts one queued
row (`insert ... on conflict do nothing`, so concurrent calls make exactly one) that the worker picks up.

Worker. A separate process from the same image (`visuals-worker` in `docker-compose.yml`). Every `VISUAL_POLL_SECONDS`
it claims one due row with `claim_due_visual_job` (`for update skip locked`, a 300 s lease, `fencing_token + 1`),
releases the transaction, then generates the image outside any transaction while renewing the lease every 100 s. The
output is normalized (1024x420 WebP canvas, ground baseline at y=366, metadata stripped, decoded size and bytes
re-checked, at most 400 KiB), written as a draft at `drafts/{variant}/workshop-v1/{fencing_token}.webp`, and the row is
finalized into `review`. A stale worker (lost lease) cannot finalize because the SQL function requires the current
fencing token. Once at startup and then at most hourly the worker sweeps the store: drafts and objects older than one hour that no row references
are deleted; drafts of rows in `review` are always kept.

Operator. `scripts/visuals_admin.py` is a CLI, not an HTTP route. Nothing is served to users until an operator approves.

### Storage and signing decision

- Storage is a local disk volume (`visuals`), mounted at `/data/visuals` in both `api` and `visuals-worker`. The stack is
  one VPS with plain PostgreSQL, so there is no object store to use and none to pay for; the media store already works
  the same way. Published objects are named by their SHA-256, so they are immutable and a re-upload is a no-op.
- Downloads are HMAC-signed URLs, because the mobile client fetches `download_url` with a plain GET and no
  Authorization header. The URL is
  `{PUBLIC_BASE_URL}/api/v1/visuals/files/{sha256}.webp?exp={unix seconds}&sig={hex}`. The signature is an HMAC-SHA-256
  over `{sha256}.{exp}` with a key derived from `JWT_SECRET_KEY` (a fixed label is mixed in), so no new secret exists.
  Verification is constant time and rejects expired URLs, tampered URLs and an expiry further away than the TTL
  (plus 60 s of skew). The file name must match `[0-9a-f]{64}.webp` exactly, so a path cannot leave `objects/`.
- The URL is a capability for one shared image, not for a user: two users with the same variant receive the same
  `sha256`. Ownership of the car is checked when the descriptor is requested and again when the URL is signed.
- Every refusal of a URL (bad name, bad or expired signature, missing or corrupt object) is the same `404 File not
  found`; the route also answers 429 (rate limit) and 405 (wrong method). Refusals carry
  `Cache-Control: no-store, no-cache, must-revalidate, proxy-revalidate`; a successful download carries `private, max-age=<remaining lifetime, at
  most the TTL>`, `Content-Type: image/webp`, `Content-Length` and `X-Content-Type-Options: nosniff`.
- Rotating `JWT_SECRET_KEY` invalidates outstanding download URLs; they live 15 minutes by default, so
  the effect is a short window until fresh descriptors are requested.

## Environment variables

Names only; the defaults are those in `app/core/config.py`. None of these is a secret.

| Name | Default | Meaning |
| --- | --- | --- |
| `VISUALS_DIR` | `/data/visuals` | Root of the private artwork store (`objects/`, `drafts/`). Must be the `visuals` volume. |
| `PUBLIC_BASE_URL` | empty | Public https origin used in signed URLs. Empty or not `https://` means no URL can be signed and every ready variant answers `fallback` / `temporarily_unavailable`. |
| `VISUAL_URL_TTL_SECONDS` | `900` | Lifetime of a signed download URL. |
| `VISUAL_PREPARE_PER_HOUR` | `20` | Per user: a prepare call that would create a row is refused once the user's count in the last hour has reached this value. `0` turns the limit off. |
| `VISUAL_JOBS_PER_HOUR` | `60` | Global: rows created in the last hour; over it, prepare answers 429. `0` turns the limit off. |
| `VISUAL_POLL_SECONDS` | `10` | Worker sleep between polls when nothing is due. |
| `VISUAL_PROVIDER_ENABLED` | `false` | Build the paid-provider generator instead of the curated one. Leave off (see Behaviours to know). |
| `VISUAL_PROVIDER_DAILY_BUDGET` | `20` | Provider calls allowed per rolling day; at the ceiling the worker stops calling and fails the job `budget_exhausted`. |
| `VISUAL_CURATED_DIR` | `/data/visuals/curated` | Where the curated generator looks for `{variant_key}.webp\|png\|jpg`. |

## Running the worker

Compose runs it with the rest of the stack (`docker compose up -d --build`). By hand, the service command is:

```
docker compose up -d visuals-worker
docker compose logs --tail 20 visuals-worker
```

It stops cleanly on SIGTERM (`stop_grace_period: 130s`, longer than the 120 s provider timeout). Outside compose:
`uv run python -m app.workers.vehicle_visuals` with `DATABASE_URL` and `VISUALS_DIR` set, plus the app's other required
settings (`JWT_SECRET_KEY` of 32+ characters and `DEEPSEEK_API_KEY`), or it fails at its first settings load;
`VISUAL_CURATED_DIR` matters too when the curated generator is in use. Run one replica: see
"Behaviours to know" (g). With the provider flag off the worker uses the curated generator, which reads
`{VISUAL_CURATED_DIR}/{variant_key}.webp|png|jpg`. A queued variant without such a file is retried (60 to 75 s, then 120
to 150 s) and, after the third attempt, fails with `no_source`; seed it with the CLI instead.

## Seeding and publishing artwork

The reference artwork for the Toyota Echo is `assets/vehicles/toyota-echo.webp` in the mobile repo. Pass its host path
as an argument; the file is read inside the container through a read-only mount, so no path is hard-coded:

```
docker compose run --rm -v <host path>/assets/vehicles/toyota-echo.webp:/seed/toyota-echo.webp:ro visuals-worker uv run --no-sync python scripts/visuals_admin.py seed toyota-echo-2001-sedan --image /seed/toyota-echo.webp --approve
```

`seed` normalizes the image to the published contract, writes it as the draft and puts the row in `review`. With
`--approve` it then approves that exact draft at the row's current revision. Without `--approve`, review and approve
as a second step:

```
docker compose run --rm visuals-worker uv run --no-sync python scripts/visuals_admin.py list-review
docker compose run --rm visuals-worker uv run --no-sync python scripts/visuals_admin.py approve toyota-echo-2001-sedan --revision 0
```

`--revision` is the `revision=` value `list-review` printed (the number of times the variant has been published so far;
0 for the first publish). Approve writes the immutable object first and activates the revision second, so a crash in
between leaves only an unreferenced object that the hourly sweep removes.

## Operator commands

Run every command as `docker compose run --rm visuals-worker uv run --no-sync python scripts/visuals_admin.py <command>`.
Output is one short line per result (`seed --approve` prints two: `seeded ...` then `approved ...`); shas, signed URLs and filesystem paths are never printed.

| Command | What it does |
| --- | --- |
| `list-review` | One line per row in `review`: `variant revision=N state=review attempts=N`. Exit 0. |
| `approve VARIANT --revision N` | Publishes the draft of a row in `review`, if `N` is its current `published_revision`. Prints `approved VARIANT revision=N+1`. |
| `reject VARIANT` | Discards the draft of a row in `review` and deletes the draft file. A row that was published before returns to `ready` and keeps serving its old artwork; a never-published row becomes `failed` (`rejected`). |
| `requeue VARIANT` | Resets a `failed` or `review` row to `queued` with zero attempts so the worker tries again. |
| `seed VARIANT --image PATH [--approve]` | Puts a curated image into `review` for an allowlisted variant (see above). |

Exit codes: 0 success, 1 usage error / not found / nothing changed, 2 refused, 3 internal error (the output is
`error: <exception class name>` only).

| Message | Exit | Meaning |
| --- | --- | --- |
| `not found` | 1 | approve or seed found no row for that variant. |
| `unknown variant` | 1 | seed: the key is not in the catalog allowlist. |
| `unreadable image: need a regular file of at most 20 MB` | 1 | seed: the path is missing, not a regular file (a symlink is refused), or too large. |
| `nothing to reject` / `nothing to requeue` | 1 | The row is not in a state the command applies to (reject needs `review`; requeue needs `failed` or `review`). |
| `refused: not in review` | 2 | approve: the row is not in `review`. |
| `refused: stale revision` | 2 | approve: `--revision` is not the row's current revision, or the row changed between the check and the publish. Run `list-review` again. |
| `refused: draft missing` | 2 | approve: the draft file is not on disk (swept, or never written). Reject or requeue, or seed again. |
| `refused: draft mismatch` | 2 | approve: the draft file's hash is not the one recorded for the row. Nothing is published. |
| `refused: draft changed` | 2 | seed `--approve`: the row's draft changed after the seed wrote it (a second seed or a worker). Nothing is published; review again. |
| `refused: job running` | 2 | seed: the row is `generating` (a worker holds it, or a crashed one has not been reclaimed yet). |
| `refused: <code>` | 2 | The image failed validation. Codes: `too_large`, `not_image`, `bad_dimensions`, `bad_margins`, `too_many_bytes`. |
| `error: <ClassName>` | 3 | Unexpected failure (for example the database is unreachable). |

## Descriptor contract

Both endpoints return this JSON. Fields that do not apply are omitted. The client discards anything that breaks a hard
rule below.

| Field | Type | Present when |
| --- | --- | --- |
| `schema_version` | `1` | always |
| `state` | `ready`, `pending`, `fallback` | always |
| `variant_key` | string | `ready`, `pending` |
| `style_version` | string (`workshop-v1`) | `ready`, `pending` |
| `preset` | `garage_roll_v1` | `ready`, `pending` (absent on `fallback`) |
| `reason` | `unknown_variant`, `choose_variant`, `temporarily_unavailable` | `fallback` only |
| `retry_after_ms` | integer | `pending` only: 5000, or 30000 while a draft waits in review |
| `options` | `[{variant_key, label}]` | `fallback` with `choose_variant` only; labels, no other data |
| `artwork` | object | `ready` only |

`artwork`: `format` `flat_webp`, `width` 1024, `height` 420, `baseline_y` 366, `sha256` (64 lowercase hex),
`download_url` (https), `expires_at` (ISO-8601, UTC).

States. `ready`: an artwork is published and its object is on disk. `pending`: the row is `queued`, `generating` or
`review` with nothing published yet, or the variant has no row yet. `fallback`: `unknown_variant` (the catalog has no
match), `choose_variant` (several variants fit; pick one by sending `variant_key` to prepare),
`temporarily_unavailable` (anything else: a `failed` row, a missing object, no signable URL, tables not migrated).

Status codes. GET: always 200 for the three states, 404 for a car that does not exist or is not the caller's (never
403). Prepare: 200 (existing row) or 202 (this call created the job); 404 as above; 422 when `variant_key` is not
allowlisted or belongs to another make or model than the car (the body names the field, never the value); 429 with an
integer `Retry-After` when throttled. The body is `{}` or `{"variant_key": "<key>"}`; other fields are rejected.

The download: plain GET, no Authorization header, `200`, `Content-Type: image/webp`, `Content-Length` at most 409,600
bytes, bytes whose SHA-256 equals `artwork.sha256`, decoded as 1024x420. The route is limited to 120 requests a minute
per client address.

## Failure and fallback behaviour

| Situation | What happens |
| --- | --- |
| Visuals tables not migrated | GET and prepare answer 200 `fallback` / `temporarily_unavailable`; the api logs `vehicle visuals unavailable: <ClassName>`; the worker logs `visual claim failed` every poll. Apply the migration. |
| Database unreachable | The app's standard database-unavailable response, as for every other route. |
| Published object missing on disk | `fallback` / `temporarily_unavailable` and the warning `published artwork is missing from the store` in the api log (operator-visible). Seed and approve again. |
| `PUBLIC_BASE_URL` empty or not https | No URL is signed: `fallback` / `temporarily_unavailable`, warning `could not sign a download url`. |
| Generation fails (no source, provider error, timeout) | Up to 3 attempts, backoff `min(30 * 2^attempt, 900)` seconds plus up to 25% jitter; then `failed` with a stable `last_error_code`. A `failed` row answers `fallback` / `temporarily_unavailable` and is not retried until an operator runs `requeue`. |
| Invalid image from the generator | The row goes straight to `failed` with the validation code; no draft is kept. |
| Provider budget reached | The worker stops calling the provider and fails the job `budget_exhausted` (descriptor: `temporarily_unavailable`). Raise the budget or wait, then `requeue`. |
| Worker crashes mid-job | The row stays `generating` until its 300 s lease expires, then the next claim hands it out again with a new fencing token (an attempt is counted). The crashed worker, if it were alive, could not finalize. A job reclaimed past the 3rd attempt becomes `failed` (`max_attempts`). |
| Crash between writing the object and activating it | Only an unreferenced object remains; the sweep deletes it after an hour. |
| Variant has a published revision and a new draft fails or waits in review | The published artwork keeps being served (see (b) below). |
| A car is removed while a request is in flight | The descriptor service re-checks ownership before signing; the answer is 404. |

## Behaviours to know

- (a) `GET /cars/{id}/visual` never creates a job; only `POST .../prepare` does (ruling R8). For a variant with no row,
  GET answers `pending` forever, so the mobile client must call prepare to get artwork produced.
- (b) A row that was ever published keeps serving its published artwork whatever its state while a new draft is
  reviewed, is regenerated or fails (R7). `reject` restores `ready` for such a row.
- (c) Approve is fenced on the reviewed draft (R12): `publish_visual_revision` requires `draft_sha256 = p_sha` and
  `published_revision = N`, so a re-seed between review and approve makes approve refuse (`stale revision`).
- (d) Seed refuses every `generating` row (`refused: job running`), even one whose worker died. `requeue` only applies to
  `failed` or `review` rows, so an orphaned `generating` row cannot be requeued by hand: keep the worker running; it
  reclaims the row once the lease (300 s) has expired, after which the row is `review`, `queued` or `failed` and can be
  seeded.
- (e) A review draft lives at `drafts/{variant}/workshop-v1/{fencing_token}.webp`, and the sweep (at startup, then at
  most hourly) keeps the drafts
  of rows in `review` (R11); other drafts and unreferenced objects are removed after one hour.
- (f) uvicorn's access log is redacted for `sig` and `exp` by `app/utils/log_redaction.py`, but if Caddy access logging
  is ever enabled it WOULD record the signed URL. Keep it off, or redact `sig` and `exp` for `/api/v1/visuals/files/*`
  (R10). The README's Caddy block does not enable it.
- (g) The provider budget check and its recording are not atomic across several workers. That is latent while one
  replica runs and the provider call is unwired. Upgrade path before running more replicas: one SQL function that takes
  an advisory lock, counts the day's calls, and inserts the ledger row in the same statement.
- (h) `ProviderGenerator._call` is intentionally unimplemented: it raises `provider_not_configured`, and the flag
  `VISUAL_PROVIDER_ENABLED` defaults to off. Turning the flag on today only makes each job record a ledger row and
  fail; wire a real service into `_call` first. No real provider key or URL is documented because none is used.
- (i) The rate limits (including the download route's) and the per-user prepare throttle live in process memory: they
  reset on restart and multiply with the number of api workers. The image runs one.
- (j) The descriptor service re-checks car ownership when signing, so a car removed mid-flight gives 404.
- After a republish the previous revision's object keeps its old mtime, so the next sweep deletes it, and any
  still-live signed URL for the old image (up to the 15-minute TTL) answers 404 once. The client falls back to its
  silhouette and picks up the new artwork on its next descriptor fetch.
- The host file passed to `seed` must be world-readable: the container user is uid 10001.
- Adding the `02-` init mount to `db` makes `docker compose up -d --build` recreate the `db` container (a few seconds of
  downtime; the data volume is kept).
- The `visuals` volume is not part of `scripts/backup.sh`. The rows live in the database dump; the images can be
  re-created by seeding again from the source artwork.

## Owner deploy checklist

Existing VPS, run from the repository directory. Nothing here has been run against the real server by the implementer.

1. Pull the branch: `git fetch && git checkout feat/vehicle-visuals && git pull`.
2. Apply the migration by hand. The file only creates what is missing and replaces its functions, so it is safe to run
   again, and it must be RE-RUN after pulling this branch if an earlier revision of the file was applied (the
   `publish_visual_revision` function changed):
   ```
   docker compose exec -T db psql -U autocare -d autocare -v ON_ERROR_STOP=1 -f - < app/db/sql/vehicle_visuals.sql
   ```
   (A brand-new `pgdata` volume applies it by itself on first start.)
3. In the VPS `.env` (edit it; do not print it) set `PUBLIC_BASE_URL=https://api.192-255-184-148.sslip.io`. Keep the other
   visual variables at their defaults unless you want to change them.
4. `docker compose up -d --build`. This adds the `visuals` volume and the `visuals-worker` service. The Dockerfile now
   creates and chowns `/data/visuals`. If a `visuals` volume was created before that Dockerfile change, fix its owner
   once:
   ```
   docker compose run --rm --user root api chown autocare:autocare /data/visuals
   ```
5. Seed and approve the artwork with the command in "Seeding and publishing artwork".
6. Check: `docker compose logs --tail 20 visuals-worker`, then request the descriptor from the mobile app, or with curl
   and a user's access token: `GET https://api.192-255-184-148.sslip.io/api/v1/cars/<car id>/visual`. A ready
   descriptor has an https `download_url`; open it without a token to confirm a 200 `image/webp`.

## What could not be verified here

- The Docker image build and `docker compose up` (the Docker engine hung twice on the development machine, so the
  Dockerfile and compose changes were reviewed but not built or started).
- A real provider call (none is wired) and the provider budget under real concurrency.
- The mobile client against the live API; the descriptor contract is tested with a re-implementation of the client's
  validation checklist, not with the app.
- Caddy's behaviour with the new route (`/api/v1/visuals/files/*`, no bearer token), including its logging.
- The Linux-only file-permission assertions (modes `0700` / `0600`) beyond the store unit test, which the implementer
  ran under WSL.
- Lint and type checking: the repository configures neither, and no ruff or flake8 is installed in its environment.
