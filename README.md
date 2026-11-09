# AutoCare Backend

FastAPI backend for an AI car-diagnosis assistant. A user signs up, adds a car, opens a diagnosis session and chats about a problem with text, photos, audio or video. Each turn streams back as Server-Sent Events. Answers are grounded in a Postgres full-text knowledge base (car manuals and books) and use DeepSeek as the LLM (at most 2 LLM calls per turn: one streamed reply, one state update). Audio is transcribed with faster-whisper and engine sounds are tagged with CLAP.

It runs as three containers from `docker-compose.yml`: `db` (Postgres 16, not published to the host), `api` (this app, published on `127.0.0.1:8000` only, for a reverse proxy on the same host) and `visuals-worker` (vehicle artwork jobs, same image, no published port). Uploaded media lives on a Docker volume.

## Configuration

Settings come from `.env` (`.env.example` lists every name with its default; `app/core/config.py` is the source of truth).

Required: `JWT_SECRET_KEY` (32+ chars), `POSTGRES_PASSWORD`, `DATABASE_URL`, `DEEPSEEK_API_KEY`.

Optional: `REGISTRATION_CODE`, `ENABLE_DOCS`, `DAILY_MESSAGE_LIMIT`, `MAX_CONCURRENT_MEDIA_TURNS`, `GOOGLE_CLIENT_IDS`, `FORWARDED_ALLOW_IPS`, `ENV`, `DEBUG`, `ALLOWED_ORIGINS`, `DEEPSEEK_BASE_URL`, `DEEPSEEK_MODEL`, `WHISPER_MODEL`, `CLAP_MODEL`, `SOUND_MIN_SCORE`, `MAX_FRAMES`, `MAX_MEDIA_SECONDS`, `MAX_MEDIA_MB`, `MAX_IMAGES`, `MAX_IMAGE_MB`, `KB_TOP_K`, `HISTORY_MESSAGES`, `MEDIA_DIR`, `MEDIA_MIN_FREE_MB`, `MEDIA_USER_QUOTA_MB`, `VISUALS_DIR`, `PUBLIC_BASE_URL`, `VISUAL_URL_TTL_SECONDS`, `VISUAL_PREPARE_PER_HOUR`, `VISUAL_JOBS_PER_HOUR`, `VISUAL_POLL_SECONDS`, `VISUAL_PROVIDER_ENABLED`, `VISUAL_PROVIDER_DAILY_BUDGET`, `VISUAL_CURATED_DIR`.

Vehicle artwork (the `visuals-worker` service, signed download URLs, the operator CLI and the deploy steps, including the migration `app/db/sql/vehicle_visuals.sql`) is documented in [docs/VEHICLE_VISUALS.md](docs/VEHICLE_VISUALS.md).

- `POSTGRES_PASSWORD` is also pasted into the api's `DATABASE_URL`, so it must be URL-safe (letters, digits, `-_.~`). Generate it with `openssl rand -hex 24`.
- Inside compose, the api's `DATABASE_URL` is set by `docker-compose.yml` (host `db`) and overrides `.env`. The `.env` value is only for running the app or scripts outside compose, e.g. `postgresql://autocare:<password>@localhost:5432/autocare`.
- `JWT_SECRET_KEY` signs the app's own tokens (HS256): `openssl rand -hex 32`.
- Set `ENV=production` on the server (adds the HSTS header).
- Set `REGISTRATION_CODE` on any public deployment: it is how sign-up is restricted (see [Abuse protection](#abuse-protection)). Left empty, anyone can register.

## VPS setup

Needs Docker with the compose plugin.

```
git clone <repo> autocare-backend && cd autocare-backend
cp .env.example .env              # fill in the required values (openssl rand -hex 32 / -hex 24)
docker compose up -d --build
curl http://localhost:8000/health # {"status":"ok","database":true,"deepseek":true}
```

The first build bakes the Whisper and CLAP models into the image, so it takes a while; the api health check allows 90 s to start. `/health` is public and not rate limited. It answers `503` with `"status":"unavailable","database":false` while Postgres is unreachable, so `docker compose ps` shows the api as unhealthy; a DeepSeek outage alone keeps `200` with `"deepseek":false`.

First-run knowledge base: put the PDFs in `car_data/` on the host (`<car_id>_manual.pdf` is that car's manual, any other PDF is a book) and run:

```
docker compose run --rm -v ./car_data:/app/car_data:ro api uv run --no-sync python scripts/ingest_kb.py
```

`car_data/` is never copied into the image; it is only mounted for this command. Re-running replaces the passages of each PDF it finds.

### Schema changes

`app/db/sql/schema.sql` is mounted into Postgres' init directory, so it runs **only when the `pgdata` volume is empty** (the very first start). The app never changes the schema. Later changes are applied by hand:

```
docker compose exec -T db psql -U autocare -d autocare -v ON_ERROR_STOP=1 < app/db/sql/schema.sql
```

The file is idempotent (`if not exists` / `or replace`), so re-running it adds new tables, indexes and functions, but it does not add columns to tables that already exist: write those `alter table` statements by hand with `docker compose exec db psql -U autocare -d autocare`.

A database created before the session lease existed needs its columns (new databases already have them). The same command also fixes one that already has `busy_until` but lacks `busy_by`: each `add column if not exists` skips a column that is there.

```
docker compose exec -T db psql -U autocare -d autocare -v ON_ERROR_STOP=1 -c 'alter table "Chatsession" add column if not exists busy_until timestamptz, add column if not exists busy_by uuid;'
```

### Reverse proxy (Caddy)

Put Caddy in front for TLS. Response buffering must be off for SSE, and the body limit must allow the largest upload (`max(MAX_IMAGES * MAX_IMAGE_MB, MAX_MEDIA_MB) + 1` MB, 101 MB with the defaults):

```
api.example.com {
    request_body {
        max_size 101MB
    }
    reverse_proxy 127.0.0.1:8000 {
        flush_interval -1
    }
}
```

Caddy sends `X-Forwarded-For`/`-Proto`/`-Host` by default. The api runs uvicorn with `--proxy-headers`, which trusts those headers only from `FORWARDED_ALLOW_IPS` (IPs or CIDRs, comma-separated; default `127.0.0.1`). Rate limits are per client address, so set it to the address Caddy's requests arrive from, or every user shares one limit. With Caddy on the host that is the compose network's gateway:

```
docker network inspect autocare-backend_default -f '{{(index .IPAM.Config 0).Gateway}}'
```

Put that in `.env` as `FORWARDED_ALLOW_IPS` and run `docker compose up -d`. Never trust a range that untrusted clients can connect from. Compose publishes the api on `127.0.0.1:8000` only, so plain HTTP never leaves the host. To expose it directly without Caddy (plain HTTP, not recommended), change the mapping in `docker-compose.yml` to `"8000:8000"`.

### Workers

The image runs a single uvicorn worker, and that is the recommended setting. The lock that stops two turns (or a turn and a delete) from running at once in one session (`409`) is a lease in the database (`"Chatsession".busy_until` and its holder `busy_by`: 5 minutes, renewed every minute while a turn runs, however long the model takes, and expiring on its own if a worker dies; a turn whose lease was taken over stops without saving), so several uvicorn workers or api replicas are safe for it. Two things still multiply per worker:

- **Rate limits** are counted in each process's memory, so with N workers a client gets up to N times every limit (10/min for sign-in, 20/min for messages).
- **Whisper and CLAP** are loaded once per worker, so each worker needs their RAM again. Keep 1 worker unless the host has the memory for more.

API replicas on other hosts must keep their clocks NTP-synced with each other and with the database host. Token revocation compares the app's clock (token `iat`, the logout cutoff) with the database's (a Google link), so a skew of a few seconds lets tokens minted that long before a logout on another host survive it, or refuses fresh sign-ins on a slower host for that long (PyJWT also rejects an `iat` from the future).

## Backup and restore

```
bash scripts/backup.sh
```

It writes `backups/YYYY-MM-DD.sql.gz` (`pg_dump --clean --if-exists`) and `backups/YYYY-MM-DD-media.tar.gz` (the media volume), keeps the newest 14 of each and prints a crontab line for a daily run. `backups/` is git-ignored; copy it off the server. The script is committed as executable, but `bash scripts/backup.sh` works without relying on the file mode.

Restore the database (the dump drops and recreates its tables, so stop the api first):

```
docker compose stop api
gunzip -c backups/2026-11-06.sql.gz | docker compose exec -T db psql -U autocare -d autocare -v ON_ERROR_STOP=1
```

Restore the media (the volume is `<project>_media`, by default `autocare-backend_media`):

```
docker run --rm -i -v autocare-backend_media:/data postgres:16-alpine tar -xzf - -C /data < backups/2026-11-06-media.tar.gz
docker compose start api
```

## Auth

- **Email + password**: `POST /api/v1/auth/register` and `POST /api/v1/auth/login` with `{"email", "password"}` (8 to 1024 chars, argon2id). There is no email confirmation, so email+password accounts are **not verified**. Register returns the same tokens as login; a taken email gets `409`. With `REGISTRATION_CODE` set, register also needs `"invite_code"` (see [Abuse protection](#abuse-protection)). Unknown email and wrong password get the same `401`.
- **Google**: in Google Cloud Console create OAuth client ids for each client you ship (Web, Android, iOS) and put them comma-separated in `GOOGLE_CLIENT_IDS`. The client signs in with Google, gets an **ID token** and sends `{"id_token": "..."}` to `POST /api/v1/auth/google`; the server checks signature, audience, issuer, expiry and `email_verified` and returns the app's tokens. With `GOOGLE_CLIENT_IDS` empty the route returns `404`.
- **Linking**: when the email already has a password account, Google sign-in links to it only if Google is authoritative for the address (`@gmail.com`, `@googlemail.com`, or a Workspace token with an `hd` claim). Linking removes the password and invalidates every token issued before it, so someone who pre-registered the address loses access; the Google user inherits the account's cars and sessions. For any other address Google sign-in returns `401` instead of linking. New accounts are created for any verified Google email.
- **Tokens and refresh**: every sign-in returns `access_token` (30 minutes) and `refresh_token` (7 days), stateless HS256 JWTs. Before the access token expires, send `{"refresh_token": "..."}` to `POST /api/v1/auth/refresh` (no bearer header) for a new pair in the same shape. Only a refresh token is accepted there, and one issued before a logout or a Google link, or for a deleted account, is refused; every failure is the same `401` (`Invalid refresh token`). The old refresh token keeps working until it expires, so clients should store the new one.
- **Logout**: `POST /api/v1/auth/logout` (bearer header) signs the account out on **every device**: all of its access and refresh tokens issued until then stop working (`401`, `/auth/refresh` included), and the next sign-in works as usual. Per-device logout is not implemented. Clients should still discard both tokens. One narrow race is accepted: a refresh that is mid-flight at the very moment of the logout (a sub-second window) can still return a new pair, and a client holding that pair can keep refreshing it; anything later is refused.
- **Database outages**: while Postgres is unreachable, every route that needs it (sign-in, refresh, logout, any token check, cars, sessions) answers `503` (`{"error_code": "SYS_002", "message": "Service temporarily unavailable", ...}`), never `401`, `404` or an empty list. Clients should keep their tokens and retry; only a `401` means sign in again.
- Register, login, Google sign-in and refresh are each limited to 10 requests per minute per client address, chat messages to 20.

Accounts have no delete route. Deleting one by hand (`delete from users where id = '...'`) cascades to the profile and sessions, but leaves the user's uploaded media in `MEDIA_DIR/{user_id}`; remove it with `docker compose exec api rm -rf /data/media/<user_id>`.

## API

Interactive reference: `http://localhost:8000/docs`, served only when `ENABLE_DOCS=true` (off by default: `/docs`, `/redoc` and `/openapi.json` answer `404`). When on, they are public. Every route except `/api/v1/auth/register|login|google|refresh`, `/health`, the docs and `GET /api/v1/visuals/files/{sha256}.webp` (public on purpose: its HMAC-signed `exp`/`sig` URL is the credential) needs `Authorization: Bearer <access_token>`; the user id always comes from the token.

| Area | Routes |
| --- | --- |
| Auth | `POST /api/v1/auth/register`, `POST /api/v1/auth/login`, `POST /api/v1/auth/google`, `POST /api/v1/auth/refresh`, `POST /api/v1/auth/logout` |
| Cars | `GET /api/v1/cars`, `POST /api/v1/cars` (`{"make", "model", "year", "transmission"?}`, transmission is `manual`, `automatic`, `cvt` or `dual_clutch`), `PATCH /api/v1/cars/{car_id}` (`{"transmission"}`), `DELETE /api/v1/cars/{car_id}` |
| Vehicle visuals | `GET /api/v1/cars/{car_id}/visual`, `POST /api/v1/cars/{car_id}/visual/prepare` (`{}` or `{"variant_key"}`), `GET /api/v1/visuals/files/{sha256}.webp?exp&sig` (no token; see [docs/VEHICLE_VISUALS.md](docs/VEHICLE_VISUALS.md)) |
| Sessions | `POST /api/v1/sessions` (`{"car_id"}`), `GET /api/v1/sessions`, `GET /api/v1/sessions/{id}`, `DELETE /api/v1/sessions/{id}` |
| Chat | `POST /api/v1/sessions/{id}/messages` (multipart form: `text`, `files`; replies as SSE) |

Car `make` and `model` must be 1 to 64 characters and `year` between 1900 and next year; car ids are at most 200 characters; NUL characters are refused. Such input gets `422`. Adding a car starts a background ingest of its manuals: startmycar first, then up to `MANUAL_MAX_SOURCES` (default 3) PDFs in total found with the Brave Search API when `BRAVE_API_KEY` is set; the car's `kb_status` (`pending`, `ready`, `failed`, `no_manual`) tracks progress. Deleting a session deletes its media.

**Errors.** Every error response, whatever the status (validation, body too large, rate limit, unknown route, server error), has the same JSON body: `{"error_code": "VAL_001", "message": "...", "correlation_id": "...", "timestamp": "..."}` (FastAPI's `{"detail": ...}` is not used). A `422` from request validation adds `fields`, the names of the invalid fields; the rejected values are never echoed. `correlation_id` is the request's id, also sent on every response as the `X-Correlation-ID` header and written on the server's log lines for that request (`[ID: ...]`); quote it when reporting a problem. A `409` (`BIZ_002`) on a session means a reply is being generated or the session is being deleted; retry later.

### SSE events

| Event | Data |
| --- | --- |
| `stage` | `{"name": "processing_media" \| "searching"}` (`processing_media` only when files were sent) |
| `token` | `{"text": "..."}` a piece of the reply |
| `state` | updated diagnosis state: `suspected_causes` (cause, likelihood, evidence, next_check), `safety_flags`, `title` |
| `error` | `{"message": "..."}` ends the stream |
| `done` | `{"message_id": "..."}` ends the stream |

```
event: stage
data: {"name": "searching"}

event: token
data: {"text": "A squeal "}

event: token
data: {"text": "when braking usually points to worn pads."}

event: state
data: {"suspected_causes": [{"cause": "Worn brake pads", "likelihood": 0.7, "evidence": "Squeal when braking", "next_check": "Inspect pad thickness"}], "safety_flags": [], "title": "Brake squeal"}

event: done
data: {"message_id": "..."}
```

## Media

A message needs text (up to 5000 chars) or at least one file: up to 4 images of at most 10 MB each, or one audio/video file of at most 60 s and 100 MB (`MAX_IMAGES`, `MAX_IMAGE_MB`, `MAX_MEDIA_SECONDS`, `MAX_MEDIA_MB`). Bodies are capped before they are read and before auth: the chat route takes up to `max(MAX_IMAGES * MAX_IMAGE_MB, MAX_MEDIA_MB) + 1` MB, every other route 1 MB; larger gets `413` (`VAL_005`).

Originals are kept in `MEDIA_DIR/{user_id}/{session_id}/{message_id}/` (default `/data/media`, the `media` volume) until the session is deleted or `scripts/prune_media.py` removes them, so the volume grows with use; nothing deletes media on its own, so watch the disk, schedule the prune below and include the volume in backups. Two limits keep it in check; when either refuses a file, that original is not stored, the turn still runs and a warning is logged (class name only). When the volume has less than `MEDIA_MIN_FREE_MB` free (default 2048) new originals are not stored. At startup the api logs a warning if `MEDIA_DIR` is not writable.

### Per-user quota

`MEDIA_USER_QUOTA_MB` (default 500, `0` turns the cap off) limits what one user may keep in `MEDIA_DIR/{user_id}`. An upload that would push the user past it is refused like a low-disk one: the file is left out of the message's `media`. The check sums the user's regular files (symlinks are not followed) just before each write, so uploads racing each other can overshoot by a file or two.

### Pruning old media

```bash
uv run python scripts/prune_media.py --days 90 --dry-run   # print what would go
uv run python scripts/prune_media.py --days 90             # default is 90
```

It deletes files under `MEDIA_DIR` whose modification time is older than `--days` days (only paths shaped `{user_id}/{session_id}/{message_id}/{n}.jpg|mp4|m4a`; symlinks are never followed), removes their refs from the `media` arrays of the affected messages (one update per session, message order and other fields kept; `updated_at` is not changed), then removes emptied folders. It prints one summary line (files, bytes, messages updated) and never logs content. Running it again is harmless. `--dry-run` only counts files and bytes and does not touch the database. Pruned originals are gone for good, so keep the backups. The api image includes `scripts/`, and `docker compose run` reuses the `api` service's `media:/data/media` mount, so a daily cron entry on the host is:

```
0 3 * * * cd /path/to/autocare-backend && docker compose run --rm api uv run --no-sync python scripts/prune_media.py --days 90
```

## Abuse protection

The API is meant to be reachable from the internet, so these limits are on by default or one setting away. Defaults are in `app/core/config.py`.

- **Invitation code** (`REGISTRATION_CODE`, empty by default = open registration; a value that is only whitespace counts as empty, and leading and trailing whitespace is ignored on both the setting and the client's `invite_code`). Set it on a public deployment. While it is set, `POST /api/v1/auth/register` needs the same value in the body field `invite_code`; a missing and a wrong code both get `403` `Registration is by invitation.`, checked before any hashing or database work, and the code is never logged. `POST /api/v1/auth/google` then signs in or links existing accounts as usual but refuses to create a new one (the same `403`). Existing accounts are not affected.
- **API docs off** (`ENABLE_DOCS`, default `false`). `/docs`, `/redoc` and `/openapi.json` answer the normal `404` unless it is `true`.
- **Daily message cap** (`DAILY_MESSAGE_LIMIT`, default 200, `0` turns it off). Each accepted `POST /sessions/{id}/messages` counts one for the account per UTC day. Past the cap the route answers `429` (`EXT_003`) with `Retry-After` set to the seconds until midnight UTC, after the files are validated and before any model work starts. Only a message that gets as far as the model counts: one refused for an unknown session, invalid input or files, a busy session (`409`) or a database outage is not counted.
- **Media ceiling** (`MAX_CONCURRENT_MEDIA_TURNS`, default 2, `0` turns both limits off). At most this many turns that carry files (images, audio, video) run at once in the process, and one account never has more than one running (so it cannot hold every slot). A media turn over either limit is answered `429` at once (`Retry-After: 15`, `The server is busy processing other media. Try again shortly.`) before anything is claimed or stored. Text-only turns are not limited.
- **Per process.** The daily counts and the media slots live in the process's memory: they reset when it restarts, and with several workers each would allow its own share (like the rate limits, see [Workers](#workers)). The image runs one worker.

## Development and tests

```
uv sync
uv run uvicorn main:app --reload          # needs DATABASE_URL pointing at a Postgres with schema.sql applied
uv run pytest -q                          # offline: fakes only, no database or network
```

The real-database tests are marked `db` and need a throwaway database in `TEST_DATABASE_URL` (they apply the schema and write rows):

```
docker run --rm -d --name autocare-testdb -e POSTGRES_PASSWORD=test -p 127.0.0.1:54329:5432 postgres:16-alpine
TEST_DATABASE_URL=postgresql://postgres:test@127.0.0.1:54329/postgres uv run pytest -q -m db
docker stop autocare-testdb
```

## Scripts

- `scripts/ingest_kb.py [--dir car_data]`: load PDFs into the knowledge base (see VPS setup for the compose command).
- `scripts/prune_media.py [--days 90] [--dry-run]`: deletes media originals older than N days and strips their refs from messages (see Media).
- `scripts/e2e.py --base https://api.example.com --email you@example.com --password ... [--register] [--media-dir DIR] [--allow-skip]`: checks `/health` (`database` and `deepseek` must be true), signs in (`--register` creates the account when login answers `401`), adds a car, sends a text, photo, audio and video turn, and deletes the session. Sample media is read from `tests/fixtures/e2e/` (git-ignored); a missing file fails the run unless `--allow-skip`.
- `scripts/run_cases.py [--only id,id] [--out path]`: diagnosis quality cases against the real DeepSeek and knowledge base, written as a Markdown report to `reports/`. It needs the dev dependencies (`uv sync`) and a `DATABASE_URL` that reaches the database, which is not published; on the VPS host use the db container's address, e.g. `$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' $(docker compose ps -q db))`.
- `scripts/backup.sh`: see Backup and restore.
