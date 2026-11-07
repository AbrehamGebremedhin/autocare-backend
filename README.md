# AutoCare Backend

FastAPI backend for an AI car-diagnosis assistant. A user signs up, adds a car, opens a diagnosis session and chats about a problem with text, photos, audio or video. Each turn streams back as Server-Sent Events. Answers are grounded in a Postgres full-text knowledge base (car manuals and books) and use DeepSeek as the LLM (at most 2 LLM calls per turn: one streamed reply, one state update). Audio is transcribed with faster-whisper and engine sounds are tagged with CLAP.

It runs as two containers from `docker-compose.yml`: `db` (Postgres 16, not published to the host) and `api` (this app, published on `127.0.0.1:8000` only, for a reverse proxy on the same host). Uploaded media lives on a Docker volume.

## Configuration

Settings come from `.env` (`.env.example` lists every name with its default; `app/core/config.py` is the source of truth).

Required: `JWT_SECRET_KEY` (32+ chars), `POSTGRES_PASSWORD`, `DATABASE_URL`, `DEEPSEEK_API_KEY`.

Optional: `GOOGLE_CLIENT_IDS`, `FORWARDED_ALLOW_IPS`, `ENV`, `DEBUG`, `ALLOWED_ORIGINS`, `DEEPSEEK_BASE_URL`, `DEEPSEEK_MODEL`, `WHISPER_MODEL`, `CLAP_MODEL`, `SOUND_MIN_SCORE`, `MAX_FRAMES`, `MAX_MEDIA_SECONDS`, `MAX_MEDIA_MB`, `MAX_IMAGES`, `MAX_IMAGE_MB`, `KB_TOP_K`, `HISTORY_MESSAGES`, `MEDIA_DIR`, `MEDIA_MIN_FREE_MB`.

- `POSTGRES_PASSWORD` is also pasted into the api's `DATABASE_URL`, so it must be URL-safe (letters, digits, `-_.~`). Generate it with `openssl rand -hex 24`.
- Inside compose, the api's `DATABASE_URL` is set by `docker-compose.yml` (host `db`) and overrides `.env`. The `.env` value is only for running the app or scripts outside compose, e.g. `postgresql://autocare:<password>@localhost:5432/autocare`.
- `JWT_SECRET_KEY` signs the app's own tokens (HS256): `openssl rand -hex 32`.
- Set `ENV=production` on the server (adds the HSTS header).

## VPS setup

Needs Docker with the compose plugin.

```
git clone <repo> autocare-backend && cd autocare-backend
cp .env.example .env              # fill in the required values (openssl rand -hex 32 / -hex 24)
docker compose up -d --build
curl http://localhost:8000/health # {"status":"ok","database":true,"deepseek":true}
```

The first build bakes the Whisper and CLAP models into the image, so it takes a while; the api health check allows 90 s to start. `/health` is public and not rate limited.

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

### One worker

The image runs a single uvicorn worker. The lock that stops two turns from running at once in one session (`409`) lives in process memory, so do not add `--workers` or run more than one api replica.

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

- **Email + password**: `POST /api/v1/auth/register` and `POST /api/v1/auth/login` with `{"email", "password"}` (8 to 1024 chars, argon2id). There is no email confirmation, so email+password accounts are **not verified**. Register returns the same tokens as login; a taken email gets `409`. Unknown email and wrong password get the same `401`.
- **Google**: in Google Cloud Console create OAuth client ids for each client you ship (Web, Android, iOS) and put them comma-separated in `GOOGLE_CLIENT_IDS`. The client signs in with Google, gets an **ID token** and sends `{"id_token": "..."}` to `POST /api/v1/auth/google`; the server checks signature, audience, issuer, expiry and `email_verified` and returns the app's tokens. With `GOOGLE_CLIENT_IDS` empty the route returns `404`.
- **Linking**: when the email already has a password account, Google sign-in links to it only if Google is authoritative for the address (`@gmail.com`, `@googlemail.com`, or a Workspace token with an `hd` claim). Linking removes the password and invalidates every token issued before it, so someone who pre-registered the address loses access; the Google user inherits the account's cars and sessions. For any other address Google sign-in returns `401` instead of linking. New accounts are created for any verified Google email.
- **Tokens and logout**: every sign-in returns `access_token` (30 minutes) and `refresh_token` (7 days), stateless HS256 JWTs. `POST /api/v1/auth/logout` only checks the bearer token; nothing is revoked, so clients must discard both tokens.
- Register, login and Google sign-in are limited to 10 requests per minute per client address, chat messages to 20.

Accounts have no delete route. Deleting one by hand (`delete from users where id = '...'`) cascades to the profile and sessions, but leaves the user's uploaded media in `MEDIA_DIR/{user_id}`; remove it with `docker compose exec api rm -rf /data/media/<user_id>`.

## API

Interactive reference: `http://localhost:8000/docs`. `/docs`, `/redoc` and `/openapi.json` are public. Every route except `/api/v1/auth/register|login|google`, `/health` and the docs needs `Authorization: Bearer <access_token>`; the user id always comes from the token.

| Area | Routes |
| --- | --- |
| Auth | `POST /api/v1/auth/register`, `POST /api/v1/auth/login`, `POST /api/v1/auth/google`, `POST /api/v1/auth/logout` |
| Cars | `GET /api/v1/cars`, `POST /api/v1/cars` (`{"make", "model", "year"}`), `DELETE /api/v1/cars/{car_id}` |
| Sessions | `POST /api/v1/sessions` (`{"car_id"}`), `GET /api/v1/sessions`, `GET /api/v1/sessions/{id}`, `DELETE /api/v1/sessions/{id}` |
| Chat | `POST /api/v1/sessions/{id}/messages` (multipart form: `text`, `files`; replies as SSE) |

Adding a car starts a background ingest of its manual; the car's `kb_status` (`pending`, `ready`, `failed`, `no_manual`) tracks progress. Deleting a session deletes its media.

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

Originals are kept in `MEDIA_DIR/{user_id}/{session_id}/{message_id}/` (default `/data/media`, the `media` volume) until the session is deleted, so the volume grows with use; watch its disk and include it in backups. When the volume has less than `MEDIA_MIN_FREE_MB` free (default 2048), new originals are not stored (the turn still runs and a warning is logged). At startup the api logs a warning if `MEDIA_DIR` is not writable.

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
- `scripts/e2e.py --base https://api.example.com --email you@example.com --password ... [--register] [--media-dir DIR] [--allow-skip]`: checks `/health` (`database` and `deepseek` must be true), signs in (`--register` creates the account when login answers `401`), adds a car, sends a text, photo, audio and video turn, and deletes the session. Sample media is read from `tests/fixtures/e2e/` (git-ignored); a missing file fails the run unless `--allow-skip`.
- `scripts/run_cases.py [--only id,id] [--out path]`: diagnosis quality cases against the real DeepSeek and knowledge base, written as a Markdown report to `reports/`. It needs the dev dependencies (`uv sync`) and a `DATABASE_URL` that reaches the database, which is not published; on the VPS host use the db container's address, e.g. `$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' $(docker compose ps -q db))`.
- `scripts/backup.sh`: see Backup and restore.
