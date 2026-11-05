# AutoCare Backend

FastAPI backend for an AI car-diagnosis assistant. A user registers, adds a car, opens a diagnosis session and chats about a problem with text, photos, audio or video. Each turn streams back as Server-Sent Events. Answers are grounded in a Postgres full-text knowledge base (car manuals and books) and use DeepSeek as the LLM (at most 2 LLM calls per turn: one streamed reply, one state update). Audio is transcribed with faster-whisper and engine sounds are tagged with CLAP.

## Configuration

Put these in `.env` (names only; see `app/core/config.py` for defaults).

Required: `JWT_SECRET_KEY` (min 32 chars), `SUPABASE_URL`, `SUPABASE_KEY`, `BASE_URL`, `DEEPSEEK_API_KEY`.

`SUPABASE_KEY` must be the project's **service-role** key, not the anon key. The schema enables RLS with no policies (so the public anon key reads nothing), and logout uses the auth admin API. Keep it server-side only.

Optional: `ENV`, `DEBUG`, `ALLOWED_ORIGINS`, `DEEPSEEK_BASE_URL`, `DEEPSEEK_MODEL`, `WHISPER_MODEL`, `CLAP_MODEL`, `SOUND_MIN_SCORE`, `MAX_FRAMES`, `MAX_MEDIA_SECONDS`, `MAX_MEDIA_MB`, `MAX_IMAGES`, `MAX_IMAGE_MB`, `KB_TOP_K`, `HISTORY_MESSAGES`, `MEDIA_BUCKET`.

## Supabase setup

1. Review `app/db/sql/001_v2_schema.sql`. It is **destructive**: it drops `Chatsession` and `Groundknowledge` and removes two `Car` columns. It enables RLS on `kb_passages` and `Chatsession` with no policies, so only the service-role key can reach them. It does not change RLS on the existing `Car` and `user_profiles` tables: check those in the dashboard and enable RLS on them too.
2. Run it by hand in the Supabase SQL editor. The backend never applies schema changes.
3. In Storage, create a **private** bucket named `media` (or whatever `MEDIA_BUCKET` is set to). This is a manual step.

## Run

```
uv sync
uv run scripts/ingest_kb.py            # optional: ingest PDFs from car_data/ (--dir to change)
uv run uvicorn main:app --reload
```

Or with Docker (reads `.env`, serves on port 8000, health check on `/health`):

```
docker compose up
```

## API

Interactive reference: `http://localhost:8000/docs`. `/docs`, `/redoc` and `/openapi.json` are public (no token), so the API schema is visible to anyone who can reach the server. Every route except `/api/v1/auth/*`, `/health` and those docs pages needs `Authorization: Bearer <token>` from login, and so does `POST /api/v1/auth/logout`; the user id always comes from the token.

| Area | Routes |
| --- | --- |
| Auth | `POST /api/v1/auth/register`, `POST /api/v1/auth/login`, `POST /api/v1/auth/logout`, `GET\|POST /api/v1/auth/confirm`, `GET /api/v1/auth/confirm-page` |
| Cars | `GET /api/v1/cars`, `POST /api/v1/cars`, `DELETE /api/v1/cars/{car_id}` |
| Sessions | `POST /api/v1/sessions`, `GET /api/v1/sessions`, `GET /api/v1/sessions/{id}`, `DELETE /api/v1/sessions/{id}` |
| Chat | `POST /api/v1/sessions/{id}/messages` (multipart form: `text`, `files`; replies as SSE) |

Adding a car starts a background ingest of its manual; the car's `kb_status` tracks progress.

### SSE events

| Event | Data |
| --- | --- |
| `stage` | `{"name": "processing_media" \| "searching"}` (`processing_media` only when files were sent) |
| `token` | `{"text": "..."}` a piece of the reply |
| `state` | updated diagnosis state: `suspected_causes` (cause, likelihood, evidence, next_check), `safety_flags`, `title` |
| `error` | `{"message": "..."}` ends the stream |
| `done` | `{"message_id": "..."}` ends the stream |

Example (text only):

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

## Media limits

A message needs text (up to 5000 chars) or at least one file. Files are either up to 4 images of at most 10 MB each, or a single audio/video file of at most 60 s and 100 MB. Limits come from `MAX_IMAGES`, `MAX_IMAGE_MB`, `MAX_MEDIA_SECONDS` and `MAX_MEDIA_MB`.

Request bodies are capped before they are read, and before auth runs: the chat route accepts up to `max(MAX_IMAGES * MAX_IMAGE_MB, MAX_MEDIA_MB) + 1` MB, and every other route accepts 1 MB. A larger `Content-Length`, or a chunked body that grows past the cap, gets `413` with error code `VAL_005`.

## Tests and checks

```
uv run pytest
uv run scripts/run_cases.py [--only id,id] [--out path]   # diagnosis quality cases -> Markdown report in reports/ (uses real DeepSeek and the KB)
uv run scripts/e2e.py --email you@example.com --password ... [--base URL] [--media-dir DIR] [--allow-skip]   # against a running stack
```

`e2e.py` looks for sample media in `tests/fixtures/e2e/` (git-ignored) and exits non-zero if a file is missing unless `--allow-skip` is given.
