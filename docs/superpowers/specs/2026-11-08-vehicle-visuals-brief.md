# AutoCare vehicle visuals: owner's brief (verbatim requirements)

Source: the owner's prompt of 2026-11-08, saved so the plan's executors can read it. Where the brief says "Supabase" it
is wrong for this deployment: the stack is a single VPS with plain PostgreSQL (see the plan's Decisions).

## Context
AutoCare is a mobile app for diagnosing car problems. Its Garage screen shows a workshop-manual illustration of the user's
selected car. The backend resolves which car variant a user's car is, prepares one shared illustration per variant,
reviews and publishes it, and serves it through short-lived signed URLs. The mobile client is finished and already calls the
endpoints below. If artwork is missing, it uses a bundled silhouette, so artwork must never block diagnosis, sessions or
manual indexing.

Do not merge, push, deploy, or touch any production database or environment files. Small conventional commits at each
working checkpoint.

## Endpoints (additive, prefix /api/v1)
### GET /cars/{car_id}/visual
Descriptor for the authenticated user's car. 404 if the car does not exist or does not belong to the caller (never 403,
never reveal existence). Always HTTP 200 for ready/pending/fallback.

### POST /cars/{car_id}/visual/prepare
Idempotently reuse existing artwork or queue one job. Body: {} or {"variant_key": "<allowlisted key>"} for a confirmed
choice. Returns the same descriptor shape, HTTP 200 or 202. 404 inaccessible car; 422 if variant_key belongs to a different
make/model than the car; 429 with Retry-After (integer seconds) when throttled. Rate-limit per user and globally dedupe
jobs per variant.

### Descriptor shape (the client rejects anything else)
```
{
  "schema_version": 1,
  "state": "ready" | "pending" | "fallback",
  "variant_key": "toyota-echo-2001-sedan",
  "style_version": "workshop-v1",
  "preset": "garage_roll_v1",
  "reason": "<stable code, fallback only>",
  "retry_after_ms": <bounded int, pending only>,
  "options": [{"variant_key": "...", "label": "..."}],   // only for reason "choose_variant"
  "artwork": {                                            // ready only
    "format": "flat_webp", "width": 1024, "height": 420, "baseline_y": 366,
    "sha256": "<64 hex lowercase>",
    "download_url": "https://...", "expires_at": "<ISO-8601>"
  }
}
```
Hard client rules (violating any makes the client discard the descriptor):
- schema_version must be exactly 1; preset must be "garage_roll_v1" or absent.
- ready requires artwork with format "flat_webp", width 1024, height 420, baseline_y 366, a 64-hex sha256, a non-empty
  variant_key and style_version, and an https download_url.
- Internal states queued/generating/review map to "pending". failed or unknown map to "fallback" with reason in:
  unknown_variant, choose_variant, temporarily_unavailable.
- Never include signed URLs in logs.

## Artifact download (what the client does)
The client downloads download_url with a plain GET and NO Authorization header. It requires: HTTP 200, Content-Type exactly
image/webp, Content-Length present and <= 400 KiB (409,600 bytes), body bytes whose SHA-256 equals the descriptor's sha256,
decoded as 1024x420. So the signed URL must be self-authenticating (HMAC over path+expiry, default 15 minutes,
configurable), verified in constant time, reject expired or tampered URLs, and never expose paths outside the visuals
directory (no traversal). Serve with Cache-Control: private, max-age bounded by the remaining expiry, and
X-Content-Type-Options: nosniff.

## Data model (additive SQL)
- vehicle_visuals: one row per (variant_key, style_version) UNIQUE. Internal state queued|generating|review|ready|failed.
  Columns for provenance, artifact sha256, width/height, recipe_version, object key, published_revision (int),
  attempt_count, next_attempt_at, lease_expires_at, fencing_token, last_error_code (no raw provider payloads), timestamps.
- user_car_visual_choices: (user_id, car_id) -> variant_key, for ambiguous matches. Validate against the car's candidate
  list on write; re-check car ownership on every read and ignore/clear the choice if the car is no longer the user's.
- Narrow DB functions for claim and finalize (claim_due_visual_job, finalize_visual_job, publish_visual_revision) so worker
  logic is atomic. Unique constraint plus atomic insert (INSERT ... ON CONFLICT DO NOTHING) deduplicates concurrent
  prepares for the same variant.
- Do NOT apply migrations to any real database. Provide the SQL and apply it only to a disposable test database.

## Variant catalog
A versioned in-repo catalog mapping normalized make/model/year (with alias handling, case/whitespace/punctuation
normalization) to verified variants with explicit ambiguity markers. Start with: Toyota Echo (2000-2005 sedan, variant
toyota-echo-2001-sedan) and Honda Civic (one reviewed demo variant). Auto-select only when the catalog marks the match
unambiguous; if several variants are valid return fallback reason "choose_variant" with options (labels only); unknown ->
"unknown_variant". Never put VINs, plates, user ids or owner photos into the shared pipeline or into provider prompts.

## Worker (separate process from the same image)
- Entry point python -m app.workers.vehicle_visuals; configurable poll interval.
- Claim due rows with FOR UPDATE SKIP LOCKED in a short transaction, set token + lease, release the transaction BEFORE
  calling the generator. Renew the lease on long jobs.
- Max 3 attempts, bounded exponential backoff with jitter. Provider timeouts count as potentially charged attempts against
  a configurable budget ceiling; when the ceiling is hit, stop calling the provider and move variants to fallback
  "temporarily_unavailable". Operators can requeue terminal failures.
- Generator is an interface with at least: (a) a curated/preseeded source that loads reviewed images from a private
  directory, and (b) a provider-backed generator behind an environment flag that is OFF by default. Tests must never call a
  real provider or spend money. Use a fake in tests and document required env vars without inventing values.
- Post-processing: normalize to a 1024x420 WebP canvas with the ground baseline at y=366, strip all metadata, verify actual
  decoded dimensions and bytes (do not trust headers or extensions), check alpha margins, enforce <= 400 KiB, compute
  SHA-256. Reject and route invalid output to failed/review.
- Provider output goes to a private draft area. Only an approved, reviewed image is published: upload the immutable object
  under a checksum-addressed key, THEN atomically activate the manifest, conditioned on the lease/fencing token and expected
  published_revision so a stale worker or stale approval cannot publish. Keep the previous approved revision until the new
  one is active. Clean up orphaned drafts/objects after a grace period. If a published object is missing on disk, serve
  fallback "temporarily_unavailable" and log an operator-visible event.

## Operator boundary
A CLI command (not an HTTP route) to list drafts in review, approve/publish a specific variant at an expected revision,
reject, and requeue failed jobs. Include a command that seeds the Toyota Echo variant from a provided image file path (the
mobile repo's assets/vehicles/toyota-echo.webp is the reference artwork; accept the path as an argument, no hardcoded
machine path).

## Security and privacy checklist
- Authorize car ownership on every request AND again when signing a URL.
- A user without the car gets 404 for both endpoints. Two users with the same variant share one vehicle_visuals row and
  receive the same sha256.
- No user data in logs or provider prompts; correlation ids in logs; no secrets, signed URLs or tokens in logs or error
  bodies.
- Validate variant_key against the allowlist and the car's make/model (422 otherwise).
- Rate-limit prepare per user and globally; return Retry-After.

## Tests (disposable PostgreSQL, never the real one)
1. Auth: no token 401; other user's car 404 on both endpoints; own car 200.
2. Descriptor schema: every state serializes to the exact shape; the ready descriptor passes the client rules (include a
   test that re-implements the client's validation checklist).
3. Dedup: concurrent prepare calls for the same variant create exactly one job row; two users with the same variant share
   one published checksum.
4. Variant resolution: unambiguous, ambiguous (choose_variant with options), unknown, wrong-model variant_key 422.
5. Worker: SKIP LOCKED does not double-claim; lease expiry lets another worker reclaim; a stale fencing token cannot
   finalize or publish; retry/backoff to the max-attempts limit; budget ceiling stops provider calls; malformed, oversized
   or wrong-dimension image is rejected; crash between upload and commit is reconciled; missing published object falls back.
6. Review/publish: nothing is served until published; publish with a stale expected revision is refused; previous revision
   stays until the new one activates.
7. Signed downloads: valid URL returns 200 with image/webp, Content-Length, correct bytes/sha256; expired, tampered and
   traversal attempts are refused; no Authorization header is needed.
8. Rate limiting: 429 with Retry-After on prepare.
9. Existing test suite still passes. Run the linter/type checker and the full suite.

## Docs and handoff
docs/VEHICLE_VISUALS.md: architecture, chosen storage and signing approach, env vars (names only), how to run the worker,
how to seed and publish artwork, operator commands, the descriptor contract, failure/fallback behavior. At the end report:
branch, commits, files changed, exact commands run with results, what could NOT be verified, and the exact manual steps
for the owner to deploy later (migration SQL, worker process/service unit, env vars).

If blocked (missing credentials, ambiguous convention), implement everything independent of it, record the blocker in the
docs, and continue. If two attempts at the same thing fail, stop and summarize.
