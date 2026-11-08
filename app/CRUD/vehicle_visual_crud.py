"""Repository over vehicle_visuals.sql: thin wrappers, every value a bind parameter.

Each function is one statement through the pool helpers, so no connection is held across anything else.
Database errors (DatabaseUnavailable included) pass through; callers decide what an outage means.
"""
from uuid import UUID

from app.db import pool

STYLE_VERSION = "workshop-v1"


async def get_visual(variant_key: str) -> dict | None:
    return await pool.fetchrow(
        "select * from vehicle_visuals where variant_key = $1 and style_version = $2",
        variant_key, STYLE_VERSION)


async def ensure_job(variant_key: str, provenance: str = "curated") -> tuple[dict, bool]:
    """Queue a job for the variant unless one exists. (row, True) when this call created it."""
    row = await pool.fetchrow(
        "insert into vehicle_visuals (variant_key, style_version, state, provenance) values ($1, $2, 'queued', $3)"
        " on conflict (variant_key, style_version) do nothing returning *",
        variant_key, STYLE_VERSION, provenance)
    if row is not None:
        return row, True
    return await get_visual(variant_key), False


async def jobs_created_last_hour() -> int:
    return await pool.fetchval("select count(*) from vehicle_visuals where created_at > now() - interval '1 hour'")


async def get_choice(user_id: str, car_id: str) -> str | None:
    return await pool.fetchval(
        "select variant_key from user_car_visual_choices where user_id = $1::uuid and car_id = $2",
        user_id, car_id)


async def set_choice(user_id: str, car_id: str, variant_key: str) -> None:
    await pool.execute(
        "insert into user_car_visual_choices (user_id, car_id, variant_key) values ($1::uuid, $2, $3)"
        " on conflict (user_id, car_id) do update set variant_key = excluded.variant_key",
        user_id, car_id, variant_key)


async def clear_choice(user_id: str, car_id: str) -> None:
    await pool.execute("delete from user_car_visual_choices where user_id = $1::uuid and car_id = $2", user_id, car_id)


async def claim_job(lease_seconds: int) -> dict | None:
    row = await pool.fetchrow("select * from claim_due_visual_job($1)", lease_seconds)
    # Nothing due comes back as one all-NULL composite row.
    return None if row is None or row["id"] is None else row


async def renew_lease(job_id: str, token: int, lease_seconds: int) -> bool:
    row = await pool.fetchrow(
        "update vehicle_visuals set lease_expires_at = now() + make_interval(secs => $3), updated_at = now()"
        " where id = $1::uuid and fencing_token = $2 and state = 'generating' returning id",
        job_id, token, lease_seconds)
    return row is not None


async def finalize_job(job_id: str, token: int, state: str, draft_sha: str | None, error: str | None,
                       next_attempt_seconds: int | None) -> bool:
    # NULL seconds makes a NULL timestamp, which the SQL function reads as "keep next_attempt_at".
    return bool(await pool.fetchval(
        "select finalize_visual_job($1::uuid, $2, $3, $4, $5, now() + make_interval(secs => $6::float8))",
        job_id, token, state, draft_sha, error, next_attempt_seconds))


async def publish(job_id: str, expected_revision: int, sha: str, width: int, height: int) -> bool:
    return bool(await pool.fetchval(
        "select publish_visual_revision($1::uuid, $2, $3, $4, $5)", job_id, expected_revision, sha, width, height))


async def list_review() -> list[dict]:
    return await pool.fetch("select * from vehicle_visuals where state = 'review' order by updated_at, variant_key")


async def reject(variant_key: str) -> bool:
    """Discard the draft of a row in review. A row that was published before goes back to ready, keeping its
    published artwork; a never-published row fails with 'rejected'."""
    row = await pool.fetchrow(
        "update vehicle_visuals set"
        " state = case when published_revision > 0 then 'ready' else 'failed' end,"
        " last_error_code = case when published_revision > 0 then null else 'rejected' end,"
        " draft_sha256 = null, updated_at = now() where variant_key = $1 and style_version = $2 and state = 'review' returning id",
        variant_key, STYLE_VERSION)
    return row is not None


async def requeue(variant_key: str) -> bool:
    row = await pool.fetchrow(
        "update vehicle_visuals set state = 'queued', attempt_count = 0, next_attempt_at = now(),"
        " lease_expires_at = null, last_error_code = null, updated_at = now()"
        " where variant_key = $1 and style_version = $2 and state in ('failed', 'review') returning id",
        variant_key, STYLE_VERSION)
    return row is not None


async def record_provider_call(variant_key: str) -> None:
    await pool.execute("insert into visual_provider_calls (variant_key) values ($1)", variant_key)


async def provider_calls_last_day() -> int:
    return await pool.fetchval("select count(*) from visual_provider_calls where called_at > now() - interval '1 day'")


async def referenced_shas() -> set[str]:
    rows = await pool.fetch(
        "select sha256 as sha from vehicle_visuals where sha256 is not null"
        " union select draft_sha256 as sha from vehicle_visuals where state = 'review' and draft_sha256 is not null")
    return {r["sha"] for r in rows}


async def seed_review(variant_key: str, draft_sha: str, width: int, height: int) -> dict:
    """Put a curated draft into review. The published revision, sha and size are untouched. A job that is generating
    under a live lease is left alone: ValueError."""
    row = await pool.fetchrow(
        "insert into vehicle_visuals (variant_key, style_version, state, provenance, draft_sha256, width, height)"
        " values ($1, $2, 'review', 'curated', $3, $4, $5)"
        " on conflict (variant_key, style_version) do update set state = 'review', provenance = 'curated',"
        " draft_sha256 = excluded.draft_sha256, width = case when vehicle_visuals.published_revision = 0 then excluded.width else vehicle_visuals.width end,"
        " height = case when vehicle_visuals.published_revision = 0 then excluded.height else vehicle_visuals.height end,"
        " lease_expires_at = null, last_error_code = null, updated_at = now()"
        " where not (vehicle_visuals.state = 'generating' and vehicle_visuals.lease_expires_at > now())"
        " returning *",
        variant_key, STYLE_VERSION, draft_sha, width, height)
    if row is None:
        raise ValueError("variant is being generated under a live lease; try again after it expires")
    return row


async def owned_car(user_id: str, car_id: str) -> dict | None:
    try:
        user_id = str(UUID(user_id))  # the form Postgres's ::uuid reads the same way
    except ValueError:
        return None
    return await pool.fetchrow(
        'select c.* from user_profiles p join "Car" c on c.id = any(p.cars) where p.user_id = $1::uuid and c.id = $2',
        user_id, car_id)
