-- vehicle_visuals.sql
-- Purpose: vehicle artwork ("visuals"): one row per (variant, style) with the job state
--          the visuals worker drives, the owner's choice among ambiguous variants, a
--          provider call ledger, and three narrow functions that keep the worker atomic.
--
-- Apply: after schema.sql (it references users). The postgres image runs it on a fresh
--        volume (docker-entrypoint-initdb.d), or by hand:
--        psql -v ON_ERROR_STOP=1 -f vehicle_visuals.sql "$DATABASE_URL".
--        It only creates what is missing (if not exists / or replace), drops
--        nothing and is safe to run again.

begin;

-- One row per (variant_key, style_version); concurrent prepares dedupe with
-- insert ... on conflict do nothing. fencing_token grows on every claim so a stale
-- worker cannot finalize; published_revision guards publishing against a stale approval.
create table if not exists vehicle_visuals (
    id                 uuid primary key default gen_random_uuid(),
    variant_key        text not null,
    style_version      text not null,
    state              text not null check (state in ('queued', 'generating', 'review', 'ready', 'failed')),
    provenance         text not null default 'curated' check (provenance in ('curated', 'provider')),
    recipe_version     text not null default 'r1',
    sha256             text,
    draft_sha256       text,
    width              int,
    height             int,
    published_revision int not null default 0,
    attempt_count      int not null default 0,
    next_attempt_at    timestamptz not null default now(),
    lease_expires_at   timestamptz,
    fencing_token      bigint not null default 0,
    last_error_code    text,
    created_at         timestamptz not null default now(),
    updated_at         timestamptz not null default now(),
    unique (variant_key, style_version)
);

create index if not exists vehicle_visuals_due_idx on vehicle_visuals (state, next_attempt_at);

-- The owner's pick when a car matches several variants.
create table if not exists user_car_visual_choices (
    user_id     uuid not null references users (id) on delete cascade,
    car_id      text not null,
    variant_key text not null,
    created_at  timestamptz not null default now(),
    primary key (user_id, car_id)
);

-- One row per provider call, for the budget ceiling.
create table if not exists visual_provider_calls (
    id          bigserial primary key,
    variant_key text not null,
    called_at   timestamptz not null default now()
);

create index if not exists visual_provider_calls_called_at_idx on visual_provider_calls (called_at);

-- Claim one due job: select * from claim_due_visual_job(lease_seconds).
-- Due = queued and ready to retry, or generating with an expired lease. Returns the
-- updated row (state generating, fencing_token + 1, a fresh lease), or an all-NULL row
-- when nothing is due. The 3-attempt cap is enforced by the worker, not here.
create or replace function claim_due_visual_job(p_lease_seconds int)
returns vehicle_visuals
language plpgsql
as $$
declare
    claimed vehicle_visuals;
begin
    update vehicle_visuals v
    set state            = 'generating',
        fencing_token    = v.fencing_token + 1,
        lease_expires_at = now() + make_interval(secs => p_lease_seconds),
        attempt_count    = v.attempt_count + 1,
        updated_at       = now()
    where v.id = (
        select d.id
        from vehicle_visuals d
        where (d.state = 'queued' and d.next_attempt_at <= now())
           or (d.state = 'generating' and d.lease_expires_at < now())
        order by d.next_attempt_at
        for update skip locked
        limit 1
    )
    returning v.* into claimed;
    return claimed;
end;
$$;

-- Record a job's outcome. Only the holder of the current fencing token, while the row
-- is still generating, can write it. Returns whether a row changed.
create or replace function finalize_visual_job(
    p_id           uuid,
    p_token        bigint,
    p_state        text,
    p_draft_sha    text,
    p_error        text,
    p_next_attempt timestamptz
)
returns boolean
language plpgsql
as $$
begin
    update vehicle_visuals
    set state            = p_state,
        draft_sha256     = p_draft_sha,
        last_error_code  = p_error,
        next_attempt_at  = coalesce(p_next_attempt, next_attempt_at),
        lease_expires_at = null,
        updated_at       = now()
    where id = p_id and fencing_token = p_token and state = 'generating';
    return found;
end;
$$;

-- Publish an approved draft. Only succeeds from review at the expected revision, so a
-- stale approval cannot overwrite a newer one. Returns whether a row changed.
create or replace function publish_visual_revision(
    p_id                uuid,
    p_expected_revision int,
    p_sha               text,
    p_width             int,
    p_height            int
)
returns boolean
language plpgsql
as $$
begin
    update vehicle_visuals
    set state              = 'ready',
        sha256             = p_sha,
        width              = p_width,
        height             = p_height,
        published_revision = published_revision + 1,
        updated_at         = now()
    where id = p_id and published_revision = p_expected_revision and state = 'review';
    return found;
end;
$$;

commit;
