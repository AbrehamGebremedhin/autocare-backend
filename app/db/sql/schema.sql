-- schema.sql
-- Purpose: the whole AutoCare schema on plain Postgres (13+, for the built-in
--          gen_random_uuid): accounts (users), user_profiles, the shared "Car"
--          catalogue, "Chatsession", kb_passages and the search_kb function.
--
-- Apply: the postgres image runs it on a fresh volume (docker-entrypoint-initdb.d),
--        or by hand: psql -v ON_ERROR_STOP=1 -f schema.sql "$DATABASE_URL".
--        It only creates what is missing (if not exists / or replace), drops
--        nothing and is safe to run again. Later column changes are applied by
--        hand; this file does not migrate existing tables.

begin;

-- Accounts ----------------------------------------------------------------------
-- email is stored lower-cased; a row signs in with a password, Google, or both.
-- App tokens issued before tokens_valid_after are rejected (it moves when Google is linked).
create table if not exists users (
    id                 uuid primary key default gen_random_uuid(),
    email              text not null,
    password_hash      text,
    google_sub         text,
    created_at         timestamptz not null default now(),
    tokens_valid_after timestamptz not null default now(),
    constraint users_login_method_check
        check (password_hash is not null or google_sub is not null)
);

create unique index if not exists users_email_lower_idx on users (lower(email));
create unique index if not exists users_google_sub_idx on users (google_sub) where google_sub is not null;

-- User profiles -----------------------------------------------------------------
-- cars holds "Car".id values; the app adds and removes single ids with array_append/array_remove.
create table if not exists user_profiles (
    user_id     uuid primary key references users (id) on delete cascade,
    cars        text[] not null default '{}',
    preferences jsonb  not null default '{}',
    created_at  timestamptz not null default now(),
    updated_at  timestamptz not null default now()
);

-- Cars --------------------------------------------------------------------------
-- id is built by CarCRUD.unique_logic: "<make>-<model>-<year>", lowercased.
create table if not exists "Car" (
    id                 text primary key,
    make               text not null,
    model              text not null,
    year               int  not null,
    owner_manual_url   text default '',
    service_manual_url text default '',
    car_guide_links    jsonb not null default '[]',
    kb_status          text not null default 'pending'
        constraint car_kb_status_check
        check (kb_status in ('pending', 'ready', 'failed', 'no_manual')),
    kb_passage_count   int  not null default 0,
    created_at         timestamptz not null default now()
);

-- Chat sessions -----------------------------------------------------------------
-- Deleting an account deletes its sessions. busy_until is the lease a running turn or a delete holds
-- (ChatSessionCRUD.claim/renew/release), shared by every api worker; null or past means free. busy_by
-- is the holder's token: only that holder may renew or release the lease.
create table if not exists "Chatsession" (
    id         uuid primary key default gen_random_uuid(),
    user_id    uuid not null references users (id) on delete cascade,
    car_id     text not null,
    title      text not null default 'New diagnosis',
    messages   jsonb not null default '[]',
    state      jsonb not null default '{}',
    created_at timestamptz default now(),
    updated_at timestamptz default now(),
    busy_until timestamptz,
    busy_by    uuid
);

create index if not exists chatsession_user_id_idx on "Chatsession" (user_id);

-- Knowledge base passages -------------------------------------------------------
create table if not exists kb_passages (
    id          bigint generated always as identity primary key,
    source_type text not null check (source_type in ('manual', 'book')),
    car_id      text,
    title       text not null,
    page        int,
    content     text not null,
    tsv         tsvector generated always as (to_tsvector('english', content)) stored
);

create index if not exists kb_passages_tsv_idx on kb_passages using gin (tsv);
create index if not exists kb_passages_car_id_idx on kb_passages (car_id);

-- Full-text search: select * from search_kb($1 tsq, $2 p_car_id, $3 k)
create or replace function search_kb(tsq text, p_car_id text, k int default 8)
returns table (
    source_type text,
    car_id      text,
    title       text,
    page        int,
    content     text,
    rank        real
)
language sql
stable
as $$
    select kb.source_type,
           kb.car_id,
           kb.title,
           kb.page,
           kb.content,
           ts_rank(kb.tsv, to_tsquery('english', tsq)) as rank
    from kb_passages kb
    where kb.tsv @@ to_tsquery('english', tsq)
      and (kb.source_type = 'book' or kb.car_id = p_car_id)
    order by ts_rank(kb.tsv, to_tsquery('english', tsq)) desc
    limit k;
$$;

-- What an owner says about their own car. "Car" rows describe the model and are shared by every owner,
-- so per-owner facts live here; the row goes with the car (removed from the garage) or the account.
create table if not exists user_car_details (
    user_id      uuid not null references users (id) on delete cascade,
    car_id       text not null,
    transmission text not null
        constraint user_car_details_transmission_check
        check (transmission in ('manual', 'automatic', 'cvt', 'dual_clutch')),
    updated_at   timestamptz not null default now(),
    primary key (user_id, car_id)
);

-- What an owner has established about their own car, across sessions (app/services/findings.py). Per owner
-- and car, never shared. user_confirmed rows carry the owner's own words in evidence_quote; model_suggested
-- rows are hypotheses the model proposed and are shown to it as such. A contradicted row is marked superseded
-- and linked, never deleted. A deleted session keeps its findings. backfill_key is set only by
-- scripts/backfill_findings.py, so running it twice adds nothing.
create table if not exists car_findings (
    id             uuid primary key default gen_random_uuid(),
    user_id        uuid not null references users (id) on delete cascade,
    car_id         text not null,
    kind           text not null
        constraint car_findings_kind_check
        check (kind in ('confirmed_finding', 'suspected_cause', 'ruled_out', 'test_done', 'repair_done')),
    text           text not null
        constraint car_findings_text_check check (length(text) between 1 and 300),
    norm           text not null,
    status         text not null default 'active'
        constraint car_findings_status_check check (status in ('active', 'superseded', 'retracted')),
    source         text not null
        constraint car_findings_source_check check (source in ('user_confirmed', 'model_suggested')),
    evidence_quote text,
    session_id     uuid references "Chatsession" (id) on delete set null,
    created_at     timestamptz not null default now(),
    updated_at     timestamptz not null default now(),
    superseded_by  uuid references car_findings (id) on delete set null,
    backfill_key   text unique,
    constraint car_findings_quote_check check (source = 'model_suggested' or evidence_quote is not null)
);

create index if not exists car_findings_owner_idx on car_findings (user_id, car_id, status);
create unique index if not exists car_findings_active_dedupe_idx
    on car_findings (user_id, car_id, kind, norm) where status = 'active';

commit;
