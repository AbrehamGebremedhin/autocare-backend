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
create table if not exists users (
    id            uuid primary key default gen_random_uuid(),
    email         text not null,
    password_hash text,
    google_sub    text,
    created_at    timestamptz not null default now(),
    constraint users_login_method_check
        check (password_hash is not null or google_sub is not null)
);

create unique index if not exists users_email_lower_idx on users (lower(email));
create unique index if not exists users_google_sub_idx on users (google_sub) where google_sub is not null;

-- User profiles -----------------------------------------------------------------
-- cars holds "Car".id values; the app reads and rewrites the whole array.
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
create table if not exists "Chatsession" (
    id         uuid primary key default gen_random_uuid(),
    user_id    uuid not null,
    car_id     text not null,
    title      text not null default 'New diagnosis',
    messages   jsonb not null default '[]',
    state      jsonb not null default '{}',
    created_at timestamptz default now(),
    updated_at timestamptz default now()
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

commit;
