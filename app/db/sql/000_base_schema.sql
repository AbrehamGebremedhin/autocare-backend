-- 000_base_schema.sql
-- Purpose: base tables for a fresh Supabase project: "Car" (the shared car
--          catalogue, one row per make-model-year) and user_profiles (one row
--          per auth user, holding the ids of the user's cars).
--
-- Order: run this FIRST on a fresh project, then 001_v2_schema.sql. 001 alters
--        "Car", so it fails without this file. This file only creates what is
--        missing (create table if not exists) and is safe to run again.
--
-- RLS: this file enables row level security on "Car" and user_profiles and adds
--      no policies, the same as 001 does for its tables, so the anon and
--      authenticated roles get no rows through PostgREST. The backend must use
--      the service-role key (it bypasses RLS) in SUPABASE_KEY.
--
-- Review before running, then run it in the Supabase SQL editor.

begin;

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

alter table "Car" enable row level security;

-- User profiles -----------------------------------------------------------------
-- cars holds "Car".id values; the app reads and rewrites the whole array.
create table if not exists user_profiles (
    user_id     uuid primary key references auth.users (id) on delete cascade,
    cars        text[] not null default '{}',
    preferences jsonb  not null default '{}',
    created_at  timestamptz not null default now(),
    updated_at  timestamptz not null default now()
);

alter table user_profiles enable row level security;

commit;
