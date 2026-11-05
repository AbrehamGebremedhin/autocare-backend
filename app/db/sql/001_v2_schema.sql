-- 001_v2_schema.sql
-- Purpose: v2 schema for the AutoCare backend. Adds kb_passages (full-text
--          knowledge base) and the search_kb RPC, recreates Chatsession, and
--          moves the knowledge-base status onto "Car".
--
-- DESTRUCTIVE: drops Chatsession and Groundknowledge (old data is test data per
--              the design doc), removes two Car columns (is_vectorized,
--              vector_chunk_count).
--
-- Review before running: RLS is not configured by this file; the backend must
-- use the service-role key or you must add policies.
--
-- Run in the Supabase SQL editor after review.

begin;

-- Knowledge base passages -----------------------------------------------------
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

-- Full-text search RPC: supabase.rpc('search_kb', {'tsq':..., 'p_car_id':..., 'k':...})
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

-- Chat sessions (dropped and recreated) ---------------------------------------
drop table if exists "Chatsession" cascade;

create table "Chatsession" (
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

-- Car: knowledge-base status replaces the vector columns ------------------------
alter table "Car" add column if not exists kb_status text not null default 'pending';
alter table "Car" drop constraint if exists car_kb_status_check;
alter table "Car" add constraint car_kb_status_check
    check (kb_status in ('pending', 'ready', 'failed', 'no_manual'));
alter table "Car" add column if not exists kb_passage_count int not null default 0;
alter table "Car" drop column if exists is_vectorized;
alter table "Car" drop column if exists vector_chunk_count;

-- Old vector knowledge table ---------------------------------------------------
drop table if exists "Groundknowledge" cascade;

commit;
