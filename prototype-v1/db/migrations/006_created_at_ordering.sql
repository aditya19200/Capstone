-- 006_created_at_ordering.sql
-- Brings the live database in line with Copilot's fix in 001/002:
-- adds created_at to batch_items/xai_jobs for true FIFO ordering, and
-- rebuilds the claim-ordering indexes and RPCs to use it.

alter table batch_items
    add column if not exists created_at timestamptz not null default now();

alter table xai_jobs
    add column if not exists created_at timestamptz not null default now();

drop index if exists idx_batch_items_claim;
create index idx_batch_items_claim
    on batch_items (status, created_at, id);

drop index if exists idx_xai_jobs_claim;
create index idx_xai_jobs_claim
    on xai_jobs (status, created_at, id);

create or replace function claim_batch_items(n integer)
returns setof batch_items
language plpgsql
as $$
begin
    return query
    with claimed as (
        select id
        from batch_items
        where status = 'pending'
        order by created_at, id
        limit n
        for update skip locked
    )
    update batch_items
    set status = 'processing',
        locked_at = now(),
        attempts = attempts + 1
    from claimed
    where batch_items.id = claimed.id
    returning batch_items.*;
end;
$$;

create or replace function claim_xai_job()
returns setof xai_jobs
language plpgsql
as $$
begin
    return query
    with claimed as (
        select id
        from xai_jobs
        where status = 'pending'
        order by created_at, id
        limit 1
        for update skip locked
    )
    update xai_jobs
    set status = 'processing',
        locked_at = now(),
        attempts = attempts + 1
    from claimed
    where xai_jobs.id = claimed.id
    returning xai_jobs.*;
end;
$$;
