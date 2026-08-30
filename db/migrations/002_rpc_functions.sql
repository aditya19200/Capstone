-- 002_rpc_functions.sql
-- The job-queue mechanism, per docs/SCHEMA_REQUEST.md. FOR UPDATE SKIP
-- LOCKED lets multiple worker processes claim disjoint rows without
-- blocking each other or double-claiming — this is what replaces
-- Celery/Redis in this design (see CLAUDE.md: "the database table IS
-- the job queue").
--
-- Note: stuck-job recovery (rows stuck in 'processing') is NOT a DB
-- function — repositories/batch_items.py:sweep_stuck() and
-- repositories/xai_jobs.py:sweep_stuck() do that from the app side with
-- plain UPDATE calls each worker loop. These RPCs only claim.

-- ============================================================
-- claim_batch_items(n) — classify worker claims up to n pending items
-- ============================================================
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

-- ============================================================
-- claim_xai_job() — SHAP worker claims a single pending job
-- ============================================================
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

-- ============================================================
-- increment_batch_completed(batch_id) — atomic progress-bar increment
-- Called by repositories/batches.py:increment_completed(). A plain
-- read-modify-write from the app would race under concurrent worker
-- completions; this does it in one statement.
-- ============================================================
create or replace function increment_batch_completed(batch_id uuid)
returns batches
language plpgsql
as $$
declare
    updated batches;
begin
    update batches
    set completed_items = completed_items + 1
    where id = increment_batch_completed.batch_id
    returning * into updated;

    if updated is null then
        raise exception 'batches row % does not exist', batch_id;
    end if;

    return updated;
end;
$$;
