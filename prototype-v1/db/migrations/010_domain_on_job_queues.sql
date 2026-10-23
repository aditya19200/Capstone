-- 010_domain_on_job_queues.sql
-- Lets two domains share one database without taking each other's work.
--
-- Migration 007 put a domain on predictions and annotations, which made the
-- RESULTS separable. The queues that produce them were left domain-less:
--
--   batches / batch_items   claim_batch_items(n) takes ANY pending item, so a
--                           medical worker could claim a legal upload and
--                           classify it with the medical model.
--   xai_jobs                claim_xai_job() likewise, explaining a legal
--                           prediction with medical weights.
--   retrain_jobs            the worker reads "the latest job", whichever
--                           domain triggered it.
--   model_versions          one global single-active index, so activating a
--                           medical version de-activated the legal one.
--
-- Until this runs, the medical instance is kept on an in-memory store.
-- backend/repositories/_client.domain_queues_available() detects this
-- migration and the code switches to the domain-scoped paths below; without
-- it, every query behaves exactly as before.
--
-- Additive and idempotent: existing rows take domain 'legal' (all of them are
-- legal today), the original claim functions are left in place, and running
-- this twice is harmless.
--
-- Run it in the Supabase dashboard: SQL Editor -> paste -> Run.

begin;

-- ---------------------------------------------------------------------------
-- 1. A domain on the job tables. batch_items and xai_jobs inherit theirs
--    through batches and predictions, so they need no column of their own.
-- ---------------------------------------------------------------------------
alter table batches       add column if not exists domain text not null default 'legal';
alter table retrain_jobs  add column if not exists domain text not null default 'legal';
alter table model_versions add column if not exists domain text not null default 'legal';

create index if not exists idx_batches_domain       on batches (domain);
create index if not exists idx_retrain_jobs_domain  on retrain_jobs (domain, triggered_at desc);

-- The medical baseline is the only non-legal checkpoint registered so far.
update model_versions set domain = 'medical'
 where file_path = './models/MedicalModelShared' and domain <> 'medical';

-- ---------------------------------------------------------------------------
-- 2. One active version PER DOMAIN, not one overall.
-- ---------------------------------------------------------------------------
drop index if exists idx_model_versions_one_active;
create unique index if not exists idx_model_versions_one_active_per_domain
    on model_versions (domain) where is_active = true;

-- ---------------------------------------------------------------------------
-- 3. Domain-scoped claims. Same shape as claim_batch_items / claim_xai_job
--    (006), with a join to the row that carries the domain.
-- ---------------------------------------------------------------------------
create or replace function claim_batch_items_for(n integer, d text)
returns setof batch_items
language plpgsql
as $$
begin
    return query
    with claimed as (
        select bi.id
        from batch_items bi
        join batches b on b.id = bi.batch_id
        where bi.status = 'pending'
          and b.domain = d
        order by bi.created_at, bi.id
        limit n
        for update of bi skip locked
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

create or replace function claim_xai_job_for(d text)
returns setof xai_jobs
language plpgsql
as $$
begin
    return query
    with claimed as (
        select x.id
        from xai_jobs x
        join predictions p on p.id = x.prediction_id
        where x.status = 'pending'
          and p.domain = d
        order by x.created_at, x.id
        limit 1
        for update of x skip locked
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

-- ---------------------------------------------------------------------------
-- 4. Activation scoped to the target's own domain. Every UPDATE carries a
--    WHERE clause, so this also replaces the unfiltered UPDATE that made
--    migration 003's activate_model_version fail on Supabase (see 009).
-- ---------------------------------------------------------------------------
create or replace function activate_model_version_for(target_id uuid)
returns setof model_versions
language plpgsql
as $$
declare
    d text;
begin
    select domain into d from model_versions where id = target_id;
    if not found then
        raise exception 'model version % does not exist', target_id;
    end if;

    update model_versions
       set is_active = false
     where is_active = true and domain = d and id <> target_id;

    update model_versions
       set is_active = true
     where id = target_id;

    return query select * from model_versions where id = target_id;
end;
$$;

commit;
