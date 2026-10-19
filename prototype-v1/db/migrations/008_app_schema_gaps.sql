-- 008_app_schema_gaps.sql
-- Two places where what the app writes did not fit the schema. Neither
-- showed up locally, because the in-memory mock has no column types and
-- no column list.
--
-- 1. User ids
--
-- 001 typed annotations.annotator_id and retrain_jobs.triggered_by as uuid,
-- anticipating Supabase Auth user ids. The app has no auth yet: who is
-- acting comes from the X-User-Id header, and the frontend's mock login
-- sends ids like "mock-user-1". Postgres rejects those for a uuid column,
-- so with a real database every POST /annotate and POST /retrain failed
-- with "invalid input syntax for type uuid" — while the in-memory mock,
-- which has no types, accepted them, so nothing showed it locally.
--
-- text holds both: today's header ids, and real auth uuids later (a uuid
-- casts to text losslessly, which is also what USING does to existing
-- rows). Nothing references either column — no foreign key, no RLS policy
-- (004 only enables RLS) — and the annotator_id index is rebuilt by the
-- type change automatically.
--
-- 2. Retrain job parameters
--
-- POST /retrain accepts notes and a min_annotations threshold and checks
-- the threshold before queueing. retrain_jobs had neither column, so the
-- threshold was dropped on the way into the table and the worker fell back
-- to its built-in default of 50 — refusing a job the API had just accepted
-- (asked for 10, had 20). Both are nullable: older rows simply lack them,
-- and the worker already treats a missing threshold as "use the default".
--
-- Safe to re-run: altering a text column to text is a no-op, and the
-- columns are added with IF NOT EXISTS.

alter table annotations
    alter column annotator_id type text using annotator_id::text;

alter table retrain_jobs
    alter column triggered_by type text using triggered_by::text;

comment on column annotations.annotator_id is
    'Who submitted the annotation: the X-User-Id header today, an auth user id later.';
comment on column retrain_jobs.triggered_by is
    'Who triggered the retrain: the X-User-Id header today, an auth user id later.';

alter table retrain_jobs
    add column if not exists notes text;

alter table retrain_jobs
    add column if not exists min_annotations integer
        check (min_annotations is null or min_annotations >= 1);
