-- 004_rls_policies.sql
-- Aditya's backend talks to Supabase with the SERVICE ROLE key, which
-- bypasses RLS entirely (repositories/_client.py: get_client() uses
-- SUPABASE_SERVICE_KEY). RLS here only governs direct access via the
-- anon/authenticated key, which per the API contract in CLAUDE.md the
-- frontend never uses ("The frontend talks only to FastAPI. It never
-- queries Neo4j or Postgres directly.") — so this is a safety net, not
-- a load-bearing part of the request path today. Enabled anyway so a
-- leaked anon key can't read or write these tables.

alter table predictions          enable row level security;
alter table batches              enable row level security;
alter table batch_items          enable row level security;
alter table xai_jobs             enable row level security;
alter table annotations          enable row level security;
alter table model_versions       enable row level security;
alter table dataset_versions     enable row level security;
alter table retrain_jobs         enable row level security;

-- No policies are defined for `authenticated` or `anon` on any table.
-- With RLS enabled and zero matching policies, every operation (select,
-- insert, update, delete) is denied by default for those roles — only
-- the service role (which bypasses RLS) can touch these tables.
--
-- If the frontend ever needs direct Supabase reads (e.g. a future
-- Realtime subscription instead of polling GET /explain/{prediction_id}),
-- add a narrowly-scoped `for select ... to authenticated` policy on
-- that one table rather than opening things up broadly.
