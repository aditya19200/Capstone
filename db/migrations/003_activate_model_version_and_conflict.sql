-- 003_activate_model_version_and_conflict.sql
-- Straight from docs/MIGRATION_activate_model_version.md (Aditya's spec) —
-- applied verbatim rather than re-derived, since he already wrote and
-- reasoned through this exact SQL for repositories/model_versions.py
-- and services/supabase_service.detect_and_flag_conflict() to call into.

-- ============================================================
-- 1. Atomic model-version rollback RPC
-- Called by repositories/model_versions.set_active().
-- ============================================================
create or replace function activate_model_version(target_id uuid)
returns setof model_versions
language plpgsql
as $$
begin
  if not exists (select 1 from model_versions where id = target_id) then
    raise exception 'model_versions row % does not exist', target_id;
  end if;

  update model_versions set is_active = false;
  update model_versions set is_active = true where id = target_id;

  return query select * from model_versions where id = target_id;
end;
$$;

-- ============================================================
-- 2. Conflict flag on annotations
-- Set by services/supabase_service.detect_and_flag_conflict().
-- Deliberately independent of `status` — a row can be
-- status='validated' AND has_conflict=true at the same time.
-- ============================================================
alter table annotations
  add column if not exists has_conflict boolean not null default false;

-- Partial index for the reviewer conflict queue — only flagged rows
-- are indexed, since only a small fraction of rows will ever be true.
create index if not exists idx_annotations_has_conflict
  on annotations (has_conflict)
  where has_conflict = true;
