-- 009_activate_model_version_where_clause.sql
--
-- Fixes activate_model_version(), which could not run on Supabase at all.
--
-- 003 wrote the deactivate step as:
--
--     update model_versions set is_active = false;
--
-- Supabase runs a safeguard that rejects any UPDATE or DELETE with no WHERE
-- clause, and it applies inside plpgsql too, so every call came back as
--
--     400  {'code': '21000', 'message': 'UPDATE requires a WHERE clause'}
--
-- and the admin activate endpoint returned 500. Nothing caught it because the
-- in-memory mock takes a different path entirely: repositories/model_versions
-- .set_active() calls mock_db.set_active_model_version() when Supabase is not
-- configured, and only reaches this RPC against a live database. So the
-- feature was exercised by every test and by every local demo, and was broken
-- the first time it ran for real.
--
-- `where is_active = true` is also the correct statement on its own terms: the
-- partial unique index idx_model_versions_one_active means at most one row is
-- ever active, so touching the rest was always wasted work.

create or replace function activate_model_version(target_id uuid)
returns setof model_versions
language plpgsql
as $$
begin
  if not exists (select 1 from model_versions where id = target_id) then
    raise exception 'model_versions row % does not exist', target_id;
  end if;

  -- Deactivate first: idx_model_versions_one_active is a unique partial index
  -- on (is_active) where is_active = true, so activating before deactivating
  -- would collide with the row already holding it.
  update model_versions set is_active = false where is_active = true;
  update model_versions set is_active = true  where id = target_id;

  return query select * from model_versions where id = target_id;
end;
$$;
