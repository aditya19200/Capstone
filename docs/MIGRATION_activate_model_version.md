# Migration: activate_model_version RPC + annotations.has_conflict column

**Author:** Aditya (backend) → Arjun (data layer)  
**Context:** see `docs/SCHEMA_REQUEST.md` for the full schema spec.

---

## 1. Atomic model-version rollback RPC

Called by `repositories/model_versions.set_active()`.

The two UPDATEs are safe from the atomicity concern precisely because they run
inside a single `plpgsql` function body: Postgres wraps every `CALL` / `SELECT`
of a `plpgsql` function in an implicit subtransaction, so both statements
commit together or neither does.  A crash between the two statements rolls back
the first UPDATE automatically.  This is the fix — not something still open.

The existence check fires **before** any data is touched, so a bad `target_id`
fails loudly with a clear message instead of silently deactivating every model
version.

```sql
CREATE OR REPLACE FUNCTION activate_model_version(target_id uuid)
RETURNS SETOF model_versions
LANGUAGE plpgsql
AS $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM model_versions WHERE id = target_id) THEN
    RAISE EXCEPTION 'model_versions row % does not exist', target_id;
  END IF;

  UPDATE model_versions SET is_active = false;
  UPDATE model_versions SET is_active = true  WHERE id = target_id;

  RETURN QUERY SELECT * FROM model_versions WHERE id = target_id;
END;
$$;
```

### Why a function and not two separate UPDATE calls from the app?

Without a transaction wrapper in the app, a process crash between the two
`supabase-py` calls leaves every model version inactive and the API cannot
load weights at the next startup.  A single function call is one round-trip
and one implicit transaction — no client-side transaction management needed.

---

## 2. Conflict flag on annotations

`has_conflict` is orthogonal to `status`: a row can be `status='validated'`
*and* `has_conflict=true` simultaneously (e.g. two annotators chose different
labels; a reviewer accepted one but the flag stays so the disagreement is
visible in the dashboard).

```sql
ALTER TABLE annotations
  ADD COLUMN IF NOT EXISTS has_conflict boolean NOT NULL DEFAULT false;

-- Partial index for the reviewer conflict queue (only flagged rows are indexed)
CREATE INDEX IF NOT EXISTS idx_annotations_has_conflict
  ON annotations (has_conflict)
  WHERE has_conflict = true;
```

Backend sets this via:
- **Mock mode:** `mock_db.set_annotation_has_conflict(annotation_id, has_conflict)`
- **Supabase:** `UPDATE annotations SET has_conflict = $1 WHERE id = $2`

The conflict-detection logic lives in
`services/supabase_service.detect_and_flag_conflict()`.
