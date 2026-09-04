-- 005_batch_items_prediction_id.sql
--
-- Bug fix (found in PR #2 review by Aditya): batch_items writes
-- predicted_label/confidence directly onto itself but never inserts a
-- row into predictions. Since xai_jobs.prediction_id and
-- annotations.prediction_id both reference predictions(id), any item
-- that arrived via CSV/paste upload (batch_items) has nothing for
-- /explain or annotation-correction to attach to — only the single
-- /predict endpoint (which inserts into predictions directly) works
-- today.
--
-- This migration only adds the column. The classify worker itself
-- needs to be updated (Aditya's side, in backend/workers/) to:
--   1. insert a predictions row after predict_batch() returns a result
--      for each item (same shape as /predict's insert)
--   2. write the returned id back onto batch_items.prediction_id
--   3. set batch_items.status = 'classified'
-- so single-predict and batch-predict converge on the same
-- predictions table.

alter table batch_items
    add column if not exists prediction_id uuid references predictions(id);

-- GET /batches/{id}/items and the review queue will need to join
-- through this to find explain/annotation state for batch-sourced items
create index if not exists idx_batch_items_prediction_id
    on batch_items (prediction_id);
