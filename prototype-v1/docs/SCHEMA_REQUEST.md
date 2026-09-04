# Schema Request — Backend Dependencies

**From:** Aditya (backend)  
**To:** Arjun (data layer)  
**Date:** 2026-08-20

This document is a spec, not DDL. Do not treat any SQL here as canonical — you own
the DDL. Every column name below is `snake_case`; Postgres folds unquoted identifiers
to lowercase so camelCase is a silent bug. All timestamps must be `timestamptz`
(Supabase runs UTC; the team is in IST). All primary keys are UUIDs unless noted.

---

## Tables

### `predictions`

Populated by the API (`POST /predict`) and the classify worker. Read by the XAI
worker (needs text + label to run SHAP) and the low-confidence queue endpoint.

| Column | Type | Nullable | Written by | Read by |
|--------|------|----------|-----------|---------|
| `id` | `uuid` PK default `gen_random_uuid()` | no | API/worker on insert | everywhere |
| `text_content` | `text` | no | API/worker | xai_worker (SHAP needs the original text) |
| `predicted_label` | `text` | no | API/worker | xai_worker, annotations |
| `label_id` | `smallint` | no | API/worker | xai_worker (maps label→int for SHAP) |
| `confidence` | `real` | no | API/worker | low-confidence queue, admin metrics |
| `all_probabilities` | `jsonb` | no | API/worker | explain endpoint, admin metrics |
| `model_version` | `text` | yes | API/worker | admin metrics |
| `created_at` | `timestamptz` | no | default `now()` | admin metrics, ordering |

**Indexes needed:**
- `predictions(confidence)` — `GET /queue/low-confidence` filters `confidence < 0.55`
  on every call; without an index this is a full table scan that grows with every
  prediction.
- `predictions(created_at DESC)` — admin metrics and queue ordering.

---

### `batches`

One row per ingestion event (paste / CSV / PDF). Created by the batch ingestion
endpoints; updated as items are processed.

| Column | Type | Nullable | Written by | Read by |
|--------|------|----------|-----------|---------|
| `id` | `uuid` PK | no | API on insert | API, worker |
| `source` | `text` CHECK IN `('paste','csv','pdf')` | no | API | API |
| `filename` | `text` | yes | API (null for paste) | `GET /batches/{id}` |
| `status` | `text` CHECK IN `('pending','processing','done','failed')` | no | API, worker | `GET /batches/{id}` |
| `total_items` | `integer` | no | API on insert | `GET /batches/{id}`, progress bar |
| `completed_items` | `integer` | no | worker (increments per item) | `GET /batches/{id}`, progress bar |
| `created_at` | `timestamptz` | no | default `now()` | ordering |

**Indexes needed:**
- `batches(status)` — worker and admin dashboard filter by status.

---

### `batch_items`

One row per text unit within a batch. This table **is the job queue** — the worker
polls it directly. No Celery, no Redis.

| Column | Type | Nullable | Written by | Read by |
|--------|------|----------|-----------|---------|
| `id` | `uuid` PK | no | API on insert | worker, export |
| `batch_id` | `uuid` FK → `batches.id` | no | API | `GET /batches/{id}/items`, export |
| `seq` | `integer` | no | API (1-based position in batch) | export (preserve order) |
| `text_content` | `text` | no | API | worker, export |
| `predicted_label` | `text` | yes | classify worker | export, reviewer UI |
| `label_id` | `smallint` | yes | classify worker | export |
| `confidence` | `real` | yes | classify worker | export, routing |
| `all_probabilities` | `jsonb` | yes | classify worker | export |
| `validated_label` | `text` | yes | annotation (human) | export, retraining |
| `status` | `text` CHECK IN `('pending','processing','classified','validated','failed')` | no | worker, API | `claim_batch_items` RPC, `GET /batches/{id}/items` |
| `attempts` | `smallint` | no | worker (increments on each claim) | stuck-job sweep |
| `locked_at` | `timestamptz` | yes | worker (set on claim, null after done/failed) | stuck-job sweep |
| `error_message` | `text` | yes | worker on failure | admin dashboard |

**Indexes needed:**
- `batch_items(status, id)` — `claim_batch_items(n)` scans for the oldest `n`
  `pending` rows. A composite index on `(status, id)` lets Postgres satisfy the
  filter and the ordering in a single index scan. Without it the RPC does a full
  table scan on every worker tick (every 5 seconds).
- `batch_items(batch_id)` — `GET /batches/{id}/items` and the export endpoint both
  filter by `batch_id`; this index keeps both O(items_in_batch) not O(all_items).

---

### `xai_jobs`

One row per SHAP explanation request. Also a job queue polled by the XAI worker.

| Column | Type | Nullable | Written by | Read by |
|--------|------|----------|-----------|---------|
| `id` | `uuid` PK | no | API on enqueue | worker, `GET /explain/{job_id}` |
| `prediction_id` | `uuid` FK → `predictions.id` | no | API | worker (fetches text + label) |
| `status` | `text` CHECK IN `('pending','processing','done','failed')` | no | API, worker | `GET /explain/{job_id}`, frontend polling |
| `token_importance` | `jsonb` | yes | xai_worker on completion | `GET /explain/{job_id}` |
| `summary` | `text` | yes | xai_worker (optional prose summary) | `GET /explain/{job_id}` |
| `attempts` | `smallint` | no | worker (increments per claim) | stuck-job sweep |
| `locked_at` | `timestamptz` | yes | worker (set on claim) | stuck-job sweep |
| `error_message` | `text` | yes | worker on failure | admin dashboard |

**Indexes needed:**
- `xai_jobs(status, id)` — same reason as `batch_items`: `claim_xai_job()` scans
  for oldest pending row.
- `xai_jobs(prediction_id)` — `GET /explain/{job_id}` looks up by `prediction_id`,
  not by the job's own id. Without this index every poll from the frontend (every 3s)
  scans the full table.

---

### `annotations`

Human annotation decisions. Written by the annotation router, read for conflict
detection and retraining dataset assembly.

| Column | Type | Nullable | Written by | Read by |
|--------|------|----------|-----------|---------|
| `id` | `uuid` PK | no | API on insert | API |
| `prediction_id` | `uuid` FK → `predictions.id` | no | API | conflict detection, export |
| `validated_label` | `text` | no | annotator | retraining, export |
| `annotator_id` | `uuid` | yes | API (from X-User-Id header; null if missing) | `GET /annotations` user-scoped filter |
| `status` | `text` CHECK IN `('pending','validated','rejected')` | no | API | conflict detection, `count_validated_annotations()` |
| `annotated_at` | `timestamptz` | no | default `now()` | ordering |

**Indexes needed:**
- `annotations(prediction_id)` — conflict detection calls `list_annotations(document_id)`
  on every annotation write; this index keeps that lookup fast.
- `annotations(annotator_id)` — annotators see only their own rows; without this
  the filter is a full table scan per page load.
- `annotations(status)` — `count_validated_annotations()` is called before every
  retraining trigger to check the threshold.

---

### `model_versions`

One row per trained model checkpoint. `is_active` is the rollback mechanism —
flip it without touching `.env` or restarting anything except the API/workers.

| Column | Type | Nullable | Written by | Read by |
|--------|------|----------|-----------|---------|
| `id` | `uuid` PK | no | retrain worker | API (active model lookup) |
| `version_number` | `text` | no | retrain worker (e.g. `"v2"`) | admin metrics, response payload |
| `accuracy` | `real` | no | retrain worker | admin metrics |
| `f1_per_class` | `jsonb` | no | retrain worker (`{label: f1_score}`) | admin metrics |
| `trained_at` | `timestamptz` | no | default `now()` | ordering |
| `file_path` | `text` | no | retrain worker (path to saved weights) | API startup (load model from this path) |
| `dataset_version_id` | `uuid` FK → `dataset_versions.id` | yes | retrain worker | audit trail |
| `is_active` | `boolean` | no | retrain worker (true = currently serving) | API startup, admin metrics |

**Why `is_active` matters:** The API reads the `file_path` of the row where
`is_active = true` at startup. To roll back from v3 to v2 without changing `.env`:
set `is_active = false` on v3, `is_active = true` on v2, restart. No config change
needed. The constraint `CHECK (is_active IN (true, false))` is not enough — you
should add a partial unique index `WHERE is_active = true` to enforce that exactly
one version is active at any time.

**Index needed:**
- Partial unique index on `model_versions(is_active) WHERE is_active = true` — enforces
  single-active-version invariant at the DB level.

---

### `dataset_versions`

Snapshot of the labelled dataset used for each training run.

| Column | Type | Nullable | Written by | Read by |
|--------|------|----------|-----------|---------|
| `id` | `uuid` PK | no | retrain worker before training | `model_versions.dataset_version_id` |
| `version_id` | `text` | no | retrain worker (human-readable tag) | admin metrics |
| `sample_count` | `integer` | no | retrain worker | admin metrics |
| `label_distribution` | `jsonb` | no | retrain worker (`{label: count}`) | admin metrics |
| `created_at` | `timestamptz` | no | default `now()` | ordering |

No additional indexes needed beyond the PK.

---

### `retrain_jobs`

Tracks each retraining pipeline run from trigger to completion.

| Column | Type | Nullable | Written by | Read by |
|--------|------|----------|-----------|---------|
| `id` | `uuid` PK | no | API on trigger | `GET /retrain/status` |
| `status` | `text` CHECK IN `('pending','running','complete','failed')` | no | API, retrain worker | `GET /retrain/status`, concurrent-run guard |
| `triggered_by` | `uuid` | yes | API (from X-User-Id header) | admin audit |
| `triggered_at` | `timestamptz` | no | default `now()` | ordering |
| `completed_at` | `timestamptz` | yes | retrain worker on finish | `GET /retrain/status` |
| `model_version_id` | `uuid` FK → `model_versions.id` | yes | retrain worker on completion | `GET /retrain/status` |

**Index needed:**
- `retrain_jobs(status)` — the concurrent-run guard before every trigger checks for
  any row with `status IN ('pending','running')`.

---

## RPC Functions

Both RPCs must run inside a transaction and use `FOR UPDATE SKIP LOCKED`.

### `claim_batch_items(n int)`

**Purpose:** Atomically hand `n` batch items to the classify worker.

**Behaviour:**
1. `SELECT … FROM batch_items WHERE status = 'pending' ORDER BY id LIMIT n FOR UPDATE SKIP LOCKED`
2. `UPDATE batch_items SET status = 'processing', locked_at = now(), attempts = attempts + 1 WHERE id IN (<claimed ids>)`
3. `RETURN` the updated rows (all columns).

**Why `FOR UPDATE SKIP LOCKED`:**
- The classify worker runs as a separate OS process. If two worker processes ever
  run concurrently (e.g. during a deploy overlap), a plain `SELECT … UPDATE` without
  locking lets both processes claim the same row, causing double-classification and
  corrupted results.
- `FOR UPDATE` takes a row-level lock at claim time. `SKIP LOCKED` makes the second
  worker skip already-locked rows rather than blocking — keeping throughput high and
  avoiding deadlocks.
- Even with a single worker process, this pattern is required for the stuck-job
  recovery sweep: rows with `locked_at < now() - interval` go back to `pending`.
  Without the lock, a race between the sweep and the normal poll could flip a
  mid-flight row back to `pending` while the worker is still processing it.

**Stuck-job handling (done by the worker, not this RPC):** Rows that stay in
`processing` with `locked_at` older than `STUCK_JOB_MINUTES` are swept back to
`pending` with `attempts + 1`. Past `MAX_ATTEMPTS` they flip to `failed` with an
`error_message`. The RPC only claims — the worker owns the sweep.

---

### `claim_xai_job()`

**Purpose:** Atomically claim one pending SHAP job for the XAI worker.

**Behaviour:**
1. `SELECT … FROM xai_jobs WHERE status = 'pending' ORDER BY id LIMIT 1 FOR UPDATE SKIP LOCKED`
2. `UPDATE xai_jobs SET status = 'processing', locked_at = now(), attempts = attempts + 1 WHERE id = <claimed id>`
3. `RETURN` the updated row (all columns).

**Why limit 1:** SHAP on a single document takes 30–90 seconds on CPU and holds
the model in memory throughout. Processing two SHAP jobs concurrently on a single
GPU is not beneficial. The worker always processes one at a time.

**Why `FOR UPDATE SKIP LOCKED`:** Same reasoning as `claim_batch_items`. The XAI
worker is a separate process; locking prevents a restart-overlap from re-claiming
an in-flight job.

---

## Summary of Indexes

| Table | Index | Why |
|-------|-------|-----|
| `predictions` | `(confidence)` | low-confidence queue filter |
| `predictions` | `(created_at DESC)` | ordering for metrics and queue |
| `batches` | `(status)` | worker and dashboard filters |
| `batch_items` | `(status, id)` | `claim_batch_items` RPC — filter + order in one scan |
| `batch_items` | `(batch_id)` | item listing and CSV export |
| `xai_jobs` | `(status, id)` | `claim_xai_job` RPC |
| `xai_jobs` | `(prediction_id)` | frontend polling via `GET /explain/{job_id}` |
| `annotations` | `(prediction_id)` | conflict detection on every annotation write |
| `annotations` | `(annotator_id)` | user-scoped annotation listing |
| `annotations` | `(status)` | `count_validated_annotations()` pre-retrain check |
| `model_versions` | partial unique `WHERE is_active = true` | enforce single active model |
| `retrain_jobs` | `(status)` | concurrent-run guard on trigger |
