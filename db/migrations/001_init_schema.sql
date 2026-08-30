-- 001_init_schema.sql
-- XAI Legal Annotation Framework — core Postgres schema (7 tables)
-- Built against docs/SCHEMA_REQUEST.md and CLAUDE.md's schema block —
-- treat those as the frozen interface; if anything here doesn't match,
-- the docs win and this file needs fixing, not the other way round.

create extension if not exists pgcrypto;

-- ============================================================
-- dataset_versions — one row per training data snapshot
-- ============================================================
create table if not exists dataset_versions (
    id                  uuid primary key default gen_random_uuid(),
    version_id          text not null unique,      -- e.g. "v1", "v2"
    sample_count        integer not null,
    label_distribution  jsonb not null default '{}'::jsonb,
    created_at          timestamptz not null default now()
);

-- ============================================================
-- model_versions — one row per trained model checkpoint
-- is_active is the rollback mechanism (see 003_activate_model_version.sql)
-- ============================================================
create table if not exists model_versions (
    id                   uuid primary key default gen_random_uuid(),
    version_number       text not null,
    accuracy             real not null,
    f1_per_class         jsonb not null default '{}'::jsonb,
    trained_at           timestamptz not null default now(),
    file_path            text not null,
    dataset_version_id   uuid references dataset_versions(id),
    is_active            boolean not null default false
);

-- Enforces the single-active-version invariant at the DB level.
create unique index if not exists idx_model_versions_one_active
    on model_versions (is_active) where is_active = true;

-- ============================================================
-- predictions — one row per classification, the anchor table
-- ============================================================
create table if not exists predictions (
    id                 uuid primary key default gen_random_uuid(),
    text_content       text not null,
    predicted_label    text not null,
    label_id           smallint not null check (label_id between 0 and 9),
    confidence         real not null check (confidence >= 0 and confidence <= 1),
    all_probabilities  jsonb not null default '{}'::jsonb,
    model_version      text,
    created_at         timestamptz not null default now()
);

-- GET /queue/low-confidence filters confidence < REVIEW_THRESHOLD on every call.
create index if not exists idx_predictions_confidence
    on predictions (confidence);
-- admin metrics + queue ordering
create index if not exists idx_predictions_created_at
    on predictions (created_at desc);

-- ============================================================
-- batches — one row per ingestion event (paste / csv / pdf)
-- ============================================================
create table if not exists batches (
    id               uuid primary key default gen_random_uuid(),
    source           text not null check (source in ('paste', 'csv', 'pdf')),
    filename         text,
    status           text not null default 'pending'
                       check (status in ('pending', 'processing', 'done', 'failed')),
    total_items      integer not null,
    completed_items  integer not null default 0,
    created_at       timestamptz not null default now()
);

-- worker + admin dashboard filter by status
create index if not exists idx_batches_status
    on batches (status);

-- ============================================================
-- batch_items — one row per text unit in a batch.
-- This table IS the classify job queue (no Celery/Redis).
-- ============================================================
create table if not exists batch_items (
    id                 uuid primary key default gen_random_uuid(),
    batch_id           uuid not null references batches(id) on delete cascade,
    seq                integer not null,
    text_content       text not null,
    predicted_label    text,
    label_id           smallint,
    confidence         real,
    all_probabilities  jsonb,
    validated_label    text,
    status             text not null default 'pending'
                         check (status in ('pending', 'processing', 'classified', 'validated', 'failed')),
    attempts           smallint not null default 0,
    locked_at          timestamptz,
    error_message      text,
    unique (batch_id, seq)
);

-- claim_batch_items(n) scans for the oldest n 'pending' rows — composite
-- index satisfies the filter AND the ordering in a single index scan.
create index if not exists idx_batch_items_claim
    on batch_items (status, id);
-- GET /batches/{id}/items and CSV export both filter by batch_id
create index if not exists idx_batch_items_batch_id
    on batch_items (batch_id);

-- ============================================================
-- xai_jobs — one row per SHAP explain request; also a job queue
-- ============================================================
create table if not exists xai_jobs (
    id                uuid primary key default gen_random_uuid(),
    prediction_id     uuid not null references predictions(id) on delete cascade,
    status            text not null default 'pending'
                        check (status in ('pending', 'processing', 'done', 'failed')),
    token_importance  jsonb,
    summary           text,
    attempts          smallint not null default 0,
    locked_at         timestamptz,
    error_message     text
);

-- claim_xai_job() scans for the oldest pending row
create index if not exists idx_xai_jobs_claim
    on xai_jobs (status, created_at, id);
-- GET /explain/{job_id} looks up by prediction_id, polled every 3s by the frontend
create index if not exists idx_xai_jobs_prediction_id
    on xai_jobs (prediction_id);

-- ============================================================
-- annotations — human corrections, feeds retraining
-- (has_conflict is added in 003_activate_model_version_and_conflict.sql)
-- ============================================================
create table if not exists annotations (
    id               uuid primary key default gen_random_uuid(),
    prediction_id    uuid not null references predictions(id) on delete cascade,
    validated_label  text not null,
    annotator_id     uuid,
    status           text not null default 'pending'
                       check (status in ('pending', 'validated', 'rejected')),
    annotated_at     timestamptz not null default now()
);

-- conflict detection calls list_annotations(prediction_id) on every write
create index if not exists idx_annotations_prediction_id
    on annotations (prediction_id);
-- annotators see only their own rows
create index if not exists idx_annotations_annotator_id
    on annotations (annotator_id);
-- count_validated_annotations() runs before every retrain trigger
create index if not exists idx_annotations_status
    on annotations (status);

-- ============================================================
-- retrain_jobs — retraining pipeline state machine
-- ============================================================
create table if not exists retrain_jobs (
    id                uuid primary key default gen_random_uuid(),
    status            text not null default 'pending'
                        check (status in ('pending', 'running', 'complete', 'failed')),
    triggered_by      uuid,
    triggered_at      timestamptz not null default now(),
    completed_at      timestamptz,
    model_version_id  uuid references model_versions(id)
);

-- concurrent-run guard checks for any row with status in ('pending','running')
create index if not exists idx_retrain_jobs_status
    on retrain_jobs (status);
