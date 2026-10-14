-- 007_domain_column.sql
-- Makes the schema hold more than one label set.
--
-- Two things in 001 assumed there would only ever be ten legal labels:
--
--   1. predictions.label_id carried  check (label_id between 0 and 9).
--      The upper bound is a property of the domain, not of the database — a
--      domain with eight or twelve classes is a different domains/<key>.json,
--      and the schema has no way to know which one is loaded. Keeping the
--      bound here would mean editing a migration every time a domain is
--      added, so the check is narrowed to what is actually invariant:
--      a label_id is a non-negative index into the model's output layer.
--
--   2. Nothing recorded WHICH label set a row's label_id indexes. That is
--      the part that actually corrupts data: label_id 1 is "Criminal Law"
--      under domains/legal.json and something entirely different under
--      another pack, so a predictions table holding both is unreadable
--      without the domain. Retraining reads validated annotations straight
--      out of these rows, which means an unlabelled mix would silently
--      train one model on another domain's labels.
--
-- Existing rows are all legal: that is the only pack that existed when they
-- were written, hence the default and the backfill.

alter table predictions
    drop constraint if exists predictions_label_id_check;

alter table predictions
    add constraint predictions_label_id_check
    check (label_id >= 0);

alter table predictions
    add column if not exists domain text not null default 'legal';

alter table annotations
    add column if not exists domain text not null default 'legal';

-- Queries are almost always "this domain's rows, newest first".
create index if not exists idx_predictions_domain
    on predictions (domain, created_at desc);

comment on column predictions.domain is
    'Key of the domains/<key>.json pack whose label set label_id indexes.';
comment on column annotations.domain is
    'Key of the domains/<key>.json pack whose label set the labels belong to.';
