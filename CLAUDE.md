# XAI Legal Annotation Framework

Indian legal text -> 10-class domain classification (InLegalBERT) -> SHAP
explanation -> human validation -> retraining loop.

Stack: FastAPI, Supabase (Postgres), Neo4j, React, PyTorch/HuggingFace, Docker.

---

## Ownership — READ THIS BEFORE EDITING ANYTHING

Four people work in this repo. Stay inside the boundary of the current session
owner. If a task needs a change outside that boundary, STOP and say so — do
not edit across the line, and do not create files in someone else's area.

| Owner   | Area                    | Paths                                   |
|---------|-------------------------|-----------------------------------------|
| Aditya  | Backend API + workers   | `app/`, `workers/`, `docker-compose.yml` |
| Arjun   | Data layer: Postgres schema, migrations, RPC functions, Neo4j | `db/`, `graph/` |
| Ankush  | ML pipeline             | `ml/` (`inference.py`, `scripts/retrain.py`, `models/`) |
| Bidisha | Frontend                | `frontend/`                             |

### Session scoping

Each developer keeps a `CLAUDE.local.md` at the repo root (gitignored)
declaring which boundary their session is in. If that file is absent, ask
which area you are working in before editing anything.

Universal rule: if a task needs a change outside the current owner's
boundary, STOP and state the exact change needed. Do not edit across the
line, and do not create files in someone else's area.

- `db/` — only Arjun writes migrations or SQL DDL. If a column or table is
  missing, report the exact spec needed; never create it.
- `ml/inference.py` — read-only dependency for everyone. Import, never edit.

---

## Architecture decisions (settled — do not redesign)

**The database table IS the job queue.** No Celery, no Redis, no RabbitMQ.
A row with `status='pending'` is a queued job. Workers claim rows atomically
via Postgres RPC functions using `FOR UPDATE SKIP LOCKED`.
Rationale: single GPU (horizontal scaling is impossible), and job state must
stay queryable for the progress bar and admin dashboard.

**Three processes, one Docker image, three commands:**

| Process           | Loop                    | Batch | Holds model |
|-------------------|-------------------------|-------|-------------|
| `api` (uvicorn)   | HTTP                    | -     | yes         |
| `worker-classify` | polls `batch_items`     | 16    | yes         |
| `worker-shap`     | polls `xai_jobs`        | 1     | yes         |

Separate because classification is ~1s per batch and SHAP is 30-90s per item.
Merged, one SHAP click freezes everyone's classification progress.

**SHAP is user-triggered only.** Never explain a whole batch — 200 items is
3-5 hours of GPU. One `xai_jobs` row per reviewer click.

**Async results reach the frontend by polling** `GET /explain/{job_id}` every
3s until `status='done'`. Supabase Realtime is a possible v2 upgrade, not v1.
Do NOT build FastAPI websockets — the worker is a separate process and cannot
reach uvicorn's in-memory connections.

**Ingestion order: paste -> CSV -> PDF.** All three end at the same
`insert_items(batch_id, list[str])`. PDF parsing is last priority.

**Confidence routing:** below `REVIEW_THRESHOLD` (0.5) goes to the human
review queue. This is the active-learning loop.

---

## Hard constraints

- `max_length=256` everywhere. The model was trained at 256. 512 is a bug.
- All Postgres identifiers `snake_case`. Postgres folds unquoted identifiers
  to lowercase, so `annotationStatus` silently becomes `annotationstatus`.
- `timestamptz`, never `timestamp`. Supabase runs UTC; the team is in IST.
- `REVIEW_THRESHOLD` lives in backend config. Never hardcode 0.5 anywhere else.
- Softmax confidence is not a calibrated probability. User-facing strings say
  "model certainty", never a percentage.
- Pydantic for all request validation. pandas ONLY for CSV parsing and
  export building — never in a single-request path.

---

## The 10 labels

Source of truth is `ml/models/v1/config.json` (`id2label`). Never hardcode
this list — read it from the loaded model.

```
0 Contract Law              5 Family Law
1 Criminal Law              6 Labour & Employment Law
2 Constitutional Law        7 Intellectual Property Law
3 Corporate / Company Law   8 Taxation Law
4 Property / Real Estate    9 Civil Procedure / Other
```

---

## Existing backend code

- `app/services/model_service.py` — works. **Bug: `max_length=512` at line 92.**
- `app/services/shap_service.py` — works. **Bug: `max_length=512` at line 36**
  (inside `_predict_fn`).
- `workers/xai_worker.py` — polling architecture correct, needs `MODEL_PATH`
  and the atomic claim RPC.
- `workers/prediction_worker.py` — stub. Becomes `worker-classify`.

---

## Schema the backend depends on (Arjun builds this in `db/`)

The backend reads and writes these. It does not create them. Treat this as a
fixed interface; if it doesn't match reality, report the mismatch.

```
predictions      id, text_content, predicted_label, label_id, confidence,
                 all_probabilities jsonb, model_version, created_at
batches          id, source ('paste'|'csv'|'pdf'), filename, status,
                 total_items, completed_items, created_at
batch_items      id, batch_id -> batches, seq, text_content, predicted_label,
                 label_id, confidence, all_probabilities jsonb, validated_label,
                 status ('pending'|'processing'|'classified'|'validated'|'failed'),
                 attempts, locked_at, error_message
xai_jobs         id, prediction_id -> predictions, status
                 ('pending'|'processing'|'done'|'failed'),
                 token_importance jsonb, summary, attempts, locked_at, error_message
annotations      id, prediction_id -> predictions, validated_label, annotator_id,
                 status ('pending'|'validated'|'rejected'), annotated_at
model_versions   id, version_number, accuracy, f1_per_class jsonb, trained_at,
                 file_path, dataset_version_id, is_active
dataset_versions id, version_id, sample_count, label_distribution jsonb, created_at
retrain_jobs     id, status ('pending'|'running'|'complete'|'failed'),
                 triggered_by, triggered_at, completed_at, model_version_id
```

Two RPC functions the workers call:

```
claim_batch_items(n int)  -> flips n oldest 'pending' rows to 'processing',
                             sets locked_at = now(), RETURNS the rows.
                             Must use FOR UPDATE SKIP LOCKED.
claim_xai_job()           -> same, limit 1, on xai_jobs.
```

---

## API contract (frozen — frontend builds against this)

```
POST /predict              {text} -> prediction object
POST /batches/paste        {texts:[...]} -> {batch_id, total_items}
POST /batches/csv          multipart file -> {batch_id, total_items}
GET  /batches/{id}         -> {status, total_items, completed_items}
GET  /batches/{id}/items   -> paginated items
GET  /batches/{id}/export  -> text/csv download
POST /explain              {prediction_id} -> 202 {job_id, status}
GET  /explain/{job_id}     -> {status, token_importance[], summary}
GET  /queue/low-confidence -> predictions below REVIEW_THRESHOLD
POST /annotations          {prediction_id, validated_label}
GET  /concepts/{domain}    -> proxied Neo4j lookup (Arjun's graph)
POST /retrain              -> 202 {job_id}
GET  /admin/metrics        -> dashboard payload
```

Prediction object:

```json
{"prediction_id":"uuid","label":"Criminal Law","label_id":1,
 "confidence":0.751,"all_probabilities":{"Contract Law":0.03},
 "needs_review":false,"truncated":false,"model_version":"v1"}
```

The frontend talks only to FastAPI. It never queries Neo4j or Postgres directly.

---

## Backend conventions

- All Supabase access goes through `app/repositories/` — no raw client calls
  scattered in route handlers. One module per table.
- `app/utils/text.py: normalize()` runs before every tokenization. Ankush
  calls the identical function in `retrain.py` or train/serve text drifts.
- Every worker loop starts with a stuck-job sweep: rows in `processing` with
  `locked_at` older than `STUCK_JOB_MINUTES` go back to `pending`,
  `attempts + 1`. Past `MAX_ATTEMPTS` -> `failed` with `error_message`.
- Never commit `.env`. It holds the Supabase `service_role` key.

## Commands

```bash
source ~/venv/bin/activate
uvicorn app.main:app --reload          # api
python -m workers.classify_worker      # worker-classify
python -m workers.xai_worker           # worker-shap
pytest -q                              # tests
```
