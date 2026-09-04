# Ownership map — Prototype V1

Who owns what, so anyone can find their own work and know what they're
reviewing.

## Why this file exists instead of per-person folders

The obvious thing would be one folder per person. It doesn't work here,
because `backend/` is a single Python package whose imports cross ownership
lines in both directions:

- Aditya's `routers/predict.py` does `from services.model_service import model_service` — that's ML code.
- Ankush's `workers/retrain_worker.py` does `from config.settings import settings` and `from repositories import ...` — that's backend code.

Split those into sibling folders and both sides stop importing. Making it
work would mean turning each area into a separately installable package —
a real refactor, and one that buys nothing while the app is this size.

So the code stays organised by **component** (which is how it runs), and
ownership is written down here (which is how we work). `.github/CODEOWNERS` (repo root, where GitHub reads it) makes
GitHub enforce it automatically on pull requests.

---

## By area

| Path | Owner | What it is |
|------|-------|------------|
| `frontend/` | **Bidisha** | React app — annotate, review, conflicts, admin, metrics screens |
| `db/` | **Arjun** | Postgres schema, migrations, RPC functions, RLS policies |
| `graph/` | **Arjun** | Neo4j legal ontology — seed and loader |
| `backend/routers/` | **Aditya** | FastAPI HTTP layer |
| `backend/repositories/` | **Aditya** | Data access (Supabase, with mock fallback) |
| `backend/models/*.py` | **Aditya** | Pydantic request/response schemas |
| `backend/config/`, `backend/utils/`, `backend/tests/` | **Aditya** | Settings, shared text helpers, tests |
| `backend/services/` | **split** | See the file-level table below |
| `backend/workers/` | **split** | See the file-level table below |

## File-level, where a folder is shared

| File | Owner | Notes |
|------|-------|-------|
| `backend/services/model_service.py` | **Ankush** | Loads InLegalBERT, runs inference, active-learning signals |
| `backend/services/shap_service.py` | **Ankush** | SHAP token attributions |
| `backend/services/active_learning.py` | **Aditya** | Confidence routing thresholds (config-driven) |
| `backend/services/supabase_service.py` | **Aditya** | Legacy data-access layer |
| `backend/services/neo4j_service.py` | **Aditya** | Ontology access (currently mock-backed) |
| `backend/services/mock_db.py` | **Aditya** | In-memory stand-in for Postgres |
| `backend/workers/retrain_worker.py` | **Ankush** | Retraining pipeline |
| `backend/workers/batch_worker.py` | **Ankush** | Batch classification — see note below |
| `backend/workers/xai_worker.py` | **Aditya** | SHAP job queue worker |
| `backend/workers/prediction_worker.py` | **Aditya** | Legacy — targets the old `prediction_jobs` queue, superseded by `batch_worker.py` |
| `backend/local_workers.py` | **Ankush** | Runs workers in-thread for local dev only |
| `backend/models/LegalModelShared/` | **Ankush** | Model weights — **not in git**, distributed separately |
| `seed_retrain_data.py`, `run_local_demo.sh` | **Ankush** | Local demo tooling |

---

## Cross-boundary changes needing review

Several fixes in this prototype touch files their nominal owner didn't write.
Flagging them explicitly rather than burying them in the diff:

| File | Nominal owner | What changed and why |
|------|---------------|----------------------|
| `backend/routers/explain.py` | Aditya | Fixed a stale field name (`jobId` → `id`) and a status vocabulary mismatch (`completed` vs `done`) — both caused HTTP 500s |
| `backend/routers/batches.py` | Aditya | Added `GET /batches`; exposed `prediction_id` on batch items |
| `backend/routers/annotate.py` | Aditya | Exposed `prediction_id` so conflicts can be resolved |
| `backend/routers/retrain.py` | Aditya | Added `GET /retrain/eligibility` |
| `backend/repositories/batch_items.py` | Aditya | Added the missing mock fallbacks for `claim()` and `update()` |
| `backend/services/mock_db.py` | Aditya | Added `claim_batch_items()` and `update_batch_item()` |
| `backend/models/response_models.py` | Aditya | New response models + two missing `prediction_id` fields |
| `backend/main.py` | Aditya | Starts in-thread workers when no real DB is configured |
| `frontend/src/api/client.js` | Bidisha | Wired seven functions to real endpoints that had since been built |
| `frontend/src/pages/*.jsx`, `components/**` | Bidisha | Pagination fix, Recent Batches list, Explain in review queue, retrain counter |
| `frontend/src/index.css`, `index.html` | Bidisha | Dark theme (pure palette remap, no component changes) |

Full reasoning for each is in the session changelog.

---

## What's still owned and unfinished

| Item | Owner |
|------|-------|
| Admin + Metrics pages are mock data — `/admin/metrics` needs a model-versions list | **Aditya** |
| Dashboard is 100% hardcoded — never calls the API | **Bidisha** |
| A newly trained model is never loaded into memory | **Ankush** |
| `retrain_jobs` is missing the `notes` and `min_annotations` columns the API sends | **Arjun** |
| No real Supabase/Neo4j connected — everything runs on the in-memory mock | **Aditya + Arjun** |
