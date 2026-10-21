"""
main.py — FastAPI application entry point.

Initializes the app, loads the InLegalBERT model once at startup,
registers all routers, and exposes a /health endpoint.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from repositories._client import is_configured
from routers import admin, annotate, batches, domains, explain, ontology, predict, queue, retrain, stats
from services import neo4j_service
from services.model_service import model_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    On startup:  load InLegalBERT into memory once.
    On shutdown: release model resources.
    """
    # --- Startup ---
    # Named from the active domain pack, not hardcoded: with ACTIVE_DOMAIN=medical
    # this line used to read "Loading InLegalBERT from ./models/MedicalModelShared".
    from config.domains import active_domain as _active_domain
    print(f"[startup] Loading the {_active_domain().key} model from: {settings.MODEL_PATH}")
    model_service.load()
    print("[startup] Model loaded successfully.")

    # The served model has no model_versions row of its own; register it so
    # the admin table, the F1 chart and retrain's regression guard all have
    # a baseline to work from. No-ops if versions already exist. See
    # baseline.py.
    # Ontology consistency check — catches confidently-wrong predictions that
    # confidence-based routing is structurally blind to. Degrades to a no-op
    # if the markers file is absent.
    from services.ontology_check import ontology_check
    ontology_check.load()
    print(f"[startup] Ontology check: {'enabled' if ontology_check.is_loaded() else 'DISABLED (no markers file)'}")

    from baseline import register_baseline_if_absent, reconcile_served_version
    register_baseline_if_absent()

    # Rollback has to survive a restart: model_versions.get_active() is the
    # record of which weights should be serving, so honour it rather than
    # always booting MODEL_PATH. Normally this is a no-op — the baseline row
    # registered above points at MODEL_PATH, which is already loaded.
    from repositories import model_versions as _model_versions_repo
    _active = _model_versions_repo.get_active()
    if _active and _active.get("file_path") and _active["file_path"] != settings.MODEL_PATH:
        print(f"[startup] Active version is {_active['version_number']}; loading its weights.")
        try:
            model_service.load(
                model_path=_active["file_path"],
                version_number=_active["version_number"],
            )
        except Exception as exc:
            # Don't refuse to boot over a bad saved version — MODEL_PATH is
            # already loaded and serving, which is the safe fallback.
            print(f"[startup] WARNING: could not load active version: {exc}")
            print(f"[startup] Continuing with {settings.MODEL_PATH}.")

    # Whatever came out of that, the table should now describe it: the right
    # name on the row, and is_active on the row whose weights are loaded.
    _served_name = reconcile_served_version(
        model_service.loaded_path or settings.MODEL_PATH)
    if _served_name:
        # Without this the weights loaded straight from MODEL_PATH have no
        # name, so every prediction they make is stamped "baseline" while
        # the version table calls them something else.
        model_service.name_loaded_version(_served_name)
    print(f"[startup] Serving {model_service.loaded_version or 'baseline'} "
          f"from {model_service.loaded_path or settings.MODEL_PATH}")

    # No real Supabase configured (local dev) -> batch/xai workers as
    # separate processes would each get their own empty mock_db and never
    # see what this process inserts. Run them in-thread here instead. See
    # local_workers.py. Real deployments (is_configured() True) are
    # unaffected — they keep running workers as separate processes.
    # RUN_WORKERS_IN_PROCESS lets run_local_demo.sh keep that behaviour once
    # Supabase is connected, since it starts no separate worker processes.
    if not is_configured() or settings.RUN_WORKERS_IN_PROCESS:
        from local_workers import start_local_workers
        start_local_workers()

    yield

    # --- Shutdown ---
    print("[shutdown] Releasing model resources.")
    model_service.unload()


app = FastAPI(
    title="XAI Legal Annotation API",
    description=(
        "Backend for the XAI-Enabled Knowledge Graph Driven Legal Annotation Framework. "
        "Provides legal text classification, explainability (SHAP), active learning routing, "
        "human annotation management, and retraining pipeline controls."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS — allow the React frontend (and dev tooling) to reach the API
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(predict.router,   prefix="/predict",   tags=["Prediction"])
app.include_router(explain.router,   prefix="/explain",   tags=["Explainability"])
app.include_router(annotate.router,  prefix="/annotate",  tags=["Annotation"])
app.include_router(retrain.router,   prefix="/retrain",   tags=["Retraining"])
app.include_router(ontology.router,  prefix="/ontology",  tags=["Ontology"])
app.include_router(domains.router,   prefix="/domains",   tags=["Domains"])
app.include_router(batches.router,   prefix="/batches",   tags=["Batches"])
app.include_router(queue.router,     prefix="/queue",     tags=["Queue"])
app.include_router(admin.router,     prefix="/admin",     tags=["Admin"])
app.include_router(stats.router,     prefix="/stats",     tags=["Stats"])


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Health"])
async def health_check():
    """
    Returns the API status, whether the model is loaded, and which data layer
    this process is actually using.

    Used by monitoring, Docker health checks, and the frontend status banner.

    `data_layer` reports what the RUNNING process is doing, which is the point
    of it being here rather than only in check_data_layer.py. That script
    reads the same .env the app does, so it passes whether or not the server
    was restarted after the file appeared — "the script says live, the server
    is still in memory" is a real way to mislead yourself. This answers for
    the process serving the request.
    """
    return {
        "status": "ok",
        "model_loaded": model_service.is_loaded(),
        # What is actually resident right now, which is not necessarily
        # MODEL_PATH once a retrained version has been activated.
        "model_path": model_service.loaded_path or settings.MODEL_PATH,
        "model_version": model_service.loaded_version or "baseline",
        "configured_model_path": settings.MODEL_PATH,
        "data_layer": {
            # Supabase has no fallback: configured means every write goes to
            # Postgres, and a failure surfaces as an error rather than
            # quietly becoming an in-memory write.
            "supabase": "live" if is_configured() else "in-memory",
            # Neo4j does fall back, silently and by design — see its status().
            "neo4j": neo4j_service.status(),
        },
    }
