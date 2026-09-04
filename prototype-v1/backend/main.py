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
from routers import admin, annotate, batches, explain, ontology, predict, queue, retrain, stats
from services.model_service import model_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    On startup:  load InLegalBERT into memory once.
    On shutdown: release model resources.
    """
    # --- Startup ---
    print(f"[startup] Loading InLegalBERT from: {settings.MODEL_PATH}")
    model_service.load()
    print("[startup] Model loaded successfully.")

    # The served model has no model_versions row of its own; register it so
    # the admin table, the F1 chart and retrain's regression guard all have
    # a baseline to work from. No-ops if versions already exist. See
    # baseline.py.
    from baseline import register_baseline_if_absent
    register_baseline_if_absent()

    # No real Supabase configured (local dev) -> batch/xai workers as
    # separate processes would each get their own empty mock_db and never
    # see what this process inserts. Run them in-thread here instead. See
    # local_workers.py. Real deployments (is_configured() True) are
    # unaffected — they keep running workers as separate processes.
    if not is_configured():
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
    Returns the API status and whether the model is loaded.

    Used by monitoring, Docker health checks, and the frontend status banner.
    """
    return {
        "status": "ok",
        "model_loaded": model_service.is_loaded(),
        "model_path": settings.MODEL_PATH,
    }
