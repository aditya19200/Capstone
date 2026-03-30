"""
main.py — FastAPI application entry point.

Initializes the app, loads the InLegalBERT model once at startup,
registers all routers, and exposes a /health endpoint.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from routers import predict, explain, annotate, retrain, ontology
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
