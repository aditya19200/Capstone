"""
workers/prediction_worker.py — Batch prediction worker process.

Runs as a separate Python process (not inside FastAPI). Polls the
prediction_jobs queue, processes pending jobs in batches using InLegalBERT,
stores results, and queues SHAP jobs when needed.

Run with:
    python -m workers.prediction_worker
"""

import logging
import sys
import time
from typing import List

from config.settings import settings
from repositories import predictions as predictions_repo
from repositories import xai_jobs as xai_jobs_repo
from services.active_learning import active_learning_engine
from services.model_service import model_service
from services.supabase_service import (
    create_explanation,
    get_pending_prediction_jobs,
    update_prediction_job,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [prediction_worker] %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

POLL_INTERVAL_SECONDS = 5
BATCH_SIZE = settings.BATCH_SIZE


# ---------------------------------------------------------------------------
# Core processing logic
# ---------------------------------------------------------------------------

def process_batch(jobs: List[dict]) -> None:
    """
    Run inference on a batch of prediction jobs and persist results.

    Steps per batch:
      1. Extract text from each job.
      2. Run batched InLegalBERT inference.
      3. For each result: apply active learning, persist prediction, queue SHAP.
      4. Mark job as completed (or failed on error).
    """
    texts = [job["textContent"] for job in jobs]   # legacy prediction_jobs key

    logger.info("Processing batch of %d jobs.", len(jobs))

    try:
        inferences = model_service.predict_batch(texts)
    except Exception as exc:
        logger.error("Batch inference failed: %s", exc)
        for job in jobs:
            update_prediction_job(job_id=job["jobId"], status="failed")
        return

    for job, inference in zip(jobs, inferences):
        job_id = job["jobId"]   # legacy prediction_jobs key
        try:
            _process_single_result(job, inference)
        except Exception as exc:
            logger.error("Failed to process job %s: %s", job_id, exc)
            update_prediction_job(job_id=job_id, status="failed")


def _process_single_result(job: dict, inference: dict) -> None:
    """
    Persist a single inference result and handle downstream queuing.

    Args:
        job:       The prediction_job dict from the legacy queue.
        inference: The inference result dict from model_service.predict_batch().
    """
    job_id = job["jobId"]   # legacy prediction_jobs key

    update_prediction_job(job_id=job_id, status="processing")

    al_result = active_learning_engine.evaluate(
        confidence=inference["confidence"],
        probabilities=inference["probabilities"],
        entropy=inference["entropy"],
        margin=inference["margin"],
    )

    # Persist prediction using the repositories path (real schema)
    pred_row = predictions_repo.insert(
        text_content=job["textContent"],
        predicted_label=inference["predicted_label"],
        label_id=inference["label_id"],
        confidence=inference["confidence"],
        all_probabilities=inference["probabilities"],
        model_version=getattr(model_service, "model_version", "v1"),
    )
    prediction_id = pred_row["id"]

    xai_job_id = None
    if al_result.requires_shap:
        create_explanation(prediction_id=prediction_id)
        xai_job = xai_jobs_repo.insert(prediction_id=prediction_id)
        xai_job_id = xai_job["id"]
        logger.info("XAI job queued: %s for prediction %s", xai_job_id, prediction_id)

    result_payload = {
        "prediction_id": prediction_id,
        "document_id": None,
        "predicted_label": inference["predicted_label"],
        "confidence": inference["confidence"],
        "probabilities": inference["probabilities"],
        "routing_decision": al_result.routing_decision,
        "xai_job_id": xai_job_id,
        "entropy": inference["entropy"],
        "margin": inference["margin"],
    }

    update_prediction_job(job_id=job_id, status="completed", result=result_payload)

    logger.info(
        "Job %s completed: label='%s' conf=%.4f routing=%s",
        job_id,
        inference["predicted_label"],
        inference["confidence"],
        al_result.routing_decision,
    )


# ---------------------------------------------------------------------------
# Worker loop
# ---------------------------------------------------------------------------

def run_worker() -> None:
    """
    Main worker loop.

    Loads the model once, then continuously polls the prediction_jobs queue.
    Jobs are processed in batches of BATCH_SIZE.

    Runs until interrupted (Ctrl-C / SIGTERM).
    """
    logger.info("Prediction worker starting up.")
    logger.info("Model path: %s", settings.MODEL_PATH)
    logger.info("Batch size: %d", BATCH_SIZE)
    logger.info("Poll interval: %ds", POLL_INTERVAL_SECONDS)

    logger.info("Loading InLegalBERT...")
    model_service.load()
    logger.info("Model ready. Entering poll loop.")

    try:
        while True:
            pending = get_pending_prediction_jobs()

            if not pending:
                logger.debug("Queue empty. Sleeping %ds.", POLL_INTERVAL_SECONDS)
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            logger.info("Found %d pending job(s).", len(pending))

            for i in range(0, len(pending), BATCH_SIZE):
                batch = pending[i : i + BATCH_SIZE]
                process_batch(batch)

    except KeyboardInterrupt:
        logger.info("Prediction worker shutting down (KeyboardInterrupt).")
    finally:
        model_service.unload()
        logger.info("Model unloaded. Worker exited cleanly.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_worker()
