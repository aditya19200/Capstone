"""
workers/prediction_worker.py — Batch prediction worker process.

Runs as a separate Python process (not inside FastAPI). Polls the
prediction_jobs queue, processes pending jobs in batches using InLegalBERT,
stores results, and queues SHAP jobs when needed.

Run with:
    python -m workers.prediction_worker

or via the entry-point script:
    python worker.py
"""

import logging
import sys
import time
from typing import List

from config.settings import settings
from services.active_learning import active_learning_engine
from services.model_service import model_service
from services.supabase_service import (
    create_document,
    create_explanation,
    create_prediction,
    enqueue_xai_job,
    get_pending_prediction_jobs,
    update_document_status,
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

POLL_INTERVAL_SECONDS = 5       # how often to check for new jobs
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

    Args:
        jobs: List of prediction job dicts from the queue.
    """
    texts = [job["textContent"] for job in jobs]

    logger.info("Processing batch of %d jobs.", len(jobs))

    # --- Batch inference ---
    try:
        inferences = model_service.predict_batch(texts)
    except Exception as exc:
        logger.error("Batch inference failed: %s", exc)
        # Mark all jobs in this batch as failed
        for job in jobs:
            update_prediction_job(job_id=job["jobId"], status="failed")
        return

    # --- Per-job post-processing ---
    for job, inference in zip(jobs, inferences):
        job_id = job["jobId"]
        try:
            _process_single_result(job, inference)
        except Exception as exc:
            logger.error("Failed to process job %s: %s", job_id, exc)
            update_prediction_job(job_id=job_id, status="failed")


def _process_single_result(job: dict, inference: dict) -> None:
    """
    Persist a single inference result and handle downstream queuing.

    Args:
        job:       The prediction_job dict from the queue.
        inference: The inference result dict from model_service.predict_batch().
    """
    job_id = job["jobId"]

    # Mark job as processing
    update_prediction_job(job_id=job_id, status="processing")

    # Create or resolve document
    doc = create_document(text_content=job["textContent"], status="pending")
    document_id = doc["documentId"]

    # Active learning routing
    al_result = active_learning_engine.evaluate(
        confidence=inference["confidence"],
        probabilities=inference["probabilities"],
        entropy=inference["entropy"],
        margin=inference["margin"],
    )

    # Persist prediction row
    pred_row = create_prediction(
        document_id=document_id,
        predicted_label=inference["predicted_label"],
        confidence_score=inference["confidence"],
        probability_distribution=inference["probabilities"],
        routing_decision=al_result.routing_decision,
        entropy=inference["entropy"],
        margin=inference["margin"],
    )
    prediction_id = pred_row["predictionId"]
    update_document_status(document_id, "predicted")

    # Queue SHAP job if required
    xai_job_id = None
    if al_result.requires_shap:
        create_explanation(prediction_id=prediction_id)
        xai_job = enqueue_xai_job(prediction_id=prediction_id)
        xai_job_id = xai_job["jobId"]
        logger.info("XAI job queued: %s for prediction %s", xai_job_id, prediction_id)

    # Build the result payload stored back on the job row
    result_payload = {
        "prediction_id": prediction_id,
        "document_id": document_id,
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
    Jobs are processed in batches of BATCH_SIZE. The loop sleeps for
    POLL_INTERVAL_SECONDS when the queue is empty.

    Runs until interrupted (Ctrl-C / SIGTERM).
    """
    logger.info("Prediction worker starting up.")
    logger.info("Model path: %s", settings.MODEL_PATH)
    logger.info("Batch size: %d", BATCH_SIZE)
    logger.info("Poll interval: %ds", POLL_INTERVAL_SECONDS)

    # Load model once — reused for every batch
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

            # Process in batches
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
