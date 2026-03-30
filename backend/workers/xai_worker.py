"""
workers/xai_worker.py — SHAP explanation worker process.

Runs as a separate Python process (not inside FastAPI). Polls the xai_jobs
queue, generates SHAP token-level explanations for each pending job, and
stores the results back via supabase_service.

Run with:
    python xai_worker.py

or:
    python -m workers.xai_worker

SHAP is CPU-bound and slow (~5–30s per document depending on length).
Running it in a separate process keeps the FastAPI event loop responsive.
"""

import logging
import sys
import time

from config.settings import settings
from services.model_service import model_service
from services.shap_service import shap_service
from services.supabase_service import (
    get_pending_xai_jobs,
    get_prediction,
    get_explanation_by_prediction,
    update_explanation,
    update_xai_job,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [xai_worker] %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

POLL_INTERVAL_SECONDS = 10      # SHAP is slow; poll less aggressively than prediction worker
MAX_EVALS = 200                 # SHAP max_evals per explanation — raise for better accuracy


# ---------------------------------------------------------------------------
# Core processing logic
# ---------------------------------------------------------------------------

def process_xai_job(job: dict) -> None:
    """
    Process a single XAI job end-to-end.

    Steps:
      1. Mark job as 'processing'.
      2. Fetch the associated prediction (need text + label_id).
      3. Fetch the prediction's document text from the prediction row.
      4. Run SHAP via shap_service.explain_safe().
      5. Update the explanation row with results (completed or failed).
      6. Mark job as 'completed' or 'failed'.

    Args:
        job: XAI job dict from the queue {jobId, predictionId, status, ...}.
    """
    job_id       = job["jobId"]
    prediction_id = job["predictionId"]

    logger.info("Processing XAI job %s for prediction %s", job_id, prediction_id)

    # --- 1. Mark as processing ---
    update_xai_job(job_id=job_id, status="processing")

    # --- 2. Fetch prediction ---
    prediction = get_prediction(prediction_id=prediction_id)
    if prediction is None:
        logger.error(
            "XAI job %s: prediction '%s' not found. Marking failed.",
            job_id, prediction_id,
        )
        update_xai_job(job_id=job_id, status="failed")
        return

    # --- 3. Resolve document text ---
    # The prediction row stores the document text via documentId.
    # We stored text in the document row during prediction; retrieve it
    # via the mock_db document store through supabase_service.
    # Import here to avoid circular dependency at module level.
    from services.supabase_service import get_document

    document = get_document(document_id=prediction["documentId"])
    if document is None:
        logger.error(
            "XAI job %s: document '%s' not found. Marking failed.",
            job_id, prediction["documentId"],
        )
        update_xai_job(job_id=job_id, status="failed")
        return

    text      = document["textContent"]
    label_id  = None

    # Resolve label_id from the predicted label name
    from services.model_service import LABEL2ID
    label_id = LABEL2ID.get(prediction["predictedLabel"])
    if label_id is None:
        logger.error(
            "XAI job %s: unknown predicted label '%s'. Marking failed.",
            job_id, prediction["predictedLabel"],
        )
        update_xai_job(job_id=job_id, status="failed")
        return

    # --- 4. Run SHAP ---
    logger.info(
        "Running SHAP for job %s: label='%s' (id=%d) text_len=%d",
        job_id, prediction["predictedLabel"], label_id, len(text),
    )

    token_importances = shap_service.explain_safe(
        text=text,
        predicted_label_id=label_id,
        max_evals=MAX_EVALS,
    )

    # --- 5. Update explanation row ---
    explanation = get_explanation_by_prediction(prediction_id=prediction_id)

    if explanation is None:
        # Edge case: explanation row was never created (shouldn't happen in
        # normal flow, but handle gracefully)
        logger.warning(
            "XAI job %s: no explanation row found for prediction %s. "
            "SHAP result will not be persisted.",
            job_id, prediction_id,
        )
        update_xai_job(job_id=job_id, status="failed")
        return

    if token_importances is not None:
        update_explanation(
            explanation_id=explanation["explanationId"],
            shap_values=token_importances,
            status="completed",
        )
        update_xai_job(job_id=job_id, status="completed")
        logger.info(
            "XAI job %s completed: %d tokens explained for prediction %s.",
            job_id, len(token_importances), prediction_id,
        )
    else:
        # explain_safe returned None — SHAP failed internally, error already logged
        update_explanation(
            explanation_id=explanation["explanationId"],
            shap_values=[],
            status="failed",
        )
        update_xai_job(job_id=job_id, status="failed")
        logger.error(
            "XAI job %s failed: SHAP returned None for prediction %s.",
            job_id, prediction_id,
        )


# ---------------------------------------------------------------------------
# Worker loop
# ---------------------------------------------------------------------------

def run_worker() -> None:
    """
    Main XAI worker loop.

    Loads the model once, then continuously polls the xai_jobs queue.
    Jobs are processed one at a time (SHAP is memory-intensive; batching
    SHAP across multiple documents simultaneously is not beneficial on CPU).

    Runs until interrupted (Ctrl-C / SIGTERM).
    """
    logger.info("XAI worker starting up.")
    logger.info("Model path   : %s", settings.MODEL_PATH)
    logger.info("SHAP max_evals: %d", MAX_EVALS)
    logger.info("Poll interval : %ds", POLL_INTERVAL_SECONDS)

    # Load model once — SHAP needs it for forward passes
    logger.info("Loading InLegalBERT for SHAP...")
    model_service.load()
    logger.info("Model ready. Entering XAI poll loop.")

    try:
        while True:
            pending = get_pending_xai_jobs()

            if not pending:
                logger.debug("XAI queue empty. Sleeping %ds.", POLL_INTERVAL_SECONDS)
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            logger.info("Found %d pending XAI job(s).", len(pending))

            # Process one job at a time — SHAP is CPU/memory intensive
            for job in pending:
                try:
                    process_xai_job(job)
                except Exception as exc:
                    # Catch-all so one bad job never kills the loop
                    logger.exception(
                        "Unhandled error processing XAI job %s: %s",
                        job.get("jobId"), exc,
                    )
                    try:
                        update_xai_job(job_id=job["jobId"], status="failed")
                    except Exception:
                        pass  # best-effort

    except KeyboardInterrupt:
        logger.info("XAI worker shutting down (KeyboardInterrupt).")
    finally:
        model_service.unload()
        logger.info("Model unloaded. XAI worker exited cleanly.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_worker()
