"""
workers/batch_worker.py — Batch classification worker process.

Runs as a separate process (not inside FastAPI). Polls batch_items for
pending rows (batch_items IS the classify queue — see CLAUDE.md), classifies
them with InLegalBERT, and — per the TODO left in
db/migrations/005_batch_items_prediction_id.sql — inserts a real predictions
row for each item so batch-sourced items converge on the same predictions
table that /predict uses, letting /explain and /annotate attach to them.

Without this worker, POST /batches/paste and /batches/csv insert rows that
sit at 'pending' forever: nothing else in the codebase claims batch_items.
(workers/prediction_worker.py exists but targets an older, unrelated
'prediction_jobs' queue — it is labeled "legacy" in its own source and
doesn't touch batch_items at all.)

Run with:
    python -m workers.batch_worker
"""

import logging
import sys
import time
from typing import Dict, List

from config.settings import settings
from repositories import batch_items as batch_items_repo
from repositories import batches as batches_repo
from repositories import predictions as predictions_repo
from repositories import xai_jobs as xai_jobs_repo
from services.active_learning import active_learning_engine
from services.model_service import model_service
from services.supabase_service import create_explanation

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [batch_worker] %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 5
BATCH_SIZE = settings.BATCH_SIZE


# ---------------------------------------------------------------------------
# Core processing logic
# ---------------------------------------------------------------------------

def process_claimed_items(items: List[Dict]) -> None:
    """
    Classify a batch of claimed batch_items and persist results.

    Mirrors routers/predict.py's per-text pipeline (inference -> active
    learning -> persist prediction -> queue SHAP if needed), just applied
    to a batch of items claimed from the queue instead of one request body.
    """
    texts = [item["text_content"] for item in items]
    logger.info("Processing %d claimed batch item(s).", len(items))

    try:
        inferences = model_service.predict_batch(texts)
    except Exception:
        logger.exception("Batch inference failed; marking items failed.")
        for item in items:
            batch_items_repo.update(item["id"], status="failed", error_message="inference failed")
        return

    for item, inference in zip(items, inferences):
        try:
            _process_single_item(item, inference)
        except Exception:
            logger.exception("Failed to process batch item %s", item["id"])
            batch_items_repo.update(item["id"], status="failed", error_message="processing error")


def _process_single_item(item: Dict, inference: Dict) -> None:
    al_result = active_learning_engine.evaluate(
        confidence=inference["confidence"],
        probabilities=inference["probabilities"],
        entropy=inference["entropy"],
        margin=inference["margin"],
    )

    # Persist as a real predictions row, exactly like /predict does, so
    # /explain and /annotate have something to attach to.
    pred_row = predictions_repo.insert(
        text_content=item["text_content"],
        predicted_label=inference["predicted_label"],
        label_id=inference["label_id"],
        confidence=inference["confidence"],
        all_probabilities=inference["probabilities"],
        model_version=getattr(model_service, "model_version", "v1"),
    )
    prediction_id = pred_row["id"]

    if al_result.requires_shap:
        # Pending explanation row first, same order /predict uses, so
        # GET /explain returns 'pending' immediately instead of 404.
        create_explanation(prediction_id=prediction_id)
        xai_jobs_repo.insert(prediction_id=prediction_id)

    batch_items_repo.update(
        item["id"],
        status="classified",
        predicted_label=inference["predicted_label"],
        label_id=inference["label_id"],
        confidence=inference["confidence"],
        all_probabilities=inference["probabilities"],
        prediction_id=prediction_id,
    )

    batch = batches_repo.increment_completed(item["batch_id"])
    if batch and batch["completed_items"] >= batch["total_items"]:
        batches_repo.update_status(item["batch_id"], "done")

    logger.info(
        "Item %s -> %s (conf=%.4f, routing=%s)",
        item["id"], inference["predicted_label"], inference["confidence"], al_result.routing_decision,
    )


# ---------------------------------------------------------------------------
# Main poll loop
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("batch_worker starting up. Loading InLegalBERT...")
    if not model_service.is_loaded():
        model_service.load()
    logger.info("Model ready. Polling batch_items every %ds.", POLL_INTERVAL_SECONDS)

    while True:
        batch_items_repo.sweep_stuck()
        claimed = batch_items_repo.claim(BATCH_SIZE)
        if claimed:
            process_claimed_items(claimed)
        else:
            time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
