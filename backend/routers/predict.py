"""
routers/predict.py — Prediction endpoints.

POST /predict
  Accepts raw legal text, runs InLegalBERT inference, applies the active
  learning engine, persists results, and queues a SHAP job when needed.

GET  /predict/{job_id}
  Polls the status of a queued prediction job (used by the batch worker flow).
"""

import logging
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, status

from models.request_models import PredictRequest
from models.response_models import PredictJobResponse, PredictResponse
from services.active_learning import active_learning_engine
from services.model_service import model_service
from services.supabase_service import (
    create_document,
    create_explanation,
    create_prediction,
    enqueue_xai_job,
    get_prediction_job,
    update_document_status,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify legal text",
    description=(
        "Submit raw legal text for classification by InLegalBERT. "
        "Returns the predicted label, confidence score, full probability "
        "distribution, and an active-learning routing decision. "
        "A SHAP explanation job is automatically queued for medium- and "
        "low-confidence predictions."
    ),
)
async def predict(
    body: PredictRequest,
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
    x_role: Optional[str] = Header(default=None, alias="X-Role"),
) -> PredictResponse:
    """
    Full prediction pipeline:

      1. Create a legal_documents row (or use supplied document_id).
      2. Run InLegalBERT inference (async, thread-pool executor).
      3. Apply active learning engine → routing decision.
      4. Persist prediction row.
      5. Queue SHAP job if routing is not AUTO_ACCEPT.
      6. Return PredictResponse immediately.
    """
    # --- 1. Resolve / create document ---
    if body.document_id:
        document_id = body.document_id
    else:
        doc = create_document(text_content=body.text, status="pending")
        document_id = doc["documentId"]

    # --- 2. Run inference ---
    try:
        inference = await model_service.predict_async(body.text)
    except RuntimeError as exc:
        logger.error("Model inference failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model is not available: {exc}",
        )
    except Exception as exc:
        logger.exception("Unexpected inference error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {exc}",
        )

    # --- 3. Active learning routing ---
    al_result = active_learning_engine.evaluate(
        confidence=inference["confidence"],
        probabilities=inference["probabilities"],
        entropy=inference["entropy"],
        margin=inference["margin"],
    )

    logger.info(
        "Prediction: label='%s' conf=%.4f routing=%s doc=%s",
        inference["predicted_label"],
        inference["confidence"],
        al_result.routing_decision,
        document_id,
    )

    # --- 4. Persist prediction ---
    pred_row = create_prediction(
        document_id=document_id,
        predicted_label=inference["predicted_label"],
        confidence_score=inference["confidence"],
        probability_distribution=inference["probabilities"],
        routing_decision=al_result.routing_decision,
        entropy=inference["entropy"],
        margin=inference["margin"],
    )
    prediction_id: str = pred_row["predictionId"]

    # Update document status now that a prediction exists
    update_document_status(document_id, "predicted")

    # --- 5. Queue SHAP job if needed ---
    xai_job_id: Optional[str] = None
    if al_result.requires_shap:
        # Create a pending explanation row first so GET /explain can return
        # 'pending' immediately without waiting for the worker
        create_explanation(prediction_id=prediction_id)
        xai_job = enqueue_xai_job(prediction_id=prediction_id)
        xai_job_id = xai_job["jobId"]
        logger.info(
            "XAI job queued: job_id=%s pred_id=%s", xai_job_id, prediction_id
        )

    # --- 6. Build and return response ---
    return PredictResponse(
        prediction_id=prediction_id,
        document_id=document_id,
        predicted_label=inference["predicted_label"],
        confidence=inference["confidence"],
        probabilities=inference["probabilities"],
        routing_decision=al_result.routing_decision,
        xai_job_id=xai_job_id,
        entropy=inference["entropy"],
        margin=inference["margin"],
    )


# ---------------------------------------------------------------------------
# GET /predict/{job_id}
# ---------------------------------------------------------------------------

@router.get(
    "/{job_id}",
    response_model=PredictJobResponse,
    status_code=status.HTTP_200_OK,
    summary="Poll a prediction job",
    description=(
        "Check the status of a queued prediction job submitted via the batch "
        "worker flow. Returns the full PredictResponse once the job is completed."
    ),
)
async def get_prediction_job_status(job_id: str) -> PredictJobResponse:
    """
    Fetch a prediction job by ID and return its current status.

    The result field is populated once status == 'completed'.
    Returns 404 if the job_id does not exist.
    """
    job = get_prediction_job(job_id=job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction job '{job_id}' not found.",
        )

    result: Optional[PredictResponse] = None
    if job["status"] == "completed" and job.get("result"):
        try:
            result = PredictResponse(**job["result"])
        except Exception as exc:
            logger.error("Failed to deserialise job result for %s: %s", job_id, exc)

    return PredictJobResponse(
        job_id=job["jobId"],
        status=job["status"],
        result=result,
    )
