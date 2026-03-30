"""
routers/explain.py — Explainability endpoints.

POST /explain
  Manually trigger a SHAP explanation for an existing prediction.
  Used when the frontend wants to request an explanation on demand
  (e.g. for an AUTO_ACCEPT prediction that the user wants to inspect).

GET  /explain/{prediction_id}
  Fetch the SHAP result for a prediction. Returns status='pending' while
  the XAI worker is still processing, and the full token importances once done.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, status

from models.request_models import ExplainRequest
from models.response_models import ExplainResponse, ExplainTriggerResponse, TokenImportance
from services.supabase_service import (
    create_explanation,
    enqueue_xai_job,
    get_explanation_by_prediction,
    get_prediction,
    get_xai_job,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# POST /explain
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=ExplainTriggerResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Manually trigger a SHAP explanation",
    description=(
        "Queue an async SHAP explanation job for an existing prediction. "
        "Use GET /explain/{prediction_id} to poll for the result. "
        "If an explanation already exists (pending or completed) for this "
        "prediction, that job's ID is returned rather than creating a duplicate."
    ),
)
async def trigger_explanation(
    body: ExplainRequest,
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
    x_role: Optional[str] = Header(default=None, alias="X-Role"),
) -> ExplainTriggerResponse:
    """
    Manually queue a SHAP explanation.

      1. Verify the prediction exists.
      2. Check if an explanation already exists — avoid duplicate jobs.
      3. Create explanation row (status=pending) + enqueue XAI job.
      4. Return job ID so the frontend can poll.
    """
    # --- 1. Verify prediction exists ---
    prediction = get_prediction(prediction_id=body.prediction_id)
    if prediction is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction '{body.prediction_id}' not found.",
        )

    # --- 2. Check for existing explanation (avoid duplicate jobs) ---
    existing = get_explanation_by_prediction(prediction_id=body.prediction_id)
    if existing is not None:
        logger.info(
            "Explanation already exists for prediction %s (status=%s), "
            "returning existing job.",
            body.prediction_id, existing["status"],
        )
        return ExplainTriggerResponse(
            xai_job_id=existing["explanationId"],   # use explanation ID as reference
            prediction_id=body.prediction_id,
            status=existing["status"],
            message=(
                f"Explanation already {'completed' if existing['status'] == 'completed' else 'in progress'} "
                f"for prediction '{body.prediction_id}'. "
                "Use GET /explain/{prediction_id} to retrieve the result."
            ),
        )

    # --- 3. Create explanation row + enqueue XAI job ---
    explanation = create_explanation(prediction_id=body.prediction_id)
    xai_job     = enqueue_xai_job(prediction_id=body.prediction_id)

    logger.info(
        "SHAP job queued manually: xai_job=%s pred=%s",
        xai_job["jobId"], body.prediction_id,
    )

    return ExplainTriggerResponse(
        xai_job_id=xai_job["jobId"],
        prediction_id=body.prediction_id,
        status="pending",
        message=(
            f"SHAP explanation job queued for prediction '{body.prediction_id}'. "
            "Poll GET /explain/{prediction_id} for the result."
        ),
    )


# ---------------------------------------------------------------------------
# GET /explain/{prediction_id}
# ---------------------------------------------------------------------------

@router.get(
    "/{prediction_id}",
    response_model=ExplainResponse,
    status_code=status.HTTP_200_OK,
    summary="Fetch SHAP explanation",
    description=(
        "Retrieve the SHAP token-importance explanation for a given prediction. "
        "Returns status='pending' while the XAI worker is still processing. "
        "token_importances is populated once status='completed'."
    ),
)
async def get_explanation(prediction_id: str) -> ExplainResponse:
    """
    Fetch the SHAP explanation for a prediction.

      1. Verify the prediction exists.
      2. Look up the explanation row by prediction_id.
      3. Return status + token importances (if ready).
    """
    # --- 1. Verify prediction exists ---
    prediction = get_prediction(prediction_id=prediction_id)
    if prediction is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction '{prediction_id}' not found.",
        )

    # --- 2. Look up explanation ---
    explanation = get_explanation_by_prediction(prediction_id=prediction_id)
    if explanation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No explanation found for prediction '{prediction_id}'. "
                "POST /explain to queue one."
            ),
        )

    # --- 3. Build token importances list if completed ---
    token_importances = None
    if explanation["status"] == "completed" and explanation.get("shapValues"):
        try:
            token_importances = [
                TokenImportance(token=t["token"], importance=t["importance"])
                for t in explanation["shapValues"]
            ]
        except Exception as exc:
            logger.error(
                "Failed to deserialise SHAP values for prediction %s: %s",
                prediction_id, exc,
            )
            # Return completed status with empty list rather than crashing
            token_importances = []

    return ExplainResponse(
        explanation_id=explanation["explanationId"],
        prediction_id=prediction_id,
        status=explanation["status"],
        token_importances=token_importances,
        generated_at=explanation.get("generatedAt"),
    )
