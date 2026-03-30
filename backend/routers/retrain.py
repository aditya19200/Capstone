"""
routers/retrain.py — Retraining pipeline endpoints.

POST /retrain
  Admin-only. Validates annotation count, enqueues a retraining job, and
  returns immediately. The actual fine-tuning runs in a separate worker process.

GET  /retrain/status
  Poll the status of the most recent retraining job (or a specific job by ID).
"""

import logging
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Query, status

from models.request_models import RetrainRequest
from models.response_models import RetrainStatusResponse, RetrainTriggerResponse
from services.supabase_service import (
    count_validated_annotations,
    enqueue_retrain_job,
    get_latest_retrain_job,
    get_retrain_job,
)

logger = logging.getLogger(__name__)
router = APIRouter()

_ROLE_ADMIN = "admin"


# ---------------------------------------------------------------------------
# POST /retrain
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=RetrainTriggerResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger retraining pipeline (Admin only)",
    description=(
        "Manually kick off a retraining run. The request validates that enough "
        "accepted/modified annotations exist, enqueues the job, and returns "
        "immediately. Poll GET /retrain/status for progress."
    ),
)
async def trigger_retrain(
    body: RetrainRequest,
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
    x_role: Optional[str] = Header(default=None, alias="X-Role"),
) -> RetrainTriggerResponse:
    """
    Retraining trigger pipeline:

      1. Admin role check.
      2. Count validated annotations — reject if below min_annotations.
      3. Check no retraining job is already running.
      4. Enqueue retraining job.
      5. Return 202 Accepted + job ID.
    """
    # --- 1. Admin-only ---
    role = (x_role or "").lower()
    if role != _ROLE_ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"Role '{x_role}' is not permitted to trigger retraining. "
                "Required: admin."
            ),
        )

    # --- 2. Validate annotation count ---
    validated_count = count_validated_annotations()
    if validated_count < body.min_annotations:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Insufficient validated annotations for retraining. "
                f"Found {validated_count}, required {body.min_annotations}. "
                "Collect more accepted/modified annotations before retraining."
            ),
        )

    # --- 3. Guard against concurrent retraining jobs ---
    latest = get_latest_retrain_job()
    if latest and latest["status"] in ("pending", "processing"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"A retraining job ('{latest['jobId']}') is already "
                f"{latest['status']}. Wait for it to complete before "
                "triggering a new run."
            ),
        )

    # --- 4. Enqueue ---
    job = enqueue_retrain_job(
        notes=body.notes,
        min_annotations=body.min_annotations,
    )
    logger.info(
        "Retrain job enqueued: id=%s by user=%s validated_annotations=%d",
        job["jobId"], x_user_id, validated_count,
    )

    # --- 5. Return ---
    return RetrainTriggerResponse(
        retrain_job_id=job["jobId"],
        status="pending",
        message=(
            f"Retraining job '{job['jobId']}' queued successfully. "
            f"Using {validated_count} validated annotations. "
            "Poll GET /retrain/status for progress."
        ),
    )


# ---------------------------------------------------------------------------
# GET /retrain/status
# ---------------------------------------------------------------------------

@router.get(
    "/status",
    response_model=RetrainStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Check retraining job status",
    description=(
        "Return the status of a specific retraining job by job_id, or the most "
        "recent job if no job_id is provided. Includes accuracy and model version "
        "once the job completes."
    ),
)
async def get_retrain_status(
    job_id: Optional[str] = Query(
        default=None,
        description="Specific retrain job ID. Omit to fetch the latest job.",
    ),
    x_role: Optional[str] = Header(default=None, alias="X-Role"),
) -> RetrainStatusResponse:
    """
    Fetch retrain job status.

    Admin-only: retraining metadata is sensitive (model accuracy, versions).
    Returns 404 if the specified job_id does not exist, or if no jobs have
    been enqueued yet.
    """
    # Admin-only
    role = (x_role or "").lower()
    if role != _ROLE_ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Role 'admin' required to view retraining status.",
        )

    # Fetch the right job
    if job_id:
        job = get_retrain_job(job_id=job_id)
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Retrain job '{job_id}' not found.",
            )
    else:
        job = get_latest_retrain_job()
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No retraining jobs found. Trigger one via POST /retrain.",
            )

    # Build human-readable status message
    message = _status_message(job)

    return RetrainStatusResponse(
        retrain_job_id=job["jobId"],
        status=job["status"],
        model_version=job.get("modelVersion"),
        accuracy=job.get("accuracy"),
        started_at=job.get("startedAt"),
        completed_at=job.get("completedAt"),
        message=message,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _status_message(job: dict) -> str:
    """Build a human-readable message based on the current job state."""
    s = job["status"]
    if s == "pending":
        return f"Retraining job '{job['jobId']}' is queued and waiting to start."
    if s == "processing":
        return f"Retraining job '{job['jobId']}' is currently running."
    if s == "completed":
        acc = job.get("accuracy")
        ver = job.get("modelVersion", "unknown")
        acc_str = f"{acc:.4f}" if acc is not None else "N/A"
        return (
            f"Retraining completed. New model version: '{ver}', "
            f"accuracy: {acc_str}."
        )
    if s == "failed":
        return f"Retraining job '{job['jobId']}' failed. Check worker logs."
    return f"Unknown status: {s}"
