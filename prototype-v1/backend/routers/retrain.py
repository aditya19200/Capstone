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
from models.response_models import (
    RetrainEligibilityResponse,
    RetrainStatusResponse,
    RetrainTriggerResponse,
)
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
        "validated annotations exist, enqueues the job, and returns immediately. "
        "Poll GET /retrain/status for progress."
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
                "Collect more validated annotations before retraining."
            ),
        )

    # --- 3. Guard against concurrent retraining jobs ---
    latest = get_latest_retrain_job()
    if latest and latest["status"] in ("pending", "running"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"A retraining job ('{latest['id']}') is already "
                f"{latest['status']}. Wait for it to complete before "
                "triggering a new run."
            ),
        )

    # --- 4. Enqueue ---
    job = enqueue_retrain_job(
        triggered_by=x_user_id,
        notes=body.notes,
        min_annotations=body.min_annotations,
    )
    logger.info(
        "Retrain job enqueued: id=%s by user=%s validated_annotations=%d",
        job["id"], x_user_id, validated_count,
    )

    # --- 5. Return ---
    return RetrainTriggerResponse(
        retrain_job_id=job["id"],
        status="pending",
        message=(
            f"Retraining job '{job['id']}' queued successfully. "
            f"Using {validated_count} validated annotations. "
            "Poll GET /retrain/status for progress."
        ),
    )


# ---------------------------------------------------------------------------
# GET /retrain/eligibility
# ---------------------------------------------------------------------------

_DEFAULT_MIN_ANNOTATIONS = 50


@router.get(
    "/eligibility",
    response_model=RetrainEligibilityResponse,
    status_code=status.HTTP_200_OK,
    summary="How many validated annotations exist, and is that enough to retrain",
)
async def get_retrain_eligibility(
    min_annotations: int = Query(default=_DEFAULT_MIN_ANNOTATIONS, ge=1),
    x_role: Optional[str] = Header(default=None, alias="X-Role"),
) -> RetrainEligibilityResponse:
    """
    Report the current validated-annotation count against the retraining
    threshold, so the admin UI can show progress toward it rather than only
    surfacing the number inside a 409 when a run is refused.
    """
    role = (x_role or "").lower()
    if role != _ROLE_ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Role 'admin' required to view retraining eligibility.",
        )

    count = count_validated_annotations()
    return RetrainEligibilityResponse(
        validated_count=count,
        required=min_annotations,
        eligible=count >= min_annotations,
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
        "recent job if no job_id is provided."
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

    Admin-only: retraining metadata is sensitive (model versions).
    Returns 404 if the specified job_id does not exist, or if no jobs have
    been enqueued yet.
    """
    role = (x_role or "").lower()
    if role != _ROLE_ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Role 'admin' required to view retraining status.",
        )

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

    message = _status_message(job)

    return RetrainStatusResponse(
        retrain_job_id=job["id"],
        status=job["status"],
        model_version=job.get("model_version_id"),
        accuracy=None,           # accuracy is on model_versions, not retrain_jobs
        started_at=job.get("triggered_at"),
        completed_at=job.get("completed_at"),
        message=message,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _status_message(job: dict) -> str:
    """Build a human-readable message based on the current job state."""
    s = job["status"]
    if s == "pending":
        return f"Retraining job '{job['id']}' is queued and waiting to start."
    if s == "running":
        return f"Retraining job '{job['id']}' is currently running."
    if s == "complete":
        ver = job.get("model_version_id", "unknown")
        return f"Retraining completed. New model version id: '{ver}'."
    if s == "failed":
        return f"Retraining job '{job['id']}' failed. Check worker logs."
    return f"Unknown status: {s}"
