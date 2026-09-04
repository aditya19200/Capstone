"""
routers/admin.py — Admin dashboard and model-rollback endpoints.

POST /admin/models/{version_id}/activate
  Flip is_active onto the given model_versions row (rollback mechanism).
  Wraps the activate_model_version(target_id) Postgres RPC.

GET  /admin/metrics
  Dashboard payload: per-class F1 from the active model version, annotation
  counts by status, and per-batch throughput. Read from the DB, not JSON files.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, status

from models.response_models import (
    AdminMetricsResponse,
    BatchThroughputItem,
    ModelVersionOut,
    ModelVersionSummary,
)
from repositories import annotations as annotations_repo
from repositories import batches as batches_repo
from repositories import model_versions as model_versions_repo

logger = logging.getLogger(__name__)
router = APIRouter()

_ROLE_ADMIN = "admin"


def _require_admin(x_role: Optional[str]) -> None:
    role = (x_role or "").lower()
    if role != _ROLE_ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Role '{x_role}' is not permitted here. Required: admin.",
        )


# ---------------------------------------------------------------------------
# POST /admin/models/{version_id}/activate
# ---------------------------------------------------------------------------

@router.post(
    "/models/{version_id}/activate",
    response_model=ModelVersionOut,
    status_code=status.HTTP_200_OK,
    summary="Activate a model version (rollback mechanism)",
)
async def activate_model_version(
    version_id: str,
    x_role: Optional[str] = Header(default=None, alias="X-Role"),
) -> ModelVersionOut:
    _require_admin(x_role)

    try:
        row = model_versions_repo.set_active(version_id)
    except Exception as exc:
        if "does not exist" in str(exc):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model version '{version_id}' not found.",
            )
        raise

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model version '{version_id}' not found.",
        )

    logger.info("Model version activated: id=%s", version_id)

    return ModelVersionOut(
        id=row["id"],
        version_number=row["version_number"],
        accuracy=row["accuracy"],
        f1_per_class=row["f1_per_class"],
        file_path=row["file_path"],
        is_active=row["is_active"],
        trained_at=row["trained_at"],
    )


# ---------------------------------------------------------------------------
# GET /admin/metrics
# ---------------------------------------------------------------------------

@router.get(
    "/metrics",
    response_model=AdminMetricsResponse,
    status_code=status.HTTP_200_OK,
    summary="Admin dashboard metrics",
)
async def get_admin_metrics(
    x_role: Optional[str] = Header(default=None, alias="X-Role"),
) -> AdminMetricsResponse:
    _require_admin(x_role)

    active_model = model_versions_repo.get_active()
    annotation_counts = annotations_repo.count_by_status()
    recent_batches = batches_repo.list_all(limit=20)
    all_versions = model_versions_repo.list_all()

    return AdminMetricsResponse(
        active_model_version=active_model["version_number"] if active_model else None,
        f1_per_class=active_model["f1_per_class"] if active_model else {},
        annotation_counts=annotation_counts,
        model_versions=[
            ModelVersionSummary(
                id=v["id"],
                version_number=v["version_number"],
                accuracy=v["accuracy"],
                is_active=v.get("is_active", False),
                trained_at=v["trained_at"],
            )
            for v in all_versions
        ],
        batch_throughput=[
            BatchThroughputItem(
                batch_id=b["id"],
                source=b["source"],
                status=b["status"],
                total_items=b["total_items"],
                completed_items=b["completed_items"],
                created_at=b["created_at"],
            )
            for b in recent_batches
        ],
    )
