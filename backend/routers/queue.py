"""
routers/queue.py — Human review queue endpoints.

GET /queue/low-confidence
  Wraps repositories.predictions.list_low_confidence(REVIEW_THRESHOLD) so
  reviewers can see predictions the active-learning engine routed to them.
  Reviewer/Admin only — returns prediction text excerpts.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Query, status

from config.settings import settings
from models.response_models import LowConfidenceItem, LowConfidenceQueueResponse
from repositories import predictions as predictions_repo

logger = logging.getLogger(__name__)
router = APIRouter()

_EXCERPT_LENGTH = 200
_ROLES_ALLOWED = {"reviewer", "admin"}


def _require_reviewer(x_role: Optional[str]) -> None:
    role = (x_role or "").lower()
    if role not in _ROLES_ALLOWED:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Role '{x_role}' is not permitted here. Required: reviewer or admin.",
        )


@router.get(
    "/low-confidence",
    response_model=LowConfidenceQueueResponse,
    status_code=status.HTTP_200_OK,
    summary="List predictions below REVIEW_THRESHOLD (paginated)",
)
async def get_low_confidence_queue(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
    x_role: Optional[str] = Header(default=None, alias="X-Role"),
) -> LowConfidenceQueueResponse:
    _require_reviewer(x_role)

    all_rows = predictions_repo.list_low_confidence(settings.REVIEW_THRESHOLD)

    offset = (page - 1) * page_size
    page_rows = all_rows[offset : offset + page_size]

    items = [
        LowConfidenceItem(
            prediction_id=r["id"],
            text_excerpt=(
                r["text_content"][:_EXCERPT_LENGTH]
                + ("..." if len(r["text_content"]) > _EXCERPT_LENGTH else "")
            ),
            predicted_label=r["predicted_label"],
            confidence=r["confidence"],
            created_at=r["created_at"],
        )
        for r in page_rows
    ]

    return LowConfidenceQueueResponse(
        total=len(all_rows),
        page=page,
        page_size=page_size,
        items=items,
    )
