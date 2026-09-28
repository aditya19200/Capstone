"""
routers/queue.py — Human review queue endpoints.

GET /queue/low-confidence
  Predictions a human should look at. Two independent reasons put an item
  here:

    low_confidence     — the model doubted itself (confidence below
                         REVIEW_THRESHOLD). This is the original behaviour.
    ontology_conflict  — the model was confident, but the legal-concept
                         vocabulary in the text belongs to a different
                         domain (services/ontology_check.py).

  The second reason exists because the first one cannot catch confidently
  wrong predictions: those are auto-accepted, never reviewed, and so never
  corrected. Confidence is the model grading its own work; the ontology
  check is evidence from outside the model.

  The route name is kept for compatibility with the existing frontend.
  Reviewer/Admin only — returns prediction text excerpts.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Query, status

from config.settings import settings
from models.response_models import LowConfidenceItem, LowConfidenceQueueResponse
from repositories import predictions as predictions_repo
from services.ontology_check import ontology_check

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

    threshold = settings.REVIEW_THRESHOLD

    # Low confidence remains the primary route. When the ontology check is
    # unavailable (markers file absent) this degrades to exactly the old
    # behaviour rather than failing.
    if not ontology_check.is_loaded():
        all_rows = [
            (r, "low_confidence", None)
            for r in predictions_repo.list_low_confidence(threshold)
        ]
    else:
        all_rows = []
        for r in predictions_repo.list_all():
            if r["confidence"] < threshold:
                all_rows.append((r, "low_confidence", None))
                continue
            # Only confident predictions are worth checking — anything below
            # the threshold is already queued on confidence alone.
            conflict = ontology_check.check(r["text_content"], r["predicted_label"])
            if conflict:
                all_rows.append((r, "ontology_conflict", conflict))

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
            flag_reason=reason,
            suggested_label=(conflict or {}).get("suggested_label"),
            matched_terms=(conflict or {}).get("matched_terms"),
            flag_detail=(conflict or {}).get("reason"),
            evidence_score=(conflict or {}).get("evidence_score"),
            predicted_label_score=(conflict or {}).get("predicted_label_score"),
        )
        for r, reason, conflict in page_rows
    ]

    return LowConfidenceQueueResponse(
        total=len(all_rows),
        page=page,
        page_size=page_size,
        items=items,
    )
