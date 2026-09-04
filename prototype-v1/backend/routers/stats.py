"""
routers/stats.py — aggregate figures for the dashboard landing page.

GET /stats/dashboard
  One call returning everything the dashboard shows: throughput counts,
  confidence banding, label distribution and a recent-activity feed.

Why this is separate from /admin/metrics: that route is admin-only, but the
dashboard is the landing page for annotators and reviewers too. These are
aggregate counts with short text excerpts — nothing role-sensitive — so any
signed-in role may read them.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, Header, HTTPException, Query, status

from config.settings import settings
from models.response_models import (
    DashboardStatsResponse,
    LabelCount,
    RecentActivityItem,
)
from repositories import annotations as annotations_repo
from repositories import batches as batches_repo
from repositories import predictions as predictions_repo

logger = logging.getLogger(__name__)
router = APIRouter()

_KNOWN_ROLES = {"annotator", "reviewer", "admin"}

# Matches ConfidenceBadge.jsx: >=0.7 High, 0.5-0.7 Medium, <0.5 Low. Kept in
# step with the badges so the chart can't disagree with the rows it summarises.
_HIGH_BAND = 0.7

_EXCERPT_LENGTH = 90
_RECENT_LIMIT = 8
_TOP_LABELS = 5


@router.get(
    "/dashboard",
    response_model=DashboardStatsResponse,
    status_code=status.HTTP_200_OK,
    summary="Aggregate counts for the dashboard landing page",
)
async def get_dashboard_stats(
    recent_limit: int = Query(default=_RECENT_LIMIT, ge=1, le=50),
    x_role: Optional[str] = Header(default=None, alias="X-Role"),
) -> DashboardStatsResponse:
    role = (x_role or "").lower()
    if role not in _KNOWN_ROLES:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"Role '{x_role}' is not permitted here. "
                "Required: annotator, reviewer or admin."
            ),
        )

    predictions = predictions_repo.list_all()
    batches = batches_repo.list_all()
    annotation_counts = annotations_repo.count_by_status()

    # Uploaded counts every item submitted; classified counts what the model
    # has actually produced a prediction for. They differ while a batch is
    # still being worked through.
    documents_uploaded = sum(b.get("total_items", 0) for b in batches)
    documents_classified = len(predictions)

    auto_accepted = 0
    needs_review = 0
    high = medium = low = 0

    for p in predictions:
        conf = p.get("confidence") or 0.0
        if conf >= settings.CONFIDENCE_HIGH:
            auto_accepted += 1
        if conf < settings.REVIEW_THRESHOLD:
            needs_review += 1

        if conf >= _HIGH_BAND:
            high += 1
        elif conf >= settings.REVIEW_THRESHOLD:
            medium += 1
        else:
            low += 1

    label_counts: dict[str, int] = {}
    for p in predictions:
        label = p.get("predicted_label")
        if label:
            label_counts[label] = label_counts.get(label, 0) + 1

    total_labelled = sum(label_counts.values())
    label_distribution: List[LabelCount] = [
        LabelCount(
            label=label,
            count=count,
            percentage=round(count / total_labelled * 100, 1) if total_labelled else 0.0,
        )
        for label, count in sorted(label_counts.items(), key=lambda kv: -kv[1])[:_TOP_LABELS]
    ]

    # predictions_repo.list_all() already returns newest first.
    recent_activity: List[RecentActivityItem] = []
    for p in predictions[:recent_limit]:
        text = p.get("text_content") or ""
        recent_activity.append(
            RecentActivityItem(
                prediction_id=p["id"],
                text_excerpt=text[:_EXCERPT_LENGTH] + ("..." if len(text) > _EXCERPT_LENGTH else ""),
                predicted_label=p.get("predicted_label") or "",
                confidence=p.get("confidence") or 0.0,
                created_at=p.get("created_at") or "",
            )
        )

    return DashboardStatsResponse(
        documents_uploaded=documents_uploaded,
        documents_classified=documents_classified,
        auto_accepted=auto_accepted,
        needs_review=needs_review,
        conflicts=annotations_repo.count_conflicts(),
        confidence_high=high,
        confidence_medium=medium,
        confidence_low=low,
        annotations_total=sum(annotation_counts.values()),
        annotations_validated=annotation_counts.get("validated", 0),
        label_distribution=label_distribution,
        recent_activity=recent_activity,
    )
