"""
routers/annotate.py — Annotation endpoints.

POST /annotate
  Submit a human annotation decision (accept / modify / reject) for a
  prediction. Runs conflict detection automatically.

GET  /annotate
  List annotations, optionally filtered by status or conflict flag.
  Accessible to Annotators (own annotations), Reviewers (all), and Admins (all).
"""

import logging
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Query, status

from models.request_models import AnnotateRequest
from models.response_models import (
    AnnotateResponse,
    AnnotationListItem,
    AnnotationListResponse,
)
from services.neo4j_service import label_exists, get_all_labels
from services.supabase_service import (
    create_annotation,
    detect_and_flag_conflict,
    get_prediction,
    list_annotations,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# Role constants (mocked via X-Role header)
# ---------------------------------------------------------------------------

_ROLES_ALL = {"reviewer", "admin"}          # can see all annotations
_ROLES_ANNOTATE = {"annotator", "reviewer", "admin"}   # known roles for POST/GET /annotate


# ---------------------------------------------------------------------------
# POST /annotate
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=AnnotateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit a human annotation",
    description=(
        "Accept, modify, or reject a model prediction. "
        "Conflict detection runs automatically: if the submitted label differs "
        "from the model prediction, or from a prior annotation on the same "
        "prediction, the annotation's has_conflict flag is set for Reviewer "
        "attention."
    ),
)
async def submit_annotation(
    body: AnnotateRequest,
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
    x_role: Optional[str] = Header(default=None, alias="X-Role"),
) -> AnnotateResponse:
    """
    Full annotation pipeline:

      1. Role check.
      2. Validate prediction exists.
      3. Validate final_label against the ontology.
      4. Determine status from action.
      5. Persist annotation.
      6. Run conflict detection — sets has_conflict=True when conflicts found.
      7. Return AnnotateResponse.
    """
    # --- 1. Role check ---
    role = (x_role or "").lower()
    if role not in _ROLES_ANNOTATE:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"Role '{x_role}' is not permitted to submit annotations. "
                "Required: annotator, reviewer, or admin."
            ),
        )

    # --- 2. Validate prediction exists ---
    prediction = get_prediction(prediction_id=body.prediction_id)
    if prediction is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction '{body.prediction_id}' not found.",
        )

    # --- 3. Validate final_label against ontology ---
    if not label_exists(body.final_label):
        valid_labels = get_all_labels()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Label '{body.final_label}' is not a valid ontology label. "
                f"Valid labels: {valid_labels}"
            ),
        )

    # --- 4. Determine status from action ---
    # Both "accept" and "modify" map to "validated" (real schema has no
    # "accepted" or "modified" states).  "reject" maps to "rejected".
    action_to_status = {
        "accept": "validated",
        "modify": "validated",
        "reject": "rejected",
    }
    annotation_status = action_to_status[body.action]

    # --- 5. Persist annotation ---
    annotation = create_annotation(
        prediction_id=body.prediction_id,
        validated_label=body.final_label,
        annotator_id=x_user_id,
        status=annotation_status,
    )
    annotation_id: str = annotation["id"]

    logger.info(
        "Annotation created: id=%s pred=%s action=%s label='%s' user=%s",
        annotation_id, body.prediction_id, body.action, body.final_label, x_user_id,
    )

    # --- 6. Conflict detection ---
    conflict_detected = False
    if body.action != "reject":
        conflict_detected = detect_and_flag_conflict(
            prediction_id=body.prediction_id,
            new_annotation_id=annotation_id,
            new_final_label=body.final_label,
            predicted_label=prediction["predicted_label"],
        )
        if conflict_detected:
            logger.warning(
                "Conflict flagged: pred=%s annotation=%s label='%s' vs predicted='%s'",
                body.prediction_id, annotation_id,
                body.final_label, prediction["predicted_label"],
            )

    # --- 7. Return response ---
    message = _build_message(body.action, conflict_detected, body.final_label)

    return AnnotateResponse(
        annotation_id=annotation_id,
        prediction_id=body.prediction_id,
        final_label=body.final_label,
        action=body.action,
        annotation_status=annotation_status,
        conflict_detected=conflict_detected,
        message=message,
    )


# ---------------------------------------------------------------------------
# GET /annotate
# ---------------------------------------------------------------------------

@router.get(
    "",
    response_model=AnnotationListResponse,
    status_code=status.HTTP_200_OK,
    summary="List annotations",
    description=(
        "Return a paginated list of annotations. "
        "Annotators see only their own annotations (filtered by X-User-Id). "
        "Reviewers and Admins can see all annotations and apply additional filters."
    ),
)
async def list_annotations_endpoint(
    annotation_status: Optional[str] = Query(
        default=None,
        alias="status",
        description="Filter by status: pending, validated, rejected.",
    ),
    has_conflict: Optional[bool] = Query(
        default=None,
        description="Filter by conflict flag, e.g. ?has_conflict=true for the reviewer queue.",
    ),
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
    x_role: Optional[str] = Header(default=None, alias="X-Role"),
) -> AnnotationListResponse:
    """
    List annotations with optional filters.

    Role-based scoping:
      annotator → filtered to own annotator_id automatically (X-User-Id required)
      reviewer / admin → can query all annotations
    """
    role = (x_role or "").lower()

    if role not in _ROLES_ANNOTATE:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"Role '{x_role}' is not permitted to list annotations. "
                "Required: annotator, reviewer, or admin."
            ),
        )

    if role in _ROLES_ALL:
        filter_annotator_id = None
    else:
        if not x_user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="X-User-Id header is required for role 'annotator'.",
            )
        filter_annotator_id = x_user_id

    rows = list_annotations(
        annotator_id=filter_annotator_id,
        status=annotation_status,
        has_conflict=has_conflict,
    )

    # N+1 by design: one get_prediction() lookup per annotation to fetch
    # predicted_label/text_excerpt. This router is mock-backed only (see
    # CLAUDE.md audit note); not worth batching until it's on a real client.
    items = []
    for r in rows:
        prediction = get_prediction(prediction_id=r["prediction_id"])
        predicted_label = prediction["predicted_label"] if prediction else ""
        text_content = prediction["text_content"] if prediction else ""
        excerpt = text_content[:200] + ("..." if len(text_content) > 200 else "")

        items.append(
            AnnotationListItem(
                annotation_id=r["id"],
                user_id=r.get("annotator_id"),
                final_label=r["validated_label"],
                annotation_status=r["status"],
                has_conflict=r.get("has_conflict", False),
                predicted_label=predicted_label,
                text_excerpt=excerpt,
                annotated_at=r["annotated_at"],
            )
        )

    return AnnotationListResponse(total=len(items), annotations=items)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_message(action: str, conflict: bool, label: str) -> str:
    """Build a human-readable response message based on action and conflict state."""
    if action == "reject":
        return "Document rejected and marked out of scope."
    if conflict:
        return (
            f"Annotation recorded with label '{label}', but a conflict was detected. "
            "The annotation has been flagged for Reviewer resolution."
        )
    verb = "accepted" if action == "accept" else "modified"
    return f"Annotation {verb} successfully with label '{label}'."
