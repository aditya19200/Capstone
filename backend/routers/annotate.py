"""
routers/annotate.py — Annotation endpoints.

POST /annotate
  Submit a human annotation decision (accept / modify / reject) for a
  prediction. Runs conflict detection automatically.

GET  /annotate
  List annotations, optionally filtered by document, user, or status.
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
    update_document_status,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# Role constants (mocked via X-Role header)
# ---------------------------------------------------------------------------

_ROLES_ALL = {"reviewer", "admin"}          # can see all annotations
_ROLES_ANNOTATE = {"annotator", "reviewer", "admin"}


# ---------------------------------------------------------------------------
# POST /annotate
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=AnnotateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit a human annotation",
    description=(
        "Accept, modify, or reject a model prediction for a legal document. "
        "Conflict detection runs automatically: if the submitted label differs "
        "from the model prediction, or from a prior annotation on the same "
        "document, the annotation is flagged as a conflict for Reviewer attention."
    ),
)
async def submit_annotation(
    body: AnnotateRequest,
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
    x_role: Optional[str] = Header(default=None, alias="X-Role"),
) -> AnnotateResponse:
    """
    Full annotation pipeline:

      1. Role check — only annotators, reviewers, and admins may annotate.
      2. Validate prediction exists.
      3. Validate final_label against the ontology.
      4. Determine initial annotation_status from action.
      5. Persist annotation.
      6. Run conflict detection — flag conflicts if found.
      7. Update document status.
      8. Return AnnotateResponse.
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

    # Ensure the prediction belongs to the supplied document
    if prediction["documentId"] != body.document_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Prediction '{body.prediction_id}' does not belong to "
                f"document '{body.document_id}'."
            ),
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

    # --- 4. Determine initial annotation_status from action ---
    action_to_status = {
        "accept":  "accepted",
        "modify":  "modified",
        "reject":  "rejected",
    }
    annotation_status = action_to_status[body.action]

    # --- 5. Persist annotation ---
    annotation = create_annotation(
        document_id=body.document_id,
        prediction_id=body.prediction_id,
        user_id=x_user_id,
        final_label=body.final_label,
        action=body.action,
        annotation_status=annotation_status,
        notes=body.notes,
    )
    annotation_id: str = annotation["annotationId"]

    logger.info(
        "Annotation created: id=%s doc=%s action=%s label='%s' user=%s",
        annotation_id, body.document_id, body.action, body.final_label, x_user_id,
    )

    # --- 6. Conflict detection ---
    conflict_detected = False
    if body.action != "reject":
        # Conflict detection only meaningful for accept/modify decisions
        conflict_detected = detect_and_flag_conflict(
            document_id=body.document_id,
            new_annotation_id=annotation_id,
            new_final_label=body.final_label,
            predicted_label=prediction["predictedLabel"],
        )
        if conflict_detected:
            logger.warning(
                "Conflict flagged: doc=%s annotation=%s label='%s' vs predicted='%s'",
                body.document_id, annotation_id,
                body.final_label, prediction["predictedLabel"],
            )
            annotation_status = "conflict"

    # --- 7. Update document status ---
    new_doc_status = "conflict" if conflict_detected else "annotated"
    update_document_status(body.document_id, new_doc_status)

    # --- 8. Return response ---
    message = _build_message(body.action, conflict_detected, body.final_label)

    return AnnotateResponse(
        annotation_id=annotation_id,
        document_id=body.document_id,
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
    document_id: Optional[str] = Query(default=None, description="Filter by document ID."),
    annotation_status: Optional[str] = Query(
        default=None,
        alias="status",
        description="Filter by status: pending, accepted, modified, rejected, conflict.",
    ),
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
    x_role: Optional[str] = Header(default=None, alias="X-Role"),
) -> AnnotationListResponse:
    """
    List annotations with optional filters.

    Role-based scoping:
      annotator → filtered to own user_id automatically
      reviewer / admin → can query all annotations
    """
    role = (x_role or "").lower()

    # Scope user_id filter based on role
    if role in _ROLES_ALL:
        # Reviewers and admins can see everything; user_id filter is optional
        filter_user_id = None
    else:
        # Annotators (and unauthenticated) see only their own annotations
        filter_user_id = x_user_id

    rows = list_annotations(
        document_id=document_id,
        user_id=filter_user_id,
        status=annotation_status,
    )

    items = [
        AnnotationListItem(
            annotation_id=r["annotationId"],
            document_id=r["documentId"],
            user_id=r.get("userId"),
            final_label=r["finalLabel"],
            annotation_status=r["annotationStatus"],
            annotated_at=r["annotatedAt"],
        )
        for r in rows
    ]

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
