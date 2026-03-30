"""
services/supabase_service.py — Supabase data access layer.

All functions delegate to mock_db for now. When the real Supabase schema is
ready, replace each `import … from services.mock_db` call with the real
supabase-py client call — function signatures stay identical so no router or
worker code needs to change.

Swap checklist (per function):
  1. Remove the mock_db import / call.
  2. Use `supabase_client.table("<table>").<operation>(<payload>).execute()`.
  3. Map the returned `.data` list/dict back to the same return type.
"""

import logging
from typing import Dict, List, Optional

import services.mock_db as db
from config.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Real Supabase client (initialised but not used while mocked)
# ---------------------------------------------------------------------------
# Uncomment the block below when swapping to real Supabase:
#
# from supabase import create_client, Client
# supabase_client: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)


# ===========================================================================
# USERS
# ===========================================================================

def create_user(email: str, role: str) -> Dict:
    """
    Insert a new user record.

    Args:
        email: User's email address.
        role:  One of 'annotator', 'reviewer', 'admin'.

    Returns:
        The newly created user dict {userId, email, role}.
    """
    logger.debug("supabase_service.create_user: email=%s role=%s", email, role)
    return db.create_user(email=email, role=role)


def get_user(user_id: str) -> Optional[Dict]:
    """
    Fetch a user by ID.

    Returns None if the user does not exist.
    """
    return db.get_user(user_id=user_id)


def get_user_by_email(email: str) -> Optional[Dict]:
    """
    Fetch a user by email address.

    Returns None if not found.
    """
    return db.get_user_by_email(email=email)


# ===========================================================================
# LEGAL DOCUMENTS
# ===========================================================================

def create_document(text_content: str, status: str = "pending") -> Dict:
    """
    Insert a new legal document row.

    Args:
        text_content: Raw legal text submitted by the user.
        status:       Initial status — typically 'pending'.

    Returns:
        The newly created document dict {documentId, textContent, status, createdAt}.
    """
    logger.debug("supabase_service.create_document: len=%d", len(text_content))
    return db.create_document(text_content=text_content, status=status)


def get_document(document_id: str) -> Optional[Dict]:
    """Fetch a legal document by ID. Returns None if not found."""
    return db.get_document(document_id=document_id)


def update_document_status(document_id: str, status: str) -> Optional[Dict]:
    """
    Update the status field of a legal document.

    Common status values: 'pending', 'predicted', 'annotated', 'conflict'.
    Returns the updated document dict, or None if document_id does not exist.
    """
    logger.debug(
        "supabase_service.update_document_status: id=%s status=%s",
        document_id, status,
    )
    return db.update_document_status(document_id=document_id, status=status)


# ===========================================================================
# PREDICTIONS
# ===========================================================================

def create_prediction(
    document_id: str,
    predicted_label: str,
    confidence_score: float,
    probability_distribution: Dict[str, float],
    routing_decision: str,
    entropy: float,
    margin: float,
    model_version_id: Optional[str] = None,
) -> Dict:
    """
    Insert a new prediction row.

    Args:
        document_id:              ID of the source legal document.
        predicted_label:          Top-1 label from InLegalBERT.
        confidence_score:         Max softmax probability.
        probability_distribution: Full {label: probability} dict.
        routing_decision:         AUTO_ACCEPT / NEEDS_EXPLANATION / ROUTE_TO_REVIEWER.
        entropy:                  Shannon entropy of the distribution.
        margin:                   Top-1 minus top-2 probability.
        model_version_id:         Optional FK to model_versions table.

    Returns:
        The newly created prediction dict.
    """
    logger.debug(
        "supabase_service.create_prediction: doc=%s label=%s conf=%.4f routing=%s",
        document_id, predicted_label, confidence_score, routing_decision,
    )
    return db.create_prediction(
        document_id=document_id,
        predicted_label=predicted_label,
        confidence_score=confidence_score,
        probability_distribution=probability_distribution,
        routing_decision=routing_decision,
        entropy=entropy,
        margin=margin,
        model_version_id=model_version_id,
    )


def get_prediction(prediction_id: str) -> Optional[Dict]:
    """Fetch a prediction by ID. Returns None if not found."""
    return db.get_prediction(prediction_id=prediction_id)


def list_predictions(document_id: Optional[str] = None) -> List[Dict]:
    """
    List predictions, optionally filtered by document_id.

    Returns all predictions if document_id is None.
    """
    return db.list_predictions(document_id=document_id)


# ===========================================================================
# ANNOTATIONS
# ===========================================================================

def create_annotation(
    document_id: str,
    prediction_id: str,
    user_id: Optional[str],
    final_label: str,
    action: str,
    annotation_status: str,
    notes: Optional[str] = None,
) -> Dict:
    """
    Insert a new annotation row.

    Args:
        document_id:       ID of the document being annotated.
        prediction_id:     ID of the prediction this annotation responds to.
        user_id:           ID of the annotating user (None when role header missing).
        final_label:       Label chosen by the human annotator.
        action:            'accept', 'modify', or 'reject'.
        annotation_status: 'pending', 'accepted', 'modified', 'rejected', or 'conflict'.
        notes:             Optional free-text notes from the annotator.

    Returns:
        The newly created annotation dict.
    """
    logger.debug(
        "supabase_service.create_annotation: doc=%s pred=%s action=%s label=%s",
        document_id, prediction_id, action, final_label,
    )
    return db.create_annotation(
        document_id=document_id,
        prediction_id=prediction_id,
        user_id=user_id,
        final_label=final_label,
        action=action,
        annotation_status=annotation_status,
        notes=notes,
    )


def get_annotation(annotation_id: str) -> Optional[Dict]:
    """Fetch an annotation by ID. Returns None if not found."""
    return db.get_annotation(annotation_id=annotation_id)


def list_annotations(
    document_id: Optional[str] = None,
    user_id: Optional[str] = None,
    status: Optional[str] = None,
) -> List[Dict]:
    """
    List annotations with optional filters (AND-combined).

    Args:
        document_id: Filter by document.
        user_id:     Filter by annotating user.
        status:      Filter by annotation status.

    Returns:
        Filtered list of annotation dicts, newest first.
    """
    rows = db.list_annotations(
        document_id=document_id,
        user_id=user_id,
        status=status,
    )
    # Sort newest first by annotatedAt
    return sorted(rows, key=lambda r: r.get("annotatedAt", ""), reverse=True)


def update_annotation_status(annotation_id: str, new_status: str) -> Optional[Dict]:
    """
    Update the status of an existing annotation.

    Used by conflict detection logic to flag disagreements.
    Returns the updated annotation dict, or None if not found.
    """
    logger.debug(
        "supabase_service.update_annotation_status: id=%s → %s",
        annotation_id, new_status,
    )
    return db.update_annotation_status(
        annotation_id=annotation_id,
        new_status=new_status,
    )


def count_validated_annotations() -> int:
    """
    Return the number of accepted or modified annotations.

    Used by the retraining pipeline to check if there are enough validated
    samples before starting a fine-tuning run.
    """
    return db.count_validated_annotations()


# ===========================================================================
# CONFLICT DETECTION
# ===========================================================================

def detect_and_flag_conflict(
    document_id: str,
    new_annotation_id: str,
    new_final_label: str,
    predicted_label: str,
) -> bool:
    """
    Check for model-vs-human or annotator-vs-annotator conflicts and flag them.

    Conflict rules:
      1. Model vs Human — new_final_label differs from predicted_label.
      2. Annotator vs Annotator — a previous annotation for the same document
         has a different final_label.

    When a conflict is detected:
      - The new annotation is updated to status='conflict'.
      - Any prior annotation for the same document with a different label is
        also updated to status='conflict'.

    Returns:
        True if a conflict was detected and flagged, False otherwise.
    """
    conflict = False

    # Rule 1: model vs human
    if new_final_label != predicted_label:
        logger.info(
            "Conflict detected (model vs human): doc=%s predicted='%s' human='%s'",
            document_id, predicted_label, new_final_label,
        )
        db.update_annotation_status(new_annotation_id, "conflict")
        conflict = True

    # Rule 2: annotator vs annotator
    existing = db.list_annotations(document_id=document_id)
    for ann in existing:
        if (
            ann["annotationId"] != new_annotation_id
            and ann["finalLabel"] != new_final_label
            and ann["annotationStatus"] not in ("rejected",)
        ):
            logger.info(
                "Conflict detected (annotator vs annotator): doc=%s ann1='%s' ann2='%s'",
                document_id, ann["finalLabel"], new_final_label,
            )
            db.update_annotation_status(ann["annotationId"], "conflict")
            db.update_annotation_status(new_annotation_id, "conflict")
            conflict = True

    return conflict


# ===========================================================================
# EXPLANATIONS
# ===========================================================================

def create_explanation(prediction_id: str) -> Dict:
    """
    Insert a new explanation row with status='pending'.

    The XAI worker will call update_explanation() once SHAP completes.

    Returns:
        The newly created explanation dict {explanationId, predictionId, status, ...}.
    """
    logger.debug("supabase_service.create_explanation: pred=%s", prediction_id)
    return db.create_explanation(prediction_id=prediction_id)


def get_explanation(explanation_id: str) -> Optional[Dict]:
    """Fetch an explanation by its own ID. Returns None if not found."""
    return db.get_explanation(explanation_id=explanation_id)


def get_explanation_by_prediction(prediction_id: str) -> Optional[Dict]:
    """
    Fetch the SHAP explanation for a given prediction.

    Returns None if no explanation has been created yet for this prediction.
    """
    return db.get_explanation_by_prediction(prediction_id=prediction_id)


def update_explanation(
    explanation_id: str,
    shap_values: List[Dict],
    status: str = "completed",
) -> Optional[Dict]:
    """
    Store computed SHAP token importances and mark the explanation as completed.

    Args:
        explanation_id: ID of the explanation row to update.
        shap_values:    List of {token, importance} dicts from shap_service.
        status:         'completed' on success, 'failed' on error.

    Returns:
        The updated explanation dict, or None if explanation_id not found.
    """
    logger.debug(
        "supabase_service.update_explanation: id=%s status=%s tokens=%d",
        explanation_id, status, len(shap_values),
    )
    return db.update_explanation(
        explanation_id=explanation_id,
        shap_values=shap_values,
        status=status,
    )


# ===========================================================================
# PREDICTION JOBS QUEUE
# ===========================================================================

def enqueue_prediction_job(text_content: str) -> Dict:
    """
    Push a new prediction job onto the prediction_jobs queue.

    Returns:
        The new job dict {jobId, textContent, status='pending', createdAt}.
    """
    logger.debug("supabase_service.enqueue_prediction_job: len=%d", len(text_content))
    return db.enqueue_prediction_job(text_content=text_content)


def get_prediction_job(job_id: str) -> Optional[Dict]:
    """Fetch a prediction job by ID. Returns None if not found."""
    return db.get_prediction_job(job_id=job_id)


def update_prediction_job(
    job_id: str,
    status: str,
    result: Optional[Dict] = None,
) -> Optional[Dict]:
    """Update status and optional result of a prediction job."""
    return db.update_prediction_job(job_id=job_id, status=status, result=result)


def get_pending_prediction_jobs() -> List[Dict]:
    """Return all prediction jobs with status='pending'."""
    return db.get_pending_prediction_jobs()


# ===========================================================================
# XAI JOBS QUEUE
# ===========================================================================

def enqueue_xai_job(prediction_id: str) -> Dict:
    """
    Push a new SHAP job onto the xai_jobs queue.

    Returns:
        The new job dict {jobId, predictionId, status='pending', createdAt}.
    """
    logger.debug("supabase_service.enqueue_xai_job: pred=%s", prediction_id)
    return db.enqueue_xai_job(prediction_id=prediction_id)


def get_xai_job(job_id: str) -> Optional[Dict]:
    """Fetch an XAI job by ID. Returns None if not found."""
    return db.get_xai_job(job_id=job_id)


def update_xai_job(
    job_id: str,
    status: str,
    result: Optional[Dict] = None,
) -> Optional[Dict]:
    """Update status and optional result of an XAI job."""
    return db.update_xai_job(job_id=job_id, status=status, result=result)


def get_pending_xai_jobs() -> List[Dict]:
    """Return all XAI jobs with status='pending'."""
    return db.get_pending_xai_jobs()


# ===========================================================================
# RETRAIN JOBS
# ===========================================================================

def enqueue_retrain_job(notes: Optional[str], min_annotations: int) -> Dict:
    """
    Push a new retraining job onto the queue.

    Returns:
        The new job dict with status='pending'.
    """
    logger.debug(
        "supabase_service.enqueue_retrain_job: min_annotations=%d", min_annotations
    )
    return db.enqueue_retrain_job(notes=notes, min_annotations=min_annotations)


def get_retrain_job(job_id: str) -> Optional[Dict]:
    """Fetch a retrain job by ID. Returns None if not found."""
    return db.get_retrain_job(job_id=job_id)


def get_latest_retrain_job() -> Optional[Dict]:
    """Return the most recently created retrain job, or None."""
    return db.get_latest_retrain_job()


def update_retrain_job(
    job_id: str,
    status: str,
    model_version: Optional[str] = None,
    accuracy: Optional[float] = None,
    completed_at: Optional[str] = None,
) -> Optional[Dict]:
    """Update status and result fields of a retrain job."""
    return db.update_retrain_job(
        job_id=job_id,
        status=status,
        model_version=model_version,
        accuracy=accuracy,
        completed_at=completed_at,
    )


# ===========================================================================
# DATASET & MODEL VERSIONS
# ===========================================================================

def create_dataset_version(sample_count: int) -> Dict:
    """Record a new dataset snapshot created for a retraining run."""
    return db.create_dataset_version(sample_count=sample_count)


def create_model_version(accuracy: float, file_path: str) -> Dict:
    """Record a newly trained model version and its saved file path."""
    return db.create_model_version(accuracy=accuracy, file_path=file_path)


def get_latest_model_version() -> Optional[Dict]:
    """Return the most recently trained model version row, or None."""
    return db.get_latest_model_version()
