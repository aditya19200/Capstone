"""
services/supabase_service.py — Supabase data access layer.

All functions delegate to mock_db for now. When the real Supabase schema is
ready, replace each mock_db call with the real supabase-py client call —
function signatures stay identical so no router or worker code needs to change.

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
# supabase_client: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)


# ===========================================================================
# USERS  (legacy — not in real schema)
# ===========================================================================

def create_user(email: str, role: str) -> Dict:
    logger.debug("supabase_service.create_user: email=%s role=%s", email, role)
    return db.create_user(email=email, role=role)


def get_user(user_id: str) -> Optional[Dict]:
    return db.get_user(user_id=user_id)


def get_user_by_email(email: str) -> Optional[Dict]:
    return db.get_user_by_email(email=email)


# ===========================================================================
# LEGAL DOCUMENTS  (legacy — not in real schema)
# ===========================================================================

def create_document(text_content: str, status: str = "pending") -> Dict:
    logger.debug("supabase_service.create_document: len=%d", len(text_content))
    return db.create_document(text_content=text_content, status=status)


def get_document(document_id: str) -> Optional[Dict]:
    return db.get_document(document_id=document_id)


def update_document_status(document_id: str, status: str) -> Optional[Dict]:
    logger.debug(
        "supabase_service.update_document_status: id=%s status=%s",
        document_id, status,
    )
    return db.update_document_status(document_id=document_id, status=status)


# ===========================================================================
# PREDICTIONS
# ===========================================================================

def create_prediction(
    text_content: str,
    predicted_label: str,
    label_id: int,
    confidence: float,
    all_probabilities: Dict[str, float],
    model_version: str,
) -> Dict:
    """
    Insert a new prediction row.

    Returns a dict with keys: id, text_content, predicted_label, label_id,
    confidence, all_probabilities, model_version, created_at.
    """
    logger.debug(
        "supabase_service.create_prediction: label=%s conf=%.4f",
        predicted_label, confidence,
    )
    return db.create_prediction(
        text_content=text_content,
        predicted_label=predicted_label,
        label_id=label_id,
        confidence=confidence,
        all_probabilities=all_probabilities,
        model_version=model_version,
    )


def get_prediction(prediction_id: str) -> Optional[Dict]:
    """Fetch a prediction by id. Returns None if not found."""
    return db.get_prediction(prediction_id=prediction_id)


def list_predictions(predicted_label: Optional[str] = None) -> List[Dict]:
    """List predictions, optionally filtered by predicted_label."""
    return db.list_predictions(predicted_label=predicted_label)


def list_low_confidence_predictions(threshold: float) -> List[Dict]:
    """Return predictions with confidence < threshold, newest first."""
    return db.list_low_confidence_predictions(threshold=threshold)


# ===========================================================================
# ANNOTATIONS
# ===========================================================================

def create_annotation(
    prediction_id: str,
    validated_label: str,
    annotator_id: Optional[str],
    status: str,
    document_id: Optional[str] = None,
) -> Dict:
    """
    Insert a new annotation row.

    document_id is not in the real schema.  It is forwarded to mock_db so
    GET /annotate can return it in AnnotationListItem until that route is
    rewritten to join through predictions (Step 5 target).

    Returns a dict with keys: id, prediction_id, validated_label,
    annotator_id, status, has_conflict, annotated_at, document_id.
    """
    logger.debug(
        "supabase_service.create_annotation: pred=%s label=%s status=%s",
        prediction_id, validated_label, status,
    )
    return db.create_annotation(
        prediction_id=prediction_id,
        validated_label=validated_label,
        annotator_id=annotator_id,
        status=status,
        document_id=document_id,
    )


def get_annotation(annotation_id: str) -> Optional[Dict]:
    """Fetch an annotation by id. Returns None if not found."""
    return db.get_annotation(annotation_id=annotation_id)


def list_annotations(
    prediction_id: Optional[str] = None,
    annotator_id: Optional[str] = None,
    status: Optional[str] = None,
    has_conflict: Optional[bool] = None,
) -> List[Dict]:
    """
    List annotations with optional AND-combined filters.

    Sorted newest first by annotated_at.
    """
    rows = db.list_annotations(
        prediction_id=prediction_id,
        annotator_id=annotator_id,
        status=status,
        has_conflict=has_conflict,
    )
    return sorted(rows, key=lambda r: r.get("annotated_at", ""), reverse=True)


def update_annotation_status(annotation_id: str, new_status: str) -> Optional[Dict]:
    """Update the status of an existing annotation."""
    logger.debug(
        "supabase_service.update_annotation_status: id=%s → %s",
        annotation_id, new_status,
    )
    return db.update_annotation_status(annotation_id=annotation_id, new_status=new_status)


def set_annotation_has_conflict(annotation_id: str, has_conflict: bool) -> Optional[Dict]:
    """
    Set the has_conflict flag on an annotation (independent of status).

    Real Supabase path: UPDATE annotations SET has_conflict=$1 WHERE id=$2
    """
    logger.debug(
        "supabase_service.set_annotation_has_conflict: id=%s has_conflict=%s",
        annotation_id, has_conflict,
    )
    return db.set_annotation_has_conflict(annotation_id=annotation_id, has_conflict=has_conflict)


def count_validated_annotations() -> int:
    """Return the count of annotations with status='validated'."""
    return db.count_validated_annotations()


# ===========================================================================
# CONFLICT DETECTION
# ===========================================================================

def detect_and_flag_conflict(
    prediction_id: str,
    new_annotation_id: str,
    new_final_label: str,
    predicted_label: str,
) -> bool:
    """
    Check for model-vs-human or annotator-vs-annotator conflicts and flag them.

    Conflict rules:
      1. Model vs human — new_final_label differs from predicted_label.
      2. Annotator vs annotator — a previous annotation for the same
         prediction_id has a different validated_label.

    When a conflict is detected, all involved annotations are flagged with
    has_conflict=True.  Status is NOT changed — 'validated' stays 'validated'
    even when has_conflict=True.  This keeps the status enum clean and lets
    the reviewer queue query on has_conflict independently.

    Returns True if any conflict was detected.
    """
    conflict = False

    # Rule 1: model vs human
    if new_final_label != predicted_label:
        logger.info(
            "Conflict (model vs human): pred=%s predicted='%s' human='%s'",
            prediction_id, predicted_label, new_final_label,
        )
        db.set_annotation_has_conflict(new_annotation_id, True)
        conflict = True

    # Rule 2: annotator vs annotator
    existing = db.list_annotations(prediction_id=prediction_id)
    for ann in existing:
        if (
            ann["id"] != new_annotation_id
            and ann["validated_label"] != new_final_label
            and ann["status"] != "rejected"
        ):
            logger.info(
                "Conflict (annotator vs annotator): pred=%s label1='%s' label2='%s'",
                prediction_id, ann["validated_label"], new_final_label,
            )
            db.set_annotation_has_conflict(ann["id"], True)
            db.set_annotation_has_conflict(new_annotation_id, True)
            conflict = True

    return conflict


# ===========================================================================
# EXPLANATIONS  (legacy — token_importance lives in xai_jobs in real schema)
# ===========================================================================

def create_explanation(prediction_id: str) -> Dict:
    logger.debug("supabase_service.create_explanation: pred=%s", prediction_id)
    return db.create_explanation(prediction_id=prediction_id)


def get_explanation(explanation_id: str) -> Optional[Dict]:
    return db.get_explanation(explanation_id=explanation_id)


def get_explanation_by_prediction(prediction_id: str) -> Optional[Dict]:
    return db.get_explanation_by_prediction(prediction_id=prediction_id)


def update_explanation(
    explanation_id: str,
    shap_values: List[Dict],
    status: str = "completed",
) -> Optional[Dict]:
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
# PREDICTION JOBS QUEUE  (legacy — batch worker uses batch_items via RPC)
# ===========================================================================

def enqueue_prediction_job(text_content: str) -> Dict:
    logger.debug("supabase_service.enqueue_prediction_job: len=%d", len(text_content))
    return db.enqueue_prediction_job(text_content=text_content)


def get_prediction_job(job_id: str) -> Optional[Dict]:
    return db.get_prediction_job(job_id=job_id)


def update_prediction_job(
    job_id: str,
    status: str,
    result: Optional[Dict] = None,
) -> Optional[Dict]:
    return db.update_prediction_job(job_id=job_id, status=status, result=result)


def get_pending_prediction_jobs() -> List[Dict]:
    return db.get_pending_prediction_jobs()


# ===========================================================================
# XAI JOBS QUEUE
# ===========================================================================

def enqueue_xai_job(prediction_id: str) -> Dict:
    """
    Push a new SHAP job onto the xai_jobs queue.

    Returns a dict with keys: id, prediction_id, status, token_importance,
    summary, attempts, locked_at, error_message, created_at.
    """
    logger.debug("supabase_service.enqueue_xai_job: pred=%s", prediction_id)
    return db.enqueue_xai_job(prediction_id=prediction_id)


def get_xai_job(job_id: str) -> Optional[Dict]:
    """Fetch an XAI job by id. Returns None if not found."""
    return db.get_xai_job(job_id=job_id)


def update_xai_job(job_id: str, **fields) -> Optional[Dict]:
    """
    Partial update on an xai_jobs row.

    Accepts any real schema field: status, token_importance, summary,
    attempts, locked_at, error_message.
    """
    return db.update_xai_job(job_id=job_id, **fields)


def get_pending_xai_jobs() -> List[Dict]:
    """Return all XAI jobs with status='pending'."""
    return db.get_pending_xai_jobs()


# ===========================================================================
# RETRAIN JOBS
# ===========================================================================

def enqueue_retrain_job(
    triggered_by: Optional[str] = None,
    notes: Optional[str] = None,
    min_annotations: Optional[int] = None,
) -> Dict:
    """
    Push a new retraining job with status='pending'.

    Returns a dict with keys: id, status, triggered_by, triggered_at,
    completed_at, model_version_id, notes, min_annotations.
    """
    logger.debug(
        "supabase_service.enqueue_retrain_job: triggered_by=%s", triggered_by
    )
    return db.enqueue_retrain_job(
        triggered_by=triggered_by,
        notes=notes,
        min_annotations=min_annotations,
    )


def get_retrain_job(job_id: str) -> Optional[Dict]:
    """Fetch a retrain job by id. Returns None if not found."""
    return db.get_retrain_job(job_id=job_id)


def get_latest_retrain_job() -> Optional[Dict]:
    """Return the most recently created retrain job, or None."""
    return db.get_latest_retrain_job()


def update_retrain_job(
    job_id: str,
    status: str,
    model_version_id: Optional[str] = None,
    completed_at: Optional[str] = None,
) -> Optional[Dict]:
    """Update status and completion fields of a retrain job."""
    return db.update_retrain_job(
        job_id=job_id,
        status=status,
        model_version_id=model_version_id,
        completed_at=completed_at,
    )


# ===========================================================================
# DATASET & MODEL VERSIONS
# ===========================================================================

def create_dataset_version(
    version_id: str,
    sample_count: int,
    label_distribution: Dict[str, int],
) -> Dict:
    """Record a new dataset snapshot created for a retraining run."""
    return db.create_dataset_version(
        version_id=version_id,
        sample_count=sample_count,
        label_distribution=label_distribution,
    )


def create_model_version(
    version_number: str,
    accuracy: float,
    f1_per_class: Dict[str, float],
    file_path: str,
    dataset_version_id: Optional[str] = None,
) -> Dict:
    """Record a newly trained model version (is_active defaults to False)."""
    return db.create_model_version(
        version_number=version_number,
        accuracy=accuracy,
        f1_per_class=f1_per_class,
        file_path=file_path,
        dataset_version_id=dataset_version_id,
        is_active=False,
    )


def get_active_model_version() -> Optional[Dict]:
    """Return the model version row where is_active=True, or None."""
    return db.get_active_model_version()
