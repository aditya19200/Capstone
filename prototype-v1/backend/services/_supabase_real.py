"""
services/_supabase_real.py — the real-database half of supabase_service.

supabase_service.py's functions used to answer only from the in-memory mock.
That was harmless while nothing was connected, but once SUPABASE_URL was set
the app split in two: the repositories wrote predictions, batches and XAI
jobs to Postgres, while supabase_service kept reading and writing mock_db. A
prediction classified into Postgres was then "not found" by POST /annotate
and POST /explain, annotations never reached the database, and XAI jobs
queued in Postgres were never picked up by the worker.

Each supabase_service function now starts with

    if is_configured():
        return _real.<same name>(...)

and its mock body is untouched below that line, so the in-memory demo runs
exactly as before. The functions here take the same arguments and return the
same shapes as their mock counterparts, built on the repositories rather
than on new queries wherever a repository already exists.

Not here, because the real schema has no table for them and nothing reads
them from the database: users, legal documents, and the legacy prediction
job queue (POST /predict classifies synchronously). Those stay on the mock
in both modes.

Explanations
------------
The mock keeps explanations in their own store. The real schema keeps a
SHAP result on its xai_jobs row (token_importance), so here an explanation
is a view of the latest xai_jobs row for a prediction:

    explanationId   job id
    predictionId    job prediction_id
    shapValues      job token_importance
    status          'completed' when the job is done, 'failed' when failed,
                    otherwise 'pending' — the vocabulary the mock uses
    generatedAt     None (the schema records no completion time)
"""

import logging
from typing import Dict, List, Optional

from repositories import annotations as annotations_repo
from repositories import model_versions as model_versions_repo
from repositories import predictions as predictions_repo
from repositories import xai_jobs as xai_jobs_repo
from repositories._client import current_domain, get_client

logger = logging.getLogger(__name__)


# ===========================================================================
# PREDICTIONS
# ===========================================================================

def create_prediction(text_content, predicted_label, label_id, confidence,
                      all_probabilities, model_version) -> Dict:
    return predictions_repo.insert(
        text_content=text_content,
        predicted_label=predicted_label,
        label_id=label_id,
        confidence=confidence,
        all_probabilities=all_probabilities,
        model_version=model_version,
    )


def get_prediction(prediction_id: str) -> Optional[Dict]:
    return predictions_repo.get(prediction_id)


def list_predictions(predicted_label: Optional[str] = None) -> List[Dict]:
    rows = predictions_repo.list_all()
    if predicted_label:
        rows = [r for r in rows if r["predicted_label"] == predicted_label]
    return rows


def list_low_confidence_predictions(threshold: float) -> List[Dict]:
    return predictions_repo.list_low_confidence(threshold)


# ===========================================================================
# ANNOTATIONS
# ===========================================================================

def _annotation(row: Optional[Dict]) -> Optional[Dict]:
    """
    Give a real row the mock's keys. document_id is not a column (see the
    note on create_annotation in supabase_service); callers read it, so it
    is present and None rather than missing.
    """
    if row is None:
        return None
    row = dict(row)
    row.setdefault("document_id", None)
    return row


def create_annotation(prediction_id, validated_label, annotator_id, status,
                      document_id=None) -> Dict:
    return _annotation(annotations_repo.insert(
        prediction_id=prediction_id,
        validated_label=validated_label,
        annotator_id=annotator_id,
        status=status,
    ))


def get_annotation(annotation_id: str) -> Optional[Dict]:
    return _annotation(annotations_repo.get(annotation_id))


def list_annotations(prediction_id=None, annotator_id=None, status=None,
                     has_conflict=None) -> List[Dict]:
    # Same AND-combined, truthiness-based filters as mock_db.list_annotations.
    query = (
        get_client()
        .table("annotations")
        .select("*")
        .eq("domain", current_domain())
    )
    if prediction_id:
        query = query.eq("prediction_id", prediction_id)
    if annotator_id:
        query = query.eq("annotator_id", annotator_id)
    if status:
        query = query.eq("status", status)
    if has_conflict is not None:
        query = query.eq("has_conflict", has_conflict)
    rows = query.order("annotated_at", desc=True).execute().data
    return [_annotation(r) for r in rows]


def _update_annotation(annotation_id: str, fields: Dict) -> Optional[Dict]:
    result = (
        get_client()
        .table("annotations")
        .update(fields)
        .eq("id", annotation_id)
        .execute()
    )
    return _annotation(result.data[0]) if result.data else None


def update_annotation_status(annotation_id: str, new_status: str) -> Optional[Dict]:
    return _update_annotation(annotation_id, {"status": new_status})


def set_annotation_has_conflict(annotation_id: str, has_conflict: bool) -> Optional[Dict]:
    return _update_annotation(annotation_id, {"has_conflict": has_conflict})


def count_validated_annotations() -> int:
    return annotations_repo.count_validated()


def detect_and_flag_conflict(prediction_id, new_annotation_id, new_final_label,
                             predicted_label) -> bool:
    """Same two rules as the mock version, against the real table."""
    conflict = False

    if new_final_label != predicted_label:
        logger.info(
            "Conflict (model vs human): pred=%s predicted='%s' human='%s'",
            prediction_id, predicted_label, new_final_label,
        )
        set_annotation_has_conflict(new_annotation_id, True)
        conflict = True

    for ann in list_annotations(prediction_id=prediction_id):
        if (
            ann["id"] != new_annotation_id
            and ann["validated_label"] != new_final_label
            and ann["status"] != "rejected"
        ):
            logger.info(
                "Conflict (annotator vs annotator): pred=%s label1='%s' label2='%s'",
                prediction_id, ann["validated_label"], new_final_label,
            )
            set_annotation_has_conflict(ann["id"], True)
            set_annotation_has_conflict(new_annotation_id, True)
            conflict = True

    return conflict


# ===========================================================================
# EXPLANATIONS  (a view over xai_jobs — see module docstring)
# ===========================================================================

_EXPLANATION_STATUS = {"done": "completed", "failed": "failed"}


def _explanation(job: Optional[Dict]) -> Optional[Dict]:
    if job is None:
        return None
    return {
        "explanationId": job["id"],
        "predictionId": job["prediction_id"],
        "shapValues": job.get("token_importance"),
        "generatedAt": None,
        "status": _EXPLANATION_STATUS.get(job.get("status"), "pending"),
    }


def create_explanation(prediction_id: str) -> Dict:
    """
    Every caller enqueues the xai_jobs row itself straight after this call,
    so nothing is written here — writing would queue the job twice. Returns
    the existing explanation if there is one, else a pending placeholder.
    """
    existing = get_explanation_by_prediction(prediction_id)
    if existing is not None:
        return existing
    return {
        "explanationId": None,
        "predictionId": prediction_id,
        "shapValues": None,
        "generatedAt": None,
        "status": "pending",
    }


def get_explanation(explanation_id: str) -> Optional[Dict]:
    return _explanation(xai_jobs_repo.get(explanation_id))


def get_explanation_by_prediction(prediction_id: str) -> Optional[Dict]:
    return _explanation(xai_jobs_repo.get_by_prediction(prediction_id))


def update_explanation(explanation_id, shap_values, status="completed") -> Optional[Dict]:
    """
    Store the SHAP result on the job. The job's own status is set by the
    update_xai_job call the worker makes right after this one, so a failed
    explanation writes nothing here.
    """
    if status == "completed":
        return _explanation(xai_jobs_repo.update(explanation_id, token_importance=shap_values))
    return get_explanation(explanation_id)


# ===========================================================================
# XAI JOBS
# ===========================================================================

def enqueue_xai_job(prediction_id: str) -> Dict:
    return xai_jobs_repo.insert(prediction_id=prediction_id)


def get_xai_job(job_id: str) -> Optional[Dict]:
    return xai_jobs_repo.get(job_id)


def update_xai_job(job_id: str, **fields) -> Optional[Dict]:
    return xai_jobs_repo.update(job_id, **fields)


def get_pending_xai_jobs() -> List[Dict]:
    """
    Claim one job rather than listing them all. claim_xai_job() uses
    FOR UPDATE SKIP LOCKED, so two workers (an in-process one and a separate
    `python xai_worker.py`, say) never pick up the same job. Jobs left in
    'processing' by a crashed worker are reset first, as the batch worker
    does for batch_items.
    """
    xai_jobs_repo.sweep_stuck()
    job = xai_jobs_repo.claim()
    return [job] if job else []


# ===========================================================================
# RETRAIN JOBS  (no repository exists for this table)
# ===========================================================================

def enqueue_retrain_job(triggered_by=None, notes=None, min_annotations=None) -> Dict:
    from repositories._client import current_domain, domain_queues_available
    row = {
        "status": "pending",
        "triggered_by": triggered_by,
        "notes": notes,
        "min_annotations": min_annotations,
    }
    # After 010, so each domain's retrain worker picks up only its own job.
    if domain_queues_available():
        row["domain"] = current_domain()
    result = get_client().table("retrain_jobs").insert(row).execute()
    return result.data[0]


def get_retrain_job(job_id: str) -> Optional[Dict]:
    result = get_client().table("retrain_jobs").select("*").eq("id", job_id).execute()
    return result.data[0] if result.data else None


def get_latest_retrain_job() -> Optional[Dict]:
    from repositories._client import current_domain, domain_queues_available
    query = get_client().table("retrain_jobs").select("*")
    # "The latest job" is this domain's latest job once two domains share
    # the table (010); otherwise a medical worker would run a legal retrain.
    if domain_queues_available():
        query = query.eq("domain", current_domain())
    result = query.order("triggered_at", desc=True).limit(1).execute()
    return result.data[0] if result.data else None


def update_retrain_job(job_id, status, model_version_id=None, completed_at=None) -> Optional[Dict]:
    fields = {"status": status}
    if model_version_id is not None:
        fields["model_version_id"] = model_version_id
    if completed_at is not None:
        fields["completed_at"] = completed_at
    result = get_client().table("retrain_jobs").update(fields).eq("id", job_id).execute()
    return result.data[0] if result.data else None


# ===========================================================================
# DATASET & MODEL VERSIONS
# ===========================================================================

def create_dataset_version(version_id, sample_count, label_distribution) -> Dict:
    return model_versions_repo.insert_dataset_version(
        version_id=version_id,
        sample_count=sample_count,
        label_distribution=label_distribution,
    )


def create_model_version(version_number, accuracy, f1_per_class, file_path,
                         dataset_version_id=None) -> Dict:
    return model_versions_repo.insert(
        version_number=version_number,
        accuracy=accuracy,
        f1_per_class=f1_per_class,
        file_path=file_path,
        dataset_version_id=dataset_version_id,
    )


def get_active_model_version() -> Optional[Dict]:
    return model_versions_repo.get_active()
