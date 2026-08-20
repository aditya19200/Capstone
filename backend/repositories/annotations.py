"""
repositories/annotations.py — CRUD for the annotations table.
"""

from typing import Dict, List, Literal, Optional

from repositories._client import get_client, is_configured

AnnotationStatus = Literal["pending", "validated", "rejected"]


def insert(
    prediction_id: str,
    validated_label: str,
    annotator_id: Optional[str],
    status: AnnotationStatus = "pending",
    document_id: Optional[str] = None,
) -> Dict:
    """
    Insert one annotations row and return it.

    document_id is not in the real schema.  It is forwarded to mock_db so
    that GET /annotate can return it in AnnotationListItem until that route
    is rewritten to join through predictions (Step 5 target).
    """
    if not is_configured():
        from services import mock_db
        return mock_db.create_annotation(
            prediction_id=prediction_id,
            validated_label=validated_label,
            annotator_id=annotator_id,
            status=status,
            document_id=document_id,
        )

    result = get_client().table("annotations").insert({
        "prediction_id": prediction_id,
        "validated_label": validated_label,
        "annotator_id": annotator_id,
        "status": status,
    }).execute()
    return result.data[0]


def get(annotation_id: str) -> Optional[Dict]:
    """Fetch an annotation by id."""
    if not is_configured():
        from services import mock_db
        return mock_db.get_annotation(annotation_id)

    result = (
        get_client()
        .table("annotations")
        .select("*")
        .eq("id", annotation_id)
        .execute()
    )
    return result.data[0] if result.data else None


def list_by_prediction(prediction_id: str) -> List[Dict]:
    """Return all annotations for a prediction, newest first."""
    if not is_configured():
        from services import mock_db
        rows = mock_db.list_annotations(prediction_id=prediction_id)
        return sorted(rows, key=lambda r: r.get("annotated_at", ""), reverse=True)

    result = (
        get_client()
        .table("annotations")
        .select("*")
        .eq("prediction_id", prediction_id)
        .order("annotated_at", desc=True)
        .execute()
    )
    return result.data


def list_by_annotator(annotator_id: str) -> List[Dict]:
    """Return annotations submitted by a specific annotator, newest first."""
    if not is_configured():
        from services import mock_db
        rows = mock_db.list_annotations(annotator_id=annotator_id)
        return sorted(rows, key=lambda r: r.get("annotated_at", ""), reverse=True)

    result = (
        get_client()
        .table("annotations")
        .select("*")
        .eq("annotator_id", annotator_id)
        .order("annotated_at", desc=True)
        .execute()
    )
    return result.data


def count_validated() -> int:
    """
    Return the count of annotations with status='validated'.

    Used by the retraining pipeline to check whether enough human labels
    exist before starting a fine-tuning run.
    """
    if not is_configured():
        from services import mock_db
        return mock_db.count_validated_annotations()

    result = (
        get_client()
        .table("annotations")
        .select("id", count="exact")
        .eq("status", "validated")
        .execute()
    )
    return result.count or 0
