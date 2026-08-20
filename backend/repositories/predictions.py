"""
repositories/predictions.py — CRUD for the predictions table.

Falls back to mock_db when Supabase is not configured (placeholder URL).
Swap: remove the is_configured() branches once the real schema is live.
"""

from typing import Dict, List, Optional

from repositories._client import get_client, is_configured


def insert(
    text_content: str,
    predicted_label: str,
    label_id: int,
    confidence: float,
    all_probabilities: Dict[str, float],
    model_version: str,
) -> Dict:
    """Insert one predictions row and return it."""
    if not is_configured():
        from services import mock_db
        return mock_db.create_prediction(
            text_content=text_content,
            predicted_label=predicted_label,
            label_id=label_id,
            confidence=confidence,
            all_probabilities=all_probabilities,
            model_version=model_version,
        )

    result = get_client().table("predictions").insert({
        "text_content": text_content,
        "predicted_label": predicted_label,
        "label_id": label_id,
        "confidence": confidence,
        "all_probabilities": all_probabilities,
        "model_version": model_version,
    }).execute()
    return result.data[0]


def get(prediction_id: str) -> Optional[Dict]:
    """Fetch a predictions row by id."""
    if not is_configured():
        from services import mock_db
        return mock_db.get_prediction(prediction_id)

    result = (
        get_client()
        .table("predictions")
        .select("*")
        .eq("id", prediction_id)
        .execute()
    )
    return result.data[0] if result.data else None


def list_low_confidence(threshold: float) -> List[Dict]:
    """Return all predictions with confidence < threshold, newest first."""
    if not is_configured():
        from services import mock_db
        return mock_db.list_low_confidence_predictions(threshold)

    result = (
        get_client()
        .table("predictions")
        .select("*")
        .lt("confidence", threshold)
        .order("created_at", desc=True)
        .execute()
    )
    return result.data
