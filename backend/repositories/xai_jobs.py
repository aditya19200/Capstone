"""
repositories/xai_jobs.py — CRUD + queue operations for xai_jobs.

claim() wraps the claim_xai_job() Postgres RPC.
sweep_stuck() mirrors the same logic as batch_items.sweep_stuck().
"""

from typing import Dict, List, Optional

from repositories._client import get_client, is_configured


def insert(prediction_id: str) -> Dict:
    """Insert one xai_jobs row with status='pending' and return it."""
    if not is_configured():
        from services import mock_db
        return mock_db.enqueue_xai_job(prediction_id=prediction_id)

    result = get_client().table("xai_jobs").insert({
        "prediction_id": prediction_id,
        "status": "pending",
        "attempts": 0,
    }).execute()
    return result.data[0]


def get(job_id: str) -> Optional[Dict]:
    """Fetch an xai_jobs row by id."""
    if not is_configured():
        from services import mock_db
        return mock_db.get_xai_job(job_id)

    result = (
        get_client()
        .table("xai_jobs")
        .select("*")
        .eq("id", job_id)
        .execute()
    )
    return result.data[0] if result.data else None


def get_by_prediction(prediction_id: str) -> Optional[Dict]:
    """Fetch the xai_jobs row for a given prediction_id."""
    if not is_configured():
        from services import mock_db
        return mock_db.get_xai_job_by_prediction(prediction_id)

    result = (
        get_client()
        .table("xai_jobs")
        .select("*")
        .eq("prediction_id", prediction_id)
        .limit(1)
        .execute()
    )
    return result.data[0] if result.data else None


def update(job_id: str, **fields) -> Optional[Dict]:
    """
    Partial update on an xai_jobs row.

    Accepts any real schema field: status, token_importance, summary,
    attempts, locked_at, error_message.
    """
    if not is_configured():
        from services import mock_db
        return mock_db.update_xai_job(job_id=job_id, **fields)

    result = (
        get_client()
        .table("xai_jobs")
        .update(fields)
        .eq("id", job_id)
        .execute()
    )
    return result.data[0] if result.data else None


def claim() -> Optional[Dict]:
    """
    Atomically claim one pending xai_jobs row for the SHAP worker.

    Wraps the claim_xai_job() Postgres RPC (FOR UPDATE SKIP LOCKED, limit 1).
    Returns the claimed row or None if the queue is empty.
    """
    if not is_configured():
        from services import mock_db
        pending = [j for j in mock_db._xai_jobs if j["status"] == "pending"]
        if not pending:
            return None
        job = pending[0]
        mock_db.update_xai_job(job_id=job["id"], status="processing")
        return mock_db.get_xai_job(job["id"])

    result = get_client().rpc("claim_xai_job", {}).execute()
    data = result.data
    if not data:
        return None
    return data[0] if isinstance(data, list) else data


def sweep_stuck() -> None:
    """
    Reset xai_jobs rows stuck in 'processing' past STUCK_JOB_MINUTES.

    Same retry logic as batch_items.sweep_stuck(): reset to pending until
    MAX_ATTEMPTS, then mark failed.
    """
    from datetime import datetime, timezone, timedelta
    from config.settings import settings

    if not is_configured():
        return

    cutoff = (
        datetime.now(timezone.utc) - timedelta(minutes=settings.STUCK_JOB_MINUTES)
    ).isoformat()

    client = get_client()
    stuck = (
        client.table("xai_jobs")
        .select("id, attempts")
        .eq("status", "processing")
        .lt("locked_at", cutoff)
        .execute()
        .data or []
    )

    for row in stuck:
        if row["attempts"] >= settings.MAX_ATTEMPTS:
            client.table("xai_jobs").update({
                "status": "failed",
                "locked_at": None,
                "error_message": f"Exceeded {settings.MAX_ATTEMPTS} attempts",
            }).eq("id", row["id"]).execute()
        else:
            client.table("xai_jobs").update({
                "status": "pending",
                "locked_at": None,
            }).eq("id", row["id"]).execute()
