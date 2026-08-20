"""
repositories/batches.py — CRUD for the batches table.
"""

from typing import Dict, List, Literal, Optional

from repositories._client import get_client, is_configured

BatchSource = Literal["paste", "csv", "pdf"]
BatchStatus = Literal["pending", "processing", "done", "failed"]


def insert(
    source: BatchSource,
    total_items: int,
    filename: Optional[str] = None,
) -> Dict:
    """Create a new batch row and return it."""
    if not is_configured():
        from services import mock_db
        return {
            "id": mock_db._new_id(),
            "source": source,
            "filename": filename,
            "status": "pending",
            "total_items": total_items,
            "completed_items": 0,
            "created_at": mock_db._now_iso(),
        }

    result = get_client().table("batches").insert({
        "source": source,
        "filename": filename,
        "status": "pending",
        "total_items": total_items,
        "completed_items": 0,
    }).execute()
    return result.data[0]


def get(batch_id: str) -> Optional[Dict]:
    """Fetch a batch row by id."""
    if not is_configured():
        return None  # no batch store in mock_db; workers use batch_items directly

    result = (
        get_client()
        .table("batches")
        .select("*")
        .eq("id", batch_id)
        .execute()
    )
    return result.data[0] if result.data else None


def update_status(batch_id: str, status: BatchStatus) -> Optional[Dict]:
    """Flip the status field on a batch."""
    if not is_configured():
        return None

    result = (
        get_client()
        .table("batches")
        .update({"status": status})
        .eq("id", batch_id)
        .execute()
    )
    return result.data[0] if result.data else None


def increment_completed(batch_id: str) -> Optional[Dict]:
    """Atomically increment completed_items by 1."""
    if not is_configured():
        return None

    # Postgres RPC for atomic increment; avoids read-modify-write race
    result = get_client().rpc(
        "increment_batch_completed", {"batch_id": batch_id}
    ).execute()
    return result.data
