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
        return mock_db.create_batch(
            source=source, total_items=total_items, filename=filename
        )

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
        from services import mock_db
        return mock_db.get_batch(batch_id)

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
        from services import mock_db
        return mock_db.update_batch_status(batch_id, status)

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
        from services import mock_db
        return mock_db.increment_batch_completed(batch_id)

    # Postgres RPC for atomic increment; avoids read-modify-write race
    result = get_client().rpc(
        "increment_batch_completed", {"batch_id": batch_id}
    ).execute()
    return result.data


def list_all(limit: Optional[int] = None) -> List[Dict]:
    """Return batches ordered by created_at descending, optionally capped at limit."""
    if not is_configured():
        from services import mock_db
        return mock_db.list_batches(limit=limit)

    query = get_client().table("batches").select("*").order("created_at", desc=True)
    if limit is not None:
        query = query.limit(limit)
    result = query.execute()
    return result.data
