"""
repositories/batch_items.py — CRUD + queue operations for batch_items.

claim() wraps the claim_batch_items(n) Postgres RPC.
sweep_stuck() resets rows that have been in 'processing' too long.
"""

from typing import Dict, List, Optional

from repositories._client import get_client, is_configured


def insert_many(batch_id: str, texts: List[str]) -> List[Dict]:
    """
    Bulk-insert one row per text, returning all created rows.

    seq continues from the highest existing seq for this batch_id (0 if none
    exist yet) rather than always starting at 1 — batch_items has a unique
    (batch_id, seq) constraint, so a second insert call into the same batch
    that restarted at 1 would hard-fail. Mirrors mock_db.create_batch_items().
    """
    if not is_configured():
        from services import mock_db
        return mock_db.create_batch_items(batch_id=batch_id, texts=texts)

    existing = (
        get_client()
        .table("batch_items")
        .select("seq")
        .eq("batch_id", batch_id)
        .order("seq", desc=True)
        .limit(1)
        .execute()
    )
    start_seq = (existing.data[0]["seq"] if existing.data else 0) + 1

    payload = [
        {
            "batch_id": batch_id,
            "seq": seq,
            "text_content": text,
            "status": "pending",
            "attempts": 0,
        }
        for seq, text in enumerate(texts, start=start_seq)
    ]
    result = get_client().table("batch_items").insert(payload).execute()
    return result.data


def get(item_id: str) -> Optional[Dict]:
    """Fetch a single batch_item by id."""
    if not is_configured():
        from services import mock_db
        return mock_db.get_batch_item(item_id)

    result = (
        get_client()
        .table("batch_items")
        .select("*")
        .eq("id", item_id)
        .execute()
    )
    return result.data[0] if result.data else None


def list_by_batch(
    batch_id: str,
    offset: int = 0,
    limit: int = 50,
) -> List[Dict]:
    """Paginated listing of items within a batch, ordered by seq."""
    if not is_configured():
        from services import mock_db
        return mock_db.list_batch_items(batch_id, offset=offset, limit=limit)

    result = (
        get_client()
        .table("batch_items")
        .select("*")
        .eq("batch_id", batch_id)
        .order("seq")
        .range(offset, offset + limit - 1)
        .execute()
    )
    return result.data


def list_all_by_batch(batch_id: str) -> List[Dict]:
    """Return every item in a batch, ordered by seq (no pagination — used by CSV export)."""
    if not is_configured():
        from services import mock_db
        return mock_db.list_all_batch_items(batch_id)

    result = (
        get_client()
        .table("batch_items")
        .select("*")
        .eq("batch_id", batch_id)
        .order("seq")
        .execute()
    )
    return result.data


def update(item_id: str, **fields) -> Optional[Dict]:
    """Partial update on a batch_items row."""
    if not is_configured():
        from services import mock_db
        return mock_db.update_batch_item(item_id=item_id, **fields)

    result = (
        get_client()
        .table("batch_items")
        .update(fields)
        .eq("id", item_id)
        .execute()
    )
    return result.data[0] if result.data else None


def claim(n: int) -> List[Dict]:
    """
    Atomically claim up to n pending batch_items for processing.

    Wraps the claim_batch_items(n) Postgres RPC which uses FOR UPDATE SKIP LOCKED
    to prevent double-claiming across concurrent worker processes.

    Returns the claimed rows (all columns) with status already flipped to
    'processing' and locked_at set to now().
    """
    if not is_configured():
        from services import mock_db
        return mock_db.claim_batch_items(n)

    result = get_client().rpc("claim_batch_items", {"n": n}).execute()
    return result.data or []


def sweep_stuck() -> None:
    """
    Reset batch_items rows that are stuck in 'processing'.

    Rows in 'processing' with locked_at older than STUCK_JOB_MINUTES are
    retried (status → 'pending', attempts + 1). Rows past MAX_ATTEMPTS → 'failed'.
    """
    from datetime import datetime, timezone, timedelta
    from config.settings import settings

    if not is_configured():
        return  # mock_db doesn't model stuck jobs

    cutoff = (
        datetime.now(timezone.utc) - timedelta(minutes=settings.STUCK_JOB_MINUTES)
    ).isoformat()

    client = get_client()
    stuck = (
        client.table("batch_items")
        .select("id, attempts")
        .eq("status", "processing")
        .lt("locked_at", cutoff)
        .execute()
        .data or []
    )

    for row in stuck:
        if row["attempts"] >= settings.MAX_ATTEMPTS:
            client.table("batch_items").update({
                "status": "failed",
                "locked_at": None,
                "error_message": f"Exceeded {settings.MAX_ATTEMPTS} attempts",
            }).eq("id", row["id"]).execute()
        else:
            client.table("batch_items").update({
                "status": "pending",
                "locked_at": None,
            }).eq("id", row["id"]).execute()
