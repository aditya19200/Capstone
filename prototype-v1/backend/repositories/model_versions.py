"""
repositories/model_versions.py — CRUD for model_versions and dataset_versions.

is_active is the rollback mechanism: flip is_active=true on an older version,
restart the API/workers, done. No .env change needed.

set_active() in mock mode calls set_active_model_version() which deactivates
all versions then activates the target — two in-memory writes, safe because
mock_db is single-threaded.  In real Supabase it calls the
activate_model_version(target_id uuid) RPC which does this atomically in one
transaction (see docs/MIGRATION_activate_model_version.md).
"""

from typing import Dict, List, Optional

from repositories._client import get_client, is_configured


# ---------------------------------------------------------------------------
# model_versions
# ---------------------------------------------------------------------------

def get_active() -> Optional[Dict]:
    """
    Return the currently active model version row, or None.

    The API reads file_path from this row at startup to know which weights to load.
    """
    if not is_configured():
        from services import mock_db
        return mock_db.get_active_model_version()

    result = (
        get_client()
        .table("model_versions")
        .select("*")
        .eq("is_active", True)
        .limit(1)
        .execute()
    )
    return result.data[0] if result.data else None


def insert(
    version_number: str,
    accuracy: float,
    f1_per_class: Dict[str, float],
    file_path: str,
    dataset_version_id: Optional[str] = None,
) -> Dict:
    """
    Record a newly trained model version.

    Does NOT set is_active. Call set_active() after verifying the new weights
    load correctly.
    """
    if not is_configured():
        from services import mock_db
        return mock_db.create_model_version(
            version_number=version_number,
            accuracy=accuracy,
            f1_per_class=f1_per_class,
            file_path=file_path,
            dataset_version_id=dataset_version_id,
            is_active=False,
        )

    result = get_client().table("model_versions").insert({
        "version_number": version_number,
        "accuracy": accuracy,
        "f1_per_class": f1_per_class,
        "file_path": file_path,
        "dataset_version_id": dataset_version_id,
        "is_active": False,
    }).execute()
    return result.data[0]


def set_active(version_id: str) -> Optional[Dict]:
    """
    Atomically flip is_active: deactivate all versions, then activate version_id.

    Mock mode: two in-memory writes via set_active_model_version().
    Real Supabase: calls the activate_model_version(target_id) RPC.
    """
    if not is_configured():
        from services import mock_db
        return mock_db.set_active_model_version(version_id)

    result = get_client().rpc(
        "activate_model_version", {"target_id": version_id}
    ).execute()
    return result.data[0] if result.data else None


def list_all() -> List[Dict]:
    """Return all model versions ordered by trained_at descending."""
    if not is_configured():
        from services import mock_db
        return mock_db.list_model_versions()

    result = (
        get_client()
        .table("model_versions")
        .select("*")
        .order("trained_at", desc=True)
        .execute()
    )
    return result.data


# ---------------------------------------------------------------------------
# dataset_versions
# ---------------------------------------------------------------------------

def insert_dataset_version(
    version_id: str,
    sample_count: int,
    label_distribution: Dict[str, int],
) -> Dict:
    """Record a dataset snapshot created for a retraining run."""
    if not is_configured():
        from services import mock_db
        return mock_db.create_dataset_version(
            version_id=version_id,
            sample_count=sample_count,
            label_distribution=label_distribution,
        )

    result = get_client().table("dataset_versions").insert({
        "version_id": version_id,
        "sample_count": sample_count,
        "label_distribution": label_distribution,
    }).execute()
    return result.data[0]
