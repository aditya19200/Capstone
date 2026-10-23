"""
routers/admin.py — Admin dashboard and model-rollback endpoints.

POST /admin/models/{version_id}/activate
  Flip is_active onto the given model_versions row (rollback mechanism), then
  load that version's weights into the running model service so it actually
  serves predictions. Wraps the activate_model_version(target_id) Postgres RPC.

GET  /admin/metrics
  Dashboard payload: per-class F1 from the active model version, annotation
  counts by status, and per-batch throughput. Read from the DB, not JSON files.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException, status
from fastapi.responses import Response, StreamingResponse

from models.response_models import (
    AdminMetricsResponse,
    BatchComparisonResponse,
    ComparisonBatchSummary,
    BatchThroughputItem,
    ComparisonRow,
    ModelVersionOut,
    ModelVersionSummary,
)
from repositories import annotations as annotations_repo
from repositories import batch_items as batch_items_repo
from repositories import batches as batches_repo
from repositories import predictions as predictions_repo
from repositories import model_versions as model_versions_repo
from services import dataset_export
from services.model_service import model_service
from utils.text import normalize

logger = logging.getLogger(__name__)
router = APIRouter()

_ROLE_ADMIN = "admin"


def _require_admin(x_role: Optional[str]) -> None:
    role = (x_role or "").lower()
    if role != _ROLE_ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Role '{x_role}' is not permitted here. Required: admin.",
        )


# ---------------------------------------------------------------------------
# POST /admin/models/{version_id}/activate
# ---------------------------------------------------------------------------

@router.post(
    "/models/{version_id}/activate",
    response_model=ModelVersionOut,
    status_code=status.HTTP_200_OK,
    summary="Activate a model version (rollback mechanism)",
)
async def activate_model_version(
    version_id: str,
    x_role: Optional[str] = Header(default=None, alias="X-Role"),
) -> ModelVersionOut:
    _require_admin(x_role)

    # Remembered before the flip so a failed weight load can be rolled back —
    # otherwise the table would claim a version is active that never loaded.
    previous = model_versions_repo.get_active()

    try:
        row = model_versions_repo.set_active(version_id)
    except Exception as exc:
        if "does not exist" in str(exc):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model version '{version_id}' not found.",
            )
        raise

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model version '{version_id}' not found.",
        )

    # A version from another domain cannot be served here whatever the UI
    # offers: ID2LABEL and NUM_LABELS are bound when the process starts, so
    # load() would hand a 12-label checkpoint to a 10-label head and raise.
    # It fails safely, but a 500 with a state_dict traceback in it tells the
    # person nothing. Refuse before the flip, so is_active is never moved to
    # a version this process could not possibly load.
    from config.domains import active_domain, domain_for_model_path
    row_domain = domain_for_model_path(row.get("file_path"))
    serving = active_domain().key
    if row_domain and row_domain != serving:
        if previous and previous["id"] != version_id:
            try:
                model_versions_repo.set_active(previous["id"])
            except Exception:
                logger.exception("Could not restore the previously active version")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"'{row['version_number']}' belongs to the {row_domain} domain "
                f"and this process serves {serving}. The label set is fixed at "
                f"startup, so these weights can only be served by the "
                f"{row_domain} instance."
            ),
        )

    # Activation is only real once the weights are actually serving. load()
    # builds the new model before swapping it in, so on failure the previously
    # loaded model is still resident — we just have to undo the is_active flip.
    try:
        model_service.load(
            model_path=row["file_path"],
            version_number=row["version_number"],
        )
    except Exception as exc:
        logger.exception("Activation failed to load weights for %s", version_id)
        if previous and previous["id"] != version_id:
            try:
                model_versions_repo.set_active(previous["id"])
                logger.info("Rolled is_active back to %s", previous["id"])
            except Exception:
                logger.exception("Rollback of is_active to %s failed", previous["id"])
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                f"Model version '{version_id}' could not be loaded from "
                f"'{row['file_path']}': {exc}. The previously active model is "
                "still serving."
            ),
        )

    logger.info(
        "Model version activated and loaded: id=%s version=%s path=%s",
        version_id, row["version_number"], row["file_path"],
    )

    return ModelVersionOut(
        id=row["id"],
        version_number=row["version_number"],
        accuracy=row["accuracy"],
        f1_per_class=row["f1_per_class"],
        file_path=row["file_path"],
        is_active=row["is_active"],
        trained_at=row["trained_at"],
    )


# ---------------------------------------------------------------------------
# GET /admin/metrics
# ---------------------------------------------------------------------------

@router.get(
    "/metrics",
    response_model=AdminMetricsResponse,
    status_code=status.HTTP_200_OK,
    summary="Admin dashboard metrics",
)
async def get_admin_metrics(
    x_role: Optional[str] = Header(default=None, alias="X-Role"),
) -> AdminMetricsResponse:
    _require_admin(x_role)

    active_model = model_versions_repo.get_active()
    annotation_counts = annotations_repo.count_by_status()
    recent_batches = batches_repo.list_all(limit=20)
    all_versions = model_versions_repo.list_all()

    # What is loaded, as opposed to what the database says should be. The two
    # are the same almost always, and the moment they are not is exactly the
    # moment somebody needs to be told: between pressing Activate and the
    # restart that honours it, or when a flagged version's files would not
    # load and startup fell back. A table that shows only is_active cannot
    # express either, so it reports a model that is not answering anything.
    served_path = model_service.loaded_path

    # Which domain each checkpoint belongs to, recovered from its path rather
    # than a column: model_versions has none, and adding one needs a migration
    # against a database the operator may not administer. See
    # config.domains.domain_for_model_path.
    from config.domains import active_domain, domain_for_model_path
    serving_domain = active_domain().key

    return AdminMetricsResponse(
        active_model_version=active_model["version_number"] if active_model else None,
        serving_model_version=model_service.loaded_version,
        serving_domain=serving_domain,
        f1_per_class=active_model["f1_per_class"] if active_model else {},
        annotation_counts=annotation_counts,
        model_versions=[
            ModelVersionSummary(
                id=v["id"],
                version_number=v["version_number"],
                accuracy=v["accuracy"],
                is_active=v.get("is_active", False),
                trained_at=v["trained_at"],
                # A retrained version matches no pack; it is always a version
                # of whichever domain produced it, which is this one.
                domain=domain_for_model_path(v.get("file_path")) or serving_domain,
                is_serving=bool(served_path) and v.get("file_path") == served_path,
            )
            for v in all_versions
        ],
        batch_throughput=[
            BatchThroughputItem(
                batch_id=b["id"],
                source=b["source"],
                status=b["status"],
                total_items=b["total_items"],
                completed_items=b["completed_items"],
                created_at=b["created_at"],
            )
            for b in recent_batches
        ],
    )



# ---------------------------------------------------------------------------
# GET /admin/dataset  and  /admin/dataset/card
# ---------------------------------------------------------------------------

@router.get(
    "/dataset",
    status_code=status.HTTP_200_OK,
    summary="Download the validated corpus as JSONL",
    description=(
        "Every human-validated annotation as JSON Lines, one example per "
        "line, ready for datasets.load_dataset(\"json\", ...). This is the "
        "output of the framework: the model pre-labels, the routing layer "
        "decides what a human sees, and what comes out is training data. "
        "Assembled from exactly what the retrain worker consumes, so the "
        "file and the thing we retrain on cannot drift apart."
    ),
)
async def export_dataset(
    x_role: Optional[str] = Header(default=None, alias="X-Role"),
) -> StreamingResponse:
    _require_admin(x_role)
    records, skipped = dataset_export.rows()
    if skipped:
        logger.warning("Dataset export skipped %d unusable annotation(s)", skipped)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return StreamingResponse(
        dataset_export.to_jsonl(records),
        media_type="application/x-ndjson",
        headers={
            "Content-Disposition":
                f'attachment; filename="dataset-{stamp}.jsonl"',
            # Lets the UI show the count without reading the stream.
            "X-Example-Count": str(len(records)),
        },
    )


@router.get(
    "/dataset.pdf",
    status_code=status.HTTP_200_OK,
    summary="Download the annotated corpus as a PDF",
    description=(
        "The same validated annotations as GET /admin/dataset, laid out as a "
        "document: each text with its ontology concept markers highlighted in "
        "place, what the model predicted and how sure it was, and what the "
        "human decided. JSONL is what another system consumes; this is what a "
        "person reads."
    ),
)
async def export_dataset_pdf(
    x_role: Optional[str] = Header(default=None, alias="X-Role"),
) -> Response:
    _require_admin(x_role)
    records, skipped = dataset_export.rows()
    if skipped:
        logger.warning("Dataset PDF export skipped %d unusable annotation(s)", skipped)
    try:
        pdf = dataset_export.to_pdf(records)
    except ImportError as exc:
        # reportlab is in requirements.txt, but an older checkout's venv will
        # not have it. Saying so beats a 500 with a traceback in it.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "PDF export needs reportlab, which is not installed in this "
                f"environment ({exc}). Run: pip install -r requirements.txt"
            ),
        )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={
            "Content-Disposition":
                f'attachment; filename="annotated-corpus-{stamp}.pdf"',
            "X-Example-Count": str(len(records)),
        },
    )


@router.get(
    "/dataset/card",
    status_code=status.HTTP_200_OK,
    summary="Dataset card for the validated corpus",
    description=(
        "Markdown describing what the export contains — size, label "
        "distribution, fields, and the limits that matter. Returned "
        "alongside the raw counts so the UI can render either."
    ),
)
async def dataset_card(
    x_role: Optional[str] = Header(default=None, alias="X-Role"),
) -> Dict:
    _require_admin(x_role)
    records, _ = dataset_export.rows()
    return {"stats": dataset_export.stats(records),
            "card_markdown": dataset_export.card(records)}

# ---------------------------------------------------------------------------
# GET /admin/batches  and  GET /admin/compare
# ---------------------------------------------------------------------------

_COMPARE_EXCERPT = 160


def _batch_predictions(batch_id: str) -> List[Dict]:
    """Every prediction belonging to a batch, via its items."""
    rows: List[Dict] = []
    offset = 0
    while True:
        items = batch_items_repo.list_by_batch(batch_id, offset=offset, limit=200)
        if not items:
            break
        for item in items:
            pid = item.get("prediction_id")
            if not pid:
                continue
            pred = predictions_repo.get(pid)
            if pred:
                rows.append(pred)
        if len(items) < 200:
            break
        offset += 200
    return rows


@router.get(
    "/batches",
    response_model=List[ComparisonBatchSummary],
    status_code=status.HTTP_200_OK,
    summary="Batches available for comparison, newest first",
)
async def list_comparable_batches(
    x_role: Optional[str] = Header(default=None, alias="X-Role"),
) -> List[ComparisonBatchSummary]:
    _require_admin(x_role)

    out: List[ComparisonBatchSummary] = []
    for batch in batches_repo.list_all():
        preds = _batch_predictions(batch["id"])
        if not preds:
            continue
        versions = sorted({p.get("model_version") or "unknown" for p in preds})
        out.append(ComparisonBatchSummary(
            batch_id=batch["id"],
            created_at=batch.get("created_at", ""),
            item_count=len(preds),
            model_versions=versions,
        ))
    return out


@router.get(
    "/compare",
    response_model=BatchComparisonResponse,
    status_code=status.HTTP_200_OK,
    summary="Compare two batches of the same documents, classified by different models",
)
async def compare_batches(
    batch_a: str,
    batch_b: str,
    changed_only: bool = False,
    x_role: Optional[str] = Header(default=None, alias="X-Role"),
) -> BatchComparisonResponse:
    """
    Line up two runs of the same documents so the effect of a retrain is
    visible rather than asserted.

    Rows are matched on normalised text — the same normalise() every
    prediction path already applies — so the same file uploaded twice lines
    up even when the batches differ in order or length. Documents present in
    only one batch are skipped rather than shown as a change, since there is
    nothing to compare them against.
    """
    _require_admin(x_role)

    for label, bid in (("batch_a", batch_a), ("batch_b", batch_b)):
        if batches_repo.get(bid) is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"{label} '{bid}' not found.",
            )
    if batch_a == batch_b:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="batch_a and batch_b are the same batch — nothing to compare.",
        )

    preds_a = _batch_predictions(batch_a)
    preds_b = _batch_predictions(batch_b)
    by_text_b = {normalize(p["text_content"]): p for p in preds_b}

    rows: List[ComparisonRow] = []
    changed = 0
    for pa in preds_a:
        pb = by_text_b.get(normalize(pa["text_content"]))
        if pb is None:
            continue
        did_change = pa["predicted_label"] != pb["predicted_label"]
        if did_change:
            changed += 1
        if changed_only and not did_change:
            continue
        text = pa["text_content"]
        rows.append(ComparisonRow(
            text_excerpt=text[:_COMPARE_EXCERPT] + ("..." if len(text) > _COMPARE_EXCERPT else ""),
            label_a=pa["predicted_label"],
            confidence_a=pa["confidence"],
            label_b=pb["predicted_label"],
            confidence_b=pb["confidence"],
            changed=did_change,
        ))

    # Changed rows first — the whole reason to open this screen.
    rows.sort(key=lambda r: (not r.changed, r.text_excerpt))

    def _version(preds: List[Dict]) -> Optional[str]:
        versions = {p.get("model_version") for p in preds if p.get("model_version")}
        return ", ".join(sorted(versions)) if versions else None

    matched = sum(1 for pa in preds_a if normalize(pa["text_content"]) in by_text_b)
    return BatchComparisonResponse(
        batch_a=batch_a,
        batch_b=batch_b,
        model_version_a=_version(preds_a),
        model_version_b=_version(preds_b),
        matched=matched,
        changed=changed,
        rows=rows,
    )
