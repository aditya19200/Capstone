"""
routers/batches.py — Batch ingestion endpoints.

POST /batches/paste
POST /batches/csv
POST /batches/pdf
  Ingest a batch of texts. No classification happens in the request path —
  a batches row plus one 'pending' batch_items row per text is inserted and
  the response returns immediately. The classify worker picks items up from
  the batch_items queue.

  The three differ only in how they get from an upload to a list of strings.
  Past that point a PDF is indistinguishable from a pasted document: same
  queue, same routing, same ontology check, same review screen, same export.

  Every one of them runs utils.text.normalize() before insert. That is not
  cosmetic — the retrain worker normalises before training, so text stored
  un-normalised here would be classified in one form and trained on in
  another. It matters most for PDFs, whose text arrives full of the
  hyphen-breaks and soft newlines normalize() exists to remove.

GET  /batches
  List past batches, newest first — lets the UI re-open a batch after
  navigating away instead of losing track of it.

GET  /batches/{id}
  Poll batch-level progress (status, total_items, completed_items).

GET  /batches/{id}/items
  Paginated listing of items within a batch.

GET  /batches/{id}/export
  Stream the full batch (predictions included) back as a CSV download.
"""

import io
import logging

import pandas as pd
from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from fastapi.responses import StreamingResponse

from models.request_models import BatchPasteRequest
from models.response_models import (
    BatchCreateResponse,
    BatchItemOut,
    BatchItemsResponse,
    BatchListResponse,
    BatchStatusResponse,
    BatchSummary,
)
from repositories import batch_items as batch_items_repo
from repositories import batches as batches_repo
from config.settings import settings
from services import pdf_extract
from utils.text import count_tokens, normalize

logger = logging.getLogger(__name__)
router = APIRouter()

_FORMULA_TRIGGER_CHARS = ("=", "+", "-", "@")


def sanitize_csv_cell(value: str) -> str:
    """
    Neutralise CSV formula injection.

    A cell starting with =, +, -, or @ (after any leading whitespace — some
    spreadsheet importers strip it before evaluating) is interpreted as a
    formula by Excel/Sheets when the file is opened — ranging from a
    misleading link to data exfiltration via a crafted =WEBSERVICE(...)-style
    formula. Prefixing with a single quote forces spreadsheet software to
    treat the cell as literal text; it's a no-op for anything parsing the
    CSV programmatically (csv/pandas readers don't strip leading quotes).
    """
    if value.lstrip().startswith(_FORMULA_TRIGGER_CHARS):
        return "'" + value
    return value


# ---------------------------------------------------------------------------
# POST /batches/paste
# ---------------------------------------------------------------------------

@router.post(
    "/paste",
    response_model=BatchCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest a batch of pasted texts",
)
async def submit_batch_paste(body: BatchPasteRequest) -> BatchCreateResponse:
    """Create a batch row + one pending batch_items row per text."""
    texts = [t for t in (normalize(t) for t in body.texts) if t]
    if not texts:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No usable text was submitted.",
        )

    batch = batches_repo.insert(source="paste", total_items=len(texts))
    batch_items_repo.insert_many(batch_id=batch["id"], texts=texts)

    logger.info("Batch queued: id=%s source=paste items=%d", batch["id"], len(texts))

    return BatchCreateResponse(batch_id=batch["id"], total_items=len(texts))


# ---------------------------------------------------------------------------
# POST /batches/csv
# ---------------------------------------------------------------------------

@router.post(
    "/csv",
    response_model=BatchCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest a batch from an uploaded CSV file",
    description=(
        "Reads a 'text' column (case-insensitive) from the uploaded CSV. "
        "If the CSV has exactly one column, that column is used regardless "
        "of its name."
    ),
)
async def submit_batch_csv(file: UploadFile = File(...)) -> BatchCreateResponse:
    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not parse CSV: {exc}",
        )

    text_column = None
    for col in df.columns:
        if str(col).strip().lower() == "text":
            text_column = col
            break
    if text_column is None:
        if len(df.columns) == 1:
            text_column = df.columns[0]
        else:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    "CSV must have a 'text' column, or exactly one column. "
                    f"Found columns: {list(df.columns)}"
                ),
            )

    texts = df[text_column].dropna().astype(str).tolist()
    texts = [t for t in (normalize(t) for t in texts) if t]
    if not texts:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="CSV contained no usable text rows.",
        )

    batch = batches_repo.insert(source="csv", total_items=len(texts), filename=file.filename)
    batch_items_repo.insert_many(batch_id=batch["id"], texts=texts)

    logger.info(
        "Batch queued: id=%s source=csv filename=%s items=%d",
        batch["id"], file.filename, len(texts),
    )

    return BatchCreateResponse(batch_id=batch["id"], total_items=len(texts))


# ---------------------------------------------------------------------------
# POST /batches/pdf
# ---------------------------------------------------------------------------

@router.post(
    "/pdf",
    response_model=BatchCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest a batch from an uploaded PDF",
    description=(
        "Extracts the text layer from a PDF and splits it into passages that "
        "fit the model's 256-token window, breaking on paragraph boundaries "
        "rather than on page boundaries. The resulting passages join the same "
        "queue a CSV upload does. "
        "Password-protected files are rejected, and so are scans: a scanned "
        "page has no text layer, and ingesting it would queue blank documents."
    ),
)
async def submit_batch_pdf(file: UploadFile = File(...)) -> BatchCreateResponse:
    raw = await file.read()
    if not raw:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The uploaded file was empty.",
        )

    # Pack by real token count when the tokeniser is up, which it is in the
    # API process. count_tokens() raises if the model has not loaded yet —
    # fall back to the measured character budget rather than failing the
    # upload over it.
    token_counter = None
    max_tokens = None
    try:
        count_tokens("probe")
        token_counter = count_tokens
        max_tokens = settings.MAX_LENGTH
    except Exception:
        logger.warning(
            "[pdf] tokeniser unavailable — packing by character budget instead"
        )

    try:
        result = pdf_extract.extract(
            raw, token_counter=token_counter, max_tokens=max_tokens
        )
    except pdf_extract.PdfExtractionError as exc:
        # Covers encrypted, scanned, malformed and text-free files. All of
        # them are a problem with the upload, not with the server.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        )

    texts = [t for t in (normalize(d) for d in result.documents) if t]
    if not texts:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="This PDF produced no usable text.",
        )

    batch = batches_repo.insert(
        source="pdf", total_items=len(texts), filename=file.filename
    )
    batch_items_repo.insert_many(batch_id=batch["id"], texts=texts)

    logger.info(
        "Batch queued: id=%s source=pdf filename=%s pages=%d chars=%d items=%d",
        batch["id"], file.filename, result.page_count, result.chars_extracted,
        len(texts),
    )

    return BatchCreateResponse(batch_id=batch["id"], total_items=len(texts))


# ---------------------------------------------------------------------------
# GET /batches
# ---------------------------------------------------------------------------

@router.get(
    "",
    response_model=BatchListResponse,
    status_code=status.HTTP_200_OK,
    summary="List past batches, newest first",
)
async def list_batches(
    limit: int = Query(default=20, ge=1, le=100),
) -> BatchListResponse:
    rows = batches_repo.list_all(limit=limit)
    return BatchListResponse(
        batches=[
            BatchSummary(
                batch_id=r["id"],
                source=r["source"],
                status=r["status"],
                total_items=r["total_items"],
                completed_items=r["completed_items"],
                created_at=r["created_at"],
                filename=r.get("filename"),
            )
            for r in rows
        ]
    )


# ---------------------------------------------------------------------------
# GET /batches/{id}
# ---------------------------------------------------------------------------

@router.get(
    "/{batch_id}",
    response_model=BatchStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Poll batch progress",
)
async def get_batch(batch_id: str) -> BatchStatusResponse:
    batch = batches_repo.get(batch_id)
    if batch is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch '{batch_id}' not found.",
        )

    return BatchStatusResponse(
        batch_id=batch["id"],
        status=batch["status"],
        total_items=batch["total_items"],
        completed_items=batch["completed_items"],
    )


# ---------------------------------------------------------------------------
# GET /batches/{id}/items
# ---------------------------------------------------------------------------

@router.get(
    "/{batch_id}/items",
    response_model=BatchItemsResponse,
    status_code=status.HTTP_200_OK,
    summary="List items within a batch (paginated)",
)
async def list_batch_items(
    batch_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
) -> BatchItemsResponse:
    batch = batches_repo.get(batch_id)
    if batch is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch '{batch_id}' not found.",
        )

    offset = (page - 1) * page_size
    rows = batch_items_repo.list_by_batch(batch_id, offset=offset, limit=page_size)

    items = [
        BatchItemOut(
            id=r["id"],
            seq=r["seq"],
            text_content=r["text_content"],
            predicted_label=r.get("predicted_label"),
            confidence=r.get("confidence"),
            status=r["status"],
            validated_label=r.get("validated_label"),
            prediction_id=r.get("prediction_id"),
        )
        for r in rows
    ]

    return BatchItemsResponse(
        batch_id=batch_id,
        total=batch["total_items"],
        page=page,
        page_size=page_size,
        items=items,
    )


# ---------------------------------------------------------------------------
# GET /batches/{id}/export
# ---------------------------------------------------------------------------

@router.get(
    "/{batch_id}/export",
    status_code=status.HTTP_200_OK,
    summary="Download a batch as CSV",
)
async def export_batch(batch_id: str) -> StreamingResponse:
    batch = batches_repo.get(batch_id)
    if batch is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch '{batch_id}' not found.",
        )

    rows = batch_items_repo.list_all_by_batch(batch_id)
    df = pd.DataFrame(
        [
            {
                "seq": r["seq"],
                "text_content": sanitize_csv_cell(r["text_content"]),
                "predicted_label": (
                    sanitize_csv_cell(r["predicted_label"])
                    if r.get("predicted_label") is not None
                    else None
                ),
                "confidence": r.get("confidence"),
                "validated_label": (
                    sanitize_csv_cell(r["validated_label"])
                    if r.get("validated_label") is not None
                    else None
                ),
                "status": r["status"],
            }
            for r in rows
        ]
    )

    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="batch_{batch_id}.csv"'},
    )
