"""
services/mock_db.py — In-memory mock for all Supabase and Neo4j interactions.

Schema-alignment pass (Step 2):
  All tables that correspond to the real CLAUDE.md schema now use exactly
  the same snake_case field names, status enum values, and function signatures
  as the real Supabase columns will.  This makes every repository mock branch
  a thin direct call — no translation layer, no camelCase ↔ snake_case mapping.

Legacy sections below are explicitly marked.  They exist because old code
(supabase_service.py, predict.py, explain.py, test_shap.py, test_prediction.py)
still routes through them.  They will be removed when those callers are
updated in Step 5.  Do NOT modify the legacy sections as part of this
schema-alignment pass — changes there belong in the Step 5 sweep.

Swap strategy:
  supabase_service.py  imports from mock_db  (now)
  neo4j_service.py     imports from mock_db  (now)

  When the real schema is live, replace each function body with the real
  supabase-py client call — signatures stay identical so no router or worker
  needs to change.
"""

import uuid
from copy import deepcopy
from datetime import datetime, timezone
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# In-memory stores — real CLAUDE.md schema tables (snake_case, real enums)
# ---------------------------------------------------------------------------

_predictions: Dict[str, Dict] = {}
"""
predictions: {
    id → {
        id, text_content, predicted_label, label_id (int),
        confidence, all_probabilities {label→prob},
        model_version, created_at
    }
}
Keyed by id.
"""

_xai_jobs: List[Dict] = []
"""
xai_jobs: [{
    id, prediction_id,
    status ('pending'|'processing'|'done'|'failed'),
    token_importance (list|None), summary (str|None),
    attempts (int), locked_at (str|None),
    error_message (str|None), created_at
}]
"""

_annotations: Dict[str, Dict] = {}
"""
annotations: {
    id → {
        id, prediction_id, validated_label, annotator_id (str|None),
        status ('pending'|'validated'|'rejected'),
        has_conflict (bool),   ← separate dimension, not a fourth status
        annotated_at,
        document_id (str|None) ← NOT in real schema; kept only so the
                                   AnnotationListResponse can return document_id
                                   until GET /annotate is rewritten to join
                                   through predictions.  Remove in Step 5.
    }
}
Arjun needs: annotations.has_conflict boolean not null default false
"""

_model_versions: Dict[str, Dict] = {}
"""
model_versions: {
    id → {
        id, version_number, accuracy, f1_per_class {label→f1},
        file_path, dataset_version_id (str|None),
        is_active (bool), trained_at
    }
}
Keyed by id.  is_active is the rollback mechanism.
"""

_dataset_versions: Dict[str, Dict] = {}
"""
dataset_versions: {
    id → {id, version_id (str tag), sample_count, label_distribution {label→n}, created_at}
}
"""

_retrain_jobs: List[Dict] = []
"""
retrain_jobs: [{
    id, status ('pending'|'running'|'complete'|'failed'),
    triggered_by (str|None), triggered_at, completed_at (str|None),
    model_version_id (str|None),
    notes (str|None),          ← not in real schema; kept for router compat
    min_annotations (int|None) ← not in real schema; kept for router compat
}]
"""


# ===========================================================================
# PREDICTIONS
# ===========================================================================

def create_prediction(
    text_content: str,
    predicted_label: str,
    label_id: int,
    confidence: float,
    all_probabilities: Dict[str, float],
    model_version: str,
) -> Dict:
    """Insert a predictions row and return it."""
    pred = {
        "id": _new_id(),
        "text_content": text_content,
        "predicted_label": predicted_label,
        "label_id": label_id,
        "confidence": confidence,
        "all_probabilities": all_probabilities,
        "model_version": model_version,
        "created_at": _now_iso(),
    }
    _predictions[pred["id"]] = pred
    return deepcopy(pred)


def get_prediction(prediction_id: str) -> Optional[Dict]:
    """Fetch a prediction by id."""
    return deepcopy(_predictions.get(prediction_id))


def list_predictions(predicted_label: Optional[str] = None) -> List[Dict]:
    """List predictions, optionally filtered by predicted_label."""
    rows = list(_predictions.values())
    if predicted_label:
        rows = [r for r in rows if r["predicted_label"] == predicted_label]
    return [deepcopy(r) for r in rows]


def list_low_confidence_predictions(threshold: float) -> List[Dict]:
    """Return predictions with confidence < threshold, newest first."""
    rows = [p for p in _predictions.values() if p["confidence"] < threshold]
    return sorted([deepcopy(r) for r in rows], key=lambda r: r["created_at"], reverse=True)


# ===========================================================================
# XAI JOBS
# ===========================================================================

def enqueue_xai_job(prediction_id: str) -> Dict:
    """Push a new xai_job row with status='pending' and return it."""
    job = {
        "id": _new_id(),
        "prediction_id": prediction_id,
        "status": "pending",
        "token_importance": None,
        "summary": None,
        "attempts": 0,
        "locked_at": None,
        "error_message": None,
        "created_at": _now_iso(),
    }
    _xai_jobs.append(job)
    return deepcopy(job)


def get_xai_job(job_id: str) -> Optional[Dict]:
    """Fetch an xai_jobs row by id."""
    for job in _xai_jobs:
        if job["id"] == job_id:
            return deepcopy(job)
    return None


def get_xai_job_by_prediction(prediction_id: str) -> Optional[Dict]:
    """Fetch the xai_job for a given prediction_id (at most one per prediction)."""
    for job in _xai_jobs:
        if job["prediction_id"] == prediction_id:
            return deepcopy(job)
    return None


def update_xai_job(job_id: str, **fields) -> Optional[Dict]:
    """
    Generic partial update on an xai_jobs row.

    Accepts any real schema field: status, token_importance, summary,
    attempts, locked_at, error_message.
    """
    for job in _xai_jobs:
        if job["id"] == job_id:
            job.update(fields)
            return deepcopy(job)
    return None


def get_pending_xai_jobs() -> List[Dict]:
    """Return all xai_jobs rows with status='pending'."""
    return [deepcopy(j) for j in _xai_jobs if j["status"] == "pending"]


# ===========================================================================
# ANNOTATIONS
# ===========================================================================

def create_annotation(
    prediction_id: str,
    validated_label: str,
    annotator_id: Optional[str],
    status: str = "pending",
    document_id: Optional[str] = None,
) -> Dict:
    """
    Insert an annotations row and return it.

    document_id is not in the real schema.  It is accepted here solely so
    that GET /annotate can return it in AnnotationListItem.  Remove this
    parameter in Step 5 when the GET route is rewritten to join through
    predictions.
    """
    ann = {
        "id": _new_id(),
        "prediction_id": prediction_id,
        "validated_label": validated_label,
        "annotator_id": annotator_id,
        "status": status,
        "has_conflict": False,
        "annotated_at": _now_iso(),
        "document_id": document_id,
    }
    _annotations[ann["id"]] = ann
    return deepcopy(ann)


def get_annotation(annotation_id: str) -> Optional[Dict]:
    """Fetch an annotation by id."""
    return deepcopy(_annotations.get(annotation_id))


def list_annotations(
    prediction_id: Optional[str] = None,
    annotator_id: Optional[str] = None,
    status: Optional[str] = None,
    has_conflict: Optional[bool] = None,
) -> List[Dict]:
    """List annotations with optional AND-combined filters."""
    rows = list(_annotations.values())
    if prediction_id:
        rows = [r for r in rows if r["prediction_id"] == prediction_id]
    if annotator_id:
        rows = [r for r in rows if r.get("annotator_id") == annotator_id]
    if status:
        rows = [r for r in rows if r["status"] == status]
    if has_conflict is not None:
        rows = [r for r in rows if r.get("has_conflict", False) == has_conflict]
    return [deepcopy(r) for r in rows]


def update_annotation_status(annotation_id: str, new_status: str) -> Optional[Dict]:
    """Update the status field of an annotation."""
    ann = _annotations.get(annotation_id)
    if ann is None:
        return None
    ann["status"] = new_status
    return deepcopy(ann)


def set_annotation_has_conflict(annotation_id: str, has_conflict: bool) -> Optional[Dict]:
    """
    Set the has_conflict flag on an annotation.

    Independent of status — a row can be 'validated' and has_conflict=True
    simultaneously.  Real Supabase path: UPDATE annotations SET has_conflict=…
    """
    ann = _annotations.get(annotation_id)
    if ann is None:
        return None
    ann["has_conflict"] = has_conflict
    return deepcopy(ann)


def count_validated_annotations() -> int:
    """Return the count of annotations with status='validated'."""
    return sum(1 for a in _annotations.values() if a["status"] == "validated")


def count_annotations_by_status() -> Dict[str, int]:
    """Return a dict of status -> count across all annotations."""
    counts: Dict[str, int] = {}
    for a in _annotations.values():
        counts[a["status"]] = counts.get(a["status"], 0) + 1
    return counts


# ===========================================================================
# MODEL VERSIONS
# ===========================================================================

def create_model_version(
    version_number: str,
    accuracy: float,
    f1_per_class: Dict[str, float],
    file_path: str,
    dataset_version_id: Optional[str] = None,
    is_active: bool = False,
) -> Dict:
    """Insert a model_versions row and return it."""
    mv = {
        "id": _new_id(),
        "version_number": version_number,
        "accuracy": accuracy,
        "f1_per_class": f1_per_class,
        "file_path": file_path,
        "dataset_version_id": dataset_version_id,
        "is_active": is_active,
        "trained_at": _now_iso(),
    }
    _model_versions[mv["id"]] = mv
    return deepcopy(mv)


def get_active_model_version() -> Optional[Dict]:
    """Return the model_versions row where is_active=True, or None."""
    for mv in _model_versions.values():
        if mv.get("is_active"):
            return deepcopy(mv)
    return None


def set_active_model_version(version_id: str) -> Optional[Dict]:
    """
    Flip is_active: deactivate every version, then activate version_id.

    In mock mode this is two sequential in-memory writes (safe because mock_db
    is single-threaded).  The real Supabase path uses the
    activate_model_version(target_id uuid) RPC to do this in one transaction.
    """
    for mv in _model_versions.values():
        mv["is_active"] = False
    target = _model_versions.get(version_id)
    if target is None:
        return None
    target["is_active"] = True
    return deepcopy(target)


def list_model_versions() -> List[Dict]:
    """Return all model versions ordered by trained_at descending."""
    return sorted(
        [deepcopy(mv) for mv in _model_versions.values()],
        key=lambda m: m.get("trained_at", ""),
        reverse=True,
    )


# ===========================================================================
# DATASET VERSIONS
# ===========================================================================

def create_dataset_version(
    version_id: str,
    sample_count: int,
    label_distribution: Dict[str, int],
) -> Dict:
    """Insert a dataset_versions row and return it."""
    dv = {
        "id": _new_id(),
        "version_id": version_id,
        "sample_count": sample_count,
        "label_distribution": label_distribution,
        "created_at": _now_iso(),
    }
    _dataset_versions[dv["id"]] = dv
    return deepcopy(dv)


def get_dataset_version(dv_id: str) -> Optional[Dict]:
    """Fetch a dataset_versions row by id."""
    return deepcopy(_dataset_versions.get(dv_id))


# ===========================================================================
# RETRAIN JOBS  (status: 'pending'|'running'|'complete'|'failed' per CLAUDE.md)
# ===========================================================================

def enqueue_retrain_job(
    triggered_by: Optional[str] = None,
    notes: Optional[str] = None,
    min_annotations: Optional[int] = None,
) -> Dict:
    """Push a new retraining job with status='pending' and return it."""
    job = {
        "id": _new_id(),
        "status": "pending",
        "triggered_by": triggered_by,
        "triggered_at": _now_iso(),
        "completed_at": None,
        "model_version_id": None,
        # Not in real schema — kept so the router can pass them through:
        "notes": notes,
        "min_annotations": min_annotations,
    }
    _retrain_jobs.append(job)
    return deepcopy(job)


def get_retrain_job(job_id: str) -> Optional[Dict]:
    """Fetch a retrain job by id."""
    for job in _retrain_jobs:
        if job["id"] == job_id:
            return deepcopy(job)
    return None


def get_latest_retrain_job() -> Optional[Dict]:
    """Return the most recently created retraining job, or None."""
    if not _retrain_jobs:
        return None
    return deepcopy(_retrain_jobs[-1])


def update_retrain_job(
    job_id: str,
    status: str,
    model_version_id: Optional[str] = None,
    completed_at: Optional[str] = None,
) -> Optional[Dict]:
    """Update status and completion fields of a retrain job."""
    for job in _retrain_jobs:
        if job["id"] == job_id:
            job["status"] = status
            if model_version_id is not None:
                job["model_version_id"] = model_version_id
            if completed_at is not None:
                job["completed_at"] = completed_at
            return deepcopy(job)
    return None


# ===========================================================================
# BATCHES / BATCH_ITEMS
# ===========================================================================

_batches: Dict[str, Dict] = {}
"""
batches: {
    id → {
        id, source ('paste'|'csv'|'pdf'), filename (str|None),
        status ('pending'|'processing'|'done'|'failed'),
        total_items, completed_items, created_at
    }
}
"""

_batch_items: Dict[str, Dict] = {}
"""
batch_items: {
    id → {
        id, batch_id, seq, text_content,
        predicted_label (str|None), label_id (int|None), confidence (float|None),
        all_probabilities (dict|None), validated_label (str|None),
        status ('pending'|'processing'|'classified'|'validated'|'failed'),
        attempts, locked_at (str|None), error_message (str|None),
        prediction_id (str|None), created_at
    }
}
"""


def create_batch(
    source: str,
    total_items: int,
    filename: Optional[str] = None,
) -> Dict:
    """Insert a batches row (status='pending', completed_items=0) and return it."""
    batch = {
        "id": _new_id(),
        "source": source,
        "filename": filename,
        "status": "pending",
        "total_items": total_items,
        "completed_items": 0,
        "created_at": _now_iso(),
    }
    _batches[batch["id"]] = batch
    return deepcopy(batch)


def get_batch(batch_id: str) -> Optional[Dict]:
    """Fetch a batches row by id."""
    return deepcopy(_batches.get(batch_id))


def update_batch_status(batch_id: str, status: str) -> Optional[Dict]:
    """Flip the status field on a batch."""
    batch = _batches.get(batch_id)
    if batch is None:
        return None
    batch["status"] = status
    return deepcopy(batch)


def increment_batch_completed(batch_id: str) -> Optional[Dict]:
    """Atomically increment completed_items by 1 (safe: mock_db is single-threaded)."""
    batch = _batches.get(batch_id)
    if batch is None:
        return None
    batch["completed_items"] += 1
    return deepcopy(batch)


def list_batches(limit: Optional[int] = None) -> List[Dict]:
    """Return batches ordered by created_at descending, optionally capped at limit."""
    rows = sorted(
        [deepcopy(b) for b in _batches.values()],
        key=lambda b: b["created_at"],
        reverse=True,
    )
    return rows[:limit] if limit is not None else rows


def create_batch_items(batch_id: str, texts: List[str]) -> List[Dict]:
    """
    Bulk-insert one batch_items row per text (seq 1-based) and return all
    created rows in order.

    seq continues from the highest existing seq for this batch_id (0 if
    none exist yet) rather than always restarting at 1, so a second insert
    call into the same batch doesn't collide with or reorder earlier items.
    """
    existing_seqs = [
        item["seq"] for item in _batch_items.values() if item["batch_id"] == batch_id
    ]
    start_seq = max(existing_seqs, default=0) + 1

    rows = []
    for seq, text in enumerate(texts, start=start_seq):
        item = {
            "id": _new_id(),
            "batch_id": batch_id,
            "seq": seq,
            "text_content": text,
            "predicted_label": None,
            "label_id": None,
            "confidence": None,
            "all_probabilities": None,
            "validated_label": None,
            "status": "pending",
            "attempts": 0,
            "locked_at": None,
            "error_message": None,
            "prediction_id": None,
            "created_at": _now_iso(),
        }
        _batch_items[item["id"]] = item
        rows.append(deepcopy(item))
    return rows


def get_batch_item(item_id: str) -> Optional[Dict]:
    """Fetch a single batch_items row by id."""
    return deepcopy(_batch_items.get(item_id))


def list_batch_items(batch_id: str, offset: int = 0, limit: int = 50) -> List[Dict]:
    """Paginated listing of items within a batch, ordered by seq."""
    rows = sorted(
        [b for b in _batch_items.values() if b["batch_id"] == batch_id],
        key=lambda b: b["seq"],
    )
    return [deepcopy(r) for r in rows[offset : offset + limit]]


def list_all_batch_items(batch_id: str) -> List[Dict]:
    """Return every item in a batch, ordered by seq (no pagination — used by export)."""
    rows = sorted(
        [b for b in _batch_items.values() if b["batch_id"] == batch_id],
        key=lambda b: b["seq"],
    )
    return [deepcopy(r) for r in rows]


# ===========================================================================
# LEGACY: users
# NOT in real schema.  Do not modify as part of the schema-alignment pass.
# ===========================================================================

_users: Dict[str, Dict] = {}


def create_user(email: str, role: str) -> Dict:
    user = {"userId": _new_id(), "email": email, "role": role}
    _users[user["userId"]] = user
    return deepcopy(user)


def get_user(user_id: str) -> Optional[Dict]:
    return deepcopy(_users.get(user_id))


def get_user_by_email(email: str) -> Optional[Dict]:
    for u in _users.values():
        if u["email"] == email:
            return deepcopy(u)
    return None


# ===========================================================================
# LEGACY: legal_documents
# NOT in real schema — predictions store text_content directly.
# Do not modify as part of the schema-alignment pass.
# ===========================================================================

_legal_documents: Dict[str, Dict] = {}


def create_document(text_content: str, status: str = "pending") -> Dict:
    doc = {
        "documentId": _new_id(),
        "textContent": text_content,
        "status": status,
        "createdAt": _now_iso(),
    }
    _legal_documents[doc["documentId"]] = doc
    return deepcopy(doc)


def get_document(document_id: str) -> Optional[Dict]:
    return deepcopy(_legal_documents.get(document_id))


def update_document_status(document_id: str, status: str) -> Optional[Dict]:
    doc = _legal_documents.get(document_id)
    if doc is None:
        return None
    doc["status"] = status
    return deepcopy(doc)


# ===========================================================================
# LEGACY: explanations
# NOT in real schema — token_importance and summary live in xai_jobs.
# Do not modify as part of the schema-alignment pass.
# ===========================================================================

_explanations: Dict[str, Dict] = {}


def create_explanation(prediction_id: str) -> Dict:
    exp = {
        "explanationId": _new_id(),
        "predictionId": prediction_id,
        "shapValues": None,
        "generatedAt": None,
        "status": "pending",
    }
    _explanations[exp["explanationId"]] = exp
    return deepcopy(exp)


def get_explanation(explanation_id: str) -> Optional[Dict]:
    return deepcopy(_explanations.get(explanation_id))


def get_explanation_by_prediction(prediction_id: str) -> Optional[Dict]:
    for exp in _explanations.values():
        if exp["predictionId"] == prediction_id:
            return deepcopy(exp)
    return None


def update_explanation(
    explanation_id: str,
    shap_values: List[Dict],
    status: str = "completed",
) -> Optional[Dict]:
    exp = _explanations.get(explanation_id)
    if exp is None:
        return None
    exp["shapValues"] = shap_values
    exp["generatedAt"] = _now_iso()
    exp["status"] = status
    return deepcopy(exp)


# ===========================================================================
# LEGACY: prediction_jobs queue
# NOT in real schema — the batch worker uses batch_items (via Supabase RPC).
# Do not modify as part of the schema-alignment pass.
# ===========================================================================

_prediction_jobs: List[Dict] = []


def enqueue_prediction_job(text_content: str) -> Dict:
    job = {
        "jobId": _new_id(),
        "textContent": text_content,
        "status": "pending",
        "createdAt": _now_iso(),
        "result": None,
    }
    _prediction_jobs.append(job)
    return deepcopy(job)


def get_prediction_job(job_id: str) -> Optional[Dict]:
    for job in _prediction_jobs:
        if job["jobId"] == job_id:
            return deepcopy(job)
    return None


def update_prediction_job(
    job_id: str,
    status: str,
    result: Optional[Dict] = None,
) -> Optional[Dict]:
    for job in _prediction_jobs:
        if job["jobId"] == job_id:
            job["status"] = status
            if result is not None:
                job["result"] = result
            return deepcopy(job)
    return None


def get_pending_prediction_jobs() -> List[Dict]:
    return [deepcopy(j) for j in _prediction_jobs if j["status"] == "pending"]


# ===========================================================================
# NEO4J ONTOLOGY MOCK  (static dict — one entry per legal label)
# ===========================================================================

ONTOLOGY: Dict[str, Dict] = {
    "Contract Law": {
        "label": "Contract Law",
        "description": "Governs agreements between parties, including formation, enforcement, and breach.",
        "parent": None,
        "children": [],
        "related": ["Corporate / Company Law", "Civil Procedure / Other"],
    },
    "Criminal Law": {
        "label": "Criminal Law",
        "description": "Covers offences against the state or public, prosecution, and sentencing.",
        "parent": None,
        "children": [],
        "related": ["Civil Procedure / Other"],
    },
    "Constitutional Law": {
        "label": "Constitutional Law",
        "description": "Deals with the interpretation and application of a country's constitution.",
        "parent": None,
        "children": [],
        "related": ["Criminal Law", "Civil Procedure / Other"],
    },
    "Corporate / Company Law": {
        "label": "Corporate / Company Law",
        "description": "Regulates the formation, governance, and dissolution of companies.",
        "parent": None,
        "children": [],
        "related": ["Contract Law", "Taxation Law"],
    },
    "Property / Real Estate Law": {
        "label": "Property / Real Estate Law",
        "description": "Covers ownership, transfer, and use of real and personal property.",
        "parent": None,
        "children": [],
        "related": ["Contract Law", "Family Law"],
    },
    "Family Law": {
        "label": "Family Law",
        "description": "Governs marriage, divorce, child custody, and related domestic matters.",
        "parent": None,
        "children": [],
        "related": ["Property / Real Estate Law", "Civil Procedure / Other"],
    },
    "Labour & Employment Law": {
        "label": "Labour & Employment Law",
        "description": "Regulates the employer-employee relationship, wages, and workplace rights.",
        "parent": None,
        "children": [],
        "related": ["Contract Law", "Civil Procedure / Other"],
    },
    "Intellectual Property Law": {
        "label": "Intellectual Property Law",
        "description": "Covers patents, trademarks, copyrights, and trade secrets.",
        "parent": None,
        "children": [],
        "related": ["Corporate / Company Law", "Contract Law"],
    },
    "Taxation Law": {
        "label": "Taxation Law",
        "description": "Deals with tax obligations, disputes, and compliance for individuals and entities.",
        "parent": None,
        "children": [],
        "related": ["Corporate / Company Law", "Civil Procedure / Other"],
    },
    "Civil Procedure / Other": {
        "label": "Civil Procedure / Other",
        "description": "Procedural rules for civil litigation and miscellaneous legal matters.",
        "parent": None,
        "children": [],
        "related": [
            "Contract Law", "Family Law", "Labour & Employment Law",
            "Property / Real Estate Law",
        ],
    },
}


def get_ontology_nodes() -> List[Dict]:
    return [deepcopy(n) for n in ONTOLOGY.values()]


def get_ontology_node(label: str) -> Optional[Dict]:
    node = ONTOLOGY.get(label)
    return deepcopy(node) if node else None


def validate_label(label: str) -> bool:
    return label in ONTOLOGY


def validate_label_with_parent(label: str, parent_label: str) -> bool:
    node = ONTOLOGY.get(label)
    if node is None:
        return False
    if node.get("parent") == parent_label:
        return True
    return parent_label in node.get("related", [])
