"""
services/mock_db.py — In-memory mock for all Supabase and Neo4j interactions.

Simulates every DB table as a Python dict/list so the full prediction →
active-learning → SHAP → annotation flow works end-to-end with zero external
dependencies.

Swap strategy:
  supabase_service.py  imports from mock_db  (now)
  neo4j_service.py     imports from mock_db  (now)

  When the real schemas are ready, replace the import in each service file
  with the real client calls — no other code needs to change.
"""

import uuid
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    """Generate a short unique ID."""
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# In-memory stores  (one dict per Supabase table, keyed by primary-key string)
# ---------------------------------------------------------------------------

_users: Dict[str, Dict] = {}
"""
users: { userId → {userId, email, role} }
"""

_legal_documents: Dict[str, Dict] = {}
"""
legal_documents: { documentId → {documentId, textContent, status} }
"""

_predictions: Dict[str, Dict] = {}
"""
predictions: {
    predictionId → {
        predictionId, documentId, predictedLabel, confidenceScore,
        probabilityDistribution, modelVersionId, routingDecision,
        entropy, margin, createdAt
    }
}
"""

_annotations: Dict[str, Dict] = {}
"""
annotations: {
    annotationId → {
        annotationId, documentId, userId, predictionId,
        finalLabel, annotationStatus, action, notes, annotatedAt
    }
}
"""

_explanations: Dict[str, Dict] = {}
"""
explanations: {
    explanationId → {
        explanationId, predictionId, shapValues, generatedAt, status
    }
}
"""

_dataset_versions: Dict[str, Dict] = {}
"""
dataset_versions: { versionId → {versionId, createdAt, sampleCount} }
"""

_model_versions: Dict[str, Dict] = {}
"""
model_versions: { modelId → {modelId, accuracy, trainedAt, filePath} }
"""

# ---------------------------------------------------------------------------
# Queue stores  (list of job dicts, status tracked inline)
# ---------------------------------------------------------------------------

_prediction_jobs: List[Dict] = []
"""
prediction_jobs queue.
Each job: { jobId, textContent, status, createdAt, result }
status: "pending" | "processing" | "completed" | "failed"
"""

_xai_jobs: List[Dict] = []
"""
xai_jobs queue.
Each job: { jobId, predictionId, status, createdAt, result }
status: "pending" | "processing" | "completed" | "failed"
"""

_retrain_jobs: List[Dict] = []
"""
retrain_jobs queue (not a Supabase table per spec, but needed for /retrain).
Each job: { jobId, status, notes, minAnnotations, createdAt, completedAt,
            modelVersion, accuracy }
"""

# ---------------------------------------------------------------------------
# Neo4j ontology mock  (static dict — one entry per legal label)
# ---------------------------------------------------------------------------

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


# ===========================================================================
# USERS
# ===========================================================================

def create_user(email: str, role: str) -> Dict:
    """Insert a new user row and return it."""
    user = {"userId": _new_id(), "email": email, "role": role}
    _users[user["userId"]] = user
    return deepcopy(user)


def get_user(user_id: str) -> Optional[Dict]:
    """Fetch a user by ID. Returns None if not found."""
    return deepcopy(_users.get(user_id))


def get_user_by_email(email: str) -> Optional[Dict]:
    """Fetch a user by email. Returns None if not found."""
    for u in _users.values():
        if u["email"] == email:
            return deepcopy(u)
    return None


# ===========================================================================
# LEGAL DOCUMENTS
# ===========================================================================

def create_document(text_content: str, status: str = "pending") -> Dict:
    """
    Insert a new legal_documents row.

    status is usually 'pending' until a prediction is made.
    """
    doc = {
        "documentId": _new_id(),
        "textContent": text_content,
        "status": status,
        "createdAt": _now_iso(),
    }
    _legal_documents[doc["documentId"]] = doc
    return deepcopy(doc)


def get_document(document_id: str) -> Optional[Dict]:
    """Fetch a document by ID."""
    return deepcopy(_legal_documents.get(document_id))


def update_document_status(document_id: str, status: str) -> Optional[Dict]:
    """Update the status field of a document. Returns updated doc or None."""
    doc = _legal_documents.get(document_id)
    if doc is None:
        return None
    doc["status"] = status
    return deepcopy(doc)


# ===========================================================================
# PREDICTIONS
# ===========================================================================

def create_prediction(
    document_id: str,
    predicted_label: str,
    confidence_score: float,
    probability_distribution: Dict[str, float],
    routing_decision: str,
    entropy: float,
    margin: float,
    model_version_id: Optional[str] = None,
) -> Dict:
    """
    Insert a new predictions row and return it.

    probability_distribution: dict mapping label name → softmax probability.
    """
    pred = {
        "predictionId": _new_id(),
        "documentId": document_id,
        "predictedLabel": predicted_label,
        "confidenceScore": confidence_score,
        "probabilityDistribution": probability_distribution,
        "routingDecision": routing_decision,
        "entropy": entropy,
        "margin": margin,
        "modelVersionId": model_version_id,
        "createdAt": _now_iso(),
    }
    _predictions[pred["predictionId"]] = pred
    return deepcopy(pred)


def get_prediction(prediction_id: str) -> Optional[Dict]:
    """Fetch a prediction by ID."""
    return deepcopy(_predictions.get(prediction_id))


def list_predictions(document_id: Optional[str] = None) -> List[Dict]:
    """List all predictions, optionally filtered by document_id."""
    rows = list(_predictions.values())
    if document_id:
        rows = [r for r in rows if r["documentId"] == document_id]
    return [deepcopy(r) for r in rows]


# ===========================================================================
# ANNOTATIONS
# ===========================================================================

def create_annotation(
    document_id: str,
    prediction_id: str,
    user_id: Optional[str],
    final_label: str,
    action: str,
    annotation_status: str,
    notes: Optional[str] = None,
) -> Dict:
    """Insert a new annotations row and return it."""
    ann = {
        "annotationId": _new_id(),
        "documentId": document_id,
        "predictionId": prediction_id,
        "userId": user_id,
        "finalLabel": final_label,
        "action": action,
        "annotationStatus": annotation_status,
        "notes": notes,
        "annotatedAt": _now_iso(),
    }
    _annotations[ann["annotationId"]] = ann
    return deepcopy(ann)


def get_annotation(annotation_id: str) -> Optional[Dict]:
    """Fetch an annotation by ID."""
    return deepcopy(_annotations.get(annotation_id))


def list_annotations(
    document_id: Optional[str] = None,
    user_id: Optional[str] = None,
    status: Optional[str] = None,
) -> List[Dict]:
    """
    List annotations with optional filters.

    Filters are AND-combined — only rows matching all supplied filters are returned.
    """
    rows = list(_annotations.values())
    if document_id:
        rows = [r for r in rows if r["documentId"] == document_id]
    if user_id:
        rows = [r for r in rows if r["userId"] == user_id]
    if status:
        rows = [r for r in rows if r["annotationStatus"] == status]
    return [deepcopy(r) for r in rows]


def update_annotation_status(annotation_id: str, new_status: str) -> Optional[Dict]:
    """Update the status of an existing annotation (e.g. flag as 'conflict')."""
    ann = _annotations.get(annotation_id)
    if ann is None:
        return None
    ann["annotationStatus"] = new_status
    return deepcopy(ann)


def count_validated_annotations() -> int:
    """Return how many annotations have status 'accepted' or 'modified'."""
    return sum(
        1 for a in _annotations.values()
        if a["annotationStatus"] in ("accepted", "modified")
    )


# ===========================================================================
# EXPLANATIONS
# ===========================================================================

def create_explanation(prediction_id: str) -> Dict:
    """
    Insert a new explanations row with status='pending'.

    The XAI worker will call update_explanation() once SHAP is done.
    """
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
    """Fetch an explanation by its own ID."""
    return deepcopy(_explanations.get(explanation_id))


def get_explanation_by_prediction(prediction_id: str) -> Optional[Dict]:
    """Fetch the explanation for a given prediction_id (at most one per prediction)."""
    for exp in _explanations.values():
        if exp["predictionId"] == prediction_id:
            return deepcopy(exp)
    return None


def update_explanation(
    explanation_id: str,
    shap_values: List[Dict],
    status: str = "completed",
) -> Optional[Dict]:
    """
    Store the computed SHAP values and mark the explanation as completed.

    shap_values: list of {token, importance} dicts produced by shap_service.
    """
    exp = _explanations.get(explanation_id)
    if exp is None:
        return None
    exp["shapValues"] = shap_values
    exp["generatedAt"] = _now_iso()
    exp["status"] = status
    return deepcopy(exp)


# ===========================================================================
# PREDICTION JOBS QUEUE
# ===========================================================================

def enqueue_prediction_job(text_content: str) -> Dict:
    """Push a new prediction job onto the queue and return it."""
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
    """Fetch a prediction job by ID."""
    for job in _prediction_jobs:
        if job["jobId"] == job_id:
            return deepcopy(job)
    return None


def update_prediction_job(
    job_id: str,
    status: str,
    result: Optional[Dict] = None,
) -> Optional[Dict]:
    """Update the status and optional result of a prediction job."""
    for job in _prediction_jobs:
        if job["jobId"] == job_id:
            job["status"] = status
            if result is not None:
                job["result"] = result
            return deepcopy(job)
    return None


def get_pending_prediction_jobs() -> List[Dict]:
    """Return all jobs with status='pending' (consumed by the worker)."""
    return [deepcopy(j) for j in _prediction_jobs if j["status"] == "pending"]


# ===========================================================================
# XAI JOBS QUEUE
# ===========================================================================

def enqueue_xai_job(prediction_id: str) -> Dict:
    """Push a new SHAP job onto the xai_jobs queue and return it."""
    job = {
        "jobId": _new_id(),
        "predictionId": prediction_id,
        "status": "pending",
        "createdAt": _now_iso(),
        "result": None,
    }
    _xai_jobs.append(job)
    return deepcopy(job)


def get_xai_job(job_id: str) -> Optional[Dict]:
    """Fetch an XAI job by ID."""
    for job in _xai_jobs:
        if job["jobId"] == job_id:
            return deepcopy(job)
    return None


def update_xai_job(
    job_id: str,
    status: str,
    result: Optional[Dict] = None,
) -> Optional[Dict]:
    """Update the status and optional result of an XAI job."""
    for job in _xai_jobs:
        if job["jobId"] == job_id:
            job["status"] = status
            if result is not None:
                job["result"] = result
            return deepcopy(job)
    return None


def get_pending_xai_jobs() -> List[Dict]:
    """Return all XAI jobs with status='pending' (consumed by the XAI worker)."""
    return [deepcopy(j) for j in _xai_jobs if j["status"] == "pending"]


# ===========================================================================
# RETRAIN JOBS
# ===========================================================================

def enqueue_retrain_job(notes: Optional[str], min_annotations: int) -> Dict:
    """Push a new retraining job and return it."""
    job = {
        "jobId": _new_id(),
        "status": "pending",
        "notes": notes,
        "minAnnotations": min_annotations,
        "createdAt": _now_iso(),
        "startedAt": None,
        "completedAt": None,
        "modelVersion": None,
        "accuracy": None,
    }
    _retrain_jobs.append(job)
    return deepcopy(job)


def get_latest_retrain_job() -> Optional[Dict]:
    """Return the most recently created retraining job, or None."""
    if not _retrain_jobs:
        return None
    return deepcopy(_retrain_jobs[-1])


def get_retrain_job(job_id: str) -> Optional[Dict]:
    """Fetch a retrain job by ID."""
    for job in _retrain_jobs:
        if job["jobId"] == job_id:
            return deepcopy(job)
    return None


def update_retrain_job(
    job_id: str,
    status: str,
    model_version: Optional[str] = None,
    accuracy: Optional[float] = None,
    completed_at: Optional[str] = None,
) -> Optional[Dict]:
    """Update the status and result fields of a retrain job."""
    for job in _retrain_jobs:
        if job["jobId"] == job_id:
            job["status"] = status
            if model_version is not None:
                job["modelVersion"] = model_version
            if accuracy is not None:
                job["accuracy"] = accuracy
            if completed_at is not None:
                job["completedAt"] = completed_at
            return deepcopy(job)
    return None


# ===========================================================================
# DATASET & MODEL VERSIONS
# ===========================================================================

def create_dataset_version(sample_count: int) -> Dict:
    """Record a new dataset snapshot used for retraining."""
    dv = {
        "versionId": _new_id(),
        "createdAt": _now_iso(),
        "sampleCount": sample_count,
    }
    _dataset_versions[dv["versionId"]] = dv
    return deepcopy(dv)


def create_model_version(accuracy: float, file_path: str) -> Dict:
    """Record a newly trained model version."""
    mv = {
        "modelId": _new_id(),
        "accuracy": accuracy,
        "trainedAt": _now_iso(),
        "filePath": file_path,
    }
    _model_versions[mv["modelId"]] = mv
    return deepcopy(mv)


def get_latest_model_version() -> Optional[Dict]:
    """Return the most recently trained model version, or None."""
    if not _model_versions:
        return None
    latest = max(_model_versions.values(), key=lambda m: m["trainedAt"])
    return deepcopy(latest)


# ===========================================================================
# NEO4J ONTOLOGY
# ===========================================================================

def get_ontology_nodes() -> List[Dict]:
    """Return all ontology nodes as a list (mirrors a Neo4j MATCH query)."""
    return [deepcopy(n) for n in ONTOLOGY.values()]


def get_ontology_node(label: str) -> Optional[Dict]:
    """Fetch a single ontology node by label name."""
    node = ONTOLOGY.get(label)
    return deepcopy(node) if node else None


def validate_label(label: str) -> bool:
    """Return True if the label exists in the ontology."""
    return label in ONTOLOGY


def validate_label_with_parent(label: str, parent_label: str) -> bool:
    """
    Return True if label exists AND parent_label is listed as a related or
    parent node for that label.

    In the current flat ontology all top-level nodes have parent=None; this
    check uses the 'related' list as a proxy for hierarchical proximity.
    """
    node = ONTOLOGY.get(label)
    if node is None:
        return False
    # Direct parent match
    if node.get("parent") == parent_label:
        return True
    # Related-node match (used while ontology is flat)
    return parent_label in node.get("related", [])
