"""
models/response_models.py — Pydantic schemas for all outgoing API responses.

Every router uses these as its response_model so the OpenAPI docs stay accurate
and clients get consistent, typed payloads.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared / reusable types
# ---------------------------------------------------------------------------

# The 10 legal domain labels (mirrors id2label in the model)
LEGAL_LABELS = [
    "Contract Law",
    "Criminal Law",
    "Constitutional Law",
    "Corporate / Company Law",
    "Property / Real Estate Law",
    "Family Law",
    "Labour & Employment Law",
    "Intellectual Property Law",
    "Taxation Law",
    "Civil Procedure / Other",
]

RoutingDecision = Literal["AUTO_ACCEPT", "NEEDS_EXPLANATION", "ROUTE_TO_REVIEWER"]
AnnotationStatus = Literal["pending", "accepted", "modified", "rejected", "conflict"]
JobStatus = Literal["pending", "processing", "completed", "failed"]


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------

class PredictResponse(BaseModel):
    """
    Returned immediately from POST /predict.

    Contains the model's top label, confidence, the full probability
    distribution over all 10 classes, and the active-learning routing decision.
    """

    prediction_id: str = Field(..., description="Unique ID for this prediction (stored in mock DB).")
    document_id: Optional[str] = Field(default=None, description="Document ID if one was supplied in the request.")
    predicted_label: str = Field(..., description="Top-1 label from InLegalBERT.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Max softmax probability (confidence score).")
    probabilities: Dict[str, float] = Field(
        ...,
        description="Full softmax distribution keyed by label name.",
    )
    routing_decision: RoutingDecision = Field(
        ...,
        description=(
            "AUTO_ACCEPT        → confidence > CONFIDENCE_HIGH, no SHAP needed.\n"
            "NEEDS_EXPLANATION  → medium confidence, async SHAP job queued.\n"
            "ROUTE_TO_REVIEWER  → low confidence, routed to human reviewer + SHAP queued."
        ),
    )
    xai_job_id: Optional[str] = Field(
        default=None,
        description="ID of the queued SHAP job (None when routing_decision is AUTO_ACCEPT).",
    )
    entropy: float = Field(..., description="Entropy of the probability distribution.")
    margin: float = Field(..., description="Difference between top-2 probabilities.")


class PredictJobResponse(BaseModel):
    """
    Returned from GET /predict/{job_id} — status of a queued prediction job.
    """

    job_id: str
    status: JobStatus
    result: Optional[PredictResponse] = Field(
        default=None,
        description="Populated once status == 'completed'.",
    )


# ---------------------------------------------------------------------------
# /explain
# ---------------------------------------------------------------------------

class TokenImportance(BaseModel):
    """
    SHAP importance score for a single token.
    """

    token: str = Field(..., description="The text token.")
    importance: float = Field(..., description="SHAP value — positive means towards predicted class.")


class ExplainResponse(BaseModel):
    """
    Returned from GET /explain/{prediction_id} once the SHAP job is complete.
    """

    explanation_id: str
    prediction_id: str
    status: JobStatus
    token_importances: Optional[List[TokenImportance]] = Field(
        default=None,
        description="Ranked list of token-level SHAP values. None while still pending.",
    )
    generated_at: Optional[str] = Field(
        default=None,
        description="ISO-8601 timestamp of when the explanation was generated.",
    )


class ExplainTriggerResponse(BaseModel):
    """
    Returned from POST /explain — confirms the SHAP job was queued.
    """

    xai_job_id: str
    prediction_id: str
    status: JobStatus
    message: str


# ---------------------------------------------------------------------------
# /annotate
# ---------------------------------------------------------------------------

class AnnotateResponse(BaseModel):
    """
    Returned from POST /annotate — confirms the annotation was stored.

    If the submitted label differs from the model prediction, conflict_detected
    is True and the annotation is flagged for Reviewer attention.
    """

    annotation_id: str
    document_id: str
    prediction_id: str
    final_label: str
    action: Literal["accept", "modify", "reject"]
    annotation_status: AnnotationStatus
    conflict_detected: bool = Field(
        ...,
        description="True when the human label differs from the model prediction.",
    )
    message: str


class AnnotationListItem(BaseModel):
    """Single item in the GET /annotations list."""

    annotation_id: str
    document_id: str
    user_id: Optional[str]
    final_label: str
    annotation_status: AnnotationStatus
    annotated_at: str


class AnnotationListResponse(BaseModel):
    """Returned from GET /annotations."""

    total: int
    annotations: List[AnnotationListItem]


# ---------------------------------------------------------------------------
# /retrain
# ---------------------------------------------------------------------------

class RetrainTriggerResponse(BaseModel):
    """Returned from POST /retrain — confirms the retraining job was queued."""

    retrain_job_id: str
    status: JobStatus
    message: str


class RetrainStatusResponse(BaseModel):
    """Returned from GET /retrain/status."""

    retrain_job_id: Optional[str]
    status: JobStatus
    model_version: Optional[str] = None
    accuracy: Optional[float] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    message: str


# ---------------------------------------------------------------------------
# /ontology
# ---------------------------------------------------------------------------

class OntologyNode(BaseModel):
    """A single node from the Neo4j legal ontology."""

    label: str
    description: Optional[str] = None
    parent: Optional[str] = None
    children: List[str] = Field(default_factory=list)
    related: List[str] = Field(default_factory=list)


class OntologyResponse(BaseModel):
    """Returned from GET /ontology — full ontology graph."""

    total: int
    nodes: List[OntologyNode]


class OntologyValidateResponse(BaseModel):
    """Returned from POST /ontology/validate."""

    label: str
    valid: bool
    parent_label: Optional[str] = None
    parent_valid: Optional[bool] = None
    message: str


# ---------------------------------------------------------------------------
# /health  (defined inline in main.py, mirrored here for reference)
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Returned from GET /health."""

    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_path: str
