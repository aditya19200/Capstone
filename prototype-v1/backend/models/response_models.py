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

# Real schema status values (per CLAUDE.md)
AnnotationStatus = Literal["pending", "validated", "rejected"]
XaiJobStatus = Literal["pending", "processing", "done", "failed"]
RetrainJobStatus = Literal["pending", "running", "complete", "failed"]

# Legacy status values for prediction_jobs (not in real schema)
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

    prediction_id: str = Field(..., description="Unique ID for this prediction.")
    document_id: Optional[str] = Field(
        default=None,
        description="Passed through from the request body if supplied; None otherwise.",
    )
    predicted_label: str = Field(..., description="Top-1 label from InLegalBERT.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Max softmax probability.")
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
    """SHAP importance score for a single token."""

    token: str = Field(..., description="The text token.")
    importance: float = Field(..., description="SHAP value — positive means towards predicted class.")


class ExplainResponse(BaseModel):
    """
    Returned from GET /explain/{prediction_id} once the SHAP job is complete.
    """

    explanation_id: str
    prediction_id: str
    status: XaiJobStatus
    token_importances: Optional[List[TokenImportance]] = Field(
        default=None,
        description="Ranked list of token-level SHAP values. None while still pending.",
    )
    generated_at: Optional[str] = Field(
        default=None,
        description="ISO-8601 timestamp of when the explanation was generated.",
    )


class ExplainTriggerResponse(BaseModel):
    """Returned from POST /explain — confirms the SHAP job was queued."""

    xai_job_id: str
    prediction_id: str
    status: XaiJobStatus
    message: str


# ---------------------------------------------------------------------------
# /annotate
# ---------------------------------------------------------------------------

class AnnotateResponse(BaseModel):
    """
    Returned from POST /annotate — confirms the annotation was stored.

    conflict_detected=True when the human label differs from the model
    prediction or from a prior annotator's choice on the same prediction.
    The annotation's status is unaffected by conflict (see has_conflict column
    in the annotations table).
    """

    annotation_id: str
    prediction_id: str
    final_label: str
    action: Literal["accept", "modify", "reject"]
    annotation_status: AnnotationStatus
    conflict_detected: bool = Field(
        ...,
        description="True when a conflict was detected and has_conflict was set.",
    )
    message: str


class AnnotationListItem(BaseModel):
    """Single item in the GET /annotations list."""

    annotation_id: str
    prediction_id: str
    user_id: Optional[str]
    final_label: str
    annotation_status: AnnotationStatus
    has_conflict: bool
    predicted_label: str
    text_excerpt: str
    annotated_at: str


class AnnotationListResponse(BaseModel):
    """Returned from GET /annotations."""

    total: int
    annotations: List[AnnotationListItem]


# ---------------------------------------------------------------------------
# /batches
# ---------------------------------------------------------------------------

class BatchCreateResponse(BaseModel):
    """Returned immediately from POST /batches/paste and POST /batches/csv."""

    batch_id: str
    total_items: int


class BatchStatusResponse(BaseModel):
    """Returned from GET /batches/{id}."""

    batch_id: str
    status: Literal["pending", "processing", "done", "failed"]
    total_items: int
    completed_items: int


class BatchSummary(BaseModel):
    """Single row in GET /batches — enough to list and re-open a past batch."""

    batch_id: str
    source: str
    status: Literal["pending", "processing", "done", "failed"]
    total_items: int
    completed_items: int
    created_at: str


class BatchListResponse(BaseModel):
    """Returned from GET /batches."""

    batches: List[BatchSummary]


class BatchItemOut(BaseModel):
    """Single item in the GET /batches/{id}/items list."""

    id: str
    seq: int
    text_content: str
    predicted_label: Optional[str] = None
    confidence: Optional[float] = None
    status: str
    validated_label: Optional[str] = None
    prediction_id: Optional[str] = None


class BatchItemsResponse(BaseModel):
    """Returned from GET /batches/{id}/items."""

    batch_id: str
    total: int
    page: int
    page_size: int
    items: List[BatchItemOut]


# ---------------------------------------------------------------------------
# /queue
# ---------------------------------------------------------------------------

class LowConfidenceItem(BaseModel):
    """Single item in the GET /queue/low-confidence list."""

    prediction_id: str
    text_excerpt: str
    predicted_label: str
    confidence: float
    created_at: str


class LowConfidenceQueueResponse(BaseModel):
    """Returned from GET /queue/low-confidence."""

    total: int
    page: int
    page_size: int
    items: List[LowConfidenceItem]


# ---------------------------------------------------------------------------
# /admin
# ---------------------------------------------------------------------------

class ModelVersionOut(BaseModel):
    """A single model_versions row, returned by POST /admin/models/{id}/activate."""

    id: str
    version_number: str
    accuracy: float
    f1_per_class: Dict[str, float]
    file_path: str
    is_active: bool
    trained_at: str


class BatchThroughputItem(BaseModel):
    """One batch's throughput summary, part of GET /admin/metrics."""

    batch_id: str
    source: str
    status: str
    total_items: int
    completed_items: int
    created_at: str


class RecentActivityItem(BaseModel):
    """One recent prediction, for the dashboard activity feed."""

    prediction_id: str
    text_excerpt: str
    predicted_label: str
    confidence: float
    created_at: str


class LabelCount(BaseModel):
    """How many predictions fell into one category."""

    label: str
    count: int
    percentage: float


class DashboardStatsResponse(BaseModel):
    """
    Returned from GET /stats/dashboard.

    Aggregates that every role's landing page needs. Kept separate from
    /admin/metrics because that route is admin-only, while the dashboard is
    visible to annotators and reviewers too.
    """

    documents_uploaded: int = Field(
        default=0, description="Total items submitted across all batches."
    )
    documents_classified: int = Field(
        default=0, description="Predictions the model has actually produced."
    )
    auto_accepted: int = Field(
        default=0, description="Predictions at or above CONFIDENCE_HIGH."
    )
    needs_review: int = Field(
        default=0, description="Predictions below REVIEW_THRESHOLD."
    )
    conflicts: int = Field(
        default=0, description="Annotations flagged has_conflict."
    )

    # Bands match the frontend's confidence badge colours: >=0.7 green,
    # 0.5-0.7 yellow, <0.5 red — so the chart agrees with the row badges.
    confidence_high: int = 0
    confidence_medium: int = 0
    confidence_low: int = 0

    annotations_total: int = 0
    annotations_validated: int = 0

    label_distribution: List[LabelCount] = Field(default_factory=list)
    recent_activity: List[RecentActivityItem] = Field(default_factory=list)


class ModelVersionSummary(BaseModel):
    """
    One row of the admin model-versions table.

    Deliberately leaner than ModelVersionOut: file_path is a server-side
    filesystem path with no use in the UI, so it isn't exposed here.
    """

    id: str
    version_number: str
    accuracy: float
    is_active: bool
    trained_at: str


class AdminMetricsResponse(BaseModel):
    """Returned from GET /admin/metrics."""

    active_model_version: Optional[str] = Field(
        default=None,
        description="version_number of the currently active model, or None if none is active.",
    )
    f1_per_class: Dict[str, float] = Field(default_factory=dict)
    annotation_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Annotation count keyed by status ('pending', 'validated', 'rejected').",
    )
    batch_throughput: List[BatchThroughputItem] = Field(default_factory=list)
    model_versions: List[ModelVersionSummary] = Field(
        default_factory=list,
        description=(
            "Every trained model version, newest first — powers the admin "
            "version history table and its Activate action."
        ),
    )


# ---------------------------------------------------------------------------
# /retrain
# ---------------------------------------------------------------------------

class RetrainTriggerResponse(BaseModel):
    """Returned from POST /retrain — confirms the retraining job was queued."""

    retrain_job_id: str
    status: RetrainJobStatus
    message: str


class RetrainEligibilityResponse(BaseModel):
    """
    Returned from GET /retrain/eligibility.

    Lets the admin UI show how many validated annotations exist *before*
    attempting a run, instead of only learning the count from the 409 error
    text when it's too low.
    """

    validated_count: int
    required: int
    eligible: bool


class RetrainStatusResponse(BaseModel):
    """Returned from GET /retrain/status."""

    retrain_job_id: Optional[str]
    status: RetrainJobStatus
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
