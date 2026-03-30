"""
models/request_models.py — Pydantic schemas for all incoming API request bodies.

Every router imports from here. No raw dicts are accepted at the API boundary.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """
    Body for POST /predict.

    The frontend submits raw legal text; all other fields are optional
    metadata that gets stored alongside the prediction.
    """

    text: str = Field(
        ...,
        min_length=10,
        max_length=50_000,
        description="Raw legal text to classify.",
    )
    document_id: Optional[str] = Field(
        default=None,
        description="Optional reference to an existing legal_documents row.",
    )

    @field_validator("text")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        """Remove leading/trailing whitespace before any further processing."""
        return v.strip()


# ---------------------------------------------------------------------------
# /explain
# ---------------------------------------------------------------------------

class ExplainRequest(BaseModel):
    """
    Body for POST /explain — manually trigger SHAP for an existing prediction.

    The prediction must already exist in the mock DB (created via /predict).
    """

    prediction_id: str = Field(
        ...,
        description="ID of the prediction to explain.",
    )


# ---------------------------------------------------------------------------
# /annotate
# ---------------------------------------------------------------------------

class AnnotateRequest(BaseModel):
    """
    Body for POST /annotate — submit a human annotation decision.

    Annotators can accept the model's label, modify it, or reject it entirely.
    """

    document_id: str = Field(
        ...,
        description="ID of the document being annotated.",
    )
    prediction_id: str = Field(
        ...,
        description="ID of the prediction this annotation is responding to.",
    )
    final_label: str = Field(
        ...,
        description="The label the annotator has chosen for this document.",
    )
    action: Literal["accept", "modify", "reject"] = Field(
        ...,
        description=(
            "accept  → annotator agrees with the model prediction.\n"
            "modify  → annotator chooses a different label.\n"
            "reject  → document is invalid / out of scope."
        ),
    )
    notes: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Optional free-text notes from the annotator.",
    )

    @field_validator("final_label")
    @classmethod
    def validate_label(cls, v: str) -> str:
        """Normalise label casing; actual ontology validation is done in the router."""
        return v.strip()


# ---------------------------------------------------------------------------
# /retrain
# ---------------------------------------------------------------------------

class RetrainRequest(BaseModel):
    """
    Body for POST /retrain — Admin manually triggers the retraining pipeline.

    All fields are optional; omitting them uses system defaults.
    """

    notes: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Optional notes to attach to this retraining run (e.g. reason).",
    )
    min_annotations: int = Field(
        default=50,
        ge=1,
        description="Minimum number of validated annotations required before retraining starts.",
    )


# ---------------------------------------------------------------------------
# /ontology/validate
# ---------------------------------------------------------------------------

class OntologyValidateRequest(BaseModel):
    """
    Body for POST /ontology/validate — check whether a label exists in the
    Neo4j ontology hierarchy.
    """

    label: str = Field(
        ...,
        description="The label string to validate against the ontology.",
    )
    parent_label: Optional[str] = Field(
        default=None,
        description="Optional parent label to check the hierarchical relationship.",
    )
