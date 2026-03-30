"""
routers/ontology.py — Legal ontology endpoints.

GET  /ontology
  Return all ontology nodes from the Neo4j graph (mocked).

POST /ontology/validate
  Check whether a label (and optionally its asserted parent) exists in the
  ontology hierarchy.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, status

from models.request_models import OntologyValidateRequest
from models.response_models import OntologyNode, OntologyResponse, OntologyValidateResponse
from services.neo4j_service import (
    get_all_labels,
    get_all_nodes,
    get_full_hierarchy,
    label_exists,
    validate_label_hierarchy,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# GET /ontology
# ---------------------------------------------------------------------------

@router.get(
    "",
    response_model=OntologyResponse,
    status_code=status.HTTP_200_OK,
    summary="Fetch all ontology nodes",
    description=(
        "Return the full legal ontology graph as a flat list of nodes. "
        "Each node includes its label, description, parent, children, and "
        "related labels. Accessible to all authenticated roles."
    ),
)
async def get_ontology(
    x_role: Optional[str] = Header(default=None, alias="X-Role"),
) -> OntologyResponse:
    """
    Fetch all ontology nodes from Neo4j (mocked).

    No filters are applied — the full graph is returned so the frontend
    can render the ontology tree and populate label dropdowns.
    """
    try:
        raw_nodes = get_all_nodes()
    except Exception as exc:
        logger.error("Failed to fetch ontology nodes: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch ontology: {exc}",
        )

    nodes = [
        OntologyNode(
            label=n["label"],
            description=n.get("description"),
            parent=n.get("parent"),
            children=n.get("children", []),
            related=n.get("related", []),
        )
        for n in raw_nodes
    ]

    logger.debug("Returning %d ontology nodes", len(nodes))
    return OntologyResponse(total=len(nodes), nodes=nodes)


# ---------------------------------------------------------------------------
# GET /ontology/{label}
# ---------------------------------------------------------------------------

@router.get(
    "/{label}",
    response_model=OntologyNode,
    status_code=status.HTTP_200_OK,
    summary="Fetch a single ontology node",
    description=(
        "Return the full hierarchy context (parent, children, related) for a "
        "specific label. Useful for building breadcrumb navigation or showing "
        "related labels alongside a prediction."
    ),
)
async def get_ontology_node(
    label: str,
    x_role: Optional[str] = Header(default=None, alias="X-Role"),
) -> OntologyNode:
    """
    Fetch a single ontology node by label name.

    Returns 404 if the label does not exist in the ontology.
    Label matching is exact (case-sensitive) — use GET /ontology to browse
    valid labels.
    """
    node = get_full_hierarchy(label=label)
    if not node:
        valid_labels = get_all_labels()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Label '{label}' not found in the ontology. "
                f"Valid labels: {valid_labels}"
            ),
        )

    return OntologyNode(
        label=node["label"],
        description=node.get("description"),
        parent=node.get("parent"),
        children=node.get("children", []),
        related=node.get("related", []),
    )


# ---------------------------------------------------------------------------
# POST /ontology/validate
# ---------------------------------------------------------------------------

@router.post(
    "/validate",
    response_model=OntologyValidateResponse,
    status_code=status.HTTP_200_OK,
    summary="Validate a label against the ontology",
    description=(
        "Check whether a label exists in the legal ontology. "
        "Optionally supply a parent_label to verify the hierarchical relationship "
        "between the two labels. Used internally by the annotation router and "
        "exposed for frontend label validation before submission."
    ),
)
async def validate_label(
    body: OntologyValidateRequest,
    x_role: Optional[str] = Header(default=None, alias="X-Role"),
) -> OntologyValidateResponse:
    """
    Validate a label (and optional parent) against the Neo4j ontology.

    Cases:
      label invalid                   → valid=False, message explains why
      label valid, no parent supplied → valid=True
      label valid, parent supplied    → also checks parent validity and
                                        the hierarchical relationship
    """
    # --- Validate primary label ---
    label_valid = label_exists(label=body.label)

    if not label_valid:
        valid_labels = get_all_labels()
        return OntologyValidateResponse(
            label=body.label,
            valid=False,
            parent_label=body.parent_label,
            parent_valid=None,
            message=(
                f"Label '{body.label}' does not exist in the ontology. "
                f"Valid labels: {valid_labels}"
            ),
        )

    # --- No parent supplied — simple existence check ---
    if body.parent_label is None:
        return OntologyValidateResponse(
            label=body.label,
            valid=True,
            parent_label=None,
            parent_valid=None,
            message=f"Label '{body.label}' is valid.",
        )

    # --- Validate parent label ---
    parent_valid = label_exists(label=body.parent_label)
    if not parent_valid:
        valid_labels = get_all_labels()
        return OntologyValidateResponse(
            label=body.label,
            valid=True,
            parent_label=body.parent_label,
            parent_valid=False,
            message=(
                f"Label '{body.label}' is valid, but parent label "
                f"'{body.parent_label}' does not exist in the ontology. "
                f"Valid labels: {valid_labels}"
            ),
        )

    # --- Validate hierarchical relationship ---
    hierarchy_valid = validate_label_hierarchy(
        label=body.label,
        parent_label=body.parent_label,
    )

    if hierarchy_valid:
        message = (
            f"Label '{body.label}' is valid and '{body.parent_label}' "
            "is a recognised parent or related label."
        )
    else:
        message = (
            f"Label '{body.label}' is valid and '{body.parent_label}' exists, "
            "but no hierarchical or related relationship was found between them."
        )

    return OntologyValidateResponse(
        label=body.label,
        valid=True,
        parent_label=body.parent_label,
        parent_valid=True,
        message=message,
    )
