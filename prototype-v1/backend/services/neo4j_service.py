"""
services/neo4j_service.py — Neo4j legal ontology access layer.

All functions delegate to mock_db's static ONTOLOGY dict for now.
When the real Neo4j instance is ready, replace each mock_db call with
the official neo4j driver query — function signatures stay identical.

Swap checklist (per function):
  1. Uncomment the driver initialisation block below.
  2. Replace the mock_db call with a `session.run(cypher, **params)` call.
  3. Map the returned Record objects back to the same Dict/List[Dict] types.

Neo4j schema (to be created by teammate):
  Node label : OntologyNode  { label, description }
  Relationships:
    (:OntologyNode)-[:HAS_PARENT]->(:OntologyNode)
    (:OntologyNode)-[:HAS_CHILD]->(:OntologyNode)
    (:OntologyNode)-[:RELATED_TO]->(:OntologyNode)
"""

import logging
from typing import Dict, List, Optional

import services.mock_db as db
from config.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Real Neo4j driver (initialised but not used while mocked)
# ---------------------------------------------------------------------------
# Uncomment the block below when swapping to real Neo4j:
#
# from neo4j import GraphDatabase, Driver
#
# _driver: Driver = GraphDatabase.driver(
#     settings.NEO4J_URI,
#     auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
# )
#
# def _get_session():
#     return _driver.session()


# ===========================================================================
# ONTOLOGY NODE QUERIES
# ===========================================================================

def get_all_nodes() -> List[Dict]:
    """
    Return all ontology nodes.

    Mock: reads from the static ONTOLOGY dict in mock_db.
    Real: MATCH (n:OntologyNode) RETURN n

    Returns:
        List of node dicts, each with keys:
          label, description, parent, children, related
    """
    logger.debug("neo4j_service.get_all_nodes")
    return db.get_ontology_nodes()


def get_node(label: str) -> Optional[Dict]:
    """
    Fetch a single ontology node by its label name.

    Mock: dict lookup in ONTOLOGY.
    Real: MATCH (n:OntologyNode {label: $label}) RETURN n

    Args:
        label: Exact label string (e.g. "Contract Law").

    Returns:
        Node dict or None if the label does not exist.
    """
    logger.debug("neo4j_service.get_node: label=%s", label)
    return db.get_ontology_node(label=label)


def label_exists(label: str) -> bool:
    """
    Return True if the label exists as a node in the ontology.

    Mock: checks ONTOLOGY keys.
    Real: MATCH (n:OntologyNode {label: $label}) RETURN count(n) > 0

    Args:
        label: Label string to look up.
    """
    return db.validate_label(label=label)


def validate_label_hierarchy(label: str, parent_label: str) -> bool:
    """
    Return True if parent_label is a valid ancestor or related node of label.

    Mock: checks 'parent' field and 'related' list in ONTOLOGY.
    Real:
      MATCH (child:OntologyNode {label: $label})
            -[:HAS_PARENT|RELATED_TO*1..3]->
            (parent:OntologyNode {label: $parent_label})
      RETURN count(*) > 0

    Args:
        label:        The child label to validate.
        parent_label: The asserted parent/ancestor label.
    """
    logger.debug(
        "neo4j_service.validate_label_hierarchy: label=%s parent=%s",
        label, parent_label,
    )
    return db.validate_label_with_parent(label=label, parent_label=parent_label)


# ===========================================================================
# RELATIONSHIP QUERIES
# ===========================================================================

def get_children(label: str) -> List[str]:
    """
    Return the direct children of a node.

    Mock: reads 'children' list from ONTOLOGY node.
    Real: MATCH (n:OntologyNode {label: $label})-[:HAS_CHILD]->(c) RETURN c.label

    Returns:
        List of child label strings (empty list if none or label not found).
    """
    node = db.get_ontology_node(label=label)
    if node is None:
        return []
    return node.get("children", [])


def get_related(label: str) -> List[str]:
    """
    Return labels that are related (but not parent/child) to the given node.

    Mock: reads 'related' list from ONTOLOGY node.
    Real: MATCH (n:OntologyNode {label: $label})-[:RELATED_TO]->(r) RETURN r.label

    Returns:
        List of related label strings (empty list if none or label not found).
    """
    node = db.get_ontology_node(label=label)
    if node is None:
        return []
    return node.get("related", [])


def get_parent(label: str) -> Optional[str]:
    """
    Return the parent label of a node, or None if it is a top-level node.

    Mock: reads 'parent' field from ONTOLOGY node.
    Real: MATCH (n:OntologyNode {label: $label})-[:HAS_PARENT]->(p) RETURN p.label

    Returns:
        Parent label string or None.
    """
    node = db.get_ontology_node(label=label)
    if node is None:
        return None
    return node.get("parent")


def get_full_hierarchy(label: str) -> Dict:
    """
    Return a node together with its full context: parent, children, and related nodes.

    Mock: assembles from ONTOLOGY dict.
    Real: single Cypher query with OPTIONAL MATCH for each relationship type.

    Args:
        label: Label string of the node to fetch.

    Returns:
        Dict with keys: label, description, parent, children, related.
        Returns an empty dict if the label does not exist.
    """
    node = db.get_ontology_node(label=label)
    if node is None:
        logger.warning("neo4j_service.get_full_hierarchy: label '%s' not found", label)
        return {}
    return {
        "label":       node["label"],
        "description": node.get("description"),
        "parent":      node.get("parent"),
        "children":    node.get("children", []),
        "related":     node.get("related", []),
    }


# ===========================================================================
# ALL VALID LABELS  (convenience helper used by validators)
# ===========================================================================

def get_all_labels() -> List[str]:
    """
    Return a sorted list of all valid label strings in the ontology.

    Used by routers to provide descriptive error messages when an unknown
    label is submitted.

    Mock: returns sorted keys of ONTOLOGY dict.
    Real: MATCH (n:OntologyNode) RETURN n.label ORDER BY n.label
    """
    return sorted(node["label"] for node in db.get_ontology_nodes())
