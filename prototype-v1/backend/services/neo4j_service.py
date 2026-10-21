"""
services/neo4j_service.py — Neo4j legal ontology access layer.

Two backends, one set of return shapes:

  * Mock  — mock_db's ONTOLOGY dict (projected from domains/<key>.json).
            Used when NEO4J_URI still points at localhost (the default), so
            the prototype runs on a laptop with no internet.
  * Real  — a live Neo4j graph loaded by graph/load_ontology.py from
            graph/<key>_ontology_seed.cypher and <key>_concept_seed.cypher.

Several domains share one graph, so every query is scoped to the active
domain's nodes (load_ontology.py stamps them with `domain: <key>`). The mock
only ever holds the active domain, so this is what keeps the two equal.

The real path must be a query-for-query replacement for the mock: same keys,
same values, same ordering. graph/verify_ontology.py checks the graph's
content against the same domain pack the mock is built from, and
tests/test_neo4j_service.py checks that the mapping below reproduces the
mock's output exactly.

Falling back
------------
If Neo4j is configured but a query fails (instance paused, wrong password,
no network), the call logs a warning and answers from the mock instead of
raising. After a failure the live graph is skipped for _COOLDOWN_SECONDS so
an unreachable instance doesn't add a connection timeout to every request.
This mirrors the rule the Supabase side follows: the demo must keep working
when the database can't be reached.

Neo4j schema (created by graph/<key>_ontology_seed.cypher):
  Node label : OntologyNode  { label, description, domain }
  Relationships:
    (:OntologyNode)-[:HAS_PARENT]->(:OntologyNode)
    (:OntologyNode)-[:HAS_CHILD]->(:OntologyNode)
    (:OntologyNode)-[:RELATED_TO]->(:OntologyNode)
  (:Concept)-[:INDICATES]->(:OntologyNode) also exists (<key>_concept_seed)
  but nothing here reads it — the ontology consistency check reads
  graph/<key>_concept_markers.json, not the graph.
"""

import atexit
import logging
import threading
import time
from typing import Callable, Dict, List, Optional, TypeVar

import services.mock_db as db
from config.settings import settings

logger = logging.getLogger(__name__)

T = TypeVar("T")

# How long to stop trying the live graph after a failed query.
_COOLDOWN_SECONDS = 30.0
# Fail fast when the instance is unreachable instead of hanging a request.
_CONNECT_TIMEOUT_SECONDS = 5.0


# ===========================================================================
# CONNECTION
# ===========================================================================

def _is_configured() -> bool:
    """
    True when NEO4J_URI points at a real instance.

    Mirrors repositories/_client.is_configured(): the default URI
    (bolt://localhost:7687) means "no graph set up — use the mock".
    """
    return "localhost" not in settings.NEO4J_URI


def status() -> str:
    """
    Which ontology this service is answering from right now.

      "in-memory"  NEO4J_URI is the localhost default — no graph configured.
      "live"       configured, and no query has failed inside the cooldown.
      "degraded"   configured, but a query failed recently, so answers are
                   coming from the mock until the cooldown expires.

    /health reports this. The third case is the one worth surfacing: a
    configured-but-unreachable graph serves the mock *silently*, which is the
    right thing to do at runtime and the wrong thing to find out during a
    demo. Reading this issues no query and opens no connection, so it is safe
    to poll.

    Note "live" is optimistic before the first query of a process — it says
    the URI is real and nothing has failed yet, not that a round trip has
    succeeded. check_data_layer.py is what proves the connection.
    """
    if not _is_configured():
        return "in-memory"
    return "degraded" if time.monotonic() < _skip_until else "live"


_driver = None
_driver_lock = threading.Lock()
_skip_until = 0.0          # monotonic time before which the live graph is skipped
_announced = False         # whether the "using live/mock" line has been logged


def _get_driver():
    """Create the driver on first use (importing neo4j only when needed)."""
    global _driver
    if _driver is None:
        with _driver_lock:
            if _driver is None:
                from neo4j import GraphDatabase
                _driver = GraphDatabase.driver(
                    settings.NEO4J_URI,
                    auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
                    connection_timeout=_CONNECT_TIMEOUT_SECONDS,
                )
    return _driver


def close_driver() -> None:
    """Close the driver if one was opened. Safe to call more than once."""
    global _driver
    with _driver_lock:
        if _driver is not None:
            try:
                _driver.close()
            finally:
                _driver = None


atexit.register(close_driver)


def _domain() -> str:
    """The active domain's key — the same pack mock_db.ONTOLOGY is built from."""
    from config.domains import active_domain
    return active_domain().key


def _run(cypher: str, **params) -> List:
    """
    Run a read query against the configured database and return all records.
    Every query gets $domain, so each can scope itself to the active domain.
    """
    params.setdefault("domain", _domain())
    with _get_driver().session(database=settings.NEO4J_DATABASE) as session:
        return list(session.run(cypher, **params))


def _live_or_mock(name: str, live: Callable[[], T], mock: Callable[[], T]) -> T:
    """
    Answer from the live graph when it is configured and healthy, otherwise
    from the mock. Any failure on the live path falls back to the mock.
    """
    global _skip_until, _announced

    if not _is_configured():
        if not _announced:
            logger.info("neo4j_service: NEO4J_URI is localhost — using the mock ontology")
            _announced = True
        return mock()

    if time.monotonic() < _skip_until:
        return mock()

    try:
        result = live()
    except Exception as exc:  # noqa: BLE001 — any failure must degrade, not crash
        _skip_until = time.monotonic() + _COOLDOWN_SECONDS
        logger.warning(
            "neo4j_service.%s: live Neo4j query failed (%s: %s) — answering from "
            "the mock for the next %.0fs",
            name, type(exc).__name__, exc, _COOLDOWN_SECONDS,
        )
        return mock()

    if not _announced:
        logger.info(
            "neo4j_service: using live Neo4j at %s (database=%s)",
            settings.NEO4J_URI, settings.NEO4J_DATABASE,
        )
        _announced = True
    return result


# ===========================================================================
# RECORD -> DICT MAPPING
# ===========================================================================
#
# One RETURN clause builds every node dict, so all functions agree on shape.
# Pattern comprehensions avoid the row explosion you get from chaining
# OPTIONAL MATCH + collect() over three relationship types.

_NODE_RETURN = """
RETURN n.label       AS label,
       n.description AS description,
       head([(n)-[:HAS_PARENT]->(p:OntologyNode) | p.label]) AS parent,
       [(n)-[:HAS_CHILD]->(c:OntologyNode)  | c.label] AS children,
       [(n)-[:RELATED_TO]->(r:OntologyNode) | r.label] AS related
"""


def _pack_rank(order: List[str]) -> Callable[[str], tuple]:
    """
    Sort key that puts labels in the domain pack's order.

    Neo4j returns nodes and edges in no particular order, but the mock (and
    so the /ontology response the frontend renders) follows the pack. This
    only orders what the graph returned — it never adds or drops a label.
    Labels the pack doesn't know go last, alphabetically.
    """
    index = {label: i for i, label in enumerate(order)}
    return lambda label: (index.get(label, len(index)), label)


def _record_to_node(record) -> Dict:
    label = record["label"]
    pack_node = db.ONTOLOGY.get(label, {})
    return {
        "label":       label,
        "description": record["description"],
        "parent":      record["parent"],
        "children":    sorted(record["children"] or [],
                              key=_pack_rank(pack_node.get("children", []))),
        "related":     sorted(record["related"] or [],
                              key=_pack_rank(pack_node.get("related", []))),
    }


def _fetch_node(label: str) -> Optional[Dict]:
    records = _run(
        "MATCH (n:OntologyNode {label: $label, domain: $domain})" + _NODE_RETURN,
        label=label,
    )
    return _record_to_node(records[0]) if records else None


# ===========================================================================
# ONTOLOGY NODE QUERIES
# ===========================================================================

def get_all_nodes() -> List[Dict]:
    """
    Return all ontology nodes.

    Mock: reads from the static ONTOLOGY dict in mock_db.
    Real: MATCH (n:OntologyNode) RETURN n (+ its edges), in domain-pack order.

    Returns:
        List of node dicts, each with keys:
          label, description, parent, children, related
    """
    logger.debug("neo4j_service.get_all_nodes")

    def live() -> List[Dict]:
        nodes = [_record_to_node(r) for r in _run("MATCH (n:OntologyNode {domain: $domain})" + _NODE_RETURN)]
        rank = _pack_rank(list(db.ONTOLOGY))
        return sorted(nodes, key=lambda n: rank(n["label"]))

    return _live_or_mock("get_all_nodes", live, db.get_ontology_nodes)


def get_node(label: str) -> Optional[Dict]:
    """
    Fetch a single ontology node by its label name.

    Mock: dict lookup in ONTOLOGY.
    Real: MATCH (n:OntologyNode {label: $label}) RETURN n (+ its edges)

    Args:
        label: Exact label string (e.g. "Contract Law").

    Returns:
        Node dict or None if the label does not exist.
    """
    logger.debug("neo4j_service.get_node: label=%s", label)
    return _live_or_mock(
        "get_node",
        lambda: _fetch_node(label),
        lambda: db.get_ontology_node(label=label),
    )


def label_exists(label: str) -> bool:
    """
    Return True if the label exists as a node in the ontology.

    Mock: checks ONTOLOGY keys.
    Real: MATCH (n:OntologyNode {label: $label}) RETURN count(n) > 0

    Args:
        label: Label string to look up.
    """
    def live() -> bool:
        records = _run(
            "MATCH (n:OntologyNode {label: $label, domain: $domain}) "
            "RETURN count(n) > 0 AS exists",
            label=label,
        )
        return bool(records[0]["exists"])

    return _live_or_mock("label_exists", live, lambda: db.validate_label(label=label))


def validate_label_hierarchy(label: str, parent_label: str) -> bool:
    """
    Return True if parent_label is the direct parent of label, or is in
    label's related list.

    Mock: checks 'parent' field and 'related' list in ONTOLOGY.
    Real: one hop over HAS_PARENT or RELATED_TO, matching the mock exactly.
      MATCH (child:OntologyNode {label: $label})
      RETURN EXISTS {
        (child)-[:HAS_PARENT|RELATED_TO]->(:OntologyNode {label: $parent_label})
      }

    Note: an earlier version of this docstring suggested *1..3 hops. That
    would accept labels the mock rejects (e.g. two RELATED_TO steps away),
    so it is deliberately one hop.

    Args:
        label:        The child label to validate.
        parent_label: The asserted parent/ancestor label.
    """
    logger.debug(
        "neo4j_service.validate_label_hierarchy: label=%s parent=%s",
        label, parent_label,
    )

    def live() -> bool:
        records = _run(
            """
            MATCH (child:OntologyNode {label: $label, domain: $domain})
            RETURN EXISTS {
              (child)-[:HAS_PARENT|RELATED_TO]->(:OntologyNode {label: $parent_label})
            } AS ok
            """,
            label=label, parent_label=parent_label,
        )
        return bool(records[0]["ok"]) if records else False

    return _live_or_mock(
        "validate_label_hierarchy",
        live,
        lambda: db.validate_label_with_parent(label=label, parent_label=parent_label),
    )


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
    node = get_node(label)
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
    node = get_node(label)
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
    node = get_node(label)
    if node is None:
        return None
    return node.get("parent")


def get_full_hierarchy(label: str) -> Dict:
    """
    Return a node together with its full context: parent, children, and related nodes.

    Mock: assembles from ONTOLOGY dict.
    Real: one query using pattern comprehensions for each relationship type.

    Args:
        label: Label string of the node to fetch.

    Returns:
        Dict with keys: label, description, parent, children, related.
        Returns an empty dict if the label does not exist.
    """
    node = get_node(label)
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
    Real: MATCH (n:OntologyNode) RETURN n.label (sorted in Python, so the
          ordering is identical to the mock's regardless of DB collation)
    """
    def live() -> List[str]:
        return sorted(r["label"] for r in _run("MATCH (n:OntologyNode {domain: $domain}) RETURN n.label AS label"))

    return _live_or_mock(
        "get_all_labels",
        live,
        lambda: sorted(node["label"] for node in db.get_ontology_nodes()),
    )
