"""
tests/test_neo4j_service.py — the live-Neo4j path must be a drop-in
replacement for the mock, for every domain, with all domains in one graph.

No Neo4j server is needed. _run() is swapped for a fake that answers each
query from a graph parsed out of graph/<key>_ontology_seed.cypher — the
files load_ontology.py actually loads — holding legal AND medical at once,
tagged with their domain the way the loader tags them, and returning nodes
and edges in scrambled order as a real database may. Every public function
is then compared against the mock, label by label, for each domain.

What this does NOT cover is whether the Cypher text parses on a real server;
backend/check_data_layer.py runs the same comparison against a live instance.
"""

import re
from pathlib import Path

import pytest

import services.mock_db as db
import services.neo4j_service as svc
from config.domains import registry

GRAPH = Path(__file__).resolve().parents[2] / "graph"
DOMAINS = sorted(registry())


# ---------------------------------------------------------------------------
# A tiny in-memory stand-in for the seeded graph
# ---------------------------------------------------------------------------

def _parse_seed(domain):
    text = (GRAPH / f"{domain}_ontology_seed.cypher").read_text(encoding="utf-8")
    nodes = {
        label: (desc, domain)
        for label, desc in re.findall(
            r'MERGE \(n:OntologyNode \{label: "([^"]+)"\}\)\s*SET n\.description = "([^"]*)"',
            text,
        )
    }
    edges = re.findall(
        r'MATCH \(a:OntologyNode \{label: "([^"]+)"\}\), '
        r'\(b:OntologyNode \{label: "([^"]+)"\}\)\s*MERGE \(a\)-\[:(\w+)\]->\(b\)',
        text,
    )
    return nodes, edges


def _whole_graph():
    nodes, edges = {}, []
    for d in DOMAINS:
        n, e = _parse_seed(d)
        nodes.update(n)
        edges += e
    return nodes, edges


class FakeGraph:
    def __init__(self, nodes, edges):
        self.nodes = nodes      # label -> (description, domain)
        self.edges = edges      # (from, to, rel)
        self.queries = []

    def in_domain(self, domain):
        # reversed() so the fake never happens to return the pack's order
        return [l for l, (_, d) in reversed(list(self.nodes.items())) if d == domain]

    def out(self, label, rel):
        return list(reversed([b for a, b, r in self.edges if a == label and r == rel]))

    def node_record(self, label):
        parents = self.out(label, "HAS_PARENT")
        return {
            "label": label,
            "description": self.nodes[label][0],
            "parent": parents[0] if parents else None,
            "children": self.out(label, "HAS_CHILD"),
            "related": self.out(label, "RELATED_TO"),
        }

    def run(self, cypher, **params):
        self.queries.append(cypher)
        q = " ".join(cypher.split())
        assert "domain: $domain" in q, f"query is not scoped to a domain: {q}"
        domain, label = params["domain"], params.get("label")
        here = label in self.in_domain(domain)
        if "EXISTS {" in q:
            if not here:
                return []
            ok = any(a == label and b == params["parent_label"] and r in ("HAS_PARENT", "RELATED_TO")
                     for a, b, r in self.edges)
            return [{"ok": ok}]
        if "count(n) > 0" in q:
            return [{"exists": here}]
        if q.startswith("MATCH (n:OntologyNode {label: $label, domain: $domain})"):
            return [self.node_record(label)] if here else []
        if q == "MATCH (n:OntologyNode {domain: $domain}) RETURN n.label AS label":
            return [{"label": l} for l in self.in_domain(domain)]
        if q.startswith("MATCH (n:OntologyNode {domain: $domain}) RETURN n.label AS label, n.description"):
            return [self.node_record(l) for l in self.in_domain(domain)]
        raise AssertionError(f"unexpected query: {q}")


@pytest.fixture(params=DOMAINS)
def domain(request, monkeypatch):
    """Run as if this domain were active: mock and live path both switch."""
    key = request.param
    monkeypatch.setattr(db, "ONTOLOGY", registry()[key].ontology)
    monkeypatch.setattr(svc, "_domain", lambda: key)
    return key


def _go_live(monkeypatch):
    graph = FakeGraph(*_whole_graph())
    monkeypatch.setattr(svc, "_is_configured", lambda: True)
    monkeypatch.setattr(svc, "_run", lambda cypher, **p: graph.run(cypher, **dict(p, domain=svc._domain())))
    monkeypatch.setattr(svc, "_skip_until", 0.0)
    return graph


def _go_mock(monkeypatch):
    monkeypatch.setattr(svc, "_is_configured", lambda: False)


def _answers(label, other):
    return {
        "get_node": svc.get_node(label),
        "label_exists": svc.label_exists(label),
        "validate_label_hierarchy": svc.validate_label_hierarchy(label, other),
        "get_children": svc.get_children(label),
        "get_related": svc.get_related(label),
        "get_parent": svc.get_parent(label),
        "get_full_hierarchy": svc.get_full_hierarchy(label),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", DOMAINS)
def test_seed_file_matches_pack(key):
    """The fake is only a fair stand-in if the seed agrees with the pack."""
    nodes, edges = _parse_seed(key)
    ontology = registry()[key].ontology
    assert set(nodes) == set(ontology)
    for label, node in ontology.items():
        assert nodes[label][0] == node["description"]
        seeded = sorted(b for a, b, r in edges if a == label and r == "RELATED_TO")
        assert seeded == sorted(node["related"])


def test_domains_do_not_share_labels():
    """OntologyNode.label is unique graph-wide, so a shared label would merge two domains' nodes."""
    seen = {}
    for key in DOMAINS:
        for label in registry()[key].ontology:
            assert label not in seen, f"{label!r} is in both {seen[label]} and {key}"
            seen[label] = key


def test_default_uri_uses_mock():
    assert svc._is_configured() is False  # conftest pins NEO4J_URI to localhost


def test_get_all_nodes_identical_to_mock_including_order(domain, monkeypatch):
    expected = db.get_ontology_nodes()
    graph = _go_live(monkeypatch)
    assert svc.get_all_nodes() == expected
    assert graph.queries, "the fake graph was never queried"


def test_get_all_labels_identical_to_mock(domain, monkeypatch):
    _go_mock(monkeypatch)
    expected = svc.get_all_labels()
    _go_live(monkeypatch)
    assert svc.get_all_labels() == expected


def test_every_function_identical_to_mock(domain, monkeypatch):
    # Every label of every domain, so the other domain's labels are probed
    # too: they must be "not found" here, exactly as the mock says.
    labels = [l for k in DOMAINS for l in registry()[k].ontology] + ["Made Up Law"]
    for label in labels:
        for other in labels:
            _go_mock(monkeypatch)
            expected = _answers(label, other)
            _go_live(monkeypatch)
            got = _answers(label, other)
            assert got == expected, f"[{domain}] {label!r} vs {other!r}"


def test_other_domains_nodes_are_invisible(domain, monkeypatch):
    _go_live(monkeypatch)
    mine = set(registry()[domain].ontology)
    theirs = {l for k in DOMAINS if k != domain for l in registry()[k].ontology}
    assert set(svc.get_all_labels()) == mine
    for label in theirs:
        assert svc.label_exists(label) is False
        assert svc.get_node(label) is None


def test_hierarchy_is_one_hop_not_multi_hop(monkeypatch):
    # Contract Law -> Civil Procedure / Other -> Family Law is two RELATED_TO
    # hops. The mock says no; a *1..3 query would say yes.
    monkeypatch.setattr(db, "ONTOLOGY", registry()["legal"].ontology)
    monkeypatch.setattr(svc, "_domain", lambda: "legal")
    _go_live(monkeypatch)
    assert "Family Law" not in db.ONTOLOGY["Contract Law"]["related"]
    assert svc.validate_label_hierarchy("Contract Law", "Family Law") is False
    assert svc.validate_label_hierarchy("Contract Law", "Civil Procedure / Other") is True


def test_unreachable_neo4j_falls_back_to_mock_and_cools_down(monkeypatch):
    calls = []

    def broken_run(cypher, **params):
        calls.append(cypher)
        raise ConnectionError("instance paused")

    monkeypatch.setattr(svc, "_is_configured", lambda: True)
    monkeypatch.setattr(svc, "_run", broken_run)
    monkeypatch.setattr(svc, "_skip_until", 0.0)

    known = next(iter(db.ONTOLOGY))
    assert svc.get_all_nodes() == db.get_ontology_nodes()
    assert svc.label_exists(known) is True
    assert svc.label_exists("Made Up Law") is False
    assert len(calls) == 1, "after one failure the live graph should be skipped"


def test_real_driver_against_closed_port_falls_back(monkeypatch):
    """End to end through the real neo4j driver, with nothing listening."""
    pytest.importorskip("neo4j")
    monkeypatch.setattr(svc.settings, "NEO4J_URI", "bolt://127.0.0.1:1")
    monkeypatch.setattr(svc, "_driver", None)
    monkeypatch.setattr(svc, "_skip_until", 0.0)
    try:
        assert svc._is_configured() is True
        assert svc.get_all_labels() == sorted(db.ONTOLOGY)
        assert svc._skip_until > 0, "the failure should have started a cooldown"
    finally:
        svc.close_driver()
