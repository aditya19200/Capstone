"""
verify_ontology.py — sanity-checks the seeded graph against what
backend/services/neo4j_service.py expects.

The expectations are read from the domain pack (domains/<key>.json), which is
the same file backend/services/mock_db.py projects its ONTOLOGY dict from.
That is the point of the check: the mock and the graph have to agree node for
node and edge for edge, or swapping the mock out for a live Neo4j stops being
a query-for-query replacement. Comparing both against one file is what makes
"they agree" a fact rather than a hope.

Usage:
    export NEO4J_URI=... NEO4J_USER=... NEO4J_PASSWORD=... NEO4J_DATABASE=...
    python verify_ontology.py [--domain legal]
"""

import argparse
import json
import re
import sys
from pathlib import Path

from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parent.parent

# A label no pack should contain, used to check the negative case.
ABSENT_LABEL = "Made Up Law"


def load_pack(key):
    path = ROOT / "domains" / f"{key}.json"
    if not path.exists():
        available = sorted(p.stem for p in (ROOT / "domains").glob("*.json"))
        sys.exit(f"No domain pack at {path}. Available: {', '.join(available) or 'none'}")
    labels = json.loads(path.read_text(encoding="utf-8"))["labels"]
    return {l["name"]: sorted(l.get("related", [])) for l in labels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="legal", help="domain pack to verify against")
    ap.add_argument("--uri", default=None)
    args = ap.parse_args()

    import os
    uri = args.uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "password")
    database = os.environ.get("NEO4J_DATABASE", "neo4j")

    expected = load_pack(args.domain)
    print(f"Verifying against domains/{args.domain}.json — {len(expected)} labels.\n")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session(database=database) as session:
            # Several domains share one database, so only this domain's nodes
            # count. load_ontology.py stamps them with `domain`.
            rows = session.run(
                "MATCH (n:OntologyNode {domain: $domain}) RETURN n.label AS label "
                "ORDER BY n.label",
                domain=args.domain,
            )
            labels = sorted(r["label"] for r in rows)
            if not labels:
                untagged = session.run(
                    "MATCH (n:OntologyNode) WHERE n.domain IS NULL RETURN count(n) AS n"
                ).single()["n"]
                sys.exit(
                    f"No OntologyNode has domain '{args.domain}'."
                    + (f" {untagged} node(s) have no domain at all — they were loaded "
                       f"by an older load_ontology.py. Re-run it for each domain "
                       f"(it only adds the tag; nothing is duplicated)."
                       if untagged else
                       f" Load it with:  python load_ontology.py --domain {args.domain}")
                )
            expected_sorted = sorted(expected)
            assert labels == expected_sorted, (
                "Label mismatch.\n"
                f"  missing from graph: {sorted(set(expected_sorted) - set(labels))}\n"
                f"  extra in graph:     {sorted(set(labels) - set(expected_sorted))}"
            )
            print(f"OK  — {len(labels)}/{len(expected)} OntologyNode labels present "
                  f"and match the pack.")

            # Every node's edges, not a single spot check: a seed file can get
            # one domain's RELATED_TO right and another's wrong.
            for label, expected_related in sorted(expected.items()):
                result = session.run(
                    """
                    MATCH (n:OntologyNode {label: $label, domain: $domain})
                          -[:RELATED_TO]->(r)
                    RETURN r.label AS label ORDER BY r.label
                    """,
                    label=label, domain=args.domain,
                )
                related = sorted(r["label"] for r in result)
                assert related == expected_related, (
                    f"RELATED_TO mismatch for {label!r}.\n"
                    f"  got:      {related}\n  expected: {expected_related}"
                )
            edge_count = sum(len(v) for v in expected.values())
            print(f"OK  — all {edge_count} RELATED_TO edges across {len(expected)} "
                  f"labels match the pack.")

            # The concept layer: every INDICATES edge in the seed must have
            # landed on one of this domain's nodes. A MATCH that finds nothing
            # is not an error in Cypher, so this is the check that catches a
            # concept seed which "loaded" without attaching anything.
            seed = ROOT / "graph" / f"{args.domain}_concept_seed.cypher"
            if seed.exists():
                expected_edges = len(re.findall(r"\[r:INDICATES\]", seed.read_text(encoding="utf-8")))
                got_edges = session.run(
                    "MATCH (:Concept)-[r:INDICATES]->(:OntologyNode {domain: $domain}) "
                    "RETURN count(r) AS n",
                    domain=args.domain,
                ).single()["n"]
                assert got_edges == expected_edges, (
                    f"INDICATES mismatch: {got_edges} edges in the graph, "
                    f"{expected_edges} in {seed.name}."
                )
                print(f"OK  — all {got_edges} INDICATES edges from {seed.name} are present.")

            known = sorted(expected)[0]
            exists = session.run(
                "MATCH (n:OntologyNode {label: $label}) RETURN count(n) > 0 AS exists",
                label=known,
            ).single()["exists"]
            assert exists is True
            print(f"OK  — label_exists() returns True for a known label ({known!r}).")

            assert ABSENT_LABEL not in expected, (
                f"{ABSENT_LABEL!r} is in the pack — pick another ABSENT_LABEL "
                f"for the negative check."
            )
            not_exists = session.run(
                "MATCH (n:OntologyNode {label: $label}) RETURN count(n) > 0 AS exists",
                label=ABSENT_LABEL,
            ).single()["exists"]
            assert not_exists is False
            print("OK  — label_exists() returns False for an unknown label.")
    finally:
        driver.close()

    print(f"\nAll checks passed. Graph matches domains/{args.domain}.json, which is "
          f"also what mock_db.ONTOLOGY projects.")


if __name__ == "__main__":
    main()
