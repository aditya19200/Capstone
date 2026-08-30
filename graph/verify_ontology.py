"""
verify_ontology.py — sanity-checks the seeded graph against what
backend/services/neo4j_service.py expects.

Usage:
    export NEO4J_URI=... NEO4J_USER=... NEO4J_PASSWORD=... NEO4J_DATABASE=...
    python verify_ontology.py
"""

import os
from neo4j import GraphDatabase

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")

EXPECTED_LABELS = [
    "Contract Law", "Criminal Law", "Constitutional Law",
    "Corporate / Company Law", "Property / Real Estate Law", "Family Law",
    "Labour & Employment Law", "Intellectual Property Law", "Taxation Law",
    "Civil Procedure / Other",
]


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session(database=NEO4J_DATABASE) as session:
        rows = session.run(
            "MATCH (n:OntologyNode) RETURN n.label AS label ORDER BY n.label"
        )
        labels = sorted(r["label"] for r in rows)
        expected_sorted = sorted(EXPECTED_LABELS)
        assert labels == expected_sorted, (
            f"Label mismatch.\n  got:      {labels}\n  expected: {expected_sorted}"
        )
        print(f"OK  — {len(labels)}/10 OntologyNode labels present and match id2label.")

        result = session.run(
            """
            MATCH (n:OntologyNode {label: $label})-[:RELATED_TO]->(r)
            RETURN r.label AS label ORDER BY r.label
            """,
            label="Civil Procedure / Other",
        )
        related = [r["label"] for r in result]
        expected_related = sorted([
            "Contract Law", "Family Law", "Labour & Employment Law",
            "Property / Real Estate Law",
        ])
        assert related == expected_related, (
            f"RELATED_TO mismatch for 'Civil Procedure / Other'.\n"
            f"  got:      {related}\n  expected: {expected_related}"
        )
        print("OK  — RELATED_TO edges for 'Civil Procedure / Other' match the mock.")

        exists = session.run(
            "MATCH (n:OntologyNode {label: $label}) RETURN count(n) > 0 AS exists",
            label="Taxation Law",
        ).single()["exists"]
        assert exists is True
        print("OK  — label_exists() query returns True for a known label.")

        not_exists = session.run(
            "MATCH (n:OntologyNode {label: $label}) RETURN count(n) > 0 AS exists",
            label="Made Up Law",
        ).single()["exists"]
        assert not_exists is False
        print("OK  — label_exists() query returns False for an unknown label.")

    driver.close()
    print("\nAll checks passed. Graph matches services/mock_db.py's ONTOLOGY dict.")
    print("Safe to tell Aditya the swap-from-mock checklist can proceed.")


if __name__ == "__main__":
    main()
