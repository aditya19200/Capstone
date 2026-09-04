"""
load_ontology.py — runs ontology_seed.cypher against a Neo4j instance.

    NEO4J_URI      e.g. neo4j+s://xxxxxxxx.databases.neo4j.io
    NEO4J_USER     e.g. neo4j (or the instance ID, for some Aura instances)
    NEO4J_PASSWORD your instance password
    NEO4J_DATABASE defaults to "neo4j" but some Aura Free instances use
                   their instance ID as the database name instead

Usage:
    pip install neo4j
    export NEO4J_URI="neo4j+s://..."
    export NEO4J_USER="..."
    export NEO4J_PASSWORD="..."
    export NEO4J_DATABASE="..."
    python load_ontology.py
"""

import os
from pathlib import Path
from neo4j import GraphDatabase

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")

SEED_FILE = Path(__file__).parent / "ontology_seed.cypher"


def _statements(script: str):
    for block in script.split(";"):
        lines = [
            line for line in block.splitlines()
            if not line.strip().startswith("//")
        ]
        cleaned = "\n".join(lines).strip()
        if cleaned:
            yield cleaned


def run_seed():
    script = SEED_FILE.read_text()

    count = 0
    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
        with driver.session(database=NEO4J_DATABASE) as session:
            for stmt in _statements(script):
                session.run(stmt)
                count += 1

    print(f"Ontology seeded: {count} statements executed against {NEO4J_URI} (database={NEO4J_DATABASE}).")


if __name__ == "__main__":
    run_seed()
