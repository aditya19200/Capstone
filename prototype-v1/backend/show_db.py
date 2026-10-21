#!/usr/bin/env python3
"""
show_db.py — print what is actually sitting in the live databases.

Why this exists
---------------
check_data_layer.py answers "are the databases reachable and is the schema
right". That is a diagnostic: it prints row counts and migration checks, and
it is the thing to run when something is broken.

This answers a different question — "what is in there" — for the case where
somebody is watching. Uploading a document in the UI and then showing the row
it produced in cloud Postgres is the clearest evidence that the data layer is
real, and it needs the rows, not a count.

It is strictly read-only: SELECTs against Supabase and MATCH ... RETURN count
against Neo4j. It opens no file but .env (through config.settings, same as the
app) and prints no credential.

    cd backend
    ./venv/bin/python show_db.py              # summary plus a few recent rows
    ./venv/bin/python show_db.py --rows 10    # more rows per table
    ./venv/bin/python show_db.py --counts     # counts only, no row contents

Columns are not hard-coded. Each table prints whichever of its interesting
columns actually exist, so a schema that gains or loses one does not turn
this into a traceback in front of an audience.
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from config.settings import settings  # noqa: E402

# Preferred columns per table, in display order. Any that the table does not
# have is skipped rather than requested, because asking Postgres for a column
# that is not there is an error, not an empty value.
PREFERRED = {
    "batches": ["created_at", "source", "status", "total_items", "completed_items"],
    "predictions": ["created_at", "predicted_label", "confidence", "domain", "text_content"],
    "annotations": ["created_at", "validated_label", "status", "annotator_id", "domain"],
    "model_versions": ["created_at", "version_number", "is_active", "accuracy", "training_rows"],
    "retrain_jobs": ["created_at", "status", "min_annotations", "notes"],
    "batch_items": ["created_at", "seq", "status"],
    "xai_jobs": ["created_at", "status"],
    "dataset_versions": ["created_at", "version_number"],
}

ORDER = ["batches", "predictions", "annotations", "model_versions",
         "retrain_jobs", "batch_items", "xai_jobs", "dataset_versions"]


def cell(value, width=38) -> str:
    """One column's text, flattened to a single line and clipped."""
    if value is None:
        return "-"
    text = " ".join(str(value).split())
    return text if len(text) <= width else text[: width - 1] + "…"


def show_supabase(rows: int, counts_only: bool) -> int:
    from repositories._client import get_client, is_configured

    print("=" * 78)
    print(f"Supabase — {settings.SUPABASE_URL}")
    print("=" * 78)

    if not is_configured():
        print("\n  Not configured: SUPABASE_URL still contains 'placeholder', so the")
        print("  app is running on the in-memory store and there is nothing to show.")
        print("  Put Arjun's .env at backend/.env and restart the backend.\n")
        return 1

    client = get_client()
    print()
    for table in ORDER:
        try:
            res = client.table(table).select("*", count="exact").limit(1).execute()
            n = res.count
        except Exception as exc:
            print(f"  {table:<18} unavailable — {type(exc).__name__}")
            continue

        print(f"  {table:<18} {n:>6} rows")
        if counts_only or not n:
            continue

        # Newest first. created_at is on every app table (migration 006), but
        # fall back to an unordered read rather than failing if it is not.
        want = [c for c in PREFERRED.get(table, []) if c in (res.data[0] if res.data else {})]
        try:
            page = (client.table(table).select(",".join(want) or "*")
                    .order("created_at", desc=True).limit(rows).execute())
        except Exception:
            page = client.table(table).select(",".join(want) or "*").limit(rows).execute()

        for row in page.data:
            print("       " + "  ".join(cell(row.get(c)) for c in want))
        print()
    return 0


def show_neo4j() -> int:
    import services.neo4j_service as svc

    print("=" * 78)
    print(f"Neo4j — {settings.NEO4J_URI} (database={settings.NEO4J_DATABASE})")
    print("=" * 78)

    if not svc._is_configured():
        print("\n  Not configured: NEO4J_URI is localhost, so /ontology is answered")
        print("  from the in-memory ontology.\n")
        return 1

    print(f"\n  service status: {svc.status()}")
    if svc.status() == "degraded":
        print("  (configured, but a recent query failed — answers are coming from")
        print("   the mock until the cooldown expires. Do not demo in this state.)")

    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
    )
    try:
        driver.verify_connectivity()
        with driver.session(database=settings.NEO4J_DATABASE) as session:
            one = lambda q, **p: session.run(q, **p).single()[0]  # noqa: E731
            print()
            for domain in ("legal", "medical"):
                nodes = one("MATCH (n:OntologyNode {domain: $d}) RETURN count(n)", d=domain)
                if not nodes:
                    continue
                rel = one("MATCH (:OntologyNode {domain: $d})-[r:RELATED_TO]->"
                          "(:OntologyNode {domain: $d}) RETURN count(r)", d=domain)
                ind = one("MATCH (:Concept)-[r:INDICATES]->(:OntologyNode {domain: $d}) "
                          "RETURN count(r)", d=domain)
                print(f"  {domain:<9} {nodes:>3} labels   {rel:>5} RELATED_TO   "
                      f"{ind:>6} INDICATES")
            # Depth edges are not per-domain (Concept.term is unique graph-wide).
            nar = one("MATCH ()-[r:NARROWS]->() RETURN count(r)")
            co = one("MATCH ()-[r:CO_OCCURS]-() RETURN count(r)/2")
            print(f"\n  concept depth:  {nar} NARROWS   {co} CO_OCCURS")
            if nar == 0 and co == 0:
                print("  (none loaded — run graph/load_ontology.py for each domain)")
            print()
    except Exception as exc:
        print(f"\n  Could not reach it: {type(exc).__name__}: "
              f"{str(exc).splitlines()[0][:120]}")
        print("  Port 7687 blocked by the network? Try a phone hotspot.\n")
        return 1
    finally:
        driver.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--rows", type=int, default=3, help="rows to show per table")
    ap.add_argument("--counts", action="store_true", help="counts only, no row contents")
    args = ap.parse_args()

    bad = show_supabase(args.rows, args.counts)
    print()
    bad += show_neo4j()
    if bad:
        print("One or both databases are not live. check_data_layer.py says why.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
