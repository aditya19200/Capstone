"""
check_data_layer.py — is the backend really talking to Supabase and Neo4j?

The data layer falls back to the in-memory mock whenever credentials are
missing or a database can't be reached, so "the app works" proves nothing
about whether the real databases are connected. This script answers that
directly. Run it from backend/ so it reads the same .env the app does:

    cd backend
    ./venv/bin/python check_data_layer.py

It only reads (plus calling claim RPCs with n=0, which claim nothing).
Exit code is 0 when every configured database passes.
"""

import sys

from config.settings import settings

OK, FAIL, SKIP = "  OK  ", "  FAIL", "  --  "
failures = 0
skipped = 0


def report(status, msg):
    global failures, skipped
    if status == FAIL:
        failures += 1
    if status == SKIP:
        skipped += 1
    print(f"{status} {msg}")


# ---------------------------------------------------------------------------
# Supabase
# ---------------------------------------------------------------------------

def check_supabase():
    from repositories._client import get_client, is_configured

    print("\nSupabase")
    if not is_configured():
        report(SKIP, "is_configured() is False — SUPABASE_URL still contains "
                     "'placeholder', so the app is using the in-memory mock.")
        return
    report(OK, f"is_configured() is True ({settings.SUPABASE_URL})")

    key = settings.SUPABASE_SERVICE_KEY
    if key.startswith("sb_"):
        report(FAIL, "SUPABASE_SERVICE_KEY is a new-format key (sb_...). The pinned "
                     "client (supabase==2.4.6) rejects these as 'Invalid API key'. "
                     "Use Project Settings -> API Keys -> Legacy API keys -> service_role "
                     "(it starts with eyJ).")
    elif '"anon"' in _jwt_role(key):
        report(FAIL, "SUPABASE_SERVICE_KEY is the anon key. Use the service_role key.")

    try:
        client = get_client()
    except Exception as exc:
        report(FAIL, f"could not create client: {type(exc).__name__}: {exc}")
        return

    tables = ["dataset_versions", "model_versions", "predictions", "batches",
              "batch_items", "xai_jobs", "annotations", "retrain_jobs"]
    for table in tables:
        try:
            res = client.table(table).select("*", count="exact").limit(1).execute()
            report(OK, f"table {table:<17} {res.count} rows")
        except Exception as exc:
            report(FAIL, f"table {table:<17} {_short(exc)}  "
                         "(a migration was skipped or run out of order?)")

    # 007 adds the domain column; checking it catches "ran 001-006 only".
    try:
        client.table("predictions").select("domain").limit(1).execute()
        report(OK, "predictions.domain column present (migration 007 ran)")
    except Exception as exc:
        report(FAIL, f"predictions.domain missing — run 007_domain_column.sql ({_short(exc)})")

    try:
        client.table("retrain_jobs").select("notes, min_annotations").limit(1).execute()
        report(OK, "retrain_jobs.notes/min_annotations present (migration 008 ran)")
    except Exception as exc:
        report(FAIL, f"retrain_jobs columns missing — run 008_app_schema_gaps.sql ({_short(exc)})")

    for rpc, params in [("claim_batch_items", {"n": 0})]:
        try:
            client.rpc(rpc, params).execute()
            report(OK, f"rpc {rpc}() callable")
        except Exception as exc:
            report(FAIL, f"rpc {rpc}() — {_short(exc)}  (did 002 run?)")


def _jwt_role(key: str) -> str:
    import base64
    try:
        payload = key.split(".")[1]
        return base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)).decode()
    except Exception:
        return ""


def _short(exc) -> str:
    return f"{type(exc).__name__}: {str(exc).splitlines()[0][:140]}"


# ---------------------------------------------------------------------------
# Neo4j
# ---------------------------------------------------------------------------

def check_neo4j():
    import services.mock_db as db
    import services.neo4j_service as svc

    print("\nNeo4j")
    if not svc._is_configured():
        report(SKIP, "_is_configured() is False — NEO4J_URI is localhost, "
                     "so /ontology is served from the mock.")
        return
    report(OK, f"_is_configured() is True ({settings.NEO4J_URI}, "
               f"database={settings.NEO4J_DATABASE})")

    # Query directly first: the service functions swallow errors and fall
    # back to the mock, which would make every comparison below "pass".
    # Every domain pack is checked, not just the active one: they share the
    # graph, and the deliverable is all of them loaded.
    from pathlib import Path
    import re
    from config.domains import active_domain, registry
    graph_dir = Path(__file__).resolve().parent.parent / "graph"
    try:
        untagged = svc._run("MATCH (n:OntologyNode) WHERE n.domain IS NULL "
                            "RETURN count(n) AS c")[0]["c"]
        per_domain = {}
        for key, dom in sorted(registry().items()):
            ontology = dom.ontology
            seed = graph_dir / f"{key}_concept_seed.cypher"
            exp = {
                "OntologyNode": len(ontology),
                "RELATED_TO": sum(len(n["related"]) for n in ontology.values()),
                "INDICATES": len(re.findall(r"\[r:INDICATES\]", seed.read_text(encoding="utf-8")))
                             if seed.exists() else None,
            }
            got = {
                name: svc._run(q, domain=key)[0]["c"] for name, q in [
                    ("OntologyNode", "MATCH (n:OntologyNode {domain: $domain}) RETURN count(n) AS c"),
                    ("RELATED_TO", "MATCH (:OntologyNode {domain: $domain})-[r:RELATED_TO]->"
                                   "(:OntologyNode {domain: $domain}) RETURN count(r) AS c"),
                    ("INDICATES", "MATCH (:Concept)-[r:INDICATES]->(:OntologyNode {domain: $domain}) "
                                  "RETURN count(r) AS c"),
                ]
            }
            per_domain[key] = (exp, got)
    except Exception as exc:
        report(FAIL, f"cannot query Neo4j: {_short(exc)}")
        if "database" in str(exc).lower() and "not" in str(exc).lower():
            print("         Hint: some Aura Free instances use the instance ID as "
                  "NEO4J_DATABASE instead of 'neo4j'.")
        return

    for key, (exp, got) in per_domain.items():
        if got["OntologyNode"] == 0:
            report(FAIL, f"[{key}] not loaded — python graph/load_ontology.py --domain {key}")
            continue
        for name in ("OntologyNode", "RELATED_TO", "INDICATES"):
            if exp[name] is None:
                continue
            status = OK if got[name] == exp[name] else FAIL
            report(status, f"[{key}] {name:<13} {got[name]} (expected {exp[name]})")
    if untagged:
        report(FAIL, f"{untagged} OntologyNode(s) have no domain — loaded by an older "
                     f"load_ontology.py. Re-run it for each domain (it adds the tag).")

    print(f"\n  Active domain: {active_domain().key} — comparing the live answers to the mock")

    # Every public function, live vs mock, for every label plus an unknown one.
    def answers(label, other):
        return (svc.get_node(label), svc.label_exists(label),
                svc.validate_label_hierarchy(label, other), svc.get_children(label),
                svc.get_related(label), svc.get_parent(label), svc.get_full_hierarchy(label))

    labels = list(db.ONTOLOGY) + ["Made Up Law"]
    # The unknown-label probes log a "not found" warning each; mute those.
    # A live-query failure is still caught below, via the cooldown it starts.
    import logging
    svc_log = logging.getLogger(svc.__name__)
    level = svc_log.level
    svc_log.setLevel(logging.ERROR)
    try:
        live_nodes, live_labels = svc.get_all_nodes(), svc.get_all_labels()
        live = {(a, b): answers(a, b) for a in labels for b in labels}
    finally:
        svc_log.setLevel(level)
    if svc._skip_until:
        report(FAIL, "a live query failed during the comparison and the service "
                     "fell back to the mock — the instance may have gone to sleep "
                     "or dropped the connection; run this again")
        return

    real_is_configured = svc._is_configured
    svc._is_configured = lambda: False
    svc_log.setLevel(logging.ERROR)
    try:
        mock_nodes, mock_labels = svc.get_all_nodes(), svc.get_all_labels()
        mismatches = [k for k in live if live[k] != answers(*k)]
    finally:
        svc._is_configured = real_is_configured
        svc_log.setLevel(level)

    report(OK if live_nodes == mock_nodes else FAIL,
           "get_all_nodes() identical to the mock (what GET /ontology returns)")
    report(OK if live_labels == mock_labels else FAIL, "get_all_labels() identical to the mock")
    report(OK if not mismatches else FAIL,
           f"other 7 functions identical to the mock across {len(live)} label pairs"
           + (f" — first mismatch: {mismatches[0]}" if mismatches else ""))


if __name__ == "__main__":
    check_supabase()
    check_neo4j()
    if failures:
        print(f"\n{failures} check(s) failed.")
    elif skipped == 2:
        print("\nNothing is connected yet — both databases are on the mock.")
    elif skipped:
        print("\nThe connected database passed; the other is still on the mock.")
    else:
        print("\nAll checks passed. Both databases are live.")
    sys.exit(1 if failures else 0)
