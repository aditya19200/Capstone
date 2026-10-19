"""
tests/test_data_layer_wiring.py — the wiring between the app and the real
databases, checked without either database.

* Graph seeds: every statement load_ontology.py would send is real Cypher.
  It used to split on every ';', and both concept seeds have one inside a
  header comment, so the first statement began with prose and the load
  stopped before writing a single concept. --dry-run only counts, so it
  could not see this.
* supabase_service: with Supabase configured, each function that touches a
  table in the real schema must hand off to _supabase_real (otherwise reads
  go to the in-memory mock while the repositories write to Postgres, and
  POST /annotate 404s on predictions that exist); with it unconfigured,
  none may.

The real path itself was exercised end to end against Postgres + PostgREST
when this was written; see the PR description.
"""

import importlib.util
import inspect
import re
from pathlib import Path

import pytest

import services._supabase_real as real
import services.supabase_service as svc
from config.domains import registry

GRAPH = Path(__file__).resolve().parents[2] / "graph"


def _loader():
    spec = importlib.util.spec_from_file_location("load_ontology", GRAPH / "load_ontology.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("domain", sorted(registry()))
@pytest.mark.parametrize("half", ["ontology", "concept"])
def test_every_seed_statement_is_cypher(domain, half):
    text = (GRAPH / f"{domain}_{half}_seed.cypher").read_text(encoding="utf-8")
    stmts = list(_loader().statements(text))
    bad = [s for s in stmts if not s.startswith(("CREATE", "MERGE", "MATCH"))]
    assert not bad, f"{len(bad)} malformed, first: {bad[0][:120]!r}"
    # Every line ending in ';' is one statement — nothing merged or dropped.
    code = [l for l in text.splitlines() if l.strip() and not l.strip().startswith("//")]
    assert len(stmts) == sum(1 for l in code if l.rstrip().endswith(";"))


@pytest.mark.parametrize("domain", sorted(registry()))
def test_concept_seed_attaches_with_merge_not_match(domain):
    """A MATCH that finds nothing silently creates no edge; MERGE cannot."""
    text = (GRAPH / f"{domain}_concept_seed.cypher").read_text(encoding="utf-8")
    edges = [l for l in text.splitlines() if "[r:INDICATES]" in l]
    assert edges
    assert all("MERGE (d:OntologyNode" in l for l in edges)


ROUTED = [name for name, fn in inspect.getmembers(real, inspect.isfunction)
          if not name.startswith("_") and fn.__module__ == real.__name__]


def _call(fn):
    args = {}
    for p in inspect.signature(fn).parameters.values():
        if p.kind is p.VAR_KEYWORD:
            args["status"] = "done"
        elif p.default is p.empty:
            args[p.name] = {"shap_values": [], "all_probabilities": {},
                            "f1_per_class": {}, "label_distribution": {}}.get(p.name, "x")
    return fn(**args)


def test_every_real_function_has_a_service_counterpart():
    assert ROUTED
    missing = [n for n in ROUTED if not hasattr(svc, n)]
    assert not missing


@pytest.mark.parametrize("name", ROUTED)
def test_configured_routes_to_real(name, monkeypatch):
    calls = []
    monkeypatch.setattr(svc, "is_configured", lambda: True)
    monkeypatch.setattr(svc._real, name, lambda **kw: calls.append(kw) or "REAL")
    monkeypatch.setattr(svc, "db", None)  # any mock access would now raise
    assert _call(getattr(svc, name)) == "REAL"
    assert len(calls) == 1


@pytest.mark.parametrize("name", ROUTED)
def test_unconfigured_never_touches_real(name, monkeypatch):
    monkeypatch.setattr(svc, "is_configured", lambda: False)
    monkeypatch.setattr(svc, "_real", None)  # any real access would now raise
    try:
        _call(getattr(svc, name))
    except (KeyError, TypeError, ValueError):
        pass  # the mock rejecting dummy arguments is fine; reaching _real is not
