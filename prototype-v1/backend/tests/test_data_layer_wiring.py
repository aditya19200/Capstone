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


# ---------------------------------------------------------------------------
# set_active() against a database whose activate_model_version RPC predates
# migration 009.
#
# This is the shape of bug the file exists for. Migration 003 wrote the
# deactivate step as an unfiltered `update model_versions set is_active =
# false;`. Supabase rejects that (SQLSTATE 21000) even inside plpgsql, so
# every activation against a real database returned 500 -- while every test
# passed, because set_active() hands off to mock_db long before reaching the
# RPC whenever Supabase is unconfigured. The feature was fully covered and
# entirely broken.
# ---------------------------------------------------------------------------

class _FakeQuery:
    def __init__(self, log, label, data=None):
        self._log, self._label, self._data = log, label, data or []

    def update(self, payload):
        self._label += f" update {payload}"
        return self

    def eq(self, column, value):
        self._label += f" where {column}={value!r}"
        return self

    def execute(self):
        self._log.append(self._label)
        return type("Result", (), {"data": self._data})()


class _FakeRpc:
    def __init__(self, exc):
        self._exc = exc

    def execute(self):
        raise self._exc


class _FakeClient:
    def __init__(self, rpc_error, row):
        self.calls, self._rpc_error, self._row = [], rpc_error, row

    def rpc(self, name, params):
        self.calls.append(f"rpc {name}({params})")
        return _FakeRpc(self._rpc_error)

    def table(self, name):
        return _FakeQuery(self.calls, f"table {name}", [self._row])


def _patched_repo(monkeypatch, client):
    import repositories.model_versions as repo
    monkeypatch.setattr(repo, "is_configured", lambda: True)
    monkeypatch.setattr(repo, "get_client", lambda: client)
    return repo


def test_set_active_falls_back_when_the_rpc_rejects_the_unfiltered_update(monkeypatch):
    row = {"id": "abc", "version_number": "v-1", "is_active": True}
    error = Exception("{'code': '21000', 'message': 'UPDATE requires a WHERE clause'}")
    client = _FakeClient(error, row)

    result = _patched_repo(monkeypatch, client).set_active("abc")

    assert result == row
    # Deactivate before activate: idx_model_versions_one_active is a unique
    # partial index, so the other order collides with the row holding it.
    assert client.calls == [
        "rpc activate_model_version({'target_id': 'abc'})",
        "table model_versions update {'is_active': False} where is_active=True",
        "table model_versions update {'is_active': True} where id='abc'",
    ]


def test_set_active_does_not_swallow_unrelated_rpc_failures(monkeypatch):
    """A genuinely broken database must still surface, not be papered over."""
    client = _FakeClient(Exception("permission denied for relation model_versions"), {})

    with pytest.raises(Exception, match="permission denied"):
        _patched_repo(monkeypatch, client).set_active("abc")

    assert client.calls == ["rpc activate_model_version({'target_id': 'abc'})"]


def test_no_migration_issues_an_unfiltered_update_or_delete():
    """
    Supabase rejects UPDATE and DELETE with no WHERE clause. Migration 003
    shipped one and it was only found when someone pressed Activate against
    the real database for the first time.
    """
    migrations = sorted((GRAPH.parent / "db" / "migrations").glob("*.sql"))
    assert migrations, "no migrations found"

    offenders = []
    for path in migrations:
        body = re.sub(r"--[^\n]*", "", path.read_text(encoding="utf-8"))
        for statement in re.findall(r"\b(?:update|delete\s+from)\b[^;]*;", body, re.I):
            if not re.search(r"\bwhere\b", statement, re.I):
                offenders.append(f"{path.name}: {' '.join(statement.split())[:80]}")
    assert not offenders, "unfiltered UPDATE/DELETE:\n  " + "\n  ".join(offenders)
