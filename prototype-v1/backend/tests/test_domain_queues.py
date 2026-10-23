"""
Migration 010: two domains on one database without taking each other's work.

The live database cannot run the post-010 paths until someone applies the
migration, so these drive the repositories against a fake client that records
every call, and assert that each queue and the active flag are scoped to the
domain once domain_queues_available() is true -- and untouched before it.
"""

import pytest

import repositories._client as client_mod


class _Query:
    def __init__(self, log, table):
        self.log, self.table, self.filters = log, table, []
        self.payload = None

    def select(self, *_a, **_k):
        return self

    def insert(self, payload):
        self.payload = payload
        return self

    def update(self, payload):
        self.payload = payload
        return self

    def eq(self, col, val):
        self.filters.append((col, val))
        return self

    def order(self, *_a, **_k):
        return self

    def limit(self, *_a):
        return self

    def execute(self):
        self.log.append(("table", self.table, tuple(self.filters), self.payload))
        row = dict(self.payload or {}, id="row-1", is_active=False)
        return type("R", (), {"data": [row]})()


class _Rpc:
    def __init__(self, log, name, params):
        log.append(("rpc", name, params))

    def execute(self):
        return type("R", (), {"data": [{"id": "row-1"}]})()


class FakeClient:
    def __init__(self):
        self.log = []

    def table(self, name):
        return _Query(self.log, name)

    def rpc(self, name, params):
        return _Rpc(self.log, name, params)


@pytest.fixture
def fake(monkeypatch):
    """A configured, post- or pre-010 database, serving the medical domain."""
    c = FakeClient()

    def use(after_010: bool):
        import repositories.batches as batches
        import repositories.batch_items as batch_items
        import repositories.xai_jobs as xai_jobs
        import repositories.model_versions as model_versions

        for mod in (client_mod, batches, batch_items, xai_jobs, model_versions):
            monkeypatch.setattr(mod, "is_configured", lambda: True, raising=False)
            monkeypatch.setattr(mod, "get_client", lambda: c, raising=False)
            monkeypatch.setattr(mod, "domain_queues_available", lambda: after_010,
                                raising=False)
            monkeypatch.setattr(mod, "current_domain", lambda: "medical", raising=False)
        import services._supabase_real as real
        monkeypatch.setattr(real, "get_client", lambda: c)
        return c

    return use


def _rpcs(c):
    return [e for e in c.log if e[0] == "rpc"]


class TestAfterMigration010:
    def test_batch_claim_is_scoped_to_the_domain(self, fake):
        c = fake(True)
        from repositories import batch_items
        batch_items.claim(5)
        assert _rpcs(c) == [("rpc", "claim_batch_items_for", {"n": 5, "d": "medical"})]

    def test_xai_claim_is_scoped_to_the_domain(self, fake):
        c = fake(True)
        from repositories import xai_jobs
        xai_jobs.claim()
        assert _rpcs(c) == [("rpc", "claim_xai_job_for", {"d": "medical"})]

    def test_a_new_batch_is_stamped_with_the_domain(self, fake):
        c = fake(True)
        from repositories import batches
        batches.insert(source="pdf", total_items=3, filename="x.pdf")
        assert c.log[0][3]["domain"] == "medical"

    def test_batches_list_only_this_domain(self, fake):
        c = fake(True)
        from repositories import batches
        batches.list_all()
        assert ("domain", "medical") in c.log[0][2]

    def test_active_version_is_this_domains(self, fake):
        c = fake(True)
        from repositories import model_versions
        model_versions.get_active()
        assert ("domain", "medical") in c.log[0][2]

    def test_activation_uses_the_domain_scoped_function(self, fake):
        c = fake(True)
        from repositories import model_versions
        model_versions.set_active("v-1")
        assert _rpcs(c) == [("rpc", "activate_model_version_for", {"target_id": "v-1"})]

    def test_a_baseline_is_stamped_with_its_own_domain_not_the_process(self, fake):
        """One process registers every domain's baseline."""
        c = fake(True)
        from repositories import model_versions
        model_versions.insert(version_number="Legal Baseline", accuracy=0.7,
                              f1_per_class={}, file_path="./models/LegalModelShared",
                              domain="legal")
        assert c.log[0][3]["domain"] == "legal"

    def test_retrain_jobs_are_stamped_and_read_per_domain(self, fake, monkeypatch):
        c = fake(True)
        import services._supabase_real as real
        real.enqueue_retrain_job(triggered_by=None)
        real.get_latest_retrain_job()
        insert, latest = c.log[0], c.log[1]
        assert insert[3]["domain"] == "medical"
        assert ("domain", "medical") in latest[2]


class TestBeforeMigration010:
    """Nothing may change until the migration exists: the columns are not there."""

    def test_claims_use_the_original_functions(self, fake):
        c = fake(False)
        from repositories import batch_items, xai_jobs
        batch_items.claim(5)
        xai_jobs.claim()
        assert [e[1] for e in _rpcs(c)] == ["claim_batch_items", "claim_xai_job"]

    def test_no_domain_is_written_or_filtered(self, fake):
        c = fake(False)
        from repositories import batches, model_versions
        batches.insert(source="csv", total_items=1)
        model_versions.get_active()
        assert "domain" not in c.log[0][3]
        assert not any(col == "domain" for col, _ in c.log[1][2])
