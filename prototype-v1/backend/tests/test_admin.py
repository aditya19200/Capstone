"""
tests/test_admin.py — Tests for POST /admin/models/{id}/activate and
GET /admin/metrics.

Also covers repositories.annotations.count_by_status() and
mock_db.count_annotations_by_status() directly, since GET /admin/metrics
depends on them but a passing route test alone wouldn't prove the counting
logic itself is correct.
"""

import pytest
from fastapi.testclient import TestClient

from main import app
from repositories import annotations as annotations_repo
from repositories import model_versions as model_versions_repo
from models.response_models import LEGAL_LABELS
from services import mock_db

ADMIN_HEADERS = {"X-Role": "admin"}


@pytest.fixture(autouse=True)
def clear_mock_db():
    mock_db._predictions.clear()
    mock_db._annotations.clear()
    mock_db._model_versions.clear()
    mock_db._batches.clear()
    mock_db._batch_items.clear()
    yield


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _make_annotation(status: str):
    pred = mock_db.create_prediction(
        text_content="Some legal text.",
        predicted_label="Contract Law",
        label_id=0,
        confidence=0.9,
        all_probabilities={"Contract Law": 0.9},
        model_version="v1",
    )
    return mock_db.create_annotation(
        prediction_id=pred["id"],
        validated_label="Contract Law",
        annotator_id="user-1",
        status=status,
    )


# ===========================================================================
# mock_db.count_annotations_by_status — unit test
# ===========================================================================

class TestMockDbCountAnnotationsByStatus:

    def test_counts_grouped_correctly(self):
        _make_annotation("validated")
        _make_annotation("validated")
        _make_annotation("pending")
        counts = mock_db.count_annotations_by_status()
        assert counts == {"validated": 2, "pending": 1}

    def test_empty_when_no_annotations(self):
        assert mock_db.count_annotations_by_status() == {}


# ===========================================================================
# repositories.annotations.count_by_status — unit test
# ===========================================================================

class TestAnnotationsRepoCountByStatus:

    def test_counts_grouped_correctly(self):
        _make_annotation("rejected")
        _make_annotation("validated")
        counts = annotations_repo.count_by_status()
        assert counts == {"rejected": 1, "validated": 1}

    def test_zero_count_statuses_omitted(self):
        """Only 'validated' rows exist — 'pending'/'rejected' must not appear
        as zero-value keys, matching the mock branch's shape."""
        _make_annotation("validated")
        counts = annotations_repo.count_by_status()
        assert counts == {"validated": 1}
        assert "pending" not in counts
        assert "rejected" not in counts

    def test_real_branch_matches_mock_branch_shape_for_same_data(self, monkeypatch):
        """
        GET /admin/metrics must not change shape depending on whether Supabase
        is configured. Fake the real-Supabase client chain with the exact same
        underlying counts as the mock branch and assert identical output.
        """
        _make_annotation("validated")
        _make_annotation("validated")
        _make_annotation("rejected")

        mock_counts = annotations_repo.count_by_status()

        class _FakeResult:
            def __init__(self, count):
                self.count = count

        class _FakeQuery:
            def __init__(self, counts_by_status):
                self._counts = counts_by_status
                self._status = None

            def table(self, name):
                return self

            def select(self, *args, **kwargs):
                return self

            def eq(self, field, value):
                self._status = value
                return self

            def execute(self):
                return _FakeResult(self._counts.get(self._status, 0))

        fake_counts = {"validated": 2, "rejected": 1, "pending": 0}
        monkeypatch.setattr(
            "repositories.annotations.is_configured", lambda: True
        )
        monkeypatch.setattr(
            "repositories.annotations.get_client", lambda: _FakeQuery(fake_counts)
        )

        real_counts = annotations_repo.count_by_status()

        assert real_counts == mock_counts == {"validated": 2, "rejected": 1}


# ===========================================================================
# POST /admin/models/{version_id}/activate
# ===========================================================================

class TestActivateModelVersion:

    def test_non_admin_returns_403(self, client):
        resp = client.post("/admin/models/some-id/activate", headers={"X-Role": "annotator"})
        assert resp.status_code == 403

    def test_unknown_version_id_returns_404(self, client):
        resp = client.post("/admin/models/nonexistent/activate", headers=ADMIN_HEADERS)
        assert resp.status_code == 404

    def test_happy_path_activates_and_returns_row(self, client):
        version = model_versions_repo.insert(
            version_number="v2",
            accuracy=0.91,
            f1_per_class={"Contract Law": 0.9},
            file_path="/models/v2",
        )

        resp = client.post(f"/admin/models/{version['id']}/activate", headers=ADMIN_HEADERS)

        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == version["id"]
        assert data["is_active"] is True
        assert data["version_number"] == "v2"

    def test_activating_new_version_deactivates_previous(self, client):
        v1 = model_versions_repo.insert(
            version_number="v1", accuracy=0.8, f1_per_class={}, file_path="/models/v1"
        )
        model_versions_repo.set_active(v1["id"])
        v2 = model_versions_repo.insert(
            version_number="v2", accuracy=0.9, f1_per_class={}, file_path="/models/v2"
        )

        client.post(f"/admin/models/{v2['id']}/activate", headers=ADMIN_HEADERS)

        assert model_versions_repo.get_active()["id"] == v2["id"]


# ===========================================================================
# GET /admin/metrics
# ===========================================================================

class TestAdminMetrics:

    def test_non_admin_returns_403(self, client):
        resp = client.get("/admin/metrics", headers={"X-Role": "reviewer"})
        assert resp.status_code == 403

    def test_fresh_state_reports_the_baseline_and_no_activity(self, client):
        """
        What a fresh install shows: the deployed model, and nothing done yet.

        There is no longer a state with no model at all. main.py calls
        register_baseline_if_absent() at startup, which registers the shipped
        checkpoint from backend/metrics/baseline_metrics.json so the admin
        screen opens describing the model actually being served rather than
        blank. So the model fields are populated from that file, while the
        fields that record *activity* — annotations, batches — are empty,
        because none has happened.
        """
        resp = client.get("/admin/metrics", headers=ADMIN_HEADERS)
        assert resp.status_code == 200
        data = resp.json()

        # From the registered baseline.
        assert data["active_model_version"] is not None
        assert "baseline" in data["active_model_version"].lower()
        # And it is the one actually loaded, not just a row saying so.
        assert data["serving_model_version"] == data["active_model_version"]
        assert set(data["f1_per_class"]) == set(LEGAL_LABELS)

        # Nothing has been annotated or uploaded yet.
        assert data["annotation_counts"] == {}
        assert data["batch_throughput"] == []

    def test_reflects_active_model_annotations_and_batches(self, client):
        version = model_versions_repo.insert(
            version_number="v3",
            accuracy=0.95,
            f1_per_class={"Contract Law": 0.93},
            file_path="/models/v3",
        )
        model_versions_repo.set_active(version["id"])
        _make_annotation("validated")
        client.post("/batches/paste", json={"texts": ["a", "b"]})

        resp = client.get("/admin/metrics", headers=ADMIN_HEADERS)

        data = resp.json()
        assert data["active_model_version"] == "v3"
        assert data["f1_per_class"] == {"Contract Law": 0.93}
        assert data["annotation_counts"] == {"validated": 1}
        assert len(data["batch_throughput"]) == 1
        assert data["batch_throughput"][0]["total_items"] == 2


class TestServingVersusActive:
    """
    is_active is what the database says should load; is_serving is what did.

    They are separate because they really can disagree, and the screen that
    hides the difference is the one that tells an operator a model is in use
    when nothing has loaded it. This is not hypothetical: the deployed
    baseline sat in the table marked Inactive while answering every request,
    because the insert succeeded and the activation that follows it failed.
    """

    def test_metrics_marks_the_loaded_version_as_serving(self, client):
        from services.model_service import model_service

        served = model_versions_repo.insert(
            version_number="v-served", accuracy=0.9, f1_per_class={},
            file_path=model_service.loaded_path or "./models/LegalModelShared",
        )
        other = model_versions_repo.insert(
            version_number="v-elsewhere", accuracy=0.9, f1_per_class={},
            file_path="/models/somewhere-else",
        )

        rows = {v["version_number"]: v
                for v in client.get("/admin/metrics",
                                    headers=ADMIN_HEADERS).json()["model_versions"]}
        assert rows["v-served"]["is_serving"] is True
        assert rows["v-elsewhere"]["is_serving"] is False
        assert served["id"] != other["id"]

    def test_a_version_flagged_active_but_not_loaded_is_not_serving(self, client):
        version = model_versions_repo.insert(
            version_number="v-never-loaded", accuracy=0.9, f1_per_class={},
            file_path="/models/not-on-this-disk",
        )
        model_versions_repo.set_active(version["id"])

        data = client.get("/admin/metrics", headers=ADMIN_HEADERS).json()
        row = next(v for v in data["model_versions"]
                   if v["version_number"] == "v-never-loaded")
        assert row["is_active"] is True
        assert row["is_serving"] is False
        assert data["active_model_version"] == "v-never-loaded"
        assert data["serving_model_version"] != "v-never-loaded"


class TestReconcileServedVersion:
    """
    Startup makes the table describe the model that is actually loaded.

    register_baseline_if_absent() returns early once any row exists, so a row
    that landed with the wrong name, or without the activation that was meant
    to follow it, stayed wrong for the life of the database.
    """

    def test_marks_the_served_row_active_when_nothing_is(self):
        from baseline import reconcile_served_version

        row = model_versions_repo.insert(
            version_number="Baseline", accuracy=0.7, f1_per_class={},
            file_path="./models/LegalModelShared",
        )
        assert row["is_active"] is False

        reconcile_served_version("./models/LegalModelShared")

        assert model_versions_repo.get_active()["id"] == row["id"]

    def test_renames_a_row_named_after_a_directory(self, tmp_path, monkeypatch):
        import json as _json
        import baseline
        from baseline import reconcile_served_version

        metrics = tmp_path / "legal_baseline_metrics.json"
        metrics.write_text(_json.dumps({"version_number": "Baseline"}))
        monkeypatch.setattr(baseline, "_metrics_file", lambda: metrics)

        row = model_versions_repo.insert(
            version_number="baseline-LegalModelShared", accuracy=0.7,
            f1_per_class={}, file_path="./models/LegalModelShared",
        )
        reconcile_served_version("./models/LegalModelShared")

        assert model_versions_repo.list_all()[0]["version_number"] == "Baseline"
        assert model_versions_repo.get_active()["id"] == row["id"]

    def test_leaves_a_retrained_versions_name_alone(self, tmp_path, monkeypatch):
        import json as _json
        import baseline
        from baseline import reconcile_served_version

        metrics = tmp_path / "legal_baseline_metrics.json"
        metrics.write_text(_json.dumps({"version_number": "Baseline"}))
        monkeypatch.setattr(baseline, "_metrics_file", lambda: metrics)

        model_versions_repo.insert(
            version_number="v-20260929-090239", accuracy=0.81, f1_per_class={},
            file_path="./models/v-20260929-090239",
        )
        reconcile_served_version("./models/v-20260929-090239")

        assert model_versions_repo.list_all()[0]["version_number"] == "v-20260929-090239"

    def test_does_not_steal_the_flag_from_a_deliberate_activation(self):
        from baseline import reconcile_served_version

        chosen = model_versions_repo.insert(
            version_number="v-chosen", accuracy=0.9, f1_per_class={},
            file_path="/models/v-chosen",
        )
        model_versions_repo.set_active(chosen["id"])
        model_versions_repo.insert(
            version_number="Baseline", accuracy=0.7, f1_per_class={},
            file_path="./models/LegalModelShared",
        )

        # Startup tried to load v-chosen and could not, so MODEL_PATH is what
        # is serving. The flag still records what was asked for.
        reconcile_served_version("./models/LegalModelShared")

        assert model_versions_repo.get_active()["id"] == chosen["id"]

    def test_is_a_no_op_with_no_versions_at_all(self):
        from baseline import reconcile_served_version
        reconcile_served_version("./models/LegalModelShared")
        assert model_versions_repo.list_all() == []


class TestWorkersDoNotClobberTheResidentModel:
    """
    A worker must not reload MODEL_PATH over an already-resident model.

    With no Supabase configured the batch, XAI and retrain workers run as
    threads inside the API process (local_workers.py), sharing one
    model_service. The XAI worker called load() unconditionally, so seconds
    after a startup that had correctly loaded an activated version, that
    thread put settings.MODEL_PATH back — the baseline — and dropped the
    version label with it, leaving every prediction stamped "baseline"
    whatever weights produced it.
    """

    def test_ensure_loaded_does_not_reload_what_is_already_resident(self, monkeypatch):
        from services.model_service import model_service

        calls = []
        monkeypatch.setattr(model_service, "load",
                            lambda *a, **k: calls.append((a, k)))
        monkeypatch.setattr(model_service, "is_loaded", lambda: True)

        model_service.ensure_loaded()
        assert calls == []

    def test_ensure_loaded_loads_when_nothing_is_resident(self, monkeypatch):
        from services.model_service import model_service

        calls = []
        monkeypatch.setattr(model_service, "load",
                            lambda *a, **k: calls.append((a, k)))
        monkeypatch.setattr(model_service, "is_loaded", lambda: False)

        model_service.ensure_loaded()
        assert len(calls) == 1

    def test_no_worker_calls_load_directly(self):
        """
        The guard only works if every worker goes through it.

        Checked by reading the sources rather than by running the workers,
        because each one's main() is an infinite poll loop. A new worker that
        reaches for load() is the exact mistake this is here to catch.
        """
        import pathlib
        workers = pathlib.Path(__file__).resolve().parent.parent / "workers"
        offenders = [
            path.name for path in sorted(workers.glob("*.py"))
            if "model_service.load(" in path.read_text()
        ]
        assert offenders == [], (
            f"{offenders} call model_service.load() directly. Use "
            "model_service.ensure_loaded() — see its docstring."
        )


class TestACrashingWorkerDoesNotTakeTheApiDown:
    """
    The other half of the shared-model problem, found on 23 Oct.

    ensure_loaded() stopped workers *loading* over the API's model. Their
    shutdown still *unloaded* it: both xai_worker and prediction_worker end
    in `finally: model_service.unload()`, which is right for a dedicated
    worker process that owns its weights and wrong for a thread that does
    not. A DNS blip ("Temporary failure in name resolution") killed the XAI
    thread, its finally ran, and the API served /health with
    model_loaded=false until somebody restarted it.
    """

    def test_release_keeps_a_shared_model_loaded(self, monkeypatch):
        from services.model_service import model_service

        unloaded = []
        monkeypatch.setattr(model_service, "unload", lambda: unloaded.append(True))
        monkeypatch.setattr(model_service, "_shared", True, raising=False)

        model_service.release()
        assert unloaded == [], "release() unloaded a model shared with the API"

    def test_release_unloads_when_the_worker_owns_the_model(self, monkeypatch):
        from services.model_service import model_service

        unloaded = []
        monkeypatch.setattr(model_service, "unload", lambda: unloaded.append(True))
        monkeypatch.setattr(model_service, "_shared", False, raising=False)

        model_service.release()
        assert unloaded == [True], "a standalone worker process must still clean up"

    def test_no_worker_calls_unload_directly(self):
        """Same reasoning as test_no_worker_calls_load_directly."""
        import pathlib
        workers = pathlib.Path(__file__).resolve().parent.parent / "workers"
        offenders = [
            path.name for path in sorted(workers.glob("*.py"))
            if "model_service.unload(" in path.read_text()
        ]
        assert offenders == [], (
            f"{offenders} call model_service.unload() directly. Use "
            "model_service.release(), which is a no-op while the weights are "
            "shared with the API process."
        )

    def test_starting_local_workers_marks_the_model_shared(self, monkeypatch):
        import local_workers
        from services.model_service import model_service

        monkeypatch.setattr(model_service, "_shared", False, raising=False)
        monkeypatch.setattr(local_workers.threading, "Thread",
                            lambda **kw: type("T", (), {"start": lambda self: None})())
        local_workers.start_local_workers()
        assert model_service._shared is True

    def test_a_crashed_loop_is_restarted(self, monkeypatch):
        """
        A transient failure must not retire the worker permanently.

        The loop raises twice and then returns; _supervise should have called
        it three times rather than giving up after the first exception.
        """
        import local_workers

        monkeypatch.setattr(local_workers.time, "sleep", lambda _s: None)
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise ConnectionError("Temporary failure in name resolution")

        local_workers._supervise("test", flaky)
        assert len(calls) == 3

    def test_backoff_grows_and_resets_after_a_healthy_run(self, monkeypatch):
        """A blip after a long healthy run restarts fast, not at the cap."""
        import local_workers

        slept = []
        monkeypatch.setattr(local_workers.time, "sleep", lambda s: slept.append(s))
        clock = {"t": 0.0}
        monkeypatch.setattr(local_workers.time, "monotonic", lambda: clock["t"])

        calls = []

        def flaky():
            calls.append(1)
            # Third run is healthy for well over the threshold before failing.
            clock["t"] += 600 if len(calls) == 3 else 0
            if len(calls) < 5:
                raise ConnectionError("boom")

        local_workers._supervise("test", flaky)
        assert slept[:2] == [2, 5], f"backoff should grow: {slept}"
        assert slept[2] == 2, f"a healthy run should reset the backoff: {slept}"
