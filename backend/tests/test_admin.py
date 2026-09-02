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

    def test_empty_state_returns_sane_defaults(self, client):
        resp = client.get("/admin/metrics", headers=ADMIN_HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data["active_model_version"] is None
        assert data["f1_per_class"] == {}
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
