"""
tests/test_queue.py — Tests for GET /queue/low-confidence.

Covers repositories.predictions.list_low_confidence() directly (already
existed, but nothing exercised it before this route wrapped it) and the
route's pagination/threshold behaviour.
"""

import pytest
from fastapi.testclient import TestClient

from config.settings import settings
from main import app
from repositories import predictions as predictions_repo
from services import mock_db


@pytest.fixture(autouse=True)
def clear_mock_db():
    # Annotations too: the queue now filters out predictions a human has
    # already acted on, so a leftover annotation from another test would
    # silently empty the queue here.
    mock_db._predictions.clear()
    mock_db._annotations.clear()
    yield


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


REVIEWER_HEADERS = {"X-Role": "reviewer"}
ADMIN_HEADERS = {"X-Role": "admin"}


def _make_prediction(confidence: float, label: str = "Contract Law"):
    return mock_db.create_prediction(
        text_content=f"Some legal text at confidence {confidence}.",
        predicted_label=label,
        label_id=0,
        confidence=confidence,
        all_probabilities={label: confidence},
        model_version="v1",
    )


class TestPredictionsRepoListLowConfidence:

    def test_returns_only_rows_below_threshold(self):
        _make_prediction(0.3)
        _make_prediction(0.9)
        rows = predictions_repo.list_low_confidence(0.5)
        assert len(rows) == 1
        assert rows[0]["confidence"] == pytest.approx(0.3)

    def test_empty_when_nothing_below_threshold(self):
        _make_prediction(0.9)
        assert predictions_repo.list_low_confidence(0.5) == []


class TestLowConfidenceQueueRoute:

    def test_only_includes_predictions_below_review_threshold(self, client):
        below = _make_prediction(settings.REVIEW_THRESHOLD - 0.1)
        _make_prediction(settings.REVIEW_THRESHOLD + 0.1)

        resp = client.get("/queue/low-confidence", headers=REVIEWER_HEADERS)

        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["prediction_id"] == below["id"]

    def test_item_shape(self, client):
        _make_prediction(0.2, label="Criminal Law")

        resp = client.get("/queue/low-confidence", headers=REVIEWER_HEADERS)

        item = resp.json()["items"][0]
        assert item["predicted_label"] == "Criminal Law"
        assert item["confidence"] == pytest.approx(0.2)
        assert "text_excerpt" in item
        assert "created_at" in item

    def test_pagination(self, client):
        for i in range(3):
            _make_prediction(0.1 + i * 0.01)

        resp = client.get(
            "/queue/low-confidence",
            params={"page": 1, "page_size": 2},
            headers=REVIEWER_HEADERS,
        )

        data = resp.json()
        assert data["total"] == 3
        assert data["page"] == 1
        assert data["page_size"] == 2
        assert len(data["items"]) == 2

    def test_empty_queue_returns_empty_list(self, client):
        resp = client.get("/queue/low-confidence", headers=REVIEWER_HEADERS)
        assert resp.status_code == 200
        assert resp.json() == {"total": 0, "page": 1, "page_size": 50, "items": []}

    def test_annotated_predictions_do_not_come_back(self, client):
        """
        A prediction a human has dealt with must not reappear in the queue.

        The queue is built from predictions, which carry no record of having
        been reviewed. The Review screen removes a row as soon as its POST
        succeeds, so without this filter a single sitting looks correct and a
        page refresh brings every corrected item back.
        """
        reviewed = _make_prediction(0.2, label="Family Law")
        untouched = _make_prediction(0.2, label="Contract Law")

        assert client.get("/queue/low-confidence",
                          headers=REVIEWER_HEADERS).json()["total"] == 2

        mock_db.create_annotation(
            prediction_id=reviewed["id"],
            validated_label="Criminal Law",
            annotator_id="reviewer-1",
            status="validated",
        )

        data = client.get("/queue/low-confidence", headers=REVIEWER_HEADERS).json()
        assert data["total"] == 1
        assert data["items"][0]["prediction_id"] == untouched["id"]

    def test_rejected_predictions_do_not_come_back_either(self, client):
        """
        'rejected' means the reviewer threw the document out. That is a
        decision, not a deferral, so it must not be queued again.
        """
        thrown_out = _make_prediction(0.2)
        mock_db.create_annotation(
            prediction_id=thrown_out["id"],
            validated_label="Contract Law",
            annotator_id="reviewer-1",
            status="rejected",
        )

        assert client.get("/queue/low-confidence",
                          headers=REVIEWER_HEADERS).json()["total"] == 0


class TestLowConfidenceQueueRoleCheck:

    def test_missing_role_returns_403(self, client):
        resp = client.get("/queue/low-confidence")
        assert resp.status_code == 403

    def test_wrong_role_returns_403(self, client):
        resp = client.get("/queue/low-confidence", headers={"X-Role": "annotator"})
        assert resp.status_code == 403

    def test_reviewer_role_returns_200(self, client):
        resp = client.get("/queue/low-confidence", headers=REVIEWER_HEADERS)
        assert resp.status_code == 200

    def test_admin_role_returns_200(self, client):
        resp = client.get("/queue/low-confidence", headers=ADMIN_HEADERS)
        assert resp.status_code == 200


class TestSupersededPredictions:
    """An item predicted before a retrain says so, and stays in the queue."""

    def test_marks_items_from_a_version_no_longer_serving(self, monkeypatch):
        from fastapi.testclient import TestClient
        from main import app
        from services import mock_db
        from services.model_service import model_service

        mock_db._predictions.clear()
        mock_db._annotations.clear()
        with TestClient(app) as client:
            monkeypatch.setattr(model_service, "_loaded_version", "v-new")
            for version in ("Legal Baseline", "v-new"):
                mock_db.create_prediction(
                    text_content=f"text from {version}", predicted_label="Family Law",
                    label_id=5, confidence=0.3, all_probabilities={},
                    model_version=version)
            items = {i["predicted_by"]: i for i in client.get(
                "/queue/low-confidence", headers={"X-Role": "reviewer"}).json()["items"]}

        assert items["Legal Baseline"]["superseded"] is True
        assert items["v-new"]["superseded"] is False
