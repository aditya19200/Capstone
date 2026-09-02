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
    mock_db._predictions.clear()
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
