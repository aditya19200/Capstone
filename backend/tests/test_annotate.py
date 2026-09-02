"""
tests/test_annotate.py — Tests for POST /annotate and GET /annotate.

No annotate tests existed before this change. Covers the audited gaps:
  - document_id is no longer required on the request, nor echoed in either
    response shape.
  - GET /annotate now returns has_conflict, predicted_label, text_excerpt.
  - GET /annotate supports ?has_conflict=true filtering.
"""

import pytest
from fastapi.testclient import TestClient

from main import app
from services import mock_db

ANNOTATOR_HEADERS = {"X-User-Id": "user-1", "X-Role": "annotator"}


@pytest.fixture(autouse=True)
def clear_mock_db():
    mock_db._predictions.clear()
    mock_db._annotations.clear()
    yield


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _make_prediction(label="Contract Law", text="The parties agree to the terms of this contract."):
    return mock_db.create_prediction(
        text_content=text,
        predicted_label=label,
        label_id=0,
        confidence=0.9,
        all_probabilities={label: 0.9},
        model_version="v1",
    )


class TestSubmitAnnotation:

    def test_document_id_not_required(self, client):
        pred = _make_prediction()

        resp = client.post(
            "/annotate",
            json={
                "prediction_id": pred["id"],
                "final_label": "Contract Law",
                "action": "accept",
            },
            headers=ANNOTATOR_HEADERS,
        )

        assert resp.status_code == 201

    def test_response_has_no_document_id_field(self, client):
        pred = _make_prediction()

        resp = client.post(
            "/annotate",
            json={
                "prediction_id": pred["id"],
                "final_label": "Contract Law",
                "action": "accept",
            },
            headers=ANNOTATOR_HEADERS,
        )

        assert "document_id" not in resp.json()

    def test_unknown_prediction_returns_404(self, client):
        resp = client.post(
            "/annotate",
            json={
                "prediction_id": "nonexistent",
                "final_label": "Contract Law",
                "action": "accept",
            },
            headers=ANNOTATOR_HEADERS,
        )
        assert resp.status_code == 404

    def test_wrong_role_returns_403(self, client):
        pred = _make_prediction()
        resp = client.post(
            "/annotate",
            json={
                "prediction_id": pred["id"],
                "final_label": "Contract Law",
                "action": "accept",
            },
            headers={"X-Role": "guest"},
        )
        assert resp.status_code == 403

    def test_modify_with_different_label_flags_conflict(self, client):
        pred = _make_prediction(label="Contract Law")

        resp = client.post(
            "/annotate",
            json={
                "prediction_id": pred["id"],
                "final_label": "Criminal Law",
                "action": "modify",
            },
            headers=ANNOTATOR_HEADERS,
        )

        assert resp.status_code == 201
        assert resp.json()["conflict_detected"] is True


class TestListAnnotations:

    def test_response_items_have_no_document_id_field(self, client):
        pred = _make_prediction()
        client.post(
            "/annotate",
            json={"prediction_id": pred["id"], "final_label": "Contract Law", "action": "accept"},
            headers=ANNOTATOR_HEADERS,
        )

        resp = client.get("/annotate", headers={"X-Role": "reviewer"})

        assert "document_id" not in resp.json()["annotations"][0]

    def test_items_include_has_conflict_predicted_label_and_excerpt(self, client):
        pred = _make_prediction(label="Contract Law", text="A" * 300)
        client.post(
            "/annotate",
            json={"prediction_id": pred["id"], "final_label": "Contract Law", "action": "accept"},
            headers=ANNOTATOR_HEADERS,
        )

        resp = client.get("/annotate", headers={"X-Role": "reviewer"})

        item = resp.json()["annotations"][0]
        assert item["has_conflict"] is False
        assert item["predicted_label"] == "Contract Law"
        assert item["text_excerpt"] == "A" * 200 + "..."

    def test_has_conflict_filter(self, client):
        clean_pred = _make_prediction(label="Contract Law")
        conflict_pred = _make_prediction(label="Contract Law")

        client.post(
            "/annotate",
            json={"prediction_id": clean_pred["id"], "final_label": "Contract Law", "action": "accept"},
            headers=ANNOTATOR_HEADERS,
        )
        client.post(
            "/annotate",
            json={"prediction_id": conflict_pred["id"], "final_label": "Criminal Law", "action": "modify"},
            headers=ANNOTATOR_HEADERS,
        )

        resp = client.get("/annotate", params={"has_conflict": "true"}, headers={"X-Role": "reviewer"})

        data = resp.json()
        assert data["total"] == 1
        assert data["annotations"][0]["has_conflict"] is True

    def test_unrecognized_role_returns_403(self, client):
        resp = client.get("/annotate", headers={"X-Role": "guest"})
        assert resp.status_code == 403

    def test_missing_role_returns_403(self, client):
        resp = client.get("/annotate")
        assert resp.status_code == 403

    def test_annotator_role_without_user_id_returns_400_not_unfiltered_data(self, client):
        """Regression test: a role recognized as 'annotator' but missing X-User-Id
        must reject with 400, not silently fall through to an unfiltered query
        that would leak every annotator's data."""
        pred = _make_prediction()
        client.post(
            "/annotate",
            json={"prediction_id": pred["id"], "final_label": "Contract Law", "action": "accept"},
            headers=ANNOTATOR_HEADERS,
        )

        resp = client.get("/annotate", headers={"X-Role": "annotator"})

        assert resp.status_code == 400

    def test_status_filter_still_works(self, client):
        pred = _make_prediction()
        client.post(
            "/annotate",
            json={"prediction_id": pred["id"], "final_label": "Contract Law", "action": "reject"},
            headers=ANNOTATOR_HEADERS,
        )

        resp = client.get("/annotate", params={"status": "rejected"}, headers={"X-Role": "reviewer"})

        assert resp.json()["total"] == 1
        assert resp.json()["annotations"][0]["annotation_status"] == "rejected"
