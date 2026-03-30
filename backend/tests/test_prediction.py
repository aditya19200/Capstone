"""
tests/test_prediction.py — Tests for the prediction pipeline.

Covers:
  - POST /predict happy path (mocked model)
  - POST /predict with pre-existing document_id
  - GET  /predict/{job_id} polling (found + not found)
  - Model unavailable (503)
  - Input validation (text too short)
  - Routing decisions reflected in response
  - XAI job queued for non-AUTO_ACCEPT predictions
  - No XAI job queued for AUTO_ACCEPT predictions

The InLegalBERT model is mocked via monkeypatch so tests run without
loading the 417MB weights.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from main import app
from services import mock_db


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_mock_db():
    """Reset all in-memory stores before each test to prevent bleed-over."""
    mock_db._users.clear()
    mock_db._legal_documents.clear()
    mock_db._predictions.clear()
    mock_db._annotations.clear()
    mock_db._explanations.clear()
    mock_db._prediction_jobs.clear()
    mock_db._xai_jobs.clear()
    mock_db._retrain_jobs.clear()
    mock_db._dataset_versions.clear()
    mock_db._model_versions.clear()
    yield


@pytest.fixture
def client():
    """FastAPI TestClient with model loading bypassed."""
    with TestClient(app) as c:
        yield c


def _make_inference_result(
    label="Contract Law",
    confidence=0.92,
    margin=0.30,
    entropy=0.25,
):
    """Build a fake model_service.predict_async return value."""
    probs = {
        "Contract Law": confidence,
        "Criminal Law": confidence - margin,
        "Constitutional Law": 0.01,
        "Corporate / Company Law": 0.01,
        "Property / Real Estate Law": 0.01,
        "Family Law": 0.01,
        "Labour & Employment Law": 0.01,
        "Intellectual Property Law": 0.01,
        "Taxation Law": 0.01,
        "Civil Procedure / Other": 0.01,
    }
    return {
        "predicted_label": label,
        "label_id": 0,
        "confidence": confidence,
        "probabilities": probs,
        "entropy": entropy,
        "margin": margin,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_model(monkeypatch, inference_result: dict):
    """
    Patch model_service so predict_async returns inference_result
    and is_loaded returns True.
    """
    import services.model_service as ms

    async def fake_predict_async(text):
        return inference_result

    monkeypatch.setattr(ms.model_service, "predict_async", fake_predict_async)
    monkeypatch.setattr(ms.model_service, "is_loaded", lambda: True)


# ===========================================================================
# POST /predict — happy paths
# ===========================================================================

class TestPredictHappyPath:

    def test_returns_200_with_all_fields(self, client, monkeypatch):
        """POST /predict returns 200 and all expected response fields."""
        _patch_model(monkeypatch, _make_inference_result())

        resp = client.post("/predict", json={"text": "The parties agree to the terms of this contract."})

        assert resp.status_code == 200
        data = resp.json()
        assert data["predicted_label"] == "Contract Law"
        assert data["confidence"] == pytest.approx(0.92, abs=1e-4)
        assert "probabilities" in data
        assert len(data["probabilities"]) == 10
        assert "entropy" in data
        assert "margin" in data
        assert "routing_decision" in data
        assert "prediction_id" in data
        assert "document_id" in data

    def test_document_created_in_mock_db(self, client, monkeypatch):
        """A new document row is persisted in mock_db on each predict call."""
        _patch_model(monkeypatch, _make_inference_result())

        client.post("/predict", json={"text": "This lease agreement grants tenancy rights."})

        assert len(mock_db._legal_documents) == 1
        assert len(mock_db._predictions) == 1

    def test_prediction_stored_in_mock_db(self, client, monkeypatch):
        """Prediction row in mock_db contains correct label and routing decision."""
        _patch_model(monkeypatch, _make_inference_result(confidence=0.92, margin=0.30))

        client.post("/predict", json={"text": "The accused was charged with felony assault."})

        pred = list(mock_db._predictions.values())[0]
        assert pred["predictedLabel"] == "Contract Law"
        assert pred["routingDecision"] == "AUTO_ACCEPT"

    def test_uses_supplied_document_id(self, client, monkeypatch):
        """When document_id is supplied, no new document is created."""
        _patch_model(monkeypatch, _make_inference_result())

        # Pre-create a document
        doc = mock_db.create_document("Pre-existing legal text.", status="pending")
        doc_id = doc["documentId"]
        initial_doc_count = len(mock_db._legal_documents)

        resp = client.post(
            "/predict",
            json={
                "text": "The parties agree to the terms of this contract.",
                "document_id": doc_id,
            },
        )

        assert resp.status_code == 200
        assert resp.json()["document_id"] == doc_id
        # No new document should be created
        assert len(mock_db._legal_documents) == initial_doc_count

    def test_document_status_updated_to_predicted(self, client, monkeypatch):
        """Document status is updated to 'predicted' after successful inference."""
        _patch_model(monkeypatch, _make_inference_result())

        client.post("/predict", json={"text": "The employee was wrongfully terminated."})

        doc = list(mock_db._legal_documents.values())[0]
        assert doc["status"] == "predicted"


# ===========================================================================
# POST /predict — routing decisions
# ===========================================================================

class TestPredictRouting:

    def test_auto_accept_no_xai_job(self, client, monkeypatch):
        """HIGH confidence prediction → AUTO_ACCEPT, no XAI job queued."""
        _patch_model(monkeypatch, _make_inference_result(confidence=0.92, margin=0.35))

        resp = client.post("/predict", json={"text": "This agreement is governed by contract law."})

        data = resp.json()
        assert data["routing_decision"] == "AUTO_ACCEPT"
        assert data["xai_job_id"] is None
        assert len(mock_db._xai_jobs) == 0

    def test_needs_explanation_queues_xai_job(self, client, monkeypatch):
        """MEDIUM confidence prediction → NEEDS_EXPLANATION, XAI job queued."""
        _patch_model(monkeypatch, _make_inference_result(confidence=0.70, margin=0.20))

        resp = client.post("/predict", json={"text": "The disputed property boundary was assessed."})

        data = resp.json()
        assert data["routing_decision"] == "NEEDS_EXPLANATION"
        assert data["xai_job_id"] is not None
        assert len(mock_db._xai_jobs) == 1
        assert len(mock_db._explanations) == 1

    def test_route_to_reviewer_queues_xai_job(self, client, monkeypatch):
        """LOW confidence prediction → ROUTE_TO_REVIEWER, XAI job queued."""
        _patch_model(monkeypatch, _make_inference_result(confidence=0.40, margin=0.05))

        resp = client.post("/predict", json={"text": "Some ambiguous legal text."})

        data = resp.json()
        assert data["routing_decision"] == "ROUTE_TO_REVIEWER"
        assert data["xai_job_id"] is not None
        assert len(mock_db._xai_jobs) == 1

    def test_high_confidence_low_margin_needs_explanation(self, client, monkeypatch):
        """High raw confidence but low margin → NEEDS_EXPLANATION (not AUTO_ACCEPT)."""
        _patch_model(monkeypatch, _make_inference_result(confidence=0.87, margin=0.05))

        resp = client.post("/predict", json={"text": "The contract clause is ambiguous."})

        data = resp.json()
        # margin < 0.10 guard overrides AUTO_ACCEPT
        assert data["routing_decision"] == "NEEDS_EXPLANATION"
        assert data["xai_job_id"] is not None


# ===========================================================================
# POST /predict — error cases
# ===========================================================================

class TestPredictErrors:

    def test_model_unavailable_returns_503(self, client, monkeypatch):
        """503 is returned when the model raises RuntimeError."""
        import services.model_service as ms

        async def broken_predict(text):
            raise RuntimeError("Model not loaded")

        monkeypatch.setattr(ms.model_service, "predict_async", broken_predict)
        monkeypatch.setattr(ms.model_service, "is_loaded", lambda: False)

        resp = client.post("/predict", json={"text": "Valid legal text for testing purposes."})

        assert resp.status_code == 503
        assert "not available" in resp.json()["detail"].lower()

    def test_text_too_short_returns_422(self, client):
        """Pydantic rejects text shorter than 10 characters with 422."""
        resp = client.post("/predict", json={"text": "Short"})
        assert resp.status_code == 422

    def test_missing_text_returns_422(self, client):
        """Missing 'text' field returns 422 validation error."""
        resp = client.post("/predict", json={})
        assert resp.status_code == 422

    def test_empty_body_returns_422(self, client):
        """Empty body returns 422 validation error."""
        resp = client.post("/predict", content=b"", headers={"Content-Type": "application/json"})
        assert resp.status_code == 422


# ===========================================================================
# GET /predict/{job_id}
# ===========================================================================

class TestGetPredictionJob:

    def test_unknown_job_id_returns_404(self, client):
        """Non-existent job_id returns 404."""
        resp = client.get("/predict/nonexistent-job-id")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_pending_job_returns_correct_status(self, client):
        """A newly enqueued job has status='pending' and no result."""
        job = mock_db.enqueue_prediction_job("Some legal text for batch processing.")

        resp = client.get(f"/predict/{job['jobId']}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "pending"
        assert data["result"] is None

    def test_completed_job_returns_result(self, client, monkeypatch):
        """A completed job returns status='completed' and the full result."""
        _patch_model(monkeypatch, _make_inference_result())

        job = mock_db.enqueue_prediction_job("Completed prediction job text.")
        result_payload = {
            "prediction_id": "pred-123",
            "document_id": "doc-123",
            "predicted_label": "Contract Law",
            "confidence": 0.92,
            "probabilities": {k: 0.1 for k in [
                "Contract Law", "Criminal Law", "Constitutional Law",
                "Corporate / Company Law", "Property / Real Estate Law",
                "Family Law", "Labour & Employment Law",
                "Intellectual Property Law", "Taxation Law",
                "Civil Procedure / Other",
            ]},
            "routing_decision": "AUTO_ACCEPT",
            "xai_job_id": None,
            "entropy": 0.25,
            "margin": 0.30,
        }
        mock_db.update_prediction_job(
            job_id=job["jobId"], status="completed", result=result_payload
        )

        resp = client.get(f"/predict/{job['jobId']}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["result"]["predicted_label"] == "Contract Law"
        assert data["result"]["routing_decision"] == "AUTO_ACCEPT"

    def test_failed_job_returns_failed_status(self, client):
        """A failed job returns status='failed' with no result."""
        job = mock_db.enqueue_prediction_job("This job will fail.")
        mock_db.update_prediction_job(job_id=job["jobId"], status="failed")

        resp = client.get(f"/predict/{job['jobId']}")

        assert resp.status_code == 200
        assert resp.json()["status"] == "failed"
        assert resp.json()["result"] is None


# ===========================================================================
# GET /health — sanity check
# ===========================================================================

class TestHealth:

    def test_health_returns_ok(self, client, monkeypatch):
        """GET /health returns status=ok when model is loaded."""
        import services.model_service as ms
        monkeypatch.setattr(ms.model_service, "is_loaded", lambda: True)

        resp = client.get("/health")

        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert resp.json()["model_loaded"] is True
