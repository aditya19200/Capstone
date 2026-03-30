"""
tests/test_shap.py — Tests for the SHAP explainability service.

Covers:
  - explain() returns correct structure (list of {token, importance})
  - explain() sorts by absolute importance descending
  - explain() filters special tokens ([CLS], [SEP], [PAD])
  - explain_safe() returns None on failure (no exception raised)
  - explain_safe() returns valid result on success
  - explain() raises RuntimeError when model not loaded
  - explain() raises ValueError on empty text
  - _build_token_importances helper: ordering, special token filtering
  - POST /explain endpoint: trigger + dedup + 404 on missing prediction
  - GET  /explain/{prediction_id}: pending, completed, not-found

The SHAP computation itself is mocked so tests run without loading the model
or running actual perturbation-based explanations.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from main import app
from services import mock_db
from services.shap_service import SHAPService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_mock_db():
    """Reset all in-memory stores before each test."""
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
    """FastAPI TestClient."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def shap_svc():
    """Return a fresh SHAPService instance for unit tests."""
    return SHAPService()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_SHAP_OUTPUT = [
    {"token": "contract",   "importance":  0.42},
    {"token": "agreement",  "importance":  0.31},
    {"token": "parties",    "importance": -0.18},
    {"token": "terms",      "importance":  0.15},
    {"token": "breach",     "importance": -0.09},
]

_SAMPLE_TEXT = "The parties agree to the terms of this contract agreement."
_SAMPLE_LABEL_ID = 0  # Contract Law


def _seed_prediction(text: str = _SAMPLE_TEXT, label: str = "Contract Law") -> dict:
    """Insert a document + prediction into mock_db and return the prediction row."""
    doc  = mock_db.create_document(text_content=text)
    pred = mock_db.create_prediction(
        document_id=doc["documentId"],
        predicted_label=label,
        confidence_score=0.92,
        probability_distribution={label: 0.92},
        routing_decision="NEEDS_EXPLANATION",
        entropy=0.3,
        margin=0.25,
    )
    return pred


def _patch_shap(monkeypatch, return_value=None, raise_exc=None):
    """
    Patch shap_service.explain so it either returns return_value or raises raise_exc.
    Patches both the module-level singleton and any newly created instances.
    """
    import services.shap_service as ss

    if raise_exc is not None:
        def fake_explain(self, text, predicted_label_id, max_evals=200):
            raise raise_exc
    else:
        def fake_explain(self, text, predicted_label_id, max_evals=200):
            return return_value

    monkeypatch.setattr(ss.SHAPService, "explain", fake_explain)


# ===========================================================================
# Unit tests — SHAPService._build_token_importances
# ===========================================================================

class TestBuildTokenImportances:
    """Tests for the static helper that converts raw arrays → sorted list."""

    def test_sorted_by_absolute_importance(self, shap_svc):
        """Tokens are sorted by |importance| descending."""
        import numpy as np

        tokens = np.array(["breach", "agreement", "contract", "the"])
        scores = np.array([-0.50,      0.30,         0.80,     0.01])

        result = SHAPService._build_token_importances(tokens, scores)

        importances = [abs(r["importance"]) for r in result]
        assert importances == sorted(importances, reverse=True)

    def test_top_token_is_most_important(self, shap_svc):
        """First token in result has the highest absolute importance."""
        import numpy as np

        tokens = np.array(["small", "huge"])
        scores = np.array([0.05, -0.99])

        result = SHAPService._build_token_importances(tokens, scores)

        assert result[0]["token"] == "huge"
        assert result[0]["importance"] == pytest.approx(-0.99, abs=1e-6)

    def test_special_tokens_filtered(self, shap_svc):
        """[CLS], [SEP], [PAD] and their variants are excluded from output."""
        import numpy as np

        tokens = np.array(["[CLS]", "contract", "[SEP]", "law", "[PAD]", "<s>", "</s>"])
        scores = np.array([0.99,      0.50,      0.98,    0.30,   0.01,  0.95,   0.94])

        result = SHAPService._build_token_importances(tokens, scores)

        returned_tokens = {r["token"] for r in result}
        for special in {"[CLS]", "[SEP]", "[PAD]", "<s>", "</s>"}:
            assert special not in returned_tokens

    def test_only_content_tokens_remain(self, shap_svc):
        """After filtering specials, only content tokens are returned."""
        import numpy as np

        tokens = np.array(["[CLS]", "agreement", "[SEP]"])
        scores = np.array([0.9, 0.5, 0.8])

        result = SHAPService._build_token_importances(tokens, scores)

        assert len(result) == 1
        assert result[0]["token"] == "agreement"

    def test_importance_rounded_to_six_decimals(self, shap_svc):
        """Importance values are rounded to 6 decimal places."""
        import numpy as np

        tokens = np.array(["contract"])
        scores = np.array([0.123456789])

        result = SHAPService._build_token_importances(tokens, scores)

        assert result[0]["importance"] == pytest.approx(0.123457, abs=1e-6)

    def test_empty_input_returns_empty_list(self, shap_svc):
        """Empty token array returns empty list without error."""
        import numpy as np

        result = SHAPService._build_token_importances(np.array([]), np.array([]))
        assert result == []


# ===========================================================================
# Unit tests — SHAPService.explain() and explain_safe()
# ===========================================================================

class TestSHAPServiceExplain:

    def test_explain_raises_when_model_not_loaded(self, shap_svc, monkeypatch):
        """explain() raises RuntimeError when model_service is not loaded."""
        import services.model_service as ms
        monkeypatch.setattr(ms.model_service, "is_loaded", lambda: False)

        with pytest.raises(RuntimeError, match="not loaded"):
            shap_svc.explain(_SAMPLE_TEXT, _SAMPLE_LABEL_ID)

    def test_explain_raises_on_empty_text(self, shap_svc, monkeypatch):
        """explain() raises ValueError on empty text."""
        import services.model_service as ms
        monkeypatch.setattr(ms.model_service, "is_loaded", lambda: True)

        with pytest.raises(ValueError, match="empty"):
            shap_svc.explain("   ", _SAMPLE_LABEL_ID)

    def test_explain_safe_returns_none_on_failure(self, shap_svc, monkeypatch):
        """explain_safe() returns None when explain() raises — never propagates."""
        import services.model_service as ms
        monkeypatch.setattr(ms.model_service, "is_loaded", lambda: False)

        result = shap_svc.explain_safe(_SAMPLE_TEXT, _SAMPLE_LABEL_ID)

        assert result is None

    def test_explain_safe_returns_result_on_success(self, shap_svc, monkeypatch):
        """explain_safe() returns the token list when explain() succeeds."""
        import services.shap_service as ss

        def fake_explain(self, text, predicted_label_id, max_evals=200):
            return _FAKE_SHAP_OUTPUT

        monkeypatch.setattr(ss.SHAPService, "explain", fake_explain)

        result = shap_svc.explain_safe(_SAMPLE_TEXT, _SAMPLE_LABEL_ID)

        assert result is not None
        assert len(result) == len(_FAKE_SHAP_OUTPUT)

    def test_explain_result_has_token_and_importance_keys(self, shap_svc, monkeypatch):
        """Each item in explain() result has 'token' and 'importance' keys."""
        import services.shap_service as ss

        def fake_explain(self, text, predicted_label_id, max_evals=200):
            return _FAKE_SHAP_OUTPUT

        monkeypatch.setattr(ss.SHAPService, "explain", fake_explain)

        result = shap_svc.explain_safe(_SAMPLE_TEXT, _SAMPLE_LABEL_ID)

        for item in result:
            assert "token" in item
            assert "importance" in item
            assert isinstance(item["token"], str)
            assert isinstance(item["importance"], float)

    def test_explain_safe_passes_max_evals(self, shap_svc, monkeypatch):
        """explain_safe() forwards max_evals to explain()."""
        import services.shap_service as ss

        captured = {}

        def fake_explain(self, text, predicted_label_id, max_evals=200):
            captured["max_evals"] = max_evals
            return _FAKE_SHAP_OUTPUT

        monkeypatch.setattr(ss.SHAPService, "explain", fake_explain)

        shap_svc.explain_safe(_SAMPLE_TEXT, _SAMPLE_LABEL_ID, max_evals=50)

        assert captured["max_evals"] == 50


# ===========================================================================
# Integration tests — POST /explain
# ===========================================================================

class TestExplainTriggerEndpoint:

    def test_trigger_returns_202(self, client, monkeypatch):
        """POST /explain returns 202 Accepted and a job ID."""
        pred = _seed_prediction()
        prediction_id = pred["predictionId"]

        resp = client.post("/explain", json={"prediction_id": prediction_id})

        assert resp.status_code == 202
        data = resp.json()
        assert "xai_job_id" in data
        assert data["prediction_id"] == prediction_id
        assert data["status"] == "pending"

    def test_trigger_creates_xai_job(self, client):
        """POST /explain adds one job to the xai_jobs queue."""
        pred = _seed_prediction()

        client.post("/explain", json={"prediction_id": pred["predictionId"]})

        assert len(mock_db._xai_jobs) == 1

    def test_trigger_creates_explanation_row(self, client):
        """POST /explain creates a pending explanation row in mock_db."""
        pred = _seed_prediction()

        client.post("/explain", json={"prediction_id": pred["predictionId"]})

        assert len(mock_db._explanations) == 1
        exp = list(mock_db._explanations.values())[0]
        assert exp["status"] == "pending"

    def test_trigger_deduplicates_existing_explanation(self, client):
        """POST /explain for a prediction that already has an explanation
        returns the existing explanation without creating a duplicate job."""
        pred = _seed_prediction()
        prediction_id = pred["predictionId"]

        # First trigger
        client.post("/explain", json={"prediction_id": prediction_id})
        assert len(mock_db._xai_jobs) == 1

        # Second trigger — should NOT add another job
        resp = client.post("/explain", json={"prediction_id": prediction_id})

        assert resp.status_code == 202
        assert len(mock_db._xai_jobs) == 1         # still just one job
        assert len(mock_db._explanations) == 1     # still just one explanation

    def test_trigger_unknown_prediction_returns_404(self, client):
        """POST /explain with a non-existent prediction_id returns 404."""
        resp = client.post("/explain", json={"prediction_id": "ghost-pred-id"})

        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()


# ===========================================================================
# Integration tests — GET /explain/{prediction_id}
# ===========================================================================

class TestGetExplanationEndpoint:

    def test_get_pending_explanation(self, client):
        """GET /explain/{id} returns status='pending' when SHAP is not done."""
        pred = _seed_prediction()
        prediction_id = pred["predictionId"]

        # Create explanation row but don't complete it
        mock_db.create_explanation(prediction_id=prediction_id)

        resp = client.get(f"/explain/{prediction_id}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "pending"
        assert data["token_importances"] is None

    def test_get_completed_explanation(self, client):
        """GET /explain/{id} returns token_importances when SHAP is complete."""
        pred = _seed_prediction()
        prediction_id = pred["predictionId"]

        exp = mock_db.create_explanation(prediction_id=prediction_id)
        mock_db.update_explanation(
            explanation_id=exp["explanationId"],
            shap_values=_FAKE_SHAP_OUTPUT,
            status="completed",
        )

        resp = client.get(f"/explain/{prediction_id}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["token_importances"] is not None
        assert len(data["token_importances"]) == len(_FAKE_SHAP_OUTPUT)

    def test_completed_explanation_has_correct_token_fields(self, client):
        """Each token_importance item has 'token' and 'importance' fields."""
        pred = _seed_prediction()
        prediction_id = pred["predictionId"]

        exp = mock_db.create_explanation(prediction_id=prediction_id)
        mock_db.update_explanation(
            explanation_id=exp["explanationId"],
            shap_values=_FAKE_SHAP_OUTPUT,
            status="completed",
        )

        resp = client.get(f"/explain/{prediction_id}")
        items = resp.json()["token_importances"]

        for item in items:
            assert "token" in item
            assert "importance" in item

    def test_get_explanation_unknown_prediction_returns_404(self, client):
        """GET /explain/{id} for unknown prediction returns 404."""
        resp = client.get("/explain/nonexistent-pred-id")
        assert resp.status_code == 404

    def test_get_explanation_no_explanation_yet_returns_404(self, client):
        """GET /explain/{id} when prediction exists but no explanation queued returns 404."""
        pred = _seed_prediction()

        resp = client.get(f"/explain/{pred['predictionId']}")

        assert resp.status_code == 404
        assert "post /explain" in resp.json()["detail"].lower()

    def test_get_failed_explanation_returns_failed_status(self, client):
        """GET /explain/{id} returns status='failed' when SHAP worker failed."""
        pred = _seed_prediction()
        prediction_id = pred["predictionId"]

        exp = mock_db.create_explanation(prediction_id=prediction_id)
        mock_db.update_explanation(
            explanation_id=exp["explanationId"],
            shap_values=[],
            status="failed",
        )

        resp = client.get(f"/explain/{prediction_id}")

        assert resp.status_code == 200
        assert resp.json()["status"] == "failed"
