"""
tests/test_dataset_export.py — Tests for services/dataset_export.py.

The claim this feature makes is that the exported file *is* the training
data, not a report about it. The test that matters is the one holding the
export and workers.retrain_worker.build_training_data to the same rows.
"""

import json

import pytest
from fastapi.testclient import TestClient

from main import app
from services import dataset_export, mock_db


@pytest.fixture(autouse=True)
def clear_db():
    mock_db._predictions.clear()
    mock_db._annotations.clear()
    yield


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


ADMIN = {"X-Role": "admin"}


def _validated(text, predicted, validated, confidence=0.9, version="v-test"):
    pred = mock_db.create_prediction(
        text_content=text, predicted_label=predicted, label_id=0,
        confidence=confidence, all_probabilities={predicted: confidence},
        model_version=version,
    )
    ann = mock_db.create_annotation(
        prediction_id=pred["id"], validated_label=validated, annotator_id=None
    )
    mock_db.update_annotation_status(ann["id"], "validated")
    return pred, ann


class TestRows:
    def test_empty_corpus_is_not_an_error(self):
        records, skipped = dataset_export.rows()
        assert records == [] and skipped == 0

    def test_corrected_flag_marks_human_disagreement(self):
        _validated("A contract matter between the parties here.", "Contract Law", "Contract Law")
        _validated("A criminal matter involving the accused person.", "Family Law", "Criminal Law")
        records, _ = dataset_export.rows()
        by_label = {r["label"]: r for r in records}
        assert by_label["Contract Law"]["corrected"] is False
        assert by_label["Criminal Law"]["corrected"] is True

    def test_label_is_the_validated_one_not_the_prediction(self):
        _validated("Some text about an accused person.", "Family Law", "Criminal Law")
        record = dataset_export.rows()[0][0]
        assert record["label"] == "Criminal Law"
        assert record["predicted_label"] == "Family Law"

    def test_unknown_label_is_skipped_not_exported(self):
        _validated("Text here about something legal.", "Contract Law", "Astrology Law")
        records, skipped = dataset_export.rows()
        assert records == [] and skipped == 1

    def test_text_is_normalised(self):
        _validated("Hyphen-\nated  text\nwrapped across lines.", "Contract Law", "Contract Law")
        record = dataset_export.rows()[0][0]
        assert "\n" not in record["text"] and "  " not in record["text"]

    def test_matches_what_retraining_consumes(self):
        """
        The whole claim of this feature. If these drift, the retraining
        results stop being evidence about the exported corpus.
        """
        _validated("A contract dispute between two commercial parties.", "Contract Law", "Contract Law")
        _validated("A criminal trial of the accused before the court.", "Family Law", "Criminal Law")
        from workers.retrain_worker import build_training_data
        texts, labels = build_training_data()
        records, _ = dataset_export.rows()
        assert sorted(zip(texts, labels)) == sorted((r["text"], r["label"]) for r in records)


class TestJsonl:
    def test_one_valid_json_object_per_line(self):
        _validated("A contract matter between the parties.", "Contract Law", "Contract Law")
        _validated("A criminal matter before the court.", "Criminal Law", "Criminal Law")
        lines = list(dataset_export.to_jsonl(dataset_export.rows()[0]))
        assert len(lines) == 2
        for line in lines:
            assert line.endswith("\n")
            assert set(json.loads(line)) >= {"text", "label", "label_id", "corrected"}


class TestCard:
    def test_card_warns_when_the_corpus_is_tiny(self):
        _validated("A contract matter between the parties.", "Contract Law", "Contract Law")
        assert "not a corpus to train a production model on" in dataset_export.card(
            dataset_export.rows()[0]
        )

    def test_card_names_labels_with_no_examples(self):
        _validated("A contract matter between the parties.", "Contract Law", "Contract Law")
        assert "Family Law" in dataset_export.card(dataset_export.rows()[0]).split(
            "No examples at all for:"
        )[1]

    def test_card_flags_mixed_model_versions(self):
        _validated("A contract matter here.", "Contract Law", "Contract Law", version="v-a")
        _validated("A criminal matter here.", "Criminal Law", "Criminal Law", version="v-b")
        assert "not" in dataset_export.card(dataset_export.rows()[0]).lower()
        assert "comparable" in dataset_export.card(dataset_export.rows()[0])


class TestEndpoints:
    def test_export_requires_admin(self, client):
        assert client.get("/admin/dataset", headers={"X-Role": "reviewer"}).status_code == 403

    def test_export_streams_jsonl_with_a_count_header(self, client):
        _validated("A contract matter between the parties.", "Contract Law", "Contract Law")
        resp = client.get("/admin/dataset", headers=ADMIN)
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/x-ndjson")
        assert resp.headers["X-Example-Count"] == "1"
        assert ".jsonl" in resp.headers["content-disposition"]
        assert json.loads(resp.text.strip())["label"] == "Contract Law"

    def test_card_endpoint_returns_stats_and_markdown(self, client):
        _validated("A contract matter between the parties.", "Contract Law", "Contract Law")
        body = client.get("/admin/dataset/card", headers=ADMIN).json()
        assert body["stats"]["total"] == 1
        assert body["card_markdown"].startswith("# ")
