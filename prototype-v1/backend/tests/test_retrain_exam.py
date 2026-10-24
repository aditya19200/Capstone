"""
tests/test_retrain_exam.py — what a retrained version is scored on.

A new version used to be scored on 20% of the validated annotations and
shown beside a baseline scored on 1,701 rows: 93.3% against 70.7%, from two
different exams. These pin the replacement — the baseline's own exam, less
the rows the new version trained on — and the admin table's like-for-like
line, which reads the parent's score on those same rows back from the
worker's sidecar.
"""

import csv
import json

import pytest
from fastapi.testclient import TestClient

from main import app
from repositories import model_versions as model_versions_repo
from routers import admin
from services import mock_db
from utils.text import normalize
from workers import retrain_worker

ADMIN_HEADERS = {"X-Role": "admin"}


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label_name"])
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def exam_dirs(tmp_path, monkeypatch):
    """A repo root and metrics dir of our own, so no real file is read."""
    metrics = tmp_path / "backend" / "metrics"
    metrics.mkdir(parents=True)
    monkeypatch.setattr(retrain_worker, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(retrain_worker, "METRICS_DIR", metrics)
    return tmp_path, metrics


def _baseline(metrics, **fields):
    key = retrain_worker.ACTIVE_DOMAIN.key
    (metrics / f"{key}_baseline_metrics.json").write_text(json.dumps(fields))


class TestLoadExam:

    def test_reads_the_set_the_baseline_was_scored_on(self, exam_dirs):
        root, metrics = exam_dirs
        _write_csv(root / "datasets/v1.csv", [
            {"text": "The accused was charged with theft.", "label_name": "Criminal Law"},
            {"text": "The wife sought maintenance.", "label_name": "Family Law"},
        ])
        _baseline(metrics, dataset="datasets/v1.csv")

        name, texts, labels, excluded = retrain_worker.load_exam(set())

        assert name == "datasets/v1.csv"
        assert labels == ["Criminal Law", "Family Law"]
        assert excluded == 0

    def test_prefers_a_named_heldout_file_over_the_training_set(self, exam_dirs):
        # Medical's baseline names both: it trained on the dataset and was
        # scored on the held-out file, so the held-out file is the exam.
        root, metrics = exam_dirs
        _write_csv(root / "datasets/train.csv",
                   [{"text": "trained on", "label_name": "Criminal Law"}])
        _write_csv(root / "eval/heldout.csv",
                   [{"text": "never trained on", "label_name": "Family Law"}])
        _baseline(metrics, dataset="datasets/train.csv", heldout="eval/heldout.csv")

        name, texts, _, _ = retrain_worker.load_exam(set())

        assert name == "eval/heldout.csv"
        assert texts == [normalize("never trained on")]

    def test_drops_rows_the_new_version_trains_on(self, exam_dirs):
        # The demo's intake file shares rows with v1.csv. A corrected one
        # becomes a training row, and scoring a model on what it trained on
        # is the mistake this replaced.
        root, metrics = exam_dirs
        _write_csv(root / "datasets/v1.csv", [
            {"text": "  The accused was charged with theft. ", "label_name": "Criminal Law"},
            {"text": "The wife sought maintenance.", "label_name": "Family Law"},
        ])
        _baseline(metrics, dataset="datasets/v1.csv")

        trained_on = {normalize("The accused was charged with theft.")}
        _, texts, labels, excluded = retrain_worker.load_exam(trained_on)

        assert excluded == 1
        assert labels == ["Family Law"]

    def test_skips_labels_outside_the_domain(self, exam_dirs):
        root, metrics = exam_dirs
        _write_csv(root / "datasets/v1.csv", [
            {"text": "a", "label_name": "Not A Label"},
            {"text": "b", "label_name": "Family Law"},
            {"text": "", "label_name": "Family Law"},
        ])
        _baseline(metrics, dataset="datasets/v1.csv")

        _, _, labels, _ = retrain_worker.load_exam(set())
        assert labels == ["Family Law"]

    def test_none_without_baseline_metrics(self, exam_dirs):
        assert retrain_worker.load_exam(set()) is None

    def test_none_when_the_named_file_is_missing(self, exam_dirs):
        _, metrics = exam_dirs
        _baseline(metrics, dataset="datasets/not-here.csv")
        assert retrain_worker.load_exam(set()) is None


@pytest.fixture
def sidecars(tmp_path, monkeypatch):
    monkeypatch.setattr(admin, "_METRICS_DIR", tmp_path)
    return tmp_path


class TestParentComparison:

    def test_reads_the_parent_score_on_the_same_rows(self, sidecars):
        (sidecars / "v-20261024-101500_metrics.json").write_text(json.dumps({
            "accuracy": 0.7407, "parent_version": "Legal Baseline",
            "parent_accuracy": 0.7089, "exam_rows": 1697,
        }))
        assert admin._parent_comparison("v-20261024-101500") == {
            "parent_version": "Legal Baseline",
            "parent_accuracy": 0.7089,
            "exam_rows": 1697,
        }

    def test_empty_for_a_sidecar_from_before_the_parent_was_scored(self, sidecars):
        (sidecars / "v-20261023-140049_metrics.json").write_text(json.dumps({
            "accuracy": 0.9333, "base_accuracy": 0.7072, "test_count": 15,
        }))
        assert admin._parent_comparison("v-20261023-140049") == {}

    def test_empty_with_no_sidecar(self, sidecars):
        assert admin._parent_comparison("Legal Baseline") == {}

    @pytest.mark.parametrize("name", ["../secrets", "a/b", "", "x\x00y"])
    def test_never_builds_a_path_from_an_odd_name(self, sidecars, name):
        assert admin._parent_comparison(name) == {}


class TestAdminTableShowsTheComparison:

    @pytest.fixture(autouse=True)
    def clear_mock_db(self):
        mock_db._model_versions.clear()
        yield
        mock_db._model_versions.clear()

    def test_retrained_row_carries_its_parent_and_the_baseline_does_not(self, sidecars):
        model_versions_repo.insert(
            version_number="Legal Baseline", accuracy=0.7072, f1_per_class={},
            file_path="./models/LegalModelShared",
        )
        model_versions_repo.insert(
            version_number="v-20261024-101500", accuracy=0.7407, f1_per_class={},
            file_path="models/v-20261024-101500",
        )
        (sidecars / "v-20261024-101500_metrics.json").write_text(json.dumps({
            "parent_version": "Legal Baseline", "parent_accuracy": 0.7089,
            "exam_rows": 1697,
        }))

        with TestClient(app) as client:
            rows = {v["version_number"]: v for v in client.get(
                "/admin/metrics", headers=ADMIN_HEADERS).json()["model_versions"]}

        assert rows["v-20261024-101500"]["parent_accuracy"] == 0.7089
        assert rows["v-20261024-101500"]["exam_rows"] == 1697
        assert rows["Legal Baseline"]["parent_accuracy"] is None


class TestInterruptedJob:
    """A job a dead worker left 'running' must not block every later run."""

    def _run(self, monkeypatch, job):
        updates = []
        monkeypatch.setattr(retrain_worker, "get_latest_retrain_job", lambda: job)
        monkeypatch.setattr(retrain_worker, "update_retrain_job",
                            lambda **kw: updates.append(kw))
        retrain_worker.fail_interrupted_job()
        return updates

    def test_marks_a_running_job_failed(self, monkeypatch):
        updates = self._run(monkeypatch, {"id": "j1", "status": "running"})
        assert [(u["job_id"], u["status"]) for u in updates] == [("j1", "failed")]

    @pytest.mark.parametrize("status", ["pending", "complete", "failed"])
    def test_leaves_any_other_job_alone(self, monkeypatch, status):
        assert self._run(monkeypatch, {"id": "j1", "status": status}) == []

    def test_survives_the_database_being_unreachable(self, monkeypatch):
        def down():
            raise ConnectionError("name resolution")
        monkeypatch.setattr(retrain_worker, "get_latest_retrain_job", down)
        retrain_worker.fail_interrupted_job()   # logs, does not raise
