"""
tests/test_recorded_versions.py — versions a training script produced.

Medical's round 1 came from medical_progression.py, not the retrain worker,
so it reaches the version table through metrics/<domain>_versions.json and
baseline.register_baseline_if_absent(). These pin what that may and may not
do, above all to the other domain's rows in a shared table.
"""

import dataclasses
import json

import pytest
from fastapi.testclient import TestClient

import baseline
from config import domains
from main import app
from repositories import model_versions as model_versions_repo
from routers import admin
from services import mock_db

ADMIN_HEADERS = {"X-Role": "admin"}


@pytest.fixture(autouse=True)
def clean_store():
    mock_db._model_versions.clear()
    yield
    mock_db._model_versions.clear()


@pytest.fixture
def metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(baseline, "_METRICS_DIR", tmp_path)
    monkeypatch.setattr(admin, "_METRICS_DIR", tmp_path)
    return tmp_path


def _write(path, data):
    path.write_text(json.dumps(data))


def _baseline(metrics, key, path, accuracy=0.70):
    _write(metrics / f"{key}_baseline_metrics.json", {
        "version_number": f"{key.capitalize()} Baseline", "accuracy": accuracy,
        "f1_per_class": {}, "model_path": path,
    })


def _round1(metrics, key, path, activate=True):
    _write(metrics / f"{key}_versions.json", {"versions": [{
        "version_number": f"{key.capitalize()} Round 1", "model_path": path,
        "accuracy": 0.72, "f1_per_class": {}, "parent_version": f"{key.capitalize()} Baseline",
        "parent_accuracy": 0.70, "exam_rows": 273, "activate": activate,
    }]})


def _rows():
    return {v["version_number"]: v for v in model_versions_repo.list_all()}


class TestRegistration:

    def test_registers_the_baseline_and_activates_the_recorded_version(self, metrics):
        _baseline(metrics, "legal", "./models/base")
        _round1(metrics, "legal", "./models/round1")

        baseline.register_baseline_if_absent()

        rows = _rows()
        assert set(rows) == {"Legal Baseline", "Legal Round 1"}
        assert rows["Legal Round 1"]["is_active"] is True
        assert rows["Legal Baseline"]["is_active"] is False

    def test_registers_only_the_domain_this_process_serves(self, metrics):
        # The tests run as legal. A medical baseline on disk is the medical
        # process's to register, not this one's.
        _baseline(metrics, "legal", "./models/base")
        _baseline(metrics, "medical", "./models/medical-base")

        baseline.register_baseline_if_absent()

        assert set(_rows()) == {"Legal Baseline"}

    def test_activates_only_on_first_registration(self, metrics):
        # An admin who later activates the baseline is not overruled when the
        # process restarts and finds round 1 already registered.
        _baseline(metrics, "legal", "./models/base")
        _round1(metrics, "legal", "./models/round1")
        baseline.register_baseline_if_absent()
        model_versions_repo.set_active(_rows()["Legal Baseline"]["id"])

        baseline.register_baseline_if_absent()

        rows = _rows()
        assert len(rows) == 2
        assert rows["Legal Baseline"]["is_active"] is True
        assert rows["Legal Round 1"]["is_active"] is False

    def test_a_version_not_marked_activate_stays_inactive(self, metrics):
        _baseline(metrics, "legal", "./models/base")
        _round1(metrics, "legal", "./models/round1", activate=False)

        baseline.register_baseline_if_absent()

        assert _rows()["Legal Baseline"]["is_active"] is True
        assert _rows()["Legal Round 1"]["is_active"] is False

    def test_never_takes_the_shared_flag_before_migration_010(self, metrics, monkeypatch):
        # Live Supabase without 010: one active flag for every domain, so
        # activating would deactivate the other domain's version.
        from repositories import _client
        inserted, activated = [], []
        monkeypatch.setattr(_client, "is_configured", lambda: True)
        monkeypatch.setattr(_client, "domain_queues_available", lambda: False)
        monkeypatch.setattr(baseline.model_versions_repo, "list_all", lambda: [
            {"file_path": "./models/base", "is_active": True}])
        monkeypatch.setattr(baseline.model_versions_repo, "insert",
                            lambda **kw: inserted.append(kw) or {"id": "new", **kw})
        monkeypatch.setattr(baseline.model_versions_repo, "set_active",
                            lambda vid: activated.append(vid))
        _baseline(metrics, "legal", "./models/base")
        _round1(metrics, "legal", "./models/round1")

        baseline.register_baseline_if_absent()

        assert [kw["version_number"] for kw in inserted] == ["Legal Round 1"]
        assert activated == []


class TestRetiredCheckpoints:

    def test_a_retired_path_still_belongs_to_its_domain(self):
        # So a stale medical row in the shared table never shows on legal's tab.
        medical = domains.registry()["medical"]
        for path in medical.retired_model_paths:
            assert domains.domain_for_model_path(path) == "medical"

    def test_the_admin_table_leaves_retired_rows_out(self, metrics, monkeypatch):
        legal = domains.active_domain()
        monkeypatch.setattr(domains, "active_domain", lambda: dataclasses.replace(
            legal, retired_model_paths=("./models/old",)))
        model_versions_repo.insert(version_number="Old", accuracy=0.9,
                                   f1_per_class={}, file_path="./models/old")
        model_versions_repo.insert(version_number="Current", accuracy=0.9,
                                   f1_per_class={}, file_path="./models/current")

        with TestClient(app) as client:
            names = [v["version_number"] for v in client.get(
                "/admin/metrics", headers=ADMIN_HEADERS).json()["model_versions"]]

        assert "Old" not in names
        assert "Current" in names


class TestComparisonLine:

    def test_a_recorded_version_carries_its_parent_on_the_same_rows(self, metrics):
        _round1(metrics, "legal", "./models/round1")
        assert admin._parent_comparison("Legal Round 1") == {
            "parent_version": "Legal Baseline",
            "parent_accuracy": 0.70,
            "exam_rows": 273,
        }

    def test_nothing_for_a_name_no_record_has(self, metrics):
        _round1(metrics, "legal", "./models/round1")
        assert admin._parent_comparison("Legal Baseline") == {}
