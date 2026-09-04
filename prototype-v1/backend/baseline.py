"""
baseline.py — register the deployed model as a model_versions row at startup.

The model the API serves lives at settings.MODEL_PATH and, by itself, has no
database record. Without one:

  * the admin version-history table opens empty,
  * the per-class F1 chart has nothing to plot, and
  * retrain's regression guard compares a new model against an active
    accuracy of 0.0, so it can never fire.

Since the local setup keeps everything in an in-memory store that resets on
restart, a one-off insert wouldn't survive. This runs on every startup
instead: if no model versions exist yet, it registers the deployed model
from metrics/baseline_metrics.json (written by evaluate_baseline.py) and
marks it active, because it is in fact the model being served.

If that file is missing, nothing is registered and the table simply starts
empty — the same behaviour as before. Run evaluate_baseline.py to create it.
"""

import json
import logging
from pathlib import Path

from repositories import model_versions as model_versions_repo

logger = logging.getLogger(__name__)

METRICS_FILE = Path(__file__).resolve().parent / "metrics" / "baseline_metrics.json"


def register_baseline_if_absent() -> None:
    """Insert + activate the deployed model as a version row, if none exist."""
    try:
        existing = model_versions_repo.list_all()
    except Exception:
        logger.exception("[baseline] could not read existing model versions")
        return

    if existing:
        return  # already registered, or real versions exist from retraining

    if not METRICS_FILE.exists():
        logger.info(
            "[baseline] %s not found — the deployed model has no recorded "
            "metrics, so the admin version table will start empty. Run "
            "evaluate_baseline.py to populate it.",
            METRICS_FILE.name,
        )
        return

    try:
        metrics = json.loads(METRICS_FILE.read_text())
        row = model_versions_repo.insert(
            version_number=metrics["version_number"],
            accuracy=metrics["accuracy"],
            f1_per_class=metrics["f1_per_class"],
            file_path=metrics["model_path"],
        )
        # It is the model actually being served, so it is the active one.
        model_versions_repo.set_active(row["id"])
        logger.info(
            "[baseline] registered '%s' as the active model version "
            "(accuracy %.4f, measured on %s)",
            metrics["version_number"], metrics["accuracy"], metrics["dataset"],
        )
    except Exception:
        logger.exception("[baseline] failed to register the deployed model")
