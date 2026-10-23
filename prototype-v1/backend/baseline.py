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
from typing import Optional

from repositories import model_versions as model_versions_repo

logger = logging.getLogger(__name__)

_METRICS_DIR = Path(__file__).resolve().parent / "metrics"


def _metrics_file() -> Path:
    """
    The active domain's baseline metrics.

    Per-domain, because a second domain means a second deployed checkpoint
    with its own accuracy and its own per-class F1 — registering legal's
    numbers against a medical model would be worse than registering nothing.
    Falls back to the original un-prefixed name so an existing checkout with
    only metrics/baseline_metrics.json keeps working.
    """
    from config.domains import active_domain
    preferred = _METRICS_DIR / f"{active_domain().key}_baseline_metrics.json"
    legacy = _METRICS_DIR / "baseline_metrics.json"
    return preferred if preferred.exists() else legacy


def _metrics_for(domain_key: str):
    """That domain's metrics file, or None if it has never been evaluated."""
    f = _METRICS_DIR / f"{domain_key}_baseline_metrics.json"
    return f if f.exists() else None


def register_baseline_if_absent() -> None:
    """
    Register the deployed checkpoint of every installed domain that has no row.

    This used to return the moment ANY row existed, which was fine while there
    was one domain and wrong as soon as there were two: legal registered first
    and medical was skipped forever, so the admin table claimed the system had
    a single model while a second one sat trained on disk. The check is now per
    checkpoint — "does a row point at this file_path" — so each domain gets its
    own row and neither blocks the other.

    Only the domain this process serves is ever marked active, and only when
    nothing else is. model_versions has one global single-active index, so two
    baselines cannot both hold the flag; the other domain's row is described
    by is_serving in whichever process actually has those weights loaded.
    """
    from config.domains import active_domain, registry

    try:
        existing = model_versions_repo.list_all()
    except Exception:
        logger.exception("[baseline] could not read existing model versions")
        return

    known_paths = {v.get("file_path") for v in existing}
    something_active = any(v.get("is_active") for v in existing)
    active_key = active_domain().key

    for key in sorted(registry()):
        metrics_file = _metrics_for(key)
        if metrics_file is None:
            # Fall back to the un-prefixed name for the active domain only, so
            # an older checkout holding just metrics/baseline_metrics.json
            # still registers the model it is actually serving.
            legacy = _METRICS_DIR / "baseline_metrics.json"
            if key == active_key and legacy.exists():
                metrics_file = legacy
            else:
                logger.info(
                    "[baseline] no metrics for domain %r — it will not appear "
                    "in the version table. Run evaluate_baseline.py with "
                    "ACTIVE_DOMAIN=%s to create them.", key, key,
                )
                continue

        try:
            metrics = json.loads(metrics_file.read_text())
        except Exception:
            logger.exception("[baseline] could not read %s", metrics_file.name)
            continue

        if metrics.get("model_path") in known_paths:
            continue  # already registered

        try:
            row = model_versions_repo.insert(
                version_number=metrics["version_number"],
                accuracy=metrics["accuracy"],
                f1_per_class=metrics["f1_per_class"],
                file_path=metrics["model_path"],
            )
            known_paths.add(metrics["model_path"])
            logger.info(
                "[baseline] registered '%s' for domain %r (accuracy %.4f, "
                "measured on %s)", metrics["version_number"], key,
                metrics["accuracy"], metrics.get("dataset", "?"),
            )
        except Exception:
            logger.exception("[baseline] failed to register domain %r", key)
            continue

        if key == active_key and not something_active:
            try:
                model_versions_repo.set_active(row["id"])
                something_active = True
            except Exception:
                logger.exception(
                    "[baseline] registered '%s' but could not mark it active; "
                    "reconcile_served_version() will retry at startup.",
                    metrics["version_number"],
                )


def reconcile_served_version(served_path: str) -> Optional[str]:
    """
    Make the version table describe the model that is actually being served.

    register_baseline_if_absent() runs once and then returns early forever,
    which meant two wrong states could outlive the startup that created them:

      * **The row named after a directory.** The deployed checkpoint went in
        as 'baseline-LegalModelShared'. That is a path on somebody's laptop,
        not a version anyone reading the table needs to know about.

      * **Nothing marked active.** The insert and the activation are two
        calls, and the activation used to fail against a real database
        (migration 003's unfiltered UPDATE — see migration 009). The row
        landed, the flip did not, and the exception handler above swallowed
        it. Result: a version history whose only entry claims to be inactive
        while its weights answer every request. Re-running registration does
        not fix that, because the row exists.

    So this runs on every startup and asserts the two things that must be
    true of whatever is loaded: it is named the way the metrics file names
    it, and it is the row flagged active. Both are no-ops once correct.

    *served_path* is what model_service actually loaded, not what the config
    asked for — those differ after an activation, and the flag has to follow
    the weights rather than the intention.

    Returns the version_number of the row describing those weights, so the
    caller can name the loaded model after it, or None if no row matches.
    """
    try:
        versions = model_versions_repo.list_all()
    except Exception:
        logger.exception("[baseline] could not read model versions to reconcile")
        return None
    if not versions:
        return None

    name = None
    metrics_file = _metrics_file()
    if metrics_file.exists():
        try:
            name = json.loads(metrics_file.read_text()).get("version_number")
        except Exception:
            logger.exception("[baseline] could not read %s", metrics_file.name)

    served = next((v for v in versions if v.get("file_path") == served_path), None)

    if served and name and served["version_number"] != name:
        # Only the registered baseline is renamed. A retrained version's name
        # is its own timestamp and means something; this one never did.
        if served["version_number"].lower().startswith("baseline"):
            try:
                model_versions_repo.rename(served["id"], name)
                logger.info("[baseline] renamed '%s' to '%s'",
                            served["version_number"], name)
                served["version_number"] = name
            except Exception:
                logger.exception("[baseline] could not rename %s", served["id"])

    if served is None:
        return None
    if served.get("is_active"):
        return served["version_number"]

    if any(v.get("is_active") for v in versions):
        # Something else is flagged active while different weights are loaded.
        # Startup already tried to honour the flag and evidently could not, so
        # say so rather than overwriting a deliberate choice.
        flagged = next(v["version_number"] for v in versions if v.get("is_active"))
        logger.warning(
            "[baseline] '%s' is flagged active but '%s' is what loaded. The "
            "admin table will show the difference.", flagged,
            served["version_number"],
        )
        return served["version_number"]

    try:
        model_versions_repo.set_active(served["id"])
        logger.info("[baseline] marked '%s' active — it is what is serving",
                    served["version_number"])
    except Exception:
        logger.exception("[baseline] could not mark the served version active")
    return served["version_number"]
