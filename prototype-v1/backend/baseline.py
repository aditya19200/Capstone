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
from typing import Dict, List, Optional

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


def recorded_versions(domain_key: str) -> List[Dict]:
    """
    Versions of a domain that a training script produced and measured, from
    metrics/<domain>_versions.json, so the table can show them beside the
    baseline and whatever the retrain worker adds. Medical's round 1
    (medical_progression.py) is the one there is. Empty when there is none.
    """
    f = _METRICS_DIR / f"{domain_key}_versions.json"
    if not f.exists():
        return []
    try:
        return list(json.loads(f.read_text()).get("versions", []))
    except Exception:
        logger.exception("[baseline] could not read %s", f.name)
        return []


def _insert(record: Dict, key: str) -> Optional[Dict]:
    """One model_versions row from a metrics record, or None if it failed."""
    try:
        row = model_versions_repo.insert(
            version_number=record["version_number"],
            accuracy=record["accuracy"],
            f1_per_class=record["f1_per_class"],
            file_path=record["model_path"],
            domain=key,
        )
    except Exception:
        logger.exception("[baseline] failed to register '%s'",
                         record.get("version_number"))
        return None
    logger.info("[baseline] registered '%s' for domain %r (accuracy %.4f)",
                record["version_number"], key, record["accuracy"])
    return row


def _activate(row: Dict, name: str) -> bool:
    try:
        model_versions_repo.set_active(row["id"])
        return True
    except Exception:
        logger.exception(
            "[baseline] registered '%s' but could not mark it active; "
            "reconcile_served_version() will retry at startup.", name)
        return False


def register_baseline_if_absent() -> None:
    """
    Register this domain's deployed checkpoint, and the versions recorded for
    it, wherever no row points at them yet.

    Only the domain this process serves. It once registered every installed
    domain, while the admin table listed them all together; the table now
    shows only its own domain, and a legal process writing medical rows into
    the shared table before migration 010 left them no domain but their path.

    The check is per checkpoint — "does a row point at this file_path" —
    rather than "does any row exist", which once let the first domain to
    register block the second forever.

    The baseline is marked active only when this domain has nothing active.
    A recorded version marked "activate" takes the flag when it is first
    registered and never again, so a later choice in the admin table survives
    a restart — and only where the flag is this domain's alone: the in-memory
    store, or Supabase after migration 010. Before 010 one flag covers every
    domain, and taking it would deactivate the other domain's version.
    """
    from config.domains import active_domain
    from repositories._client import domain_queues_available, is_configured

    try:
        existing = model_versions_repo.list_all()
    except Exception:
        logger.exception("[baseline] could not read existing model versions")
        return

    per_domain = domain_queues_available()
    known_paths = {v.get("file_path") for v in existing}
    # Before migration 010 there is one active flag for the whole table, so
    # the baseline may take it only if it is free. After 010 each domain has
    # its own.
    active_domains = {v.get("domain") for v in existing if v.get("is_active")}
    something_active = bool(active_domains)
    key = active_domain().key

    # The un-prefixed name is the fallback, so an older checkout holding just
    # metrics/baseline_metrics.json still registers the model it serves.
    metrics_file = _metrics_for(key)
    if metrics_file is None:
        legacy = _METRICS_DIR / "baseline_metrics.json"
        metrics_file = legacy if legacy.exists() else None

    baseline = None
    if metrics_file is None:
        logger.info(
            "[baseline] no metrics for domain %r — it will not appear in the "
            "version table. Run evaluate_baseline.py with ACTIVE_DOMAIN=%s to "
            "create them.", key, key,
        )
    else:
        try:
            baseline = json.loads(metrics_file.read_text())
        except Exception:
            logger.exception("[baseline] could not read %s", metrics_file.name)

    if baseline and baseline.get("model_path") not in known_paths:
        row = _insert(baseline, key)
        if row:
            known_paths.add(baseline["model_path"])
            may_activate = (key not in active_domains) if per_domain else not something_active
            if may_activate and _activate(row, baseline["version_number"]):
                active_domains.add(key)
                something_active = True

    scoped = per_domain or not is_configured()
    for record in recorded_versions(key):
        if record.get("model_path") in known_paths:
            continue
        row = _insert(record, key)
        if not row:
            continue
        known_paths.add(record["model_path"])
        if not record.get("activate"):
            continue
        if scoped:
            _activate(row, record["version_number"])
        else:
            logger.warning(
                "[baseline] '%s' is marked to activate, but before migration "
                "010 the active flag is shared by every domain. Left inactive.",
                record["version_number"],
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

    from repositories._client import domain_queues_available
    from config.domains import domain_for_model_path
    if domain_queues_available():
        mine = served.get("domain") or domain_for_model_path(served.get("file_path"))
        rivals = [v for v in versions if v.get("is_active")
                  and (v.get("domain") or domain_for_model_path(v.get("file_path"))) == mine]
    else:
        rivals = [v for v in versions if v.get("is_active")]
    if rivals:
        # Something else is flagged active while different weights are loaded.
        # Startup already tried to honour the flag and evidently could not, so
        # say so rather than overwriting a deliberate choice.
        flagged = rivals[0]["version_number"]
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
