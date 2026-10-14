"""
config/domains.py — the label set, loaded from a domain pack instead of being
typed out in five places.

Why this exists
---------------
The ten legal labels used to be written out by hand in model_service
(ID2LABEL), response_models (LEGAL_LABELS), mock_db (ONTOLOGY),
graph/verify_ontology.py (EXPECTED_LABELS) and the frontend's constants.js.
Five copies that had to agree on spelling *and* order — the order is not
cosmetic, it is the index into the model's output logits, so a reordering
there silently mislabels every prediction.

They did all agree. The problem is what happens next: a second domain would
have meant five more copies, and the frontend copy already carried a comment
telling a human to remember to update it.

So the label set moved into domains/<key>.json and everything reads it from
here. The old names (ID2LABEL, LEGAL_LABELS, ONTOLOGY) still exist and still
hold the same values — they are now derived from the pack rather than typed.

Adding a domain
---------------
Drop a new domains/<key>.json next to legal.json and point ACTIVE_DOMAIN at
it. Nothing else in the backend needs editing.

Path conventions, because the two differ and it matters
-------------------------------------------------------
  model_path    relative to the backend/ directory, exactly like
                settings.MODEL_PATH, whose value it mirrors. Kept as a raw
                string because main.py compares it by equality against the
                file_path stored on a model version row.
  markers_file  relative to the repository root, like the rest of graph/.
                Exposed resolved, as markers_path, since the only consumer
                (services/ontology_check.py) wants an absolute path.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config.settings import settings

logger = logging.getLogger(__name__)

# backend/config/domains.py -> backend/config -> backend -> repo root
BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent
PACK_DIR = REPO_ROOT / "domains"


@dataclass(frozen=True)
class Label:
    """One class the model can predict, plus its ontology metadata."""

    id: int
    name: str
    description: str = ""
    related: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Domain:
    """A label set and the model that predicts it."""

    key: str
    display_name: str
    labels: Tuple[Label, ...]
    description: str = ""
    base_model: str = ""
    model_path: str = ""
    max_length: int = 256
    markers_file: str = ""
    dataset: str = ""

    # -- derived views, so callers never rebuild these ---------------------

    @property
    def names(self) -> List[str]:
        """Label names in id order. The order is the logit order."""
        return [l.name for l in self.labels]

    @property
    def id2label(self) -> Dict[int, str]:
        return {l.id: l.name for l in self.labels}

    @property
    def label2id(self) -> Dict[str, int]:
        return {l.name: l.id for l in self.labels}

    @property
    def num_labels(self) -> int:
        return len(self.labels)

    @property
    def markers_path(self) -> Optional[Path]:
        """Absolute path to the concept markers, or None if the pack has none."""
        return REPO_ROOT / self.markers_file if self.markers_file else None

    @property
    def ontology(self) -> Dict[str, Dict]:
        """
        The label set in the shape mock_db.ONTOLOGY and the /ontology endpoint
        use. parent/children are always empty here: the domain nodes are a flat
        set of ten peers, and the hierarchy below them lives in the concept
        layer (graph/concept_seed.cypher), not in this file.
        """
        return {
            l.name: {
                "label": l.name,
                "description": l.description,
                "parent": None,
                "children": [],
                "related": list(l.related),
            }
            for l in self.labels
        }


def _parse(data: Dict, source: Path) -> Domain:
    """Build a Domain from decoded pack JSON, refusing anything malformed."""
    where = source.name

    try:
        raw_labels = data["labels"]
        key = data["key"]
    except KeyError as exc:
        raise ValueError(f"{where}: missing required field {exc}") from exc

    if not raw_labels:
        raise ValueError(f"{where}: 'labels' is empty")

    labels = tuple(
        Label(
            id=int(l["id"]),
            name=str(l["name"]),
            description=str(l.get("description", "")),
            related=tuple(l.get("related", [])),
        )
        for l in raw_labels
    )

    # The ids index into the model's output layer, so they have to be exactly
    # 0..n-1 with nothing missing, duplicated or out of order. A pack that got
    # this wrong would not crash — it would quietly mislabel every prediction,
    # which is why this is checked at load rather than trusted.
    ids = [l.id for l in labels]
    if ids != list(range(len(labels))):
        raise ValueError(
            f"{where}: label ids must be 0..{len(labels) - 1} in order, got {ids}"
        )

    names = [l.name for l in labels]
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"{where}: duplicate label name(s): {', '.join(dupes)}")

    known = set(names)
    dangling = {
        l.name: [r for r in l.related if r not in known]
        for l in labels
        if any(r not in known for r in l.related)
    }
    if dangling:
        raise ValueError(f"{where}: 'related' points at unknown label(s): {dangling}")

    return Domain(
        key=str(key),
        display_name=str(data.get("display_name", key)),
        labels=labels,
        description=str(data.get("description", "")),
        base_model=str(data.get("base_model", "")),
        model_path=str(data.get("model_path", "")),
        max_length=int(data.get("max_length", 256)),
        markers_file=str(data.get("markers_file", "")),
        dataset=str(data.get("dataset", "")),
    )


def load_pack(path: Path) -> Domain:
    """Read and validate a single domain pack."""
    data = json.loads(path.read_text(encoding="utf-8"))
    domain = _parse(data, path)
    if domain.key != path.stem:
        raise ValueError(
            f"{path.name}: 'key' is {domain.key!r} but the filename says "
            f"{path.stem!r} — ACTIVE_DOMAIN looks packs up by filename"
        )
    return domain


_registry: Optional[Dict[str, Domain]] = None


def registry() -> Dict[str, Domain]:
    """All packs in domains/, keyed by their key. Read once, then cached."""
    global _registry
    if _registry is None:
        found: Dict[str, Domain] = {}
        for path in sorted(PACK_DIR.glob("*.json")):
            try:
                domain = load_pack(path)
            except (ValueError, json.JSONDecodeError, OSError) as exc:
                # One bad pack must not take down the domains that are fine.
                logger.error("[domains] ignoring %s — %s", path.name, exc)
                continue
            found[domain.key] = domain
        if not found:
            raise RuntimeError(
                f"No usable domain pack in {PACK_DIR}. Expected at least "
                f"domains/legal.json."
            )
        _registry = found
        logger.info(
            "[domains] loaded %d pack(s): %s",
            len(found),
            ", ".join(f"{d.key} ({d.num_labels} labels)" for d in found.values()),
        )
    return _registry


def get_domain(key: str) -> Domain:
    reg = registry()
    if key not in reg:
        raise KeyError(
            f"Unknown domain {key!r}. Available: {', '.join(sorted(reg))}"
        )
    return reg[key]


def active_domain() -> Domain:
    """
    The domain this process serves, per settings.ACTIVE_DOMAIN.

    Falls back to whatever single pack exists if ACTIVE_DOMAIN names one that
    does not, so a typo in .env is a loud log line rather than a dead API.
    """
    reg = registry()
    key = settings.ACTIVE_DOMAIN
    if key in reg:
        return reg[key]
    fallback = reg[sorted(reg)[0]]
    logger.error(
        "[domains] ACTIVE_DOMAIN=%r not found (have: %s) — serving %r instead",
        key, ", ".join(sorted(reg)), fallback.key,
    )
    return fallback
