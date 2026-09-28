"""
services/ontology_check.py — ontology-based consistency check on predictions.

Why this exists
---------------
The active-learning engine escalates what the model is *unsure* about. That
makes it structurally blind to the opposite failure: predictions that are
confidently wrong. Those are auto-accepted, never reach a reviewer, never
become training data, and so the error persists indefinitely — the model's
own confidence is the thing deciding whether anyone checks its work.

Where the concepts live
-----------------------
The concept layer is modelled in the ontology itself —
graph/concept_seed.cypher defines each term as

    (:Concept {term, precision, support})
        -[:INDICATES {weight}]-> (:OntologyNode {label})

and a term may indicate several domains at different weights ("violence"
genuinely points at both Family and Criminal Law), which is why it is a
graph rather than a lookup table. graph/concept_markers.json, read below, is
the flat mirror of that same model, generated in the same run so the two
cannot drift. Reading the mirror is the arrangement the rest of the graph
layer already uses: ontology_seed.cypher mirrors mock_db's ONTOLOGY exactly
so that swapping to a live Neo4j is a query-for-query replacement with no
behaviour change. No instance is stood up yet; when one is, this service
reads the same edges over bolt instead of from the export.

This check is deliberately independent of the model. It reads the text for
legal-concept markers (graph/concept_markers.json, mined from the corpus's
human gold labels — never from model predictions) and asks a different
question: does the vocabulary in this document actually belong to the domain
the model chose?

A prosecution document full of "ipc", "the prosecution", "punishable" and
"bail" that the model filed under Family Law is flagged regardless of how
confident the model was. Confidence cannot catch that; evidence can.

This is a consistency check, not a second classifier. It never overrides a
label — it only decides that a human should look. False positives cost
reviewer time; they do not corrupt data.

Known limitation, stated plainly: marker matching is shallow. It sees
vocabulary, not argument structure, and a document quoting a statute it is
not actually about can trip it. The multi-source requirement in the miner
removes corpus artifacts, but nothing here understands the text.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

MARKERS_FILE = Path(__file__).resolve().parent.parent.parent / "graph" / "concept_markers.json"

# Operating point, chosen by grid search rather than taste. Measured against
# 48 domestic-context criminal texts (the confusion this was built for) and
# all 1,701 correctly-labelled corpus samples:
#
#     recall 36/48 (75%)   false positives 7.5% of correct labels
#                          9.6% of genuine Family Law documents
#
# Reproduce with the sweep in the session log; raising MIN_EVIDENCE trades
# recall for reviewer time roughly one-for-one past this point.
#
# Note what 7.5% means: reviewers see ~7.5% more documents than confidence
# alone would route them. That is the cost of catching a class of error the
# confidence thresholds detect at 0%.

# A domain needs at least this much accumulated evidence before its
# disagreement counts for anything — one stray term is not a case.
MIN_EVIDENCE = 5.0

# And it must outweigh the predicted label's own evidence by this factor.
# Above 1.0 so a document with genuinely mixed vocabulary (a criminal matter
# that really does discuss the marriage) does not flag on a narrow margin.
DOMINANCE_RATIO = 1.2

# At least this many distinct markers must match, so a single high-scoring
# term cannot raise a flag on its own.
MIN_MATCHED_TERMS = 2


class OntologyCheck:
    """Compares the vocabulary in a text against the label the model chose."""

    def __init__(self) -> None:
        self._markers: Dict[str, List[Dict]] = {}
        self._patterns: Dict[str, List] = {}
        self._loaded = False

    def load(self) -> None:
        """Read concept_markers.json and precompile one regex per marker."""
        if not MARKERS_FILE.exists():
            logger.warning(
                "[ontology_check] %s not found — consistency checking disabled. "
                "Run graph/mine_concept_markers.py to create it.", MARKERS_FILE.name,
            )
            self._loaded = False
            return

        data = json.loads(MARKERS_FILE.read_text(encoding="utf-8"))
        self._markers = data.get("markers", {})

        # Word-boundary matching: "mark" must not fire inside "remarked",
        # and "ipc" must not fire inside "ipcc".
        self._patterns = {
            label: [
                (re.compile(r"\b" + re.escape(m["term"]) + r"\b"), m["term"], m["score"])
                for m in entries
            ]
            for label, entries in self._markers.items()
        }
        self._loaded = True
        total = sum(len(v) for v in self._markers.values())
        logger.info(
            "[ontology_check] loaded %d markers across %d domains from %s",
            total, len(self._markers), data.get("source_dataset", MARKERS_FILE.name),
        )

    def is_loaded(self) -> bool:
        return self._loaded

    def evidence(self, text: str) -> Dict[str, Dict]:
        """
        Score every domain by the markers present in *text*.

        Returns {label: {"score": float, "terms": [str, ...]}} for domains
        with at least one match.
        """
        if not self._loaded:
            return {}

        lowered = text.lower()
        found: Dict[str, Dict] = {}
        for label, patterns in self._patterns.items():
            score = 0.0
            terms: List[str] = []
            for pattern, term, weight in patterns:
                if pattern.search(lowered):
                    score += weight
                    terms.append(term)
            if terms:
                found[label] = {"score": round(score, 4), "terms": terms}
        return found

    def check(self, text: str, predicted_label: str) -> Optional[Dict]:
        """
        Return a conflict record if the text's vocabulary contradicts the
        predicted label, otherwise None.

        A conflict requires all three of:
          * some other domain accumulated MIN_EVIDENCE or more,
          * it did so on at least MIN_MATCHED_TERMS distinct markers, and
          * it outweighs the predicted label's evidence by DOMINANCE_RATIO.
        """
        ev = self.evidence(text)
        if not ev:
            return None

        predicted_score = ev.get(predicted_label, {}).get("score", 0.0)

        ranked = sorted(ev.items(), key=lambda kv: kv[1]["score"], reverse=True)
        for label, detail in ranked:
            if label == predicted_label:
                continue
            if detail["score"] < MIN_EVIDENCE:
                break                      # ranked, so nothing below qualifies
            if len(detail["terms"]) < MIN_MATCHED_TERMS:
                continue
            if detail["score"] < predicted_score * DOMINANCE_RATIO:
                continue
            return {
                "suggested_label": label,
                "evidence_score": detail["score"],
                "predicted_label_score": round(predicted_score, 4),
                "matched_terms": detail["terms"][:8],
                "reason": (
                    f"Text contains {len(detail['terms'])} {label} concept marker(s) "
                    f"({', '.join(detail['terms'][:4])}) but was classified as "
                    f"{predicted_label}."
                ),
            }
        return None


ontology_check = OntologyCheck()
