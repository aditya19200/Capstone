"""
services/xai_agreement.py — does the model's own explanation agree with the
domain ontology?

Two things in this system answer "why this label", and until now they never
met:

  * SHAP (services/shap_service.py) reports which tokens pushed the model
    toward the label it chose. It explains the model, and knows no law.
  * The ontology check (services/ontology_check.py) reports which domain
    concepts the text contains and which domain they indicate. It knows the
    law, and never looks at the model.

Putting them side by side answers a question neither can alone: are the words
the model actually relied on the vocabulary for the domain it chose, or did
it key on something else entirely?

A worked example, from PRADEEP v. STATE OF HARYANA — a murder appeal the
model files under Family Law at 0.977:

    SHAP's top drivers   tutored, age, susceptible, tutoring
    concept markers      criminal, the evidence, testimony -> Criminal Law

The model latched onto a child's suggestibility. The vocabulary is evidence
law. Neither signal says that on its own; together they do, and a reviewer
can see it rather than take it on trust.

What this is not
----------------
Not a third classifier, and not a confidence score. It reports an
observation and the evidence for it. Only a human decides anything.

The matching is deliberately shallow and will miss things. SHAP's tokens are
merged back into whole words (services/shap_service.py), and markers are
whole words and bigrams mined from the corpus, but a marker can still be a
form the text does not use. Terms are matched against the reconstructed text rather
than token-by-token, and a term is credited with the SHAP weight of the
tokens it overlaps. A term the model never looked at scores nothing.
"""

import logging
import re
from typing import Dict, List, Optional

from services.ontology_check import ontology_check

logger = logging.getLogger(__name__)

# How many of SHAP's ranked tokens to consider. Past this the importances are
# noise around zero and every marker would find something to match.
TOP_TOKENS = 25

# A term must carry at least this share of the considered SHAP weight before
# it counts as something the model relied on.
MIN_SHARE = 0.02

_WORD = re.compile(r"[a-z0-9]+")


def _clean(token: str) -> str:
    """SHAP tokens keep whitespace and wordpiece marks; markers do not."""
    return token.strip().lower().lstrip("#")


def analyse(
    text: str,
    predicted_label: str,
    token_importances: Optional[List[Dict]],
) -> Optional[Dict]:
    """
    Compare the model's own drivers against the ontology's evidence.

    Returns None when either signal is unavailable — no markers loaded, or
    SHAP has not finished — because half of this comparison is not worth
    showing.
    """
    if not token_importances or not ontology_check.is_loaded():
        return None

    ranked = sorted(
        token_importances, key=lambda t: abs(t.get("importance", 0.0)), reverse=True
    )[:TOP_TOKENS]
    if not ranked:
        return None

    total = sum(abs(t.get("importance", 0.0)) for t in ranked) or 1.0

    # Weight per cleaned token, keeping sign: positive pushed toward the
    # predicted label, negative pulled away from it.
    # A SHAP entry can be a hyphenated word ("paranoid-ideation"); each part
    # is credited with its weight, since markers are plain words.
    weight: Dict[str, float] = {}
    for entry in ranked:
        for word in _WORD.findall(_clean(str(entry.get("token", "")))):
            weight[word] = weight.get(word, 0.0) + float(entry.get("importance", 0.0))

    evidence = ontology_check.evidence(text)
    supporting, contradicting = [], []

    for label, detail in evidence.items():
        for term in detail["terms"]:
            parts = _WORD.findall(term)
            if not parts:
                continue
            # A bigram counts only if the model looked at every word in it.
            hits = [weight[p] for p in parts if p in weight]
            if len(hits) < len(parts):
                continue
            signed = sum(hits) / len(hits)
            share = abs(signed) / total
            if share < MIN_SHARE:
                continue
            record = {
                "term": term,
                "domain": label,
                "shap_weight": round(signed, 6),
                "share": round(share, 4),
                # Sign is not decoration. A positive weight means this term
                # pushed the model TOWARD the label it chose; negative means
                # the model saw it and it pushed away. A rival domain's term
                # with a negative weight is the model half-noticing the
                # conflict; the same term with a positive weight means it
                # took another domain's vocabulary as grounds for this label,
                # which is the stranger finding of the two.
                "direction": "toward" if signed > 0 else "against",
            }
            (supporting if label == predicted_label else contradicting).append(record)

    supporting.sort(key=lambda r: -abs(r["shap_weight"]))
    contradicting.sort(key=lambda r: -abs(r["shap_weight"]))

    predicted_score = evidence.get(predicted_label, {}).get("score", 0.0)
    rival = max(
        ((lab, det["score"]) for lab, det in evidence.items() if lab != predicted_label),
        key=lambda kv: kv[1],
        default=(None, 0.0),
    )

    toward = [r for r in supporting if r["direction"] == "toward"]
    rival_toward = [r for r in contradicting if r["direction"] == "toward"]
    rival_against = [r for r in contradicting if r["direction"] == "against"]

    def names(rows, n=3):
        return ", ".join(f"\u201c{r['term']}\u201d" for r in rows[:n])

    if rival_toward and not toward:
        # The strongest finding: another domain's vocabulary is what pushed
        # the model toward this label.
        verdict = "conflict"
        summary = (
            f"Nothing the model relied on is {predicted_label} vocabulary. What "
            f"pushed it toward {predicted_label} was {rival_toward[0]['domain']} "
            f"vocabulary: {names(rival_toward)}."
        )
    elif toward and rival_toward:
        verdict = "mixed"
        summary = (
            f"The model drew on {predicted_label} vocabulary ({names(toward)}) and "
            f"on {rival_toward[0]['domain']} vocabulary ({names(rival_toward)}) "
            f"to reach the same label."
        )
    elif toward and rival_against:
        # The model did see the rival domain's terms, and they argued against
        # the label it chose anyway.
        verdict = "overruled"
        summary = (
            f"{names(toward)} pushed toward {predicted_label}, while "
            f"{names(rival_against)} pushed against it. The model went with "
            f"{predicted_label} regardless."
        )
    elif toward:
        verdict = "aligned"
        summary = (
            f"The words the model relied on are {predicted_label} vocabulary: "
            f"{names(toward, 4)}."
        )
    elif rival_against:
        verdict = "overruled"
        summary = (
            f"The model's influential words include {rival_against[0]['domain']} "
            f"vocabulary ({names(rival_against)}), all of which pushed against "
            f"{predicted_label}. None of them changed the outcome."
        )
    else:
        verdict = "no_overlap"
        summary = (
            "None of the words the model relied on are concept markers for any "
            "label, so its reasoning and the ontology have nothing in common "
            "to compare. The label rests on wording the concept layer does not "
            "recognise."
        )

    return {
        "verdict": verdict,
        "summary": summary,
        "supporting": supporting[:8],
        "contradicting": contradicting[:8],
        "predicted_label": predicted_label,
        "predicted_evidence": round(predicted_score, 4),
        "rival_label": rival[0],
        "rival_evidence": round(rival[1], 4),
    }
