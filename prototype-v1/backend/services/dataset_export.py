"""
services/dataset_export.py — the annotated corpus, as a file someone can train on.

Why this exists
---------------
The product of this system is not the classifier. It is the labelled corpus
the classifier helps produce: the model pre-labels, the routing layer decides
what a human should look at, and a reviewer confirms or corrects. What comes
out the other end is training data.

That claim is worth very little while the data is only reachable as rows in a
table, so this writes it out in the form a practitioner would actually use —
JSON Lines, one object per example, ready for `datasets.load_dataset("json",
...)` — together with a card describing what is in it.

The guarantee that makes it honest
----------------------------------
`rows()` is built on exactly what `workers.retrain_worker.build_training_data`
consumes: validated annotations, joined to their prediction's text, normalised
with the same `normalize()` every prediction path uses. The export is not a
report *about* the training data, it is the training data. If the two ever
drift, the retraining results stop being evidence for the exported corpus, so
they share this module rather than each assembling their own view.

Provenance travels with every row: which model version predicted it, what it
predicted, and whether the human agreed. A corpus of labels with no record of
how they were produced is worth less than one that says.
"""

import json
import logging
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, Iterator, List, Tuple

from config.domains import active_domain
from services.supabase_service import get_prediction, list_annotations
from utils.text import normalize

logger = logging.getLogger(__name__)


def rows() -> Tuple[List[Dict], int]:
    """
    Every validated annotation as an export record.

    Returns (records, skipped). A record is skipped when its prediction row
    is missing or its label is not in the active domain — the same two
    conditions build_training_data() skips on, for the same reason.
    """
    domain = active_domain()
    known = set(domain.names)
    out: List[Dict] = []
    skipped = 0

    for ann in list_annotations(status="validated"):
        pred = get_prediction(ann["prediction_id"])
        if pred is None:
            skipped += 1
            continue
        label = ann.get("validated_label")
        if label not in known:
            logger.warning(
                "[dataset_export] annotation %s has label %r, not in domain %r",
                ann.get("id"), label, domain.key,
            )
            skipped += 1
            continue

        predicted = pred.get("predicted_label")
        out.append({
            "text": normalize(pred["text_content"]),
            "label": label,
            "label_id": domain.label2id[label],
            "domain": domain.key,
            # Provenance. "corrected" is the interesting flag: those rows are
            # the ones the model got wrong and a human fixed, which is where
            # the value of this corpus concentrates.
            "predicted_label": predicted,
            "confidence": pred.get("confidence"),
            "corrected": predicted is not None and predicted != label,
            "model_version": pred.get("model_version"),
            "annotated_at": ann.get("annotated_at"),
        })

    return out, skipped


def to_jsonl(records: List[Dict]) -> Iterator[str]:
    """One JSON object per line. Streamed so a large corpus is not held twice."""
    for record in records:
        yield json.dumps(record, ensure_ascii=False) + "\n"


def stats(records: List[Dict]) -> Dict:
    """Counts the card is built from, and that a reader would ask for first."""
    by_label = Counter(r["label"] for r in records)
    corrected = sum(1 for r in records if r["corrected"])
    confidences = [r["confidence"] for r in records if r["confidence"] is not None]
    domain = active_domain()
    return {
        "domain": domain.key,
        "display_name": domain.display_name,
        "total": len(records),
        "corrected": corrected,
        "confirmed": len(records) - corrected,
        "label_counts": {name: by_label.get(name, 0) for name in domain.names},
        "labels_unused": [n for n in domain.names if by_label.get(n, 0) == 0],
        "mean_confidence": (
            round(sum(confidences) / len(confidences), 4) if confidences else None
        ),
        "model_versions": sorted(
            {r["model_version"] for r in records if r.get("model_version")}
        ),
    }


def card(records: List[Dict]) -> str:
    """
    A dataset card, in the shape a HuggingFace reader expects.

    Deliberately states the limits. A corpus of 40 rows described as though
    it were a benchmark is worse than one that says what it is, and the
    reader of this card is likely to be the examiner asking how big it
    really is.
    """
    s = stats(records)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    pct = (100 * s["corrected"] / s["total"]) if s["total"] else 0.0

    lines = [
        f"# {s['display_name']} — annotated corpus",
        "",
        f"Exported {generated} from the annotation framework "
        f"(domain `{s['domain']}`).",
        "",
        "## What this is",
        "",
        "Human-validated labels. Each document was classified by a fine-tuned",
        "model, routed for review when the model was unsure **or** when the",
        "legal concepts in the text contradicted its answer, and then confirmed",
        "or corrected by a person. Only confirmed rows are here.",
        "",
        "## Size",
        "",
        f"| | |",
        f"|---|---|",
        f"| Examples | **{s['total']}** |",
        f"| Confirmed as predicted | {s['confirmed']} |",
        f"| Corrected by a human | {s['corrected']} ({pct:.1f}%) |",
        f"| Classes | {len(s['label_counts'])} |",
    ]
    if s["mean_confidence"] is not None:
        lines.append(f"| Mean model confidence | {s['mean_confidence']} |")
    lines += [
        "",
        "## Label distribution",
        "",
        "| Label | Examples |",
        "|---|---:|",
    ]
    for name, count in s["label_counts"].items():
        lines.append(f"| {name} | {count} |")

    lines += [
        "",
        "## Fields",
        "",
        "| Field | Meaning |",
        "|---|---|",
        "| `text` | The document, normalised exactly as the model receives it |",
        "| `label` | The human-validated label. **This is the target** |",
        "| `label_id` | Its index in the domain's label order |",
        "| `predicted_label` | What the model said before review |",
        "| `confidence` | How sure the model was |",
        "| `corrected` | True where the human disagreed with the model |",
        "| `model_version` | Which checkpoint produced the prediction |",
        "| `annotated_at` | When it was validated |",
        "",
        "## Usage",
        "",
        "```python",
        "from datasets import load_dataset",
        'ds = load_dataset("json", data_files="dataset.jsonl")["train"]',
        "```",
        "",
        "## Limits — read before using this",
        "",
    ]

    limits = [
        "`corrected: true` rows are where this corpus earns its keep — they are "
        "the cases the model got wrong and a human fixed. They are also, by "
        "construction, **not** a random sample of the domain: they are what the "
        "routing layer chose to show a reviewer.",
        "Documents the model was confident about and that raised no concept "
        "conflict were auto-accepted and never reviewed, so they are absent. "
        "This corpus is biased toward hard cases.",
        "Labels come from the reviewers available to this project. Inter-annotator "
        "agreement has not been measured.",
    ]
    if s["total"] < 200:
        limits.insert(0, f"**This export contains {s['total']} examples.** That is a "
                         "demonstration of the pipeline, not a corpus to train a "
                         "production model on.")
    if s["labels_unused"]:
        limits.append("No examples at all for: " + ", ".join(s["labels_unused"]) + ".")
    if len(s["model_versions"]) > 1:
        limits.append("Predictions come from more than one model version "
                      f"({', '.join(s['model_versions'])}), so `confidence` is not "
                      "comparable across all rows.")

    lines += [f"- {item}" for item in limits]
    lines.append("")
    return "\n".join(lines)
