#!/usr/bin/env python3
"""
evaluate_baseline.py — measure the currently deployed model on labelled data.

Why this exists
---------------
The model the API actually serves (settings.MODEL_PATH) is just a folder on
disk. It has no model_versions row, so:

  * the admin version-history table opens empty,
  * the per-class F1 chart has nothing to plot, and
  * retrain's "don't accept a model >5 points worse" guard compares against
    an active accuracy of 0.0, which means it can never fire.

This script scores the deployed model on datasets/v1.csv (1701 labelled
samples, all 10 categories) and writes metrics/baseline_metrics.json, which
the API reads at startup to register the deployed model as the baseline
version.

Honesty caveat — read before quoting these numbers
--------------------------------------------------
v1.csv is the dataset the *v1* model was trained on. Whether the currently
deployed checkpoint (LegalModelShared) was trained on the same data is NOT
known — its provenance was never recorded. If it was, some or all of these
rows were seen during training and the score below is inflated.

Treat this as an UPPER BOUND on real-world accuracy, not a held-out result.
The caveat is written into the JSON output so it travels with the numbers.

Text is normalised with utils.text.normalize — the identical function the
live prediction path uses — so this measures the model under the same
conditions it runs in production.

Run from the backend/ directory:
    venv/bin/python evaluate_baseline.py
"""

import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

from config.settings import settings                    # noqa: E402
from services.model_service import ID2LABEL, model_service  # noqa: E402
from utils.text import normalize                        # noqa: E402

DATASET = BACKEND_DIR.parent / "datasets" / "v1.csv"
OUT_DIR = BACKEND_DIR / "metrics"
OUT_FILE = OUT_DIR / "baseline_metrics.json"
BATCH_SIZE = 32


def load_dataset():
    """Return (texts, gold_labels, sources), skipping rows with unknown labels."""
    if not DATASET.exists():
        print(f"Dataset not found: {DATASET}")
        print("Recover it with:")
        print("  git show origin/feature/ml-pipeline-v1:datasets/v1.csv > datasets/v1.csv")
        sys.exit(1)

    valid = set(ID2LABEL.values())
    texts, labels, sources, skipped = [], [], [], 0

    with open(DATASET, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = (row.get("label_name") or "").strip()
            text = (row.get("text") or "").strip()
            if not text or label not in valid:
                skipped += 1
                continue
            texts.append(normalize(text))   # same normalisation as inference
            labels.append(label)
            sources.append(row.get("source", "unknown"))

    if skipped:
        print(f"Skipped {skipped} row(s) with empty text or an unrecognised label.")
    return texts, labels, sources


def score(gold, predicted):
    """
    Accuracy and per-class F1, computed the same way retrain_worker does, so
    baseline and retrained numbers are directly comparable.
    """
    tp = Counter()
    fp = Counter()
    fn = Counter()
    correct = 0

    for g, p in zip(gold, predicted):
        if g == p:
            correct += 1
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1

    accuracy = correct / len(gold) if gold else 0.0

    f1_per_class = {}
    for label in ID2LABEL.values():
        precision = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) else 0.0
        recall = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1_per_class[label] = round(f1, 4)

    return round(accuracy, 4), f1_per_class


def main():
    print(f"Loading model from {settings.MODEL_PATH} ...")
    model_service.load()

    texts, gold, sources = load_dataset()
    print(f"Scoring {len(texts)} labelled samples (batch size {BATCH_SIZE})...\n")

    predicted, confidences = [], []
    for start in range(0, len(texts), BATCH_SIZE):
        chunk = texts[start:start + BATCH_SIZE]
        for result in model_service.predict_batch(chunk):
            predicted.append(result["predicted_label"])
            confidences.append(result["confidence"])
        done = min(start + BATCH_SIZE, len(texts))
        print(f"\r  {done}/{len(texts)}", end="", flush=True)
    print("\n")

    accuracy, f1_per_class = score(gold, predicted)
    support = Counter(gold)
    mean_conf = sum(confidences) / len(confidences)

    # Accuracy split by where each sample came from — a large gap between
    # hand-written and real-world sources is worth knowing about.
    by_source = defaultdict(lambda: [0, 0])
    for g, p, s in zip(gold, predicted, sources):
        by_source[s][1] += 1
        if g == p:
            by_source[s][0] += 1

    print("=" * 62)
    print(f"  Overall accuracy : {accuracy:.4f}  ({accuracy * 100:.1f}%)")
    print(f"  Mean confidence  : {mean_conf:.4f}")
    print("=" * 62)
    print(f"\n  {'Category':32} {'F1':>7} {'support':>8}")
    print("  " + "-" * 50)
    for label in ID2LABEL.values():
        print(f"  {label:32} {f1_per_class[label]:>7.4f} {support[label]:>8}")

    print(f"\n  {'Source':20} {'accuracy':>10} {'n':>7}")
    print("  " + "-" * 40)
    for src, (ok, total) in sorted(by_source.items(), key=lambda kv: -kv[1][1]):
        print(f"  {src:20} {ok / total:>10.4f} {total:>7}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps({
        "version_number": "baseline-LegalModelShared",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "model_path": settings.MODEL_PATH,
        "dataset": "datasets/v1.csv",
        "sample_count": len(texts),
        "accuracy": accuracy,
        "mean_confidence": round(mean_conf, 4),
        "f1_per_class": f1_per_class,
        "support_per_class": dict(support),
        "accuracy_by_source": {
            s: round(ok / total, 4) for s, (ok, total) in by_source.items()
        },
        "caveat": (
            "Scored on datasets/v1.csv, which is the training set of the v1 "
            "model. Whether the deployed checkpoint was trained on this same "
            "data is unknown — its provenance was never recorded. Treat this "
            "as an upper bound, not a held-out result."
        ),
    }, indent=2))

    print(f"\nWrote {OUT_FILE.relative_to(BACKEND_DIR.parent)}")
    print("Restart the backend to register this as the baseline model version.")


if __name__ == "__main__":
    main()
