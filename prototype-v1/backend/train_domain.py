#!/usr/bin/env python3
"""
train_domain.py — fine-tune the base checkpoint for a domain pack.

This is what "the framework is domain-portable" means in practice: point it
at a pack, and it reads that pack's label set, corpus and base checkpoint and
produces a model the rest of the system can serve. No code here knows
anything about law or medicine.

    python train_domain.py --domain medical

Hyperparameters match workers/retrain_worker.py exactly — 3 epochs, batch 8,
lr 2e-5, seed 42 — so a model trained here and a model produced by the
retraining loop are comparable. They are not two different training regimes
with the same name.

Evaluation uses the held-out split named by --heldout, which must never have
been trained on. The project has already made the mistake of measuring
itself on its own training data once; see sample-data/evaluation/README.md.
"""

import argparse
import csv
import json
import pathlib
import random
import sys
from collections import Counter
from datetime import datetime, timezone

import torch
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config.domains import get_domain
from utils.text import normalize

ROOT = pathlib.Path(__file__).resolve().parent.parent
EPOCHS, BATCH_SIZE, LEARNING_RATE, SEED = 3, 8, 2e-5, 42


def pin_randomness(seed: int = SEED) -> None:
    """
    Make a training run reproducible.

    Shuffling was already seeded, but two runs of this script still differed
    by more than two points of accuracy — 70.6% and 68.3% from identical
    inputs. The shuffle was never the problem: the classifier head is
    initialised randomly, and cuDNN picks different convolution and matmul
    kernels per run, which changes floating-point summation order. Neither is
    a bug, and both make a reported number impossible to reproduce, which
    matters when the number goes in a report someone may re-run.

    What this fixes, measured: the head initialisation. Three runs gave
    0.0460, 0.0506, 0.0506 before training — the last two pinned, and
    identical.

    What it does NOT fix, also measured: final accuracy still moved 0.6897
    to 0.6828 between two pinned runs. Transformer matmuls go through cuBLAS
    rather than cuDNN, and cudnn.deterministic does not reach it. Removing
    that needs CUBLAS_WORKSPACE_CONFIG=:4096:8 in the environment plus
    torch.use_deterministic_algorithms(True), which raises on any op lacking
    a deterministic kernel.

    So a figure from this script is reproducible to about +/- 0.7 points, not
    exactly. Quote it as a range, or set the two above and accept the risk of
    a run aborting on an unsupported op.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_csv(path, known):
    rows, skipped = [], 0
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            label = (row.get("label_name") or "").strip()
            text = normalize(row.get("text") or "")
            if label not in known or not text:
                skipped += 1
                continue
            rows.append((text, label))
    return rows, skipped


def evaluate(model, tok, rows, label2id, id2label, device, max_length, batch_size=32):
    """
    Accuracy, per-class F1, and the confusion pairs.

    F1 rather than per-class accuracy, because the classes are imbalanced and
    accuracy on a rare class flatters a model that simply never predicts it:
    a class with 16 examples out of 435 scores 96% accuracy by always saying
    "no". F1 is the harmonic mean of precision and recall, so it only goes up
    when the model both finds the class and is right about it. The admin
    screen and model_versions already store f1_per_class for this reason.
    """
    model.eval()
    correct = 0
    support = Counter()          # gold occurrences
    predicted = Counter()        # times predicted
    hit = Counter()              # correct predictions
    confusion = Counter()
    with torch.no_grad():
        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i + batch_size]
            enc = tok([t for t, _ in chunk], truncation=True, max_length=max_length,
                      padding=True, return_tensors="pt").to(device)
            preds = model(**enc).logits.argmax(dim=-1).tolist()
            for (_, gold), p in zip(chunk, preds):
                support[gold] += 1
                predicted[id2label[p]] += 1
                if label2id[gold] == p:
                    correct += 1
                    hit[gold] += 1
                else:
                    confusion[(gold, p)] += 1

    f1 = {}
    for name in support:
        prec = hit[name] / predicted[name] if predicted[name] else 0.0
        rec = hit[name] / support[name] if support[name] else 0.0
        f1[name] = round(2 * prec * rec / (prec + rec), 4) if (prec + rec) else 0.0
    return correct / max(len(rows), 1), support, hit, confusion, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--heldout", default=None,
                    help="held-out CSV; defaults to sample-data/evaluation/<key>_heldout.csv")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=LEARNING_RATE)
    ap.add_argument("--out", default=None, help="where to save; defaults to the pack's model_path")
    args = ap.parse_args()

    pin_randomness()
    domain = get_domain(args.domain)
    label2id = domain.label2id
    known = set(domain.names)

    train_rows, skipped = load_csv(ROOT / domain.dataset, known)
    heldout_path = pathlib.Path(
        args.heldout or ROOT / "sample-data/evaluation" / f"{domain.key}_heldout.csv"
    )
    held_rows, _ = load_csv(heldout_path, known) if heldout_path.exists() else ([], 0)

    print(f"domain      : {domain.key} ({domain.num_labels} labels)")
    print(f"base model  : {domain.base_model}")
    print(f"train       : {len(train_rows)} rows from {domain.dataset}"
          + (f" ({skipped} skipped)" if skipped else ""))
    print(f"held out    : {len(held_rows)} rows from {heldout_path.name}"
          if held_rows else "held out    : NONE — accuracy below would be meaningless")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device      : {device}\n")

    tok = AutoTokenizer.from_pretrained(domain.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        domain.base_model, num_labels=domain.num_labels,
        id2label=domain.id2label, label2id=label2id,
    ).to(device)

    if held_rows:
        acc0, *_ = evaluate(model, tok, held_rows, label2id, domain.id2label,
                            device, domain.max_length)
        print(f"before training: {acc0:.4f} on held-out "
              f"(random would be {1/domain.num_labels:.4f})\n")
    else:
        acc0 = None

    optimizer = AdamW(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        model.train()
        order = list(range(len(train_rows)))
        random.Random(SEED + epoch).shuffle(order)
        total, batches = 0.0, 0
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            enc = tok([train_rows[i][0] for i in idx], truncation=True,
                      max_length=domain.max_length, padding=True,
                      return_tensors="pt").to(device)
            labels = torch.tensor([label2id[train_rows[i][1]] for i in idx], device=device)
            optimizer.zero_grad()
            out = model(**enc, labels=labels)
            out.loss.backward()
            optimizer.step()
            total += out.loss.item()
            batches += 1
        print(f"  epoch {epoch+1}/{args.epochs} — mean loss {total/max(batches,1):.4f}", flush=True)

    out_dir = pathlib.Path(args.out or (pathlib.Path(__file__).parent / domain.model_path))
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"\nsaved to {out_dir}")

    if not held_rows:
        return
    acc, support, hit, confusion, f1 = evaluate(
        model, tok, held_rows, label2id, domain.id2label, device, domain.max_length
    )
    print(f"\nHELD-OUT ACCURACY  {acc:.4f}  ({acc0:.4f} before training)")
    print(f"{'label':34}{'n':>5}{'correct':>9}{'recall':>8}{'F1':>7}")
    for name in domain.names:
        n = support.get(name, 0)
        if not n:
            continue
        print(f"  {name[:32]:34}{n:>5}{hit.get(name,0):>9}"
              f"{hit.get(name,0)/n:>8.2f}{f1.get(name,0.0):>7.2f}")
    print("\nmost common confusions:")
    for (gold, pid), n in confusion.most_common(6):
        print(f"  {gold[:28]:30} -> {domain.id2label[pid][:28]:30} {n:>4}")

    metrics_path = pathlib.Path(__file__).parent / "metrics" / f"{domain.key}_baseline_metrics.json"
    metrics_path.parent.mkdir(exist_ok=True)
    metrics_path.write_text(json.dumps({
        "version_number": f"baseline-{out_dir.name}",
        "domain": domain.key,
        "accuracy": round(acc, 4),
        "accuracy_before_training": round(acc0, 4) if acc0 is not None else None,
        "dataset": domain.dataset,
        "heldout": str(heldout_path.relative_to(ROOT)),
        "train_rows": len(train_rows),
        "heldout_rows": len(held_rows),
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "base_model": domain.base_model,
        # Same key the admin screen and model_versions already use, so a
        # model trained here registers exactly like a retrained one.
        "f1_per_class": {n: f1[n] for n in domain.names if n in f1},
        "model_path": domain.model_path,
        "per_label_recall": {
            n: round(hit.get(n, 0) / support[n], 4)
            for n in domain.names if support.get(n)
        },
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }, indent=1))
    print(f"\nwrote {metrics_path.relative_to(pathlib.Path(__file__).parent)}")


if __name__ == "__main__":
    main()
