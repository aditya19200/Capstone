"""
retrain_experiment.py — measure what the correction loop is worth, on a
domain, under conditions that cannot flatter it.

Why this exists separately from train_domain.py and workers/retrain_worker.py:

  * train_domain.py trains one model on everything. It answers "how good is
    this model", not "does human correction improve it".
  * retrain_worker.py is the real loop, but it reads validated annotations
    out of Postgres. Reproducing a measurement through it needs a database,
    a running API and a human clicking twenty times, which is not a thing
    anyone will re-run to check a number in a report.

So this script simulates the loop offline with the worker's own recipe —
fine-tune from the *active checkpoint*, on *every validated annotation*,
3 epochs, batch 8, lr 2e-5, seed 42 — and reports the result against a
held-out set that no stage of it is allowed to touch.

The honest setup, which must be stated whenever the numbers are quoted:

  1. The pack's dataset is split into train (80%) and an incoming "queue"
     (20%), stratified by label, seeded. The queue stands in for documents
     arriving after deployment.
  2. A baseline is trained on the train part only. It is therefore a WEAKER
     model than the shipped checkpoint, which saw the queue rows too. That
     is deliberate: a loop that only improves a crippled model proves
     nothing, so the baseline has to be a model somebody would plausibly
     deploy, trained on less data rather than on bad data.
  3. The baseline predicts every queue row. Where it agrees with the gold
     label the row is a CONFIRMATION; where it disagrees it is a
     CORRECTION. No label is invented here — "the reviewer" is the dataset's
     own specialty field, so this measures the loop's mechanics and not our
     medical opinion.
  4. The baseline is fine-tuned on the validated queue, exactly as the
     worker would, and both checkpoints are evaluated on the untouched
     held-out file.

What makes the result trustworthy rather than merely positive: accuracy
alone cannot tell you whether a model got better or just got louder. A model
that learns to say 0.99 about everything scores better on any metric that
rewards confidence. So every evaluation here also reports

  * mean confidence on the predictions it got RIGHT,
  * mean confidence on the predictions it got WRONG,
  * how many predictions above the auto-accept threshold are wrong,
  * expected calibration error.

If accuracy rises while confidently-wrong predictions also rise, the gain is
not worth having and this script will show that.
"""

import argparse
import csv
import json
import math
import pathlib
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

import torch
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from config.domains import get_domain
from train_domain import SEED, evaluate, load_csv, pin_randomness
from utils.text import normalize

ROOT = pathlib.Path(__file__).resolve().parent.parent
HERE = pathlib.Path(__file__).resolve().parent

EPOCHS, BATCH_SIZE, LEARNING_RATE = 3, 8, 2e-5   # retrain_worker's own values
AUTO_ACCEPT = 0.85                                # services/active_learning.py
ECE_BINS = 10


# ---------------------------------------------------------------------------
# splitting
# ---------------------------------------------------------------------------

def stratified_split(rows, queue_frac, seed):
    """
    Hold back queue_frac of every label, not of the corpus.

    A plain random slice would take its 20% in proportion to class size and
    leave the rare classes with two or three rows in the queue, so the
    corrections would be almost entirely Cardiovascular and Orthopedic and
    the experiment would say nothing about the classes that actually need
    help. Stratifying costs nothing and keeps the queue shaped like the
    corpus.
    """
    by_label = defaultdict(list)
    for i, (_, label) in enumerate(rows):
        by_label[label].append(i)

    queue_idx = set()
    for label in sorted(by_label):                 # sorted: seed alone is not
        idx = by_label[label][:]                   # enough if dict order moves
        random.Random(seed).shuffle(idx)
        take = max(1, round(len(idx) * queue_frac))
        queue_idx.update(idx[:take])

    train = [r for i, r in enumerate(rows) if i not in queue_idx]
    queue = [r for i, r in enumerate(rows) if i in queue_idx]
    return train, queue


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def fine_tune(model, tok, rows, label2id, max_length, device, epochs, batch_size, lr,
              tag=""):
    optimizer = AdamW(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        order = list(range(len(rows)))
        random.Random(SEED + epoch).shuffle(order)
        total, batches = 0.0, 0
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            enc = tok([rows[i][0] for i in idx], truncation=True, max_length=max_length,
                      padding=True, return_tensors="pt").to(device)
            labels = torch.tensor([label2id[rows[i][1]] for i in idx], device=device)
            optimizer.zero_grad()
            out = model(**enc, labels=labels)
            out.loss.backward()
            optimizer.step()
            total += out.loss.item()
            batches += 1
        print(f"  {tag}epoch {epoch+1}/{epochs} — mean loss {total/max(batches,1):.4f}",
              flush=True)
    return model


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------

def predict_all(model, tok, rows, label2id, id2label, max_length, device, batch_size=32):
    """Return [(gold, pred, confidence)] for every row."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i + batch_size]
            enc = tok([t for t, _ in chunk], truncation=True, max_length=max_length,
                      padding=True, return_tensors="pt").to(device)
            probs = torch.softmax(model(**enc).logits, dim=-1)
            conf, pred = probs.max(dim=-1)
            for (_, gold), p, c in zip(chunk, pred.tolist(), conf.tolist()):
                out.append((gold, id2label[p], c))
    return out


def confidence_report(preds):
    """
    The honesty metrics. `preds` is [(gold, pred, conf)].

    ECE is the average gap between what the model claims and how often it is
    right, weighted by how many predictions fall in each confidence bin. A
    model can be accurate and badly calibrated at the same time, which is
    why it is reported next to accuracy rather than instead of it.
    """
    n = len(preds)
    right = [c for g, p, c in preds if g == p]
    wrong = [c for g, p, c in preds if g != p]
    loud_wrong = [c for g, p, c in preds if g != p and c > AUTO_ACCEPT]
    loud = [c for _, _, c in preds if c > AUTO_ACCEPT]

    bins = [[] for _ in range(ECE_BINS)]
    for g, p, c in preds:
        b = min(int(c * ECE_BINS), ECE_BINS - 1)
        bins[b].append(1 if g == p else 0)
    ece = sum(
        (len(b) / n) * abs(sum(b) / len(b) - ((i + 0.5) / ECE_BINS))
        for i, b in enumerate(bins) if b
    ) if n else 0.0

    return {
        "n": n,
        "accuracy": round(len(right) / n, 4) if n else 0.0,
        "mean_confidence": round(sum(c for _, _, c in preds) / n, 4) if n else 0.0,
        "mean_confidence_when_right": round(sum(right) / len(right), 4) if right else None,
        "mean_confidence_when_wrong": round(sum(wrong) / len(wrong), 4) if wrong else None,
        "above_auto_accept": len(loud),
        "above_auto_accept_wrong": len(loud_wrong),
        # The single number that decides whether an accuracy gain is real:
        # of everything the system would have accepted without a human, how
        # much of it was wrong.
        "confidently_wrong_rate": round(len(loud_wrong) / len(loud), 4) if loud else None,
        "ece": round(ece, 4),
    }


def load_model(path, domain, device):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(
        path, num_labels=domain.num_labels,
        id2label=domain.id2label, label2id=domain.label2id,
    ).to(device)
    return model, tok


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--queue-frac", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=EPOCHS,
                    help="epochs for the BASELINE. This is the model a team would have "
                         "deployed, so it gets a full training run whatever the retrain "
                         "recipe is -- passing one value for both made the baseline "
                         "undertrained and the comparison meaningless.")
    ap.add_argument("--retrain-epochs", type=int, default=None,
                    help="epochs for the fine-tune; defaults to --epochs")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=LEARNING_RATE)
    ap.add_argument("--corrections-only", action="store_true",
                    help="fine-tune on the rows the reviewer changed, not on the whole "
                         "validated queue; isolates what a correction is worth")
    ap.add_argument("--tag", default="",
                    help="suffix for the v2 checkpoint and the result file, so several "
                         "retrain variants can be compared against one baseline")
    ap.add_argument("--work-dir", default=None,
                    help="where checkpoints go; default backend/models/experiments/<key>")
    ap.add_argument("--stage", default="all",
                    choices=("all", "baseline", "queue", "retrain", "measure"))
    args = ap.parse_args()

    pin_randomness()
    domain = get_domain(args.domain)
    known = set(domain.names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    work = pathlib.Path(args.work_dir or HERE / "models" / "experiments" / domain.key)
    work.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    v1_dir = work / "v1_baseline"
    v2_dir = work / f"v2_retrained{suffix}"
    # Results go to metrics/, not into the checkpoint directory: backend/models
    # is gitignored wholesale (the weights are 400MB each), so a result file
    # written beside them is a measurement nobody else can see.
    state_path = (HERE / "metrics" / f"{domain.key}_retrain_experiment{suffix}.json")
    state_path.parent.mkdir(exist_ok=True)

    all_rows, skipped = load_csv(ROOT / domain.dataset, known)
    heldout_path = ROOT / "sample-data/evaluation" / f"{domain.key}_heldout.csv"
    held_rows, _ = load_csv(heldout_path, known)
    train_rows, queue_rows = stratified_split(all_rows, args.queue_frac, SEED)

    print(f"domain       : {domain.key} ({domain.num_labels} labels)")
    print(f"corpus       : {len(all_rows)} rows" + (f" ({skipped} skipped)" if skipped else ""))
    print(f"  train      : {len(train_rows)}")
    print(f"  queue      : {len(queue_rows)}  (stands in for post-deployment arrivals)")
    print(f"held-out     : {len(held_rows)} rows from {heldout_path.name}  — never trained on")
    print(f"device       : {device}")
    print(f"work dir     : {work.relative_to(HERE)}\n")

    if not held_rows:
        sys.exit("no held-out file — every number below would be meaningless")

    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    want = lambda s: args.stage in ("all", s)

    # -- stage 1: the baseline a team would have deployed ------------------
    if want("baseline"):
        print(f"[1/4] training baseline on {len(train_rows)} rows "
              f"from {domain.base_model}", flush=True)
        model, tok = load_model(domain.base_model, domain, device)
        fine_tune(model, tok, train_rows, domain.label2id, domain.max_length, device,
                  args.epochs, args.batch_size, args.lr, tag="v1 ")
        v1_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(v1_dir)
        tok.save_pretrained(v1_dir)
        print(f"      saved {v1_dir.name}\n")
        del model
        torch.cuda.empty_cache()

    # -- stage 2: the reviewer works the queue -----------------------------
    if want("queue"):
        print(f"[2/4] baseline predicts the {len(queue_rows)}-row queue", flush=True)
        model, tok = load_model(v1_dir, domain, device)
        qp = predict_all(model, tok, queue_rows, domain.label2id, domain.id2label,
                         domain.max_length, device)
        confirmed = [(t, l) for (t, l), (g, p, _) in zip(queue_rows, qp) if g == p]
        corrected = [(t, l) for (t, l), (g, p, _) in zip(queue_rows, qp) if g != p]
        per_label = Counter(l for _, l in corrected)
        state["queue"] = {
            "rows": len(queue_rows),
            "confirmed": len(confirmed),
            "corrected": len(corrected),
            "corrections_per_label": dict(per_label.most_common()),
            "queue_confidence": confidence_report(qp),
        }
        state_path.write_text(json.dumps(state, indent=1))
        print(f"      confirmed {len(confirmed)}  corrected {len(corrected)}")
        for l, c in per_label.most_common():
            print(f"        {c:>3}  {l}")
        print()
        del model
        torch.cuda.empty_cache()

    # -- stage 3: retrain exactly as the worker would ----------------------
    if want("retrain"):
        model, tok = load_model(v1_dir, domain, device)
        qp = predict_all(model, tok, queue_rows, domain.label2id, domain.id2label,
                         domain.max_length, device)
        validated = [
            (t, l) for (t, l), (g, p, _) in zip(queue_rows, qp)
            if (g != p) or not args.corrections_only
        ]
        print(f"[3/4] fine-tuning the baseline on {len(validated)} validated row(s)"
              + (" (corrections only)" if args.corrections_only else ""), flush=True)
        r_epochs = args.retrain_epochs if args.retrain_epochs is not None else args.epochs
        fine_tune(model, tok, validated, domain.label2id, domain.max_length, device,
                  r_epochs, args.batch_size, args.lr, tag="v2 ")
        v2_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(v2_dir)
        tok.save_pretrained(v2_dir)
        state["retrain"] = {
            "validated_rows": len(validated),
            "corrections_only": args.corrections_only,
            "from_checkpoint": v1_dir.name,
            "baseline_epochs": args.epochs, "retrain_epochs": r_epochs,
            "batch_size": args.batch_size, "lr": args.lr,
        }
        state_path.write_text(json.dumps(state, indent=1))
        print(f"      saved {v2_dir.name}\n")
        del model
        torch.cuda.empty_cache()

    # -- stage 4: both models, same untouched held-out ---------------------
    if want("measure"):
        print("[4/4] evaluating both checkpoints on the held-out set\n", flush=True)
        results = {}
        for name, path in (("v1_baseline", v1_dir), ("v2_retrained", v2_dir)):  # noqa: E501
            model, tok = load_model(path, domain, device)
            acc, support, hit, confusion, f1 = evaluate(
                model, tok, held_rows, domain.label2id, domain.id2label,
                device, domain.max_length,
            )
            preds = predict_all(model, tok, held_rows, domain.label2id, domain.id2label,
                                domain.max_length, device)
            results[name] = {
                "accuracy": round(acc, 4),
                "f1_per_class": {n: f1[n] for n in domain.names if n in f1},
                "per_label_recall": {
                    n: round(hit.get(n, 0) / support[n], 4)
                    for n in domain.names if support.get(n)
                },
                "top_confusions": [
                    [g, domain.id2label[p], n] for (g, p), n in confusion.most_common(6)
                ],
                "confidence": confidence_report(preds),
            }
            del model
            torch.cuda.empty_cache()

        state["heldout"] = {
            "file": str(heldout_path.relative_to(ROOT)),
            "rows": len(held_rows),
            "results": results,
        }
        state["measured_at"] = datetime.now(timezone.utc).isoformat()
        state_path.write_text(json.dumps(state, indent=1))

        a, b = results["v1_baseline"], results["v2_retrained"]
        pp = lambda x, y: f"{(y - x) * 100:+.1f}pp"
        print(f"{'':36}{'v1':>10}{'v2':>10}{'delta':>10}")
        print(f"{'held-out accuracy':36}{a['accuracy']:>10.4f}{b['accuracy']:>10.4f}"
              f"{pp(a['accuracy'], b['accuracy']):>10}")
        ca, cb = a["confidence"], b["confidence"]
        for key, label in (
            ("mean_confidence", "mean confidence"),
            ("mean_confidence_when_right", "  ...when right"),
            ("mean_confidence_when_wrong", "  ...when wrong"),
            ("confidently_wrong_rate", "confidently-wrong rate"),
            ("ece", "calibration error (ECE)"),
        ):
            x, y = ca.get(key), cb.get(key)
            if x is None or y is None:
                continue
            print(f"{label:36}{x:>10.4f}{y:>10.4f}{pp(x, y):>10}")
        print(f"{'wrong above auto-accept':36}"
              f"{ca['above_auto_accept_wrong']:>10}{cb['above_auto_accept_wrong']:>10}"
              f"{cb['above_auto_accept_wrong'] - ca['above_auto_accept_wrong']:>+10}")

        print(f"\n{'per-class F1':36}{'v1':>10}{'v2':>10}{'delta':>10}")
        for n in domain.names:
            if n not in a["f1_per_class"]:
                continue
            x, y = a["f1_per_class"][n], b["f1_per_class"][n]
            print(f"  {n[:32]:34}{x:>10.3f}{y:>10.3f}{pp(x, y):>10}")

        print(f"\nwrote {state_path.relative_to(HERE)}")


if __name__ == "__main__":
    main()
