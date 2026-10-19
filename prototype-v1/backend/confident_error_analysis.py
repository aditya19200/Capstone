"""
confident_error_analysis.py — when the model is confident and wrong, is there
anything in its own output that could have told us?

Background. The framework has one confidently-wrong detector that works by
contradiction: mine vocabulary that indicates each label, and flag a
prediction whose text is full of vocabulary pointing somewhere else
(services/ontology_check.py). Measured on the legal domain it catches 3 of 5
planted errors at an 8.0% false-positive rate. Measured on the medical
domain it catches 0 of 5 at every one of thirty threshold combinations,
because medical classes that get confused SHARE vocabulary rather than
contradicting each other — a spine operation reads the same whether it is
filed Orthopedic or Neurosurgery, so there is no contradicting evidence to
find.

That leaves the question this script answers: for a domain where
contradiction is unavailable, is the signal somewhere else in the softmax?

Two candidates, and the first one is a dead end for a reason worth writing
down.

  MARGIN. services/active_learning.py overrides AUTO_ACCEPT when confidence
  is above CONFIDENCE_HIGH but the top-2 gap is under 0.10. That branch
  cannot execute. The probabilities are a softmax, so they sum to one, so
  p2 <= 1 - p1, so margin = p1 - p2 >= 2*p1 - 1. At p1 > 0.85 the margin is
  necessarily above 0.70. The guard's own docstring illustrates the case it
  means to catch as "0.86 vs 0.84", which sums to 1.70 and is not a
  probability distribution. The unit test asserting the branch passes
  confidence=0.90 with margin=0.05 as independent scalars, which a real
  prediction can never produce. This script measures the minimum margin
  actually observed above the threshold, so the bound is demonstrated and
  not merely argued.

  RUNNER-UP IDENTITY. Margin says how far ahead the winner is. It says
  nothing about WHO came second, and that is the part that carries
  information when classes overlap: if the correct answer is reliably the
  second choice, then a confident prediction whose runner-up is a known
  confusable partner is worth a second look even though every
  confidence-based signal says accept it. That is a detector the ontology
  route cannot provide, and unlike margin it is not ruled out by
  arithmetic. Whether it is any good is an empirical question, so it is
  measured here rather than asserted.

Reports, over the held-out set:
  * accuracy, and the confidently-wrong population at CONFIDENCE_HIGH
  * top-2 accuracy — how often the right answer is first or second
  * of the wrong predictions, the share whose runner-up is the gold label,
    split by whether the model was confident
  * the observed margin range above the threshold, which settles the bound
  * the confusion pairs that account for the confident errors
"""

import argparse
import json
import pathlib
import sys
from collections import Counter

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from config.domains import get_domain
from config.settings import settings
from train_domain import load_csv

ROOT = pathlib.Path(__file__).resolve().parent.parent
HERE = pathlib.Path(__file__).resolve().parent


def analyse(model, tok, rows, domain, device, high, batch_size=32):
    model.eval()
    recs = []
    with torch.no_grad():
        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i + batch_size]
            enc = tok([t for t, _ in chunk], truncation=True,
                      max_length=domain.max_length, padding=True,
                      return_tensors="pt").to(device)
            probs = torch.softmax(model(**enc).logits, dim=-1)
            top2 = probs.topk(2, dim=-1)
            for (_, gold), vals, idxs in zip(chunk, top2.values.tolist(), top2.indices.tolist()):
                recs.append({
                    "gold": gold,
                    "pred": domain.id2label[idxs[0]],
                    "runner_up": domain.id2label[idxs[1]],
                    "conf": vals[0],
                    "margin": vals[0] - vals[1],
                })
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--model", default=None,
                    help="checkpoint to analyse; defaults to the pack's model_path")
    ap.add_argument("--heldout", default=None)
    ap.add_argument("--high", type=float, default=settings.CONFIDENCE_HIGH)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    domain = get_domain(args.domain)
    known = set(domain.names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = pathlib.Path(args.model or (HERE / domain.model_path))
    heldout = pathlib.Path(
        args.heldout or ROOT / "sample-data/evaluation" / f"{domain.key}_heldout.csv"
    )
    rows, _ = load_csv(heldout, known)
    if not rows:
        sys.exit(f"no usable rows in {heldout}")

    print(f"domain    : {domain.key} ({domain.num_labels} labels)")
    print(f"checkpoint: {model_path}")
    print(f"held-out  : {len(rows)} rows from {heldout.name}")
    print(f"threshold : CONFIDENCE_HIGH = {args.high}\n")

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=domain.num_labels,
        id2label=domain.id2label, label2id=domain.label2id,
    ).to(device)

    recs = analyse(model, tok, rows, domain, device, args.high)
    n = len(recs)
    wrong = [r for r in recs if r["gold"] != r["pred"]]
    right = [r for r in recs if r["gold"] == r["pred"]]
    loud = [r for r in recs if r["conf"] > args.high]
    loud_wrong = [r for r in loud if r["gold"] != r["pred"]]
    quiet_wrong = [r for r in wrong if r["conf"] <= args.high]

    # top-2: was the right answer available as the second choice at all?
    top2_hit = [r for r in recs if r["gold"] in (r["pred"], r["runner_up"])]
    runner_up_is_gold_loud = [r for r in loud_wrong if r["runner_up"] == r["gold"]]
    runner_up_is_gold_quiet = [r for r in quiet_wrong if r["runner_up"] == r["gold"]]

    pct = lambda a, b: f"{100 * a / b:.1f}%" if b else "n/a"

    print(f"accuracy                        {len(right)}/{n}  ({pct(len(right), n)})")
    print(f"top-2 accuracy                  {len(top2_hit)}/{n}  ({pct(len(top2_hit), n)})")
    print(f"  → the gold label is the first or second choice this often\n")

    print(f"above CONFIDENCE_HIGH           {len(loud)}/{n}  ({pct(len(loud), n)})")
    print(f"  of those, WRONG               {len(loud_wrong)}  ({pct(len(loud_wrong), len(loud))})")
    print(f"  of those wrong, gold was 2nd  {len(runner_up_is_gold_loud)}"
          f"  ({pct(len(runner_up_is_gold_loud), len(loud_wrong))})")
    print(f"below CONFIDENCE_HIGH, wrong    {len(quiet_wrong)}")
    print(f"  of those, gold was 2nd        {len(runner_up_is_gold_quiet)}"
          f"  ({pct(len(runner_up_is_gold_quiet), len(quiet_wrong))})\n")

    # Settle the margin-guard bound with observed numbers rather than algebra.
    if loud:
        margins = [r["margin"] for r in loud]
        print(f"margin above the threshold      min {min(margins):.4f}  "
              f"max {max(margins):.4f}")
        print(f"  the guard fires below 0.10, so it fired "
              f"{sum(1 for m in margins if m < 0.10)} time(s) out of {len(loud)}\n")

    print("confusion pairs among the confident errors:")
    for (g, p), c in Counter((r["gold"], r["pred"]) for r in loud_wrong).most_common(8):
        print(f"  {c:>3}  {g[:30]:32} predicted as  {p}")

    payload = {
        "domain": domain.key,
        "checkpoint": str(model_path),
        "heldout": str(heldout.relative_to(ROOT)),
        "rows": n,
        "confidence_high": args.high,
        "accuracy": round(len(right) / n, 4),
        "top2_accuracy": round(len(top2_hit) / n, 4),
        "above_high": len(loud),
        "above_high_wrong": len(loud_wrong),
        "above_high_wrong_runner_up_was_gold": len(runner_up_is_gold_loud),
        "below_high_wrong": len(quiet_wrong),
        "below_high_wrong_runner_up_was_gold": len(runner_up_is_gold_quiet),
        "margin_above_high_min": round(min(r["margin"] for r in loud), 4) if loud else None,
        "margin_guard_fired": sum(1 for r in loud if r["margin"] < 0.10),
        "confident_error_pairs": [
            [g, p, c] for (g, p), c in
            Counter((r["gold"], r["pred"]) for r in loud_wrong).most_common()
        ],
    }
    out = pathlib.Path(args.out or HERE / "metrics" / f"{domain.key}_confident_errors.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {out.relative_to(HERE)}")


if __name__ == "__main__":
    main()
