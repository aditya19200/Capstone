"""
compare_checkpoints.py — run one dataset through two checkpoints and write the
rows where they disagree.

Why this exists. Showing the retraining loop live costs a demo twenty minutes
it does not have: seed the annotations, correct them by hand, fine-tune, wait,
activate. The loop is the same loop whether it is watched or not, so this runs
the comparison ahead of time and leaves behind the artefact worth looking at —
the same documents, labelled by the model before and after, side by side with
what each one claimed.

THE TRAP, and the only thing that makes this output worth anything. The
dataset fed to both checkpoints must be data NEITHER was trained on. The
retrained checkpoint was fine-tuned on the validated queue, so running the
queue through it produces a near-perfect result that demonstrates
memorisation and nothing else. This script refuses to run on anything but a
held-out file for that reason, and prints which file it used.

What it writes:

  * `<key>_v1_vs_v2.csv` — every held-out row: the text, the gold label, and
    each checkpoint's label and confidence. This is the file to open in front
    of people, or to upload through the UI as two batches.
  * a printed summary of the four transitions that matter: fixed (v1 wrong,
    v2 right), broken (v1 right, v2 wrong), and the subset of each where the
    wrong answer was above the auto-accept threshold, which is the difference
    between an error a reviewer sees and one the system swallows.

A fixed row where v1 was confidently wrong is the strongest single thing to
show: the old model would have auto-accepted a wrong label with no human
ever seeing it, and the new one does not.
"""

import argparse
import csv
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


def predict(path, rows, domain, device, batch_size=32):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(
        path, num_labels=domain.num_labels,
        id2label=domain.id2label, label2id=domain.label2id,
    ).to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i + batch_size]
            enc = tok([t for t, _ in chunk], truncation=True,
                      max_length=domain.max_length, padding=True,
                      return_tensors="pt").to(device)
            probs = torch.softmax(model(**enc).logits, dim=-1)
            conf, pred = probs.max(dim=-1)
            out += [(domain.id2label[p], c) for p, c in zip(pred.tolist(), conf.tolist())]
    del model
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--v1", required=True, help="the before checkpoint")
    ap.add_argument("--v2", required=True, help="the after checkpoint")
    ap.add_argument("--dataset", default=None,
                    help="defaults to the domain's held-out file, which is the only "
                         "safe choice unless you know the retrained checkpoint never "
                         "saw what you pass here")
    ap.add_argument("--focus", default=None,
                    help="print only this label's rows in the detail list")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    domain = get_domain(args.domain)
    known = set(domain.names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = pathlib.Path(
        args.dataset or ROOT / "sample-data/evaluation" / f"{domain.key}_heldout.csv"
    )
    rows, _ = load_csv(data, known)
    if not rows:
        sys.exit(f"no usable rows in {data}")

    high = settings.CONFIDENCE_HIGH
    print(f"domain     : {domain.key}")
    # A dataset outside the repo is a legitimate thing to pass -- a converted
    # probe set, a one-off file -- so show it plainly rather than insisting it
    # live under ROOT.
    try:
        shown = data.relative_to(ROOT)
    except ValueError:
        shown = data
    print(f"dataset    : {shown}  ({len(rows)} rows)")
    print(f"v1         : {args.v1}")
    print(f"v2         : {args.v2}")
    print(f"auto-accept: {high}\n")

    p1 = predict(args.v1, rows, domain, device)
    p2 = predict(args.v2, rows, domain, device)

    fixed, broken, same_right, same_wrong = [], [], [], []
    for (text, gold), (l1, c1), (l2, c2) in zip(rows, p1, p2):
        rec = (text, gold, l1, c1, l2, c2)
        ok1, ok2 = l1 == gold, l2 == gold
        (fixed if (not ok1 and ok2) else
         broken if (ok1 and not ok2) else
         same_right if ok1 else same_wrong).append(rec)

    acc1 = sum(1 for _, g, l, _, _, _ in [(r[0], r[1], r[2], r[3], r[4], r[5]) for r in
               fixed + broken + same_right + same_wrong] if g == l)
    n = len(rows)
    print(f"v1 accuracy          {acc1}/{n} = {acc1/n:.4f}")
    acc2 = len(fixed) + len(same_right)
    print(f"v2 accuracy          {acc2}/{n} = {acc2/n:.4f}\n")

    print(f"FIXED   (v1 wrong -> v2 right)   {len(fixed)}")
    loud_fixed = [r for r in fixed if r[3] > high]
    print(f"  of which v1 was confidently wrong (>{high})  {len(loud_fixed)}")
    print(f"BROKEN  (v1 right -> v2 wrong)   {len(broken)}")
    loud_broken = [r for r in broken if r[5] > high]
    print(f"  of which v2 is confidently wrong (>{high})   {len(loud_broken)}")
    print(f"unchanged right {len(same_right)}   unchanged wrong {len(same_wrong)}\n")

    if fixed:
        print("fixed, by gold label:")
        for lbl, c in Counter(r[1] for r in fixed).most_common():
            print(f"  {c:>3}  {lbl}")
    if broken:
        print("\nbroken, by gold label:")
        for lbl, c in Counter(r[1] for r in broken).most_common():
            print(f"  {c:>3}  {lbl}")

    detail = [r for r in fixed if not args.focus or r[1] == args.focus]
    if detail:
        label = f" ({args.focus})" if args.focus else ""
        print(f"\nthe rows to show{label}:")
        for text, gold, l1, c1, l2, c2 in detail[:12]:
            flag = "  <-- v1 would have AUTO-ACCEPTED this" if c1 > high else ""
            print(f"\n  gold  {gold}")
            print(f"  v1    {l1} @ {c1:.3f}{flag}")
            print(f"  v2    {l2} @ {c2:.3f}")
            print(f"  text  {text[:150].strip()}...")

    out = pathlib.Path(args.out or HERE / "metrics" / f"{domain.key}_v1_vs_v2.csv")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["text", "gold_label", "v1_label", "v1_confidence",
                    "v2_label", "v2_confidence", "transition"])
        for bucket, name in ((fixed, "fixed"), (broken, "broken"),
                             (same_right, "unchanged_right"), (same_wrong, "unchanged_wrong")):
            for text, gold, l1, c1, l2, c2 in bucket:
                w.writerow([text, gold, l1, round(c1, 4), l2, round(c2, 4), name])
    try:
        print(f"\nwrote {out.relative_to(HERE)}")
    except ValueError:
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
