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
import collections
from collections import Counter

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from config.domains import get_domain
from config.settings import settings
from services import pdf_extract
from train_domain import load_csv
from utils.text import count_tokens, normalize

ROOT = pathlib.Path(__file__).resolve().parent.parent
HERE = pathlib.Path(__file__).resolve().parent


def load_pdf_manifest(manifest, pdf_dir, known, exclude):
    """
    The legal held-out set is PDFs plus a manifest, not a CSV.

    Each PDF splits into many passages and every passage inherits its
    DOCUMENT's label, which is weak by construction -- a tax judgment
    contains procedural passages that are not tax law. The source filename
    comes back with each passage so the caller can aggregate to the
    document level, which is the figure docs/RESULTS.md quotes.
    """
    rows, sources, skipped = [], [], []
    for row in csv.DictReader(open(manifest, encoding="utf-8")):
        if row["file"] in exclude:
            continue
        path = pdf_dir / row["file"]
        if not path.exists():
            skipped.append((row["file"], "missing"))
            continue
        if row["gold_label"] not in known:
            skipped.append((row["file"], f"unknown label {row['gold_label']!r}"))
            continue
        try:
            extracted = pdf_extract.extract(
                path.read_bytes(), token_counter=count_tokens,
                max_tokens=settings.MAX_LENGTH,
            )
        except pdf_extract.PdfExtractionError as exc:
            # Scans with no text layer are refused by the guard, not a bug.
            skipped.append((row["file"], type(exc).__name__))
            continue
        for doc in extracted.documents:
            rows.append((normalize(doc), row["gold_label"]))
            sources.append(row["file"])
    return rows, sources, skipped


def document_level(rows, sources, preds):
    """
    One verdict per document: the label most of its passages got.

    Aggregating ~80 passages smooths out individual errors, which is why
    this number is higher than the passage figure and why both are reported.
    """
    by_doc = collections.defaultdict(list)
    gold_of = {}
    for (_, gold), src, (label, _) in zip(rows, sources, preds):
        by_doc[src].append(label)
        gold_of[src] = gold
    verdict = {}
    for src, labels in by_doc.items():
        verdict[src] = Counter(labels).most_common(1)[0][0]
    correct = sum(1 for src, v in verdict.items() if v == gold_of[src])
    return correct, len(verdict), verdict, gold_of


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
    ap.add_argument("--pdf-manifest", default=None,
                    help="a manifest CSV (file,gold_label) beside a directory of "
                         "PDFs, as the legal held-out set is shaped. Reports both "
                         "passage-level and document-level accuracy.")
    ap.add_argument("--pdf-dir", default=None,
                    help="where the manifest's PDFs live; defaults to a 'heldout' "
                         "directory beside the manifest")
    ap.add_argument("--exclude", default="",
                    help="comma-separated filenames to drop from the manifest")
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
    sources = None
    if args.pdf_manifest:
        manifest = pathlib.Path(args.pdf_manifest)
        pdf_dir = pathlib.Path(args.pdf_dir) if args.pdf_dir else manifest.parent / "heldout"
        exclude = {e.strip() for e in args.exclude.split(",") if e.strip()}
        rows, sources, skipped = load_pdf_manifest(manifest, pdf_dir, known, exclude)
        data = manifest
        for name, why in skipped:
            print(f"  skipped {name}: {why}")
        if exclude:
            print(f"  excluded by request: {', '.join(sorted(exclude))}")
    else:
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

    if sources:
        n_docs = len(set(sources))
        print(f"  {len(rows)} passages from {n_docs} documents\n")
        for name, preds in (("v1", p1), ("v2", p2)):
            c, n, verdict, gold_of = document_level(rows, sources, preds)
            print(f"{name} DOCUMENT-level  {c}/{n} = {c/n:.4f}")
            for src in sorted(verdict):
                if verdict[src] != gold_of[src]:
                    print(f"     miss: {src}  gold {gold_of[src]} -> said {verdict[src]}")
        print()

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
