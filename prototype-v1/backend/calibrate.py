#!/usr/bin/env python3
"""
calibrate.py — is the model's confidence honest, and by how much?

Accuracy asks whether the predicted label is right. Calibration asks a
different question: when the model says 0.98, is it right 98% of the time?
Both matter here, and the second one more than usual, because every routing
decision this system makes is a threshold on confidence. Auto-accepting
everything above CONFIDENCE_HIGH is only defensible if that number means
what it says.

What this measures
------------------
Expected Calibration Error. Predictions are binned by confidence; within
each bin the mean confidence is compared with the actual accuracy, and the
gaps are averaged weighted by bin size. 0.0 is perfect honesty. A negative
gap means overconfidence.

Then it fits a temperature: a single scalar the logits are divided by before
softmax. Fitted on half the data, measured on the other half, because tuning
and evaluating on the same rows is the mistake this whole script exists to
avoid. Temperature cannot change which logit is largest, so accuracy is
identical before and after — only the confidence numbers move.

Which data, and why it matters
------------------------------
Run with --set heldout (the default) it uses sample-data/evaluation/, which
is the only data in this repository the model has definitely never seen.

Run with --set v1 it uses datasets/v1.csv, whose relationship to the deployed
checkpoint was never recorded — see the warning at the top of
evaluate_baseline.py. Numbers from it are not evidence about unseen text.

--exclude drops a document from the held-out set. It exists because the
Indian Stamp Act is 356 pages and splits into 821 passages, 56% of the whole
set on its own, so it decides the result by weight of size alone — and its
gold label is the one label in the manifest we have agreed is arguable. The
default excludes it. Pass --exclude "" to include everything and watch the
error triple.

    python calibrate.py                     # held-out, Stamp Act excluded
    python calibrate.py --set v1
    python calibrate.py --exclude ""
"""

import argparse
import csv
import json
import pathlib
import sys

import torch

from config.settings import settings
from services import pdf_extract
from services.model_service import ID2LABEL, model_service
from utils.text import count_tokens, normalize

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_EXCLUDE = "Indian Stamp Act 1899 eng.pdf"


@torch.no_grad()
def logits_for(texts, model, tok, device, batch_size=16):
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], truncation=True,
                  max_length=settings.MAX_LENGTH, padding=True,
                  return_tensors="pt").to(device)
        out.append(model(**enc).logits.float().cpu())
    return torch.cat(out)


def load_v1(label_to_id):
    rows = list(csv.DictReader(open(ROOT / "datasets/v1.csv", encoding="utf-8")))
    return ([normalize(r["text"]) for r in rows],
            torch.tensor([label_to_id[r["label_name"].strip()] for r in rows]))


def load_heldout(label_to_id, exclude):
    """
    Passages from the held-out PDFs, each carrying its DOCUMENT's label.

    That label is weak by construction: a tax judgment contains procedural
    passages that are not tax law, so perfect agreement is neither expected
    nor desirable. It is still the only labelled text here the model has not
    seen.
    """
    manifest = ROOT / "sample-data/evaluation/heldout_manifest.csv"
    texts, labels = [], []
    for row in csv.DictReader(open(manifest, encoding="utf-8")):
        if row["file"] in exclude:
            continue
        path = ROOT / "sample-data/evaluation/heldout" / row["file"]
        if not path.exists():
            print(f"  missing, skipped: {row['file']}")
            continue
        try:
            extracted = pdf_extract.extract(
                path.read_bytes(), token_counter=count_tokens,
                max_tokens=settings.MAX_LENGTH,
            )
        except pdf_extract.PdfExtractionError as exc:
            print(f"  refused, skipped: {row['file']} ({type(exc).__name__})")
            continue
        for doc in extracted.documents:
            texts.append(normalize(doc))
            labels.append(label_to_id[row["gold_label"]])
    return texts, torch.tensor(labels)


def ece(logits, y, temperature=1.0, bins=10):
    probs = torch.softmax(logits / temperature, dim=-1)
    conf, pred = probs.max(dim=-1)
    correct = (pred == y).float()
    edges = torch.linspace(0, 1, bins + 1)
    error, table = 0.0, []
    for i in range(bins):
        mask = (conf > edges[i]) & (conf <= edges[i + 1])
        if mask.sum() == 0:
            continue
        error += (mask.float().mean()
                  * (correct[mask].mean() - conf[mask].mean()).abs()).item()
        table.append((float(edges[i]), float(edges[i + 1]), int(mask.sum()),
                      float(conf[mask].mean()), float(correct[mask].mean())))
    return error, table, correct.mean().item(), conf.mean().item()


def fit_temperature(logits, y):
    """Minimise NLL over a single scalar. LBFGS converges in a few steps."""
    temperature = torch.ones(1, requires_grad=True)
    optimiser = torch.optim.LBFGS([temperature], lr=0.05, max_iter=200)
    loss_fn = torch.nn.CrossEntropyLoss()

    def closure():
        optimiser.zero_grad()
        loss = loss_fn(logits / temperature.clamp(min=1e-3), y)
        loss.backward()
        return loss

    optimiser.step(closure)
    return float(temperature.detach().clamp(min=1e-3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=("heldout", "v1"), default="heldout")
    ap.add_argument("--exclude", default=DEFAULT_EXCLUDE,
                    help='document to drop from the held-out set; "" keeps all')
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="write results as JSON")
    args = ap.parse_args()

    model_service.load()
    model, tok = model_service._snapshot()
    device = next(model.parameters()).device
    label_to_id = {v: k for k, v in ID2LABEL.items()}

    exclude = {e for e in [args.exclude] if e}
    if args.set == "v1":
        texts, y = load_v1(label_to_id)
        print("WARNING: v1.csv may be this checkpoint's training data — see "
              "evaluate_baseline.py. These numbers say nothing about unseen text.")
    else:
        texts, y = load_heldout(label_to_id, exclude)
        if exclude:
            print(f"excluded: {', '.join(sorted(exclude))}")

    if len(texts) < 50:
        sys.exit(f"Only {len(texts)} passages — too few to bin meaningfully.")

    print(f"{len(texts)} passages; computing logits ...", flush=True)
    logits = logits_for(texts, model, tok, device)

    generator = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(y), generator=generator)
    half = len(y) // 2
    fit_idx, eval_idx = order[:half], order[half:]

    temperature = fit_temperature(logits[fit_idx], y[fit_idx])
    before, table, accuracy, mean_conf = ece(logits[eval_idx], y[eval_idx], 1.0, args.bins)
    after, _, _, mean_conf_cal = ece(logits[eval_idx], y[eval_idx], temperature, args.bins)

    print(f"\nfit on {half}, evaluated on {len(y) - half}")
    print(f"  accuracy                {accuracy:.4f}   (temperature cannot change this)")
    print(f"  mean confidence         {mean_conf:.4f} -> {mean_conf_cal:.4f}")
    print(f"  expected calib. error   {before:.4f} -> {after:.4f}   at T = {temperature:.3f}")
    print(f"\n  {'bin':>10} {'n':>6} {'confidence':>11} {'accuracy':>9} {'gap':>8}")
    for lo, hi, n, c, a in table:
        print(f"  {lo:.1f}-{hi:.1f} {n:>10} {c:>11.3f} {a:>9.3f} {a - c:>+8.3f}")
    print("\n  A negative gap is overconfidence: the model claimed more than it delivered.")
    print(f"  Set CALIBRATION_TEMPERATURE={temperature:.3f} in backend/.env to apply this.")

    if args.out:
        pathlib.Path(args.out).write_text(json.dumps({
            "set": args.set, "excluded": sorted(exclude), "n": len(y),
            "temperature": round(temperature, 4), "accuracy": round(accuracy, 4),
            "ece_before": round(before, 4), "ece_after": round(after, 4),
            "mean_confidence_before": round(mean_conf, 4),
            "mean_confidence_after": round(mean_conf_cal, 4),
            "bins": table,
        }, indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
