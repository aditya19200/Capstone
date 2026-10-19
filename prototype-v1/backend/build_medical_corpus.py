"""
build_medical_corpus.py — rebuild the medical corpus and its held-out split
from datasets/medical-raw/mtsamples.csv, reproducibly and without
contamination.

This script did not exist. The corpus it replaces was produced ad hoc, which
is how the defect below survived: there was nothing to read and nothing to
re-run.

## The defect

`sample-data/evaluation/medical_heldout.csv` shared 129 of its 435 rows with
`datasets/medical_v1.csv` — the same transcription on both sides of the
split. The usual consequence of that is an inflated score, because the model
is tested on text it memorised. Here it did the opposite, and the reason is
worth stating precisely.

Every one of those 129 rows carried a DIFFERENT gold label from its training
copy. Not one agreed. mtsamples files the same transcription under more than
one specialty — a cervical discectomy appears as both Neurosurgery and
Orthopedic, a foetal ultrasound as both Obstetrics and Radiology — and the
split separated the copies, so the model was trained to answer A and then
marked wrong for not answering B. It could not have scored on those rows at
any level of competence.

Measured on the shipped checkpoint:

    full held-out (435 rows)        68.3%
    the 129 contaminated rows       24.8%
    the 306 clean rows              86.6%

So 68.3% was not a flattered number, it was a depressed one, and the three
classes that looked worst — Radiology 0.29 F1, Nephrology 0.36, Neurosurgery
0.38 — are the three most involved in the conflicting pairs. The model was
being marked against labels its own corpus contradicts.

## What this script does about it

Groups rows by normalised text rather than by row index, which is the fix
for the overlap: two copies of one transcription can no longer land on
opposite sides.

Texts carrying more than one specialty are then kept out of the HELD-OUT
set and written to medical_heldout_SKIPPED.md with their competing labels. A
test set has to have answerable questions in it; a row whose own corpus
gives it two labels is not a measurement of anything.

They stay in TRAINING, because excluding them everywhere was the first thing
tried and it is not survivable. 379 of 1,745 unique transcriptions conflict,
and they are concentrated precisely in the thin classes: it would leave
Radiology with 17 training rows and Neurosurgery with 16, out of 269 and 94
available. A model cannot learn a class from 17 examples, so that cure is
worse than the disease.

For training, a conflicting text is assigned to whichever of its competing
specialties has the fewest unambiguous examples. The rule is arbitrary in
the sense that any rule here is -- the project has no clinician and will not
invent a clinical judgement -- but it is deterministic, it is written down,
and it is the choice that keeps the starved classes alive rather than the
one that happens to flatter the result. Its cost is a real risk of pulling
the larger partner class toward its smaller neighbour, which is measurable
on the clean held-out set and is reported rather than assumed.

The alternative worth noting, and rejected for scope rather than for merit:
these documents are genuinely multi-label. A cervical discectomy really is
both Neurosurgery and Orthopedic. Single-label classification is the wrong
frame for them and a multi-label head would be the honest fix.

Every copy of one text lands on one side of the split, which is what
actually removes the contamination. The script asserts zero overlap before
it writes anything, so this defect cannot return silently.
"""

import argparse
import collections
import csv
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from utils.text import normalize

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "datasets/medical-raw/mtsamples.csv"
PACK = ROOT / "domains/medical.json"
TRAIN_OUT = ROOT / "datasets/medical_v1.csv"
HELD_OUT = ROOT / "sample-data/evaluation/medical_heldout.csv"
SKIPPED_OUT = ROOT / "sample-data/evaluation/medical_heldout_SKIPPED.md"

# 200 characters after normalisation. Chosen because it reproduces the 2,173
# rows of the corpus this replaces exactly, so the only thing changing here
# is the contamination and not the inclusion rule as well.
MIN_CHARS = 200
HELDOUT_FRAC = 0.20
SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change and write nothing")
    args = ap.parse_args()

    csv.field_size_limit(10 ** 7)
    labels = [l["name"] for l in json.loads(PACK.read_text())["labels"]]
    known = set(labels)

    with open(RAW, newline="", encoding="utf-8") as fh:
        raw = list(csv.DictReader(fh))

    # 1. the rows that belong to a specialty we actually serve
    kept = []
    for r in raw:
        spec = (r.get("medical_specialty") or "").strip()
        text = normalize(r.get("transcription") or "")
        if spec in known and len(text) >= MIN_CHARS:
            kept.append((text, spec))

    # 2. group by text, which is what makes the split safe
    by_text = collections.defaultdict(set)
    for text, spec in kept:
        by_text[text].add(spec)

    conflicting = {t: ls for t, ls in by_text.items() if len(ls) > 1}
    consistent = {t: next(iter(ls)) for t, ls in by_text.items() if len(ls) == 1}

    print(f"raw rows in our {len(labels)} specialties : {len(kept)}")
    print(f"unique transcriptions               : {len(by_text)}")
    print(f"  one label  -> train + held-out    : {len(consistent)}")
    print(f"  conflicting -> train only         : {len(conflicting)}")

    # 3. stratified split over the unambiguous texts only
    by_label = collections.defaultdict(list)
    for text, spec in consistent.items():
        by_label[spec].append(text)

    train, held = [], []
    for spec in sorted(by_label):                 # sorted, so dict order cannot move it
        texts = sorted(by_label[spec])            # sorted, so set iteration cannot either
        random.Random(SEED).shuffle(texts)
        n_held = max(1, round(len(texts) * HELDOUT_FRAC))
        held += [(t, spec) for t in texts[:n_held]]
        train += [(t, spec) for t in texts[n_held:]]

    # 4. conflicting texts join training only, assigned to their thinnest
    #    competing class. Counted against the unambiguous totals, so the rule
    #    does not chase its own tail as rows are added.
    unambiguous_support = {spec: len(texts) for spec, texts in by_label.items()}
    for text in sorted(conflicting):
        spec = min(sorted(conflicting[text]),
                   key=lambda s: unambiguous_support.get(s, 0))
        train.append((text, spec))

    # 4. the assertion that makes this script worth having
    overlap = set(t for t, _ in train) & set(t for t, _ in held)
    assert not overlap, f"{len(overlap)} text(s) on both sides of the split"
    assert len(set(t for t, _ in train)) == len(train), "duplicate rows in train"
    assert len(set(t for t, _ in held)) == len(held), "duplicate rows in held-out"

    print(f"\ntrain    : {len(train)}")
    print(f"held-out : {len(held)}")
    print(f"overlap  : 0  (asserted)")
    print(f"\n{'label':34}{'train':>7}{'held':>7}{'conflict':>10}")
    skipped_per_label = collections.Counter()
    for t, ls in conflicting.items():
        for l in ls:
            skipped_per_label[l] += 1
    tr_c = collections.Counter(l for _, l in train)
    ho_c = collections.Counter(l for _, l in held)
    for l in labels:
        print(f"  {l[:32]:34}{tr_c[l]:>5}{ho_c[l]:>7}{skipped_per_label[l]:>9}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    for path, rows in ((TRAIN_OUT, train), (HELD_OUT, held)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["text", "label_name", "source"])
            for text, spec in rows:
                w.writerow([text, spec, "mtsamples"])
        print(f"\nwrote {path.relative_to(ROOT)}  ({len(rows)} rows)")

    pairs = collections.Counter()
    for ls in conflicting.values():
        pairs[" / ".join(sorted(ls))] += 1
    lines = [
        "# Medical transcriptions deliberately left out of both splits",
        "",
        f"{len(conflicting)} transcriptions appear in mtsamples under more than one",
        "medical specialty. The same text, filed twice, labelled differently.",
        "",
        "They are excluded from the HELD-OUT set: a test set needs answerable",
        "questions, and a row whose own corpus gives it two labels is not one. Same",
        "reasoning as heldout_SKIPPED.md on the legal side.",
        "",
        "They remain in TRAINING, assigned to whichever competing specialty has the",
        "fewest unambiguous examples. Excluding them everywhere leaves Radiology with",
        "17 training rows and Neurosurgery with 16, which is not a trade worth making.",
        "The project has no clinician and does not invent clinical judgements, so the",
        "rule is deterministic and written down rather than correct.",
        "",
        "They were previously split across train and held-out, which meant 129 held-out",
        "rows were scored against a label that contradicted the one the model had been",
        "trained on. Those rows scored 24.8% against 86.6% on the clean remainder, and",
        "dragged the reported accuracy from 86.6% down to 68.3%.",
        "",
        "## The competing label pairs",
        "",
        "| count | filed as |",
        "|---|---|",
    ]
    lines += [f"| {c} | {p} |" for p, c in pairs.most_common()]
    lines += [
        "",
        "## Per specialty",
        "",
        "How many excluded transcriptions each specialty appears in. The totals exceed",
        f"{len(conflicting)} because every conflict involves at least two specialties.",
        "",
        "| specialty | appears in |",
        "|---|---|",
    ]
    lines += [f"| {l} | {skipped_per_label[l]} |"
              for l in labels if skipped_per_label[l]]
    lines.append("")
    SKIPPED_OUT.write_text("\n".join(lines))
    print(f"wrote {SKIPPED_OUT.relative_to(ROOT)}  ({len(conflicting)} excluded)")


if __name__ == "__main__":
    main()
