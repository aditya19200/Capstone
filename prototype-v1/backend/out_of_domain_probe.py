#!/usr/bin/env python3
"""
out_of_domain_probe.py — what happens to text from outside the domain?

Why this exists
---------------
Every accuracy number this project reports is measured on text that belongs
to the domain: legal documents for the legal pack, clinical notes for the
medical one. The label set has no "none of the above", so a document from
outside the domain cannot be classified correctly — it can only be filed
somewhere. This measures where.

It also measures the obvious fix, so that "not implemented" is a decision
with a number attached rather than an omission.

What it answers
---------------
1. What label and confidence does the deployed model give off-domain text?
2. Does the ontology consistency check flag it?
3. Would a total-marker-evidence floor separate off-domain text from real
   domain text, and what would that cost?

Question 2 has an answer that can be read off the source before running
anything: OntologyCheck.check() opens with

    ev = self.evidence(text)
    if not ev:
        return None

so text matching no marker cannot raise a conflict. The check works by
finding vocabulary that CONTRADICTS the chosen label, and off-domain text has
none to contradict. This script confirms that empirically and, more usefully,
shows that the near-miss cases do not rescue it either — a sports report
matching one stray marker at 1.04 is still far below the 5.0 evidence floor.

Question 3 is the one worth having measured. A floor sounds free until you
look at the distribution of real domain text: 13.7% of v1.csv matches no
marker at all, because the miner's precision and multi-source filters are
strict and plenty of genuine legal passages are short or paraphrased. So the
floor that catches junk also floods the review queue, which is why this ships
without one. The honest fix is an abstain class trained on negatives.

    ./venv/bin/python out_of_domain_probe.py
    ./venv/bin/python out_of_domain_probe.py --no-model   # markers only, no weights

Writes metrics/<domain>_out_of_domain.json.
"""

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from config.domains import active_domain          # noqa: E402
from config.settings import settings              # noqa: E402
from services.ontology_check import ontology_check  # noqa: E402

# Off-domain samples, chosen to span the ways text can fail to belong:
# ordinary fluent prose on an unrelated subject (recipe, football, weather),
# technical prose from another field entirely (python_docs), text from the
# OTHER domain this framework serves (clinical_note), filler that is not even
# English (lorem), and input with no content at all (empty_ish, gibberish).
# The last entry is a control: real domain text, so a run that flags nothing
# anywhere is distinguishable from a run where the check is simply broken.
SAMPLES = {
    "recipe": "Heat two tablespoons of oil in a pan. Add the chopped onions and "
              "fry until golden brown. Stir in the tomatoes, salt and turmeric, "
              "then cover and simmer for ten minutes until the gravy thickens.",
    "football": "United dominated possession in the second half but could not "
                "convert any of their six corners. The striker was substituted "
                "in the 70th minute after picking up a hamstring injury during "
                "training earlier in the week.",
    "python_docs": "A list comprehension consists of brackets containing an "
                   "expression followed by a for clause, then zero or more for "
                   "or if clauses. The result is a new list resulting from "
                   "evaluating the expression in the context of the for and if "
                   "clauses.",
    "weather": "A low pressure system over the Bay of Bengal is expected to "
               "bring heavy rainfall to coastal districts over the next 48 "
               "hours. Fishermen have been advised not to venture into the sea.",
    "lorem": "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do "
             "eiusmod tempor incididunt ut labore et dolore magna aliqua.",
    "clinical_note": "The patient underwent an anterior cervical discectomy and "
                     "fusion at C5-C6. Estimated blood loss was minimal. The "
                     "patient tolerated the procedure well and was taken to the "
                     "recovery room in stable condition.",
    "empty_ish": "ok",
    "gibberish": "qwrtp zxlm bnvc hjkl asdf ghjk uiop vbnm tyre wsxd",
}

CONTROL = {
    "legal": "The prosecution established that the accused was punishable under "
             "section 302 of the IPC. The bail application was rejected and the "
             "accused remains in judicial custody pending trial.",
}

# Floors swept for question 3. 1.0 is the lowest that means anything, since a
# single marker can score around there.
FLOORS = [1.0, 2.0, 3.0, 5.0, 8.0, 12.0]


def total_evidence(text: str) -> float:
    """Summed marker weight across all labels — how much domain vocabulary is here."""
    return sum(d["score"] for d in ontology_check.evidence(text).values())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-model", action="store_true",
                    help="skip inference; measure the markers and the floor only")
    ap.add_argument("--corpus", default=None,
                    help="labelled corpus to measure the floor's cost against "
                         "(default: the active domain pack's training corpus)")
    args = ap.parse_args()

    domain = active_domain()
    ontology_check.load()
    if not ontology_check.is_loaded():
        sys.exit(f"No markers for domain '{domain.key}' — run "
                 f"graph/mine_concept_markers.py --domain {domain.key} first.")

    labels = list(ontology_check._patterns)
    control_text = CONTROL.get(domain.key)
    out = {"domain": domain.key, "samples": {}, "floor_sweep": []}

    # --- 1 and 2: model and ontology check on off-domain text ---------------
    ms = None
    if not args.no_model:
        from services.model_service import ModelService  # torch import, so lazy
        ms = ModelService()
        ms.load()
        out["model_path"] = ms.loaded_path
        print(f"model: {ms.loaded_path}\n")

    header = f"{'sample':<15}{'predicted label':<28}{'conf':>7}  {'evid':>6}  {'ont':<6}outcome"
    print(header)
    print("-" * len(header))

    auto_accepted = flagged = 0
    confidences = []
    for name, text in SAMPLES.items():
        ev = total_evidence(text)
        row = {"total_evidence": round(ev, 4)}

        if ms is None:
            print(f"{name:<15}{'-':<28}{'-':>7}  {ev:>6.2f}  "
                  f"{'-':<6}(model skipped)")
            out["samples"][name] = row
            continue

        p = ms.predict(text)
        conflict = ontology_check.check(text, p["predicted_label"])
        # AUTO_ACCEPT is the routing decision that sends nothing to a human, so
        # "silently filed" means above the bar AND unflagged. That pairing is
        # the thing worth counting: either signal alone would have caught it.
        silent = p["confidence"] > settings.CONFIDENCE_HIGH and not conflict
        auto_accepted += silent
        flagged += bool(conflict)
        confidences.append(p["confidence"])

        row.update(predicted_label=p["predicted_label"],
                   confidence=round(p["confidence"], 4),
                   ontology_flagged=bool(conflict),
                   silently_auto_accepted=bool(silent))
        out["samples"][name] = row
        print(f"{name:<15}{p['predicted_label']:<28}{p['confidence']:>7.3f}  "
              f"{ev:>6.2f}  {'FLAG' if conflict else '-':<6}"
              f"{'filed, no human' if silent else 'routed to a human'}")

    if ms is not None and control_text:
        p = ms.predict(control_text)
        n = sum(1 for l in labels if ontology_check.check(control_text, l))
        out["control"] = {"predicted_label": p["predicted_label"],
                          "confidence": round(p["confidence"], 4),
                          "total_evidence": round(total_evidence(control_text), 4),
                          "flags_for_n_labels": n, "of_labels": len(labels)}
        print(f"\ncontrol (real {domain.key} text): {p['predicted_label']} @ "
              f"{p['confidence']:.3f}, evidence {total_evidence(control_text):.2f}, "
              f"conflicts for {n}/{len(labels)} possible predicted labels")

    if ms is not None:
        n = len(SAMPLES)
        out["summary"] = {
            "off_domain_samples": n,
            "ontology_flagged": flagged,
            "silently_auto_accepted": auto_accepted,
            "confidence_min": round(min(confidences), 4),
            "confidence_max": round(max(confidences), 4),
            "confidence_mean": round(statistics.fmean(confidences), 4),
        }
        print(f"\noff-domain samples      : {n}")
        print(f"  ontology-flagged      : {flagged}/{n}")
        print(f"  silently auto-accepted: {auto_accepted}/{n}")
        print(f"  confidence            : {min(confidences):.3f}-{max(confidences):.3f}, "
              f"mean {statistics.fmean(confidences):.3f}")

    # --- 3: would an evidence floor work? -----------------------------------
    # domain.dataset is repo-root relative ("datasets/v1.csv"); this script runs
    # from backend/, so resolve it rather than trusting the working directory.
    corpus = Path(args.corpus) if args.corpus else (
        (HERE.parent / domain.dataset) if domain.dataset else None
    )
    if corpus is None or not corpus.exists():
        print(f"\nSkipping the floor sweep — no corpus to measure its cost against. "
              f"Pass --corpus <labelled csv>.")
        _write(out, domain.key)
        return 0

    real = [total_evidence(r["text"]) for r in
            csv.DictReader(open(corpus, encoding="utf-8")) if r.get("text", "").strip()]
    real.sort()
    ood = sorted(total_evidence(t) for t in SAMPLES.values())
    pcts = {"median": statistics.median(real), "p1": real[len(real) // 100],
            "p5": real[len(real) // 20], "min": real[0]}
    out["real_text_evidence"] = {"corpus": str(corpus), "n": len(real),
                                 **{k: round(v, 4) for k, v in pcts.items()}}

    print(f"\ntotal marker evidence, real {domain.key} text "
          f"({Path(corpus).name}, n={len(real)}):")
    print(f"  median {pcts['median']:.1f}   p5 {pcts['p5']:.1f}   "
          f"p1 {pcts['p1']:.1f}   min {pcts['min']:.1f}")
    print(f"off-domain: {[round(x, 1) for x in ood]}\n")

    print(f"{'floor':>6}{'real text wrongly flagged':>28}{'off-domain caught':>20}")
    for f in FLOORS:
        fp = sum(1 for x in real if x < f)
        caught = sum(1 for x in ood if x < f)
        out["floor_sweep"].append({"floor": f, "real_flagged": fp,
                                   "real_flagged_pct": round(100 * fp / len(real), 2),
                                   "off_domain_caught": caught,
                                   "off_domain_total": len(ood)})
        print(f"{f:>6.1f}{fp:>20} ({100 * fp / len(real):>4.1f}%)"
              f"{caught:>15}/{len(ood)}")

    print("\nWhy this ships without a floor: the 5th percentile of real text is "
          f"{pcts['p5']:.1f}, so the floor that catches off-domain text also "
          "flags genuine documents in bulk. An abstain class trained on "
          "negatives is the fix; a threshold is not.")

    _write(out, domain.key)
    return 0


def _write(out: dict, key: str) -> None:
    path = HERE / "metrics" / f"{key}_out_of_domain.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {path.relative_to(HERE)}")


if __name__ == "__main__":
    sys.exit(main())
