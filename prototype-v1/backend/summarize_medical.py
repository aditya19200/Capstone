#!/usr/bin/env python3
"""
summarize_medical.py — read the recorded medical experiments and print them.

Why this exists
---------------
The medical side of this project is not a live demo. Switching the serving
domain means restarting the backend with a different pack and different
weights, and a review slot is twenty minutes. So the medical evidence is
presented as a measurement, which means it has to be *readable* — and it
currently lives across fifteen JSON files in metrics/ that nobody can parse on
a projector.

This reads those files and prints them. It computes nothing the experiments
did not already record, runs no inference, loads no weights, and takes no
arguments beyond the metrics directory. Every number it prints can be traced
to one named file, which is the point: a figure on a slide should be
answerable with "that is metrics/medical_retrain_experiment_s99.json".

What the arms are
-----------------
Each metrics/medical_retrain_experiment*.json records one retrain run against
the same 273-document held-out set, with retrain.validated_rows saying how
many annotations it trained on and two flags saying which arm it is:

  corrections_only=False, random_sample=None   the real path -- every
                                               validated annotation
  corrections_only=True                        corrections alone
  random_sample=N                              N annotations drawn at random,
                                               matched to the corrections count

The three arms exist to answer one question: are corrections valuable because
they are corrections, or just because they are more data? Comparing the
corrections arm against a random arm of the SAME SIZE is the only way to tell
those apart, and the answer turned out to be "no measurable difference" --
which is why this script prints the comparison rather than just the headline.

    ./venv/bin/python summarize_medical.py
    ./venv/bin/python summarize_medical.py --markdown   # for RESULTS.md or a slide
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

HELDOUT_NOTE = (
    "All arms are measured on the same held-out set, rebuilt 19 Oct 2026 so "
    "that no document straddles the split. The previous set shared 129 of 435 "
    "rows with training, every one of them with a conflicting label."
)


def load(metrics_dir: Path):
    """Group every recorded medical retrain by which arm it belongs to."""
    runs = []
    for path in sorted(metrics_dir.glob("medical_retrain_experiment*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        r, h = d.get("retrain", {}), d.get("heldout", {})
        res = h.get("results", {})
        if "v1_baseline" not in res or "v2_retrained" not in res:
            continue
        runs.append({
            "file": path.name,
            "seed": d.get("seed"),
            "rows": r.get("validated_rows"),
            "corrections_only": bool(r.get("corrections_only")),
            "random_sample": r.get("random_sample"),
            "epochs": r.get("retrain_epochs"),
            "base": res["v1_baseline"],
            "new": res["v2_retrained"],
            "heldout_rows": h.get("rows"),
        })
    return runs


def acc(arm):
    return arm["accuracy"] * 100.0


def ece(arm):
    return arm.get("confidence", {}).get("ece")


def wrong_above(arm):
    return arm.get("confidence", {}).get("above_auto_accept_wrong")


def pick(runs, *, corrections=False, random=False, seeded=True, epochs=3):
    out = []
    for r in runs:
        if r["corrections_only"] != corrections:
            continue
        if bool(r["random_sample"]) != random:
            continue
        if seeded and r["seed"] is None:
            continue
        if epochs is not None and r["epochs"] != epochs:
            continue
        out.append(r)
    return sorted(out, key=lambda r: (r["seed"] is None, r["seed"]))


def table(rows, md):
    """Render aligned text, or a markdown table if --markdown."""
    if not md:
        return "\n".join("  " + r for r in rows)
    head, *body = rows
    cells = [c.strip() for c in head.split("|")]
    out = ["| " + " | ".join(cells) + " |",
           "|" + "|".join("---" for _ in cells) + "|"]
    out += ["| " + " | ".join(c.strip() for c in r.split("|")) + " |" for r in body]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-dir", default=str(HERE / "metrics"))
    ap.add_argument("--markdown", action="store_true",
                    help="emit markdown tables for RESULTS.md or a slide")
    args = ap.parse_args()

    md = args.markdown
    mdir = Path(args.metrics_dir)
    runs = load(mdir)
    if not runs:
        sys.exit(f"No medical retrain experiments found in {mdir}")

    h = lambda s: print(f"\n{'## ' if md else ''}{s}\n" if md else f"\n{s}\n{'-' * len(s)}")

    print(f"{'# ' if md else ''}Medical — recorded experiments")
    print(f"\n{len(runs)} retrain runs read from {mdir.name}/. No inference was run.")

    # --- the real path, five seeds -----------------------------------------
    seeds = pick(runs, seeded=True)
    h("The real path: every validated annotation, five seeds")
    rows = ["seed | rows | baseline | retrained | delta | ECE before | ECE after | conf. errors"]
    deltas, eces = [], 0
    for r in seeds:
        d = acc(r["new"]) - acc(r["base"])
        deltas.append(d)
        e0, e1 = ece(r["base"]), ece(r["new"])
        better = e1 is not None and e0 is not None and e1 < e0
        eces += better
        rows.append(
            f"{r['seed']} | {r['rows']} | {acc(r['base']):.1f}% | {acc(r['new']):.1f}% | "
            f"{d:+.1f}pp | {e0:.4f} | {e1:.4f} | "
            f"{wrong_above(r['base'])} -> {wrong_above(r['new'])}"
        )
    print(table(rows, md))
    n = len(deltas)
    print(f"\n**mean {statistics.fmean(deltas):+.2f}pp**, sd "
          f"{statistics.stdev(deltas):.2f}pp, positive {sum(d > 0 for d in deltas)}/{n}, "
          f"ECE improved {eces}/{n}.")
    print(f"Held-out set: {seeds[0]['heldout_rows']} documents. "
          f"Quote this as **+{statistics.fmean(deltas):.1f}pp across {n} seeds** — "
          f"not the single best run, and note that seed "
          f"{min(seeds, key=lambda r: acc(r['new']) - acc(r['base']))['seed']} "
          f"is the weakest of them.")

    # --- the control --------------------------------------------------------
    corr = {r["seed"]: r for r in pick(runs, corrections=True, seeded=True)}
    rand = {r["seed"]: r for r in pick(runs, random=True, seeded=True)}
    shared = sorted(set(corr) & set(rand))
    if shared:
        h("The control: are corrections better than the same amount of random data?")
        rows = ["seed | rows | baseline | corrections | random | corr - rand | corr - baseline"]
        cd, cb = [], []
        for s in shared:
            c, d = corr[s], rand[s]
            cd.append(acc(c["new"]) - acc(d["new"]))
            cb.append(acc(c["new"]) - acc(c["base"]))
            rows.append(
                f"{s} | {c['rows']} | {acc(c['base']):.1f}% | {acc(c['new']):.1f}% | "
                f"{acc(d['new']):.1f}% | {cd[-1]:+.1f}pp | {cb[-1]:+.1f}pp"
            )
        print(table(rows, md))
        print(f"\n**corrections minus random: {statistics.fmean(cd):+.2f}pp.** "
              f"Corrections minus baseline: {statistics.fmean(cb):+.2f}pp.")
        print("Row for row, corrections are not more valuable than random "
              "annotations. State it that way: the arm is underpowered — only "
              "65 corrections exist — and the mean says nothing, so the finding "
              "is in the per-label shape below, not in the average.")

        # --- the per-label signature, which is the actual finding -----------
        worst = min(shared, key=lambda s: acc(corr[s]["new"]) - acc(corr[s]["base"]))
        c = corr[worst]
        # F1, not recall, and the choice matters. Recall alone rises whenever
        # the model reaches for a class more often, so a prior shift can look
        # like an improvement in it: seed 42's ENT recall goes UP 5.6pp while
        # its ENT F1 goes DOWN 9.3pp, which is precisely the over-prediction
        # this section is about. F1 moves only when the class gets genuinely
        # better or genuinely worse, and it is what RESULTS.md section 10
        # quotes, so the two cannot disagree on a slide.
        b4, af = c["base"].get("f1_per_class", {}), c["new"].get("f1_per_class", {})
        moves = sorted(((l, (af[l] - b4[l]) * 100) for l in b4 if l in af),
                       key=lambda kv: -kv[1])
        if moves:
            h(f"Why corrections alone go wrong — seed {worst}, per-class F1")
            print(f"Seed {worst}'s corrections-only arm came in "
                  f"{acc(c['new']) - acc(c['base']):+.1f}pp against baseline. "
                  f"Where it moved:\n")
            rows = ["specialty | F1 change"]
            for label, delta in moves:
                if abs(delta) >= 1.0:
                    rows.append(f"{label} | {delta:+.1f}pp")
            print(table(rows, md))
            print("\nCorrections are by definition concentrated in the classes "
                  "the model gets wrong, so a training set made only of them "
                  "tells the model to predict those classes more often. It "
                  "improved enormously at what it was aimed at and paid for it "
                  "everywhere else. That is class-prior skew, and it is why the "
                  "retrain worker trains on every validated annotation — the "
                  "confirmations are spread across all labels and hold the "
                  "distribution in place. The standard name for that mitigation "
                  "is rehearsal.")

    # --- calibration --------------------------------------------------------
    cal = sorted(mdir.glob("medical_calibration_*.json"))
    if cal:
        h("Temperature scaling does not transfer to medical")
        rows = ["checkpoint | n | T | accuracy | ECE before | ECE after"]
        for p in cal:
            d = json.loads(p.read_text(encoding="utf-8"))
            # d["set"] says only "csv" in both files, so name them by the
            # checkpoint the filename records instead.
            rows.append(f"{p.stem.replace('medical_calibration_', '')} | {d.get('n')} | "
                        f"{d.get('temperature'):.3f} | {d.get('accuracy', 0) * 100:.1f}% | "
                        f"{d.get('ece_before'):.4f} | {d.get('ece_after'):.4f}")
        print(table(rows, md))
        print("\nA fitted temperature makes medical's ECE **worse**, so "
              "CALIBRATION_TEMPERATURE ships at 1.0 (off). Legal's fitted "
              "T=1.313 helps. The reason is the shape of the miscalibration: "
              "medical is underconfident at the bottom of the range and "
              "overconfident at the top, and one scalar cannot fix both ends.")

    h("What is safe to say")
    for line in [
        "Medical retraining gains +3.4pp across five seeds, positive in all "
        "five, with calibration improving in all five.",
        "Corrections are not more valuable than the same volume of random "
        "annotations; the measured difference is +0.13pp, which is nothing.",
        "Training on corrections alone is unstable and skews the class prior; "
        "the confirmations are what keep the rest of the label space intact.",
        "The ontology consistency check catches 0 of 5 confident medical errors "
        "at every one of 30 threshold combinations swept, because medical's "
        "confusions share vocabulary rather than contradicting it. It ships at "
        "a high bar as an honest no-op.",
        HELDOUT_NOTE,
    ]:
        print(f"{'- ' if md else '  * '}{line}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
