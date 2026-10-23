"""
medical_progression.py — the medical model across two rounds of feedback.

What the admin table shows for medical: the model as deployed, after one
round of validated annotations, and after two. Three checkpoints, scored on
one exam, so "it went up" is a comparison rather than three unrelated numbers.

The design was fixed before anything ran, and this is the only run:

  data     datasets/medical_v1.csv (1,472 rows), split per specialty with
           the project seed (42) into 60% / 20% / 20%
  v0       Bio_ClinicalBERT fine-tuned on the 60%          "Medical Baseline"
  v1       v0 fine-tuned on round 1 (the first 20%)        one round of feedback
  v2       v1 fine-tuned on rounds 1 + 2 (all validated)   two rounds
  exam     sample-data/evaluation/medical_heldout.csv (273 rows), which no
           stage trains on

Each retrain starts from the previous version and trains on every validated
annotation so far, which is what workers/retrain_worker.py does. Same
epochs, batch size and learning rate as the worker, and the same split and
fine-tune code as retrain_experiment.py (the five-seed +3.4pp result), so
the two are measurements of one procedure.

This is not the deployed MedicalModelShared, which trained on all 1,472 rows
at once and so has nothing left to learn from. Holding 40% back is what
makes "it improved with feedback" measurable at all.

Checkpoints go to models/medical-progression/ (gitignored, ~420 MB each).
Numbers go to metrics/medical_progression.json.

    ./venv/bin/python medical_progression.py

Result, 23 Oct: v0 89.4%, v1 90.8%, v2 88.6%, with confidently-wrong answers
(wrong at 0.85 or above, so auto-accepted unseen) going 3 -> 5 -> 16.

--from-baseline, added after that run and fixed before running it
-----------------------------------------------------------------
v2 trained on round 1 a second time: v1 had already learned it, so round 1
got six epochs in all, and mean confidence climbed 0.84 -> 0.88 -> 0.92 while
the confident errors tripled. The proposed fix is that every retrain starts
from the baseline, so each annotation is learned exactly once:

  v2b      v0 fine-tuned on rounds 1 + 2 together (591 rows), same
           hyperparameters, same exam

One run, at the same seed, reported beside v2 rather than in its place, in
metrics/medical_progression_from_baseline.json. Needs v0's checkpoint from
the run above.

    ./venv/bin/python medical_progression.py --from-baseline

Result: 90.8%, tied with v1, 13 confidently wrong.
"""

import json
import pathlib
import sys
from datetime import datetime, timezone

import torch

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from config.domains import get_domain  # noqa: E402
from retrain_experiment import (  # noqa: E402
    BATCH_SIZE, EPOCHS, LEARNING_RATE,
    confidence_report, fine_tune, load_model, predict_all, stratified_split,
)
from train_domain import SEED, evaluate, load_csv, pin_randomness  # noqa: E402

OUT_DIR = HERE / "models" / "medical-progression"
RESULTS = HERE / "metrics" / "medical_progression.json"
RESULTS_FROM_BASELINE = HERE / "metrics" / "medical_progression_from_baseline.json"


def score(model, tok, rows, domain, device):
    """Accuracy and per-class F1 on the exam, plus the calibration report."""
    accuracy, _support, _hit, _confusion, f1 = evaluate(
        model, tok, rows, domain.label2id, domain.id2label, device, domain.max_length,
    )
    preds = predict_all(model, tok, rows, domain.label2id, domain.id2label,
                        domain.max_length, device)
    return {
        "accuracy": round(accuracy, 4),
        "f1_per_class": f1,
        "calibration": confidence_report(preds),
    }


def main():
    from_baseline = "--from-baseline" in sys.argv[1:]
    domain = get_domain("medical")
    known = set(domain.names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_randomness(SEED)

    rows, skipped = load_csv(ROOT / domain.dataset, known)
    exam, _ = load_csv(ROOT / "sample-data/evaluation/medical_heldout.csv", known)
    if not exam:
        sys.exit("no held-out exam — every number would be meaningless")

    # 60 / 40, then the 40 into two 20s. Stratified both times, so every
    # specialty is represented in every round rather than the rare ones
    # vanishing from the small slices.
    base, rest = stratified_split(rows, 0.4, SEED)
    round1, round2 = stratified_split(rest, 0.5, SEED)

    print(f"corpus {len(rows)} ({skipped} skipped)  ->  base {len(base)}  "
          f"round1 {len(round1)}  round2 {len(round2)}  |  exam {len(exam)}  |  {device}\n")

    if from_baseline:
        if not (OUT_DIR / "v0_baseline").exists():
            sys.exit("v0's checkpoint is missing: run without --from-baseline first")
        stages = [
            ("v2_from_baseline", "round 2, from the baseline", "v0_baseline",
             round1 + round2, "v0 retrained on rounds 1 and 2 together, each learned once"),
        ]
        results_file = RESULTS_FROM_BASELINE
    else:
        stages = [
            ("v0_baseline", "Medical Baseline", None, base,
             "Bio_ClinicalBERT fine-tuned on 60% of the training notes"),
            ("v1_round1", "round 1", "v0_baseline", round1,
             "v0 retrained on round 1 of validated annotations"),
            ("v2_round2", "round 2", "v1_round1", round1 + round2,
             "v1 retrained on rounds 1 and 2 of validated annotations"),
        ]
        results_file = RESULTS

    results = {
        "design": {
            "seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LEARNING_RATE,
            "split": {"base": len(base), "round1": len(round1), "round2": len(round2)},
            "exam": {"file": "sample-data/evaluation/medical_heldout.csv", "rows": len(exam)},
            "base_model": domain.base_model,
            "retrain_from": "the baseline" if from_baseline else "the previous version",
        },
        "versions": [],
    }

    for key, name, parent, train_rows, what in stages:
        start = domain.base_model if parent is None else OUT_DIR / parent
        print(f"[{key}] {what}: {len(train_rows)} rows", flush=True)
        model, tok = load_model(start, domain, device)
        fine_tune(model, tok, train_rows, domain.label2id, domain.max_length, device,
                  EPOCHS, BATCH_SIZE, LEARNING_RATE, tag=f"{key} ", seed=SEED)
        out = OUT_DIR / key
        out.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out)
        tok.save_pretrained(out)

        s = score(model, tok, exam, domain, device)
        cal = s["calibration"]
        print(f"   exam accuracy {s['accuracy']:.4f}  mean conf {cal['mean_confidence']}  "
              f"ECE {cal['ece']}  wrong above 0.85: {cal['above_auto_accept_wrong']}\n")
        results["versions"].append({
            "key": key, "name": name, "trained_on_rows": len(train_rows),
            "description": what,
            "model_path": f"./models/medical-progression/{key}",
            **s,
        })
        del model
        torch.cuda.empty_cache()

    results["measured_at"] = datetime.now(timezone.utc).isoformat()
    results_file.parent.mkdir(exist_ok=True)
    results_file.write_text(json.dumps(results, indent=1))

    print("version            exam accuracy")
    for v in results["versions"]:
        print(f"  {v['key']:<16} {v['accuracy']:.4f}")
    print(f"\nwrote {results_file.relative_to(HERE)}")


if __name__ == "__main__":
    main()
