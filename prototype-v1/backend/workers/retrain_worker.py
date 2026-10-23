"""
workers/retrain_worker.py — Model retraining worker process.

Runs as a separate process (not inside FastAPI), same pattern as
workers/xai_worker.py. Polls retrain_jobs for a pending job. When one shows
up: pulls validated annotations, builds a new training set normalized
identically to inference (utils.text.normalize — the exact function every
prediction path already uses, so train/serve text never drifts), fine-tunes
from the currently active model checkpoint, scores it on the exam its
domain's baseline was scored on (see load_exam), and records the result as a
new model_versions row.

New model versions are NEVER activated automatically — is_active is always
False (enforced inside supabase_service.create_model_version). A human
reviews the metrics and flips it on via the admin activate endpoint. This
worker's only guard is refusing to silently hide a large accuracy
regression: if the new model scores more than ACCURACY_DROP_GUARD below the
model it was retrained from, on the same exam rows, it is still saved
(nothing is lost) but flagged with a warning in its metrics.json sidecar
file, for a human to see.

Run with:
    python -m workers.retrain_worker
"""

import csv
import json
import logging
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config.settings import settings
from services.model_service import ACTIVE_DOMAIN, ID2LABEL, LABEL2ID, NUM_LABELS
from services.supabase_service import (
    create_dataset_version,
    create_model_version,
    get_active_model_version,
    get_latest_retrain_job,
    get_prediction,
    list_annotations,
    update_retrain_job,
)
from utils.text import normalize

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [retrain_worker] %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

POLL_INTERVAL_SECONDS = 30      # retraining is rare and slow; poll infrequently
MIN_ANNOTATIONS_DEFAULT = 50    # fallback if the job row doesn't carry
                                 # min_annotations — the real Postgres schema
                                 # doesn't have that column yet, only mock_db does
EPOCHS = 3
TEST_SPLIT = 0.2                # only when the domain has no exam; see load_exam
ACCURACY_DROP_GUARD = 0.05      # new model >5 points below its parent -> flag, don't hide
BATCH_SIZE = 8
EVAL_BATCH_SIZE = 32            # what evaluate_baseline.py scored the baseline with
LEARNING_RATE = 2e-5
RANDOM_SEED = 42

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
METRICS_DIR = Path(__file__).resolve().parent.parent / "metrics"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Job discovery
# ---------------------------------------------------------------------------

def get_pending_retrain_job() -> Optional[Dict]:
    """
    Return the pending retrain job, if any.

    No dedicated get_pending_retrain_jobs() exists yet (unlike
    get_pending_xai_jobs). POST /retrain's own concurrency guard means at
    most one job is ever pending or running at a time, so the latest job IS
    the pending one whenever one exists.
    """
    job = get_latest_retrain_job()
    if job and job["status"] == "pending":
        return job
    return None


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------

def build_training_data() -> Tuple[List[str], List[str]]:
    """
    Pull every validated annotation and its source text, normalized with the
    exact same function every prediction path uses. Returns (texts, labels)
    with labels as label *names* (matching ID2LABEL values).
    """
    annotations = list_annotations(status="validated")
    texts: List[str] = []
    labels: List[str] = []
    skipped = 0

    for ann in annotations:
        pred = get_prediction(ann["prediction_id"])
        if pred is None:
            skipped += 1
            continue

        label = ann["validated_label"]
        if label not in LABEL2ID:
            logger.warning(
                "build_training_data: annotation %s has unknown label '%s', skipping",
                ann["id"], label,
            )
            skipped += 1
            continue

        texts.append(normalize(pred["text_content"]))
        labels.append(label)

    if skipped:
        logger.warning("build_training_data: skipped %d unusable annotation(s)", skipped)
    return texts, labels


def _train_test_split(
    texts: List[str], labels: List[str], test_frac: float, seed: int
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Deterministic shuffle-and-slice split."""
    indices = list(range(len(texts)))
    random.Random(seed).shuffle(indices)
    n_test = max(1, int(len(indices) * test_frac))
    test_idx = set(indices[:n_test])

    train_texts, train_labels, test_texts, test_labels = [], [], [], []
    for i in indices:
        if i in test_idx:
            test_texts.append(texts[i])
            test_labels.append(labels[i])
        else:
            train_texts.append(texts[i])
            train_labels.append(labels[i])
    return train_texts, train_labels, test_texts, test_labels


# ---------------------------------------------------------------------------
# The exam
# ---------------------------------------------------------------------------

def load_exam(exclude: set) -> Optional[Tuple[str, List[str], List[str], int]]:
    """
    The labelled set this domain's baseline was scored on, minus every text
    in `exclude` (the rows this retrain trains on).

    New versions used to be scored on 20% of the validated annotations, and
    the admin table put that number beside a baseline scored on all 1,701
    rows of datasets/v1.csv. On 23 Oct that read 93.3% against 70.7% — and
    the 93.3% was 14 of 15 annotation rows. Two different exams, shown as if
    they were one. Every version of a domain is now scored on the same one.

    Which file is read from the baseline's own metrics, so the two cannot
    drift apart: medical names a held-out file, legal the dataset it was
    scored on. Rows the new version trained on are removed (the demo's intake
    file shares 35 rows with v1.csv, and any a reviewer validates become
    training rows), and the parent is re-scored on what is left — see
    process_retrain_job — so the comparison stays on identical rows.

    Returns (exam_path, texts, labels, excluded), or None when the domain has
    no exam on disk, in which case the caller falls back to TEST_SPLIT.
    """
    metrics_file = METRICS_DIR / f"{ACTIVE_DOMAIN.key}_baseline_metrics.json"
    if not metrics_file.exists():
        return None
    baseline = json.loads(metrics_file.read_text())
    rel = baseline.get("heldout") or baseline.get("dataset")
    if not rel or not (REPO_ROOT / rel).exists():
        return None

    texts: List[str] = []
    labels: List[str] = []
    excluded = 0
    with open(REPO_ROOT / rel, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = (row.get("label_name") or "").strip()
            text = normalize((row.get("text") or "").strip())
            if not text or label not in LABEL2ID:
                continue
            if text in exclude:
                excluded += 1
                continue
            texts.append(text)
            labels.append(label)
    return rel, texts, labels, excluded


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------

def _load(model_path: str):
    """Load a checkpoint and its tokenizer onto the GPU if there is one."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=NUM_LABELS
    )
    model.to(device)
    return model, tokenizer


def _fine_tune(
    model,
    tokenizer,
    train_texts: List[str],
    train_labels: List[str],
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
):
    """
    Fine-tune `model` in place on (train_texts, train_labels).
    Returns the fine-tuned (model, tokenizer), still resident in memory.
    """
    device = next(model.parameters()).device
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr)
    label_ids = [LABEL2ID[label] for label in train_labels]
    n = len(train_texts)

    logger.info("Fine-tuning on %d examples for %d epoch(s) on %s", n, epochs, device)

    for epoch in range(epochs):
        order = list(range(n))
        random.Random(RANDOM_SEED + epoch).shuffle(order)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            batch_idx = order[start:start + batch_size]
            batch_texts = [train_texts[i] for i in batch_idx]
            batch_labels = torch.tensor([label_ids[i] for i in batch_idx], device=device)

            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=settings.MAX_LENGTH,
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            optimizer.zero_grad()
            outputs = model(**inputs, labels=batch_labels)
            outputs.loss.backward()
            optimizer.step()

            total_loss += outputs.loss.item()
            n_batches += 1

        logger.info(
            "  epoch %d/%d — mean loss %.4f",
            epoch + 1, epochs, total_loss / max(n_batches, 1),
        )

    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate(
    model, tokenizer, texts: List[str], labels: List[str]
) -> Tuple[float, Dict[str, float]]:
    """
    Score on the exam. Returns (accuracy, f1_per_class).

    Batched like evaluate_baseline.py, because the exam is now 1,701 rows
    rather than 15 and is scored twice per retrain (parent and new version).

    No sklearn dependency (not in backend/requirements.txt) — f1 is computed
    directly from per-class TP/FP/FN counts.
    """
    device = next(model.parameters()).device
    correct = 0
    tp = {label: 0 for label in ID2LABEL.values()}
    fp = {label: 0 for label in ID2LABEL.values()}
    fn = {label: 0 for label in ID2LABEL.values()}

    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), EVAL_BATCH_SIZE):
            inputs = tokenizer(
                texts[start:start + EVAL_BATCH_SIZE],
                return_tensors="pt",
                truncation=True,
                max_length=settings.MAX_LENGTH,
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            pred_ids = model(**inputs).logits.argmax(dim=-1).tolist()

            for pred_id, true_label in zip(pred_ids, labels[start:start + EVAL_BATCH_SIZE]):
                pred_label = ID2LABEL[int(pred_id)]
                if pred_label == true_label:
                    correct += 1
                    tp[pred_label] += 1
                else:
                    fp[pred_label] += 1
                    fn[true_label] += 1

    accuracy = correct / len(texts) if texts else 0.0

    f1_per_class: Dict[str, float] = {}
    for label in ID2LABEL.values():
        precision = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) else 0.0
        recall = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1_per_class[label] = round(f1, 4)

    return round(accuracy, 4), f1_per_class


def _label_distribution(labels: List[str]) -> Dict[str, int]:
    dist: Dict[str, int] = {}
    for label in labels:
        dist[label] = dist.get(label, 0) + 1
    return dist


# ---------------------------------------------------------------------------
# Job processing
# ---------------------------------------------------------------------------

def process_retrain_job(job: Dict) -> None:
    """Process a single retrain job end-to-end."""
    job_id = job["id"]
    min_annotations = job.get("min_annotations") or MIN_ANNOTATIONS_DEFAULT

    logger.info("Processing retrain job %s (min_annotations=%d)", job_id, min_annotations)
    update_retrain_job(job_id=job_id, status="running")

    try:
        texts, labels = build_training_data()
        if len(texts) < min_annotations:
            raise ValueError(
                f"Only {len(texts)} validated annotations available, "
                f"need at least {min_annotations}."
            )

        exam = load_exam(set(texts))
        if exam is not None:
            # Every validated annotation trains: none is needed to score with.
            exam_name, test_texts, test_labels, excluded = exam
            train_texts, train_labels = texts, labels
        else:
            logger.warning(
                "No exam for domain %r (its baseline metrics name none on "
                "disk); scoring on %d%% of the annotations, which the "
                "baseline was not scored on.", ACTIVE_DOMAIN.key, TEST_SPLIT * 100,
            )
            train_texts, train_labels, test_texts, test_labels = _train_test_split(
                texts, labels, TEST_SPLIT, RANDOM_SEED
            )
            exam_name, excluded = f"{TEST_SPLIT:.0%} of the validated annotations", 0

        active = get_active_model_version()
        base_model_path = active["file_path"] if active else settings.MODEL_PATH
        parent_version = active["version_number"] if active else Path(base_model_path).name
        model, tokenizer = _load(base_model_path)

        # The parent is scored here, on these rows, rather than read from its
        # model_versions row: its stored accuracy may be on rows this retrain
        # has just removed from the exam for having trained on them.
        parent_accuracy, _ = _evaluate(model, tokenizer, test_texts, test_labels)
        logger.info(
            "Exam: %s, %d rows (%d excluded as training rows). %s scores %.4f.",
            exam_name, len(test_texts), excluded, parent_version, parent_accuracy,
        )

        _fine_tune(model, tokenizer, train_texts, train_labels)
        accuracy, f1_per_class = _evaluate(model, tokenizer, test_texts, test_labels)

        version_id = f"v-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        output_dir = MODELS_DIR / version_id
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        warning = None
        if accuracy < parent_accuracy - ACCURACY_DROP_GUARD:
            warning = (
                f"New model accuracy {accuracy:.4f} is more than "
                f"{ACCURACY_DROP_GUARD:.2f} below {parent_version}'s "
                f"{parent_accuracy:.4f} on the same exam. Saved with "
                f"is_active=False — do not activate without reviewing why."
            )
            logger.warning(warning)

        # routers/admin.py reads exam_rows / parent_version / parent_accuracy
        # from this file to print the like-for-like comparison in the table.
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        (METRICS_DIR / f"{version_id}_metrics.json").write_text(json.dumps({
            "version_id": version_id,
            "trained_at": _now_iso(),
            "base_model_path": base_model_path,
            "train_count": len(train_texts),
            "exam": exam_name,
            "exam_rows": len(test_texts),
            "excluded_from_exam": excluded,
            "parent_version": parent_version,
            "parent_accuracy": parent_accuracy,
            "accuracy": accuracy,
            "f1_per_class": f1_per_class,
            "warning": warning,
        }, indent=2))

        dv = create_dataset_version(
            version_id=version_id,
            sample_count=len(texts),
            label_distribution=_label_distribution(labels),
        )

        mv = create_model_version(
            version_number=version_id,
            accuracy=accuracy,
            f1_per_class=f1_per_class,
            file_path=str(output_dir),
            dataset_version_id=dv["id"],
        )

        update_retrain_job(
            job_id=job_id,
            status="complete",
            model_version_id=mv["id"],
            completed_at=_now_iso(),
        )
        logger.info(
            "Retrain job %s complete -> model_version %s (accuracy=%.4f against "
            "%s's %.4f on the same %d rows, is_active=False)",
            job_id, mv["id"], accuracy, parent_version, parent_accuracy, len(test_texts),
        )

    except Exception:
        logger.exception("Retrain job %s failed", job_id)
        update_retrain_job(job_id=job_id, status="failed", completed_at=_now_iso())


# ---------------------------------------------------------------------------
# Main poll loop
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("retrain_worker started. Polling every %ds.", POLL_INTERVAL_SECONDS)
    while True:
        job = get_pending_retrain_job()
        if job:
            process_retrain_job(job)
        else:
            time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
