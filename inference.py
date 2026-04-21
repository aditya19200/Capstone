"""
inference.py — ML inference module for the InLegalBERT v1 classifier.

This is the ONLY file your backend needs to import.
All model loading happens ONCE at import time.

Usage:
    from inference import predict_text, predict_paragraphs, predict_batch, explain_text

Environment:
    MODEL_PATH  — path to the model folder (default: ./models/v1)

Dependencies:
    pip install torch transformers shap numpy
"""

import os
import logging
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

# ── Model Loading (happens once at import) ────────────────────────────────────

MODEL_PATH = os.getenv("MODEL_PATH", "./models/v1")

logger.info(f"[inference] Loading model from {MODEL_PATH}...")
_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model.to(_device)
_model.eval()

# Build label map from model config (single source of truth)
LABEL_MAP = {int(k): v for k, v in _model.config.id2label.items()}
LABEL_NAMES = [LABEL_MAP[i] for i in range(len(LABEL_MAP))]

logger.info(f"[inference] Ready on {_device} | {len(LABEL_MAP)} classes")


# ── Prediction Functions ──────────────────────────────────────────────────────

def predict_text(text: str) -> dict:
    """
    Classify a single piece of legal text.

    Args:
        text: A string of legal text (sentence, paragraph, or short document).

    Returns:
        {
            "label": "Criminal Law",
            "label_id": 1,
            "confidence": 0.751,
            "all_probabilities": {"Contract Law": 0.03, "Criminal Law": 0.751, ...}
        }
    """
    if not text or not text.strip():
        raise ValueError("text must be non-empty")

    inputs = _tokenizer(
        text, return_tensors="pt", truncation=True,
        padding=True, max_length=256
    ).to(_device)

    with torch.no_grad():
        outputs = _model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)[0]
    confidence, pred_id = torch.max(probs, dim=-1)
    pid = pred_id.item()

    return {
        "label": LABEL_MAP[pid],
        "label_id": pid,
        "confidence": round(confidence.item(), 4),
        "all_probabilities": {
            LABEL_MAP[i]: round(p.item(), 4) for i, p in enumerate(probs)
        },
    }


def predict_paragraphs(text: str) -> dict:
    """
    Split text into paragraphs (by double newline) and classify each.
    Returns per-paragraph predictions plus a confidence-weighted document label.

    Args:
        text: Multi-paragraph legal text, paragraphs separated by blank lines.

    Returns:
        {
            "document_label": "Criminal Law",
            "paragraph_count": 3,
            "paragraphs": [
                {"paragraph": "...", "label": "...", "label_id": 1, "confidence": 0.75, ...},
                ...
            ]
        }
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        raise ValueError("No paragraphs found (split by blank lines)")

    results = [{"paragraph": p, **predict_text(p)} for p in paragraphs]

    # Aggregate: confidence-weighted vote
    label_scores = {}
    for r in results:
        label_scores[r["label"]] = label_scores.get(r["label"], 0) + r["confidence"]

    return {
        "document_label": max(label_scores, key=label_scores.get),
        "paragraph_count": len(results),
        "paragraphs": results,
    }


def predict_batch(texts: list) -> list:
    """
    Batch prediction for CSV uploads or bulk annotation.

    Args:
        texts: List of legal text strings.

    Returns:
        List of prediction dicts (same shape as predict_text output).
    """
    return [predict_text(t) for t in texts]


# ── SHAP Explanation ──────────────────────────────────────────────────────────

def _predict_proba(texts) -> np.ndarray:
    """Internal: SHAP-compatible prediction function. Returns (n, 10) array."""
    if isinstance(texts, str):
        texts = [texts]
    elif isinstance(texts, np.ndarray):
        texts = texts.tolist()
    else:
        texts = list(texts)

    inputs = _tokenizer(
        texts, padding=True, truncation=True,
        max_length=256, return_tensors="pt"
    ).to(_device)

    with torch.no_grad():
        outputs = _model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    return probs.cpu().numpy()


def explain_text(text: str) -> dict:
    """
    Generate SHAP token-level importance scores for a prediction.

    WARNING: This is slow (~30-90 seconds per call). Do NOT call in the
    main request path. Use from the async XAI worker only.

    Args:
        text: Legal text to explain.

    Returns:
        {
            "predicted_label": "Criminal Law",
            "predicted_label_id": 1,
            "confidence": 0.751,
            "token_importance": [
                {"token": "IPC", "importance": 0.0761},
                {"token": "charged", "importance": 0.0439},
                ...
            ],
            "top_tokens": [top 10 by absolute importance],
            "summary": "The prediction 'Criminal Law' was primarily driven by: IPC, criminal, charged."
        }
    """
    import shap

    if not text or not text.strip():
        raise ValueError("text must be non-empty")

    # Get prediction first
    pred = predict_text(text)
    pred_label = pred["label"]
    pred_id = pred["label_id"]

    # Build SHAP explainer
    masker = shap.maskers.Text(_tokenizer)
    explainer = shap.Explainer(_predict_proba, masker, output_names=LABEL_NAMES)

    # Compute SHAP values
    shap_values = explainer([text])

    tokens = shap_values.data[0]
    importance_scores = shap_values.values[0][:, pred_id]

    # Build token-importance list
    token_importance = [
        {"token": str(tok), "importance": round(float(score), 6)}
        for tok, score in zip(tokens, importance_scores)
    ]

    # Top 10 tokens by absolute importance
    top_tokens = sorted(
        token_importance, key=lambda x: abs(x["importance"]), reverse=True
    )[:10]

    # Natural-language summary
    positive = [t["token"].strip() for t in top_tokens if t["importance"] > 0][:3]
    if positive:
        summary = f"The prediction '{pred_label}' was primarily driven by: {', '.join(positive)}."
    else:
        summary = f"Prediction: '{pred_label}'. No strongly positive tokens found."

    return {
        "predicted_label": pred_label,
        "predicted_label_id": pred_id,
        "confidence": pred["confidence"],
        "token_importance": token_importance,
        "top_tokens": top_tokens,
        "shap_values_raw": importance_scores.tolist(),
        "summary": summary,
    }


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Test 1: Single text prediction ===")
    r = predict_text("The accused was charged under Section 420 IPC for criminal breach of trust.")
    print(f"  Label: {r['label']}  |  Confidence: {r['confidence']}")

    print("\n=== Test 2: Multi-paragraph prediction ===")
    r = predict_paragraphs(
        "The lease deed was executed for 11 months at Rs 50000.\n\n"
        "The accused was charged under Section 420 IPC.\n\n"
        "The petitioner seeks dissolution of marriage."
    )
    print(f"  Document label: {r['document_label']}")
    for p in r["paragraphs"]:
        print(f"    -> {p['label']} (conf {p['confidence']})")

    print("\n=== Test 3: SHAP explanation (will take 30-90 sec) ===")
    r = explain_text("The NCLT admitted the insolvency petition under Section 7 of the IBC.")
    print(f"  Label: {r['predicted_label']}  |  Confidence: {r['confidence']}")
    print(f"  Summary: {r['summary']}")
    print(f"  Top tokens:")
    for t in r["top_tokens"][:5]:
        print(f"    {t['token']:20s}  {t['importance']:+.4f}")

    print("\n=== ALL TESTS COMPLETE ===")
