"""
services/shap_service.py — SHAP token-level explainability for InLegalBERT.

Uses shap.Explainer with a text-masking pipeline approach to compute
per-token importance scores for the predicted class.

Design decisions:
  - SHAP is CPU-bound and slow; always called from the XAI worker (separate
    process) or via an async executor — never inline in a FastAPI request.
  - We use shap.Explainer with a callable model wrapper rather than
    shap.DeepExplainer, because DeepExplainer requires matching internal
    layer shapes that vary across BERT variants. The callable approach is
    model-agnostic and works reliably with any HuggingFace classifier.
  - max_evals is capped at 200 to keep latency acceptable on CPU; raise it
    in .env if you have GPU workers.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import shap
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from services.model_service import model_service

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal model wrapper for SHAP
# ---------------------------------------------------------------------------

class _BertPredictWrapper:
    """
    Callable wrapper that SHAP uses as its 'model function'.

    SHAP passes masked/perturbed strings to this callable and expects back
    a 2-D numpy array of shape (n_samples, n_classes) containing class
    probabilities for each perturbed input.
    """

    def __init__(
        self,
        model: AutoModelForSequenceClassification,
        tokenizer: AutoTokenizer,
        device: torch.device,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._device = device

    def __call__(self, texts: List[str]) -> np.ndarray:
        """
        Run batched inference on a list of (possibly masked) strings.

        Args:
            texts: List of text strings produced by SHAP's masker.

        Returns:
            np.ndarray of shape (len(texts), num_labels) with softmax probabilities.
        """
        if not texts:
            return np.empty((0, 10), dtype=np.float32)

        inputs = self._tokenizer(
            list(texts),
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._model(**inputs).logits          # (batch, num_labels)
            probs  = torch.softmax(logits, dim=-1)         # (batch, num_labels)

        return probs.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# SHAPService
# ---------------------------------------------------------------------------

class SHAPService:
    """
    Generates token-level SHAP importance scores for InLegalBERT predictions.

    The service is stateless — it reads the already-loaded model and tokenizer
    from model_service each time explain() is called, so there is nothing to
    initialise at startup.
    """

    def explain(
        self,
        text: str,
        predicted_label_id: int,
        max_evals: int = 200,
    ) -> List[Dict]:
        """
        Compute token-level SHAP values for a single text.

        Args:
            text:               The original legal text that was classified.
            predicted_label_id: Integer class index of the model's top prediction.
                                SHAP scores are computed w.r.t. this class.
            max_evals:          Maximum number of SHAP evaluations (higher = more
                                accurate but slower). Default 200 is reasonable on CPU.

        Returns:
            List of {"token": str, "importance": float} dicts, sorted by
            descending absolute importance so the most influential tokens appear first.

        Raises:
            RuntimeError: if the model has not been loaded yet.
            ValueError:   if the text is empty.
        """
        if not model_service.is_loaded():
            raise RuntimeError("Model must be loaded before running SHAP explanations.")

        text = text.strip()
        if not text:
            raise ValueError("Input text is empty — cannot generate SHAP explanation.")

        model     = model_service.get_model()
        tokenizer = model_service.get_tokenizer()
        device    = model_service.device

        logger.info(
            "Running SHAP explanation (label_id=%d, max_evals=%d, text_len=%d)",
            predicted_label_id, max_evals, len(text),
        )

        # Build the callable wrapper SHAP will use for perturbations
        predict_fn = _BertPredictWrapper(model, tokenizer, device)

        # Text masker: replaces tokens with the tokenizer's mask token
        masker = shap.maskers.Text(tokenizer=tokenizer.tokenize)

        # Partition explainer — works well for transformer text models
        explainer = shap.Explainer(predict_fn, masker, output_names=None)

        try:
            shap_values = explainer([text], max_evals=max_evals, batch_size=8)
        except Exception as exc:
            logger.error("SHAP computation failed: %s", exc)
            raise RuntimeError(f"SHAP computation failed: {exc}") from exc

        # shap_values.values shape: (1, n_tokens, n_classes)
        # shap_values.data  shape: (1, n_tokens)  — the token strings
        token_shap   = shap_values.values[0, :, predicted_label_id]  # (n_tokens,)
        token_strings = shap_values.data[0]                           # (n_tokens,)

        token_importances = self._build_token_importances(token_strings, token_shap)

        logger.info(
            "SHAP explanation complete — %d tokens, top token: '%s' (%.4f)",
            len(token_importances),
            token_importances[0]["token"] if token_importances else "N/A",
            token_importances[0]["importance"] if token_importances else 0.0,
        )

        return token_importances

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_token_importances(
        tokens: np.ndarray,
        shap_scores: np.ndarray,
    ) -> List[Dict]:
        """
        Zip token strings with their SHAP scores and sort by absolute importance.

        Special tokens ([CLS], [SEP], [PAD]) are excluded from the output
        because their SHAP values are not meaningful for human interpretation.

        Args:
            tokens:      Array of token strings from the masker.
            shap_scores: Array of SHAP values for the predicted class.

        Returns:
            Sorted list of {"token": str, "importance": float} dicts.
        """
        _SPECIAL = {"[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"}

        paired = [
            {"token": str(tok), "importance": round(float(score), 6)}
            for tok, score in zip(tokens, shap_scores)
            if str(tok) not in _SPECIAL
        ]

        # Sort by absolute value — most influential tokens first
        paired.sort(key=lambda x: abs(x["importance"]), reverse=True)
        return paired

    def explain_safe(
        self,
        text: str,
        predicted_label_id: int,
        max_evals: int = 200,
    ) -> Optional[List[Dict]]:
        """
        Fault-tolerant wrapper around explain().

        Returns None instead of raising if anything goes wrong.
        Useful for the XAI worker where a single failure should not crash
        the entire job loop.
        """
        try:
            return self.explain(text, predicted_label_id, max_evals=max_evals)
        except Exception as exc:
            logger.error(
                "explain_safe: SHAP failed for label_id=%d — %s",
                predicted_label_id, exc,
            )
            return None


# ---------------------------------------------------------------------------
# Shared singleton
# ---------------------------------------------------------------------------

shap_service = SHAPService()
