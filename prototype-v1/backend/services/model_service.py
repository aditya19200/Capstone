"""
services/model_service.py — InLegalBERT load + inference.

Loads BertForSequenceClassification weights once at startup (via main.py lifespan)
and exposes a single predict() method used by every prediction path.

Design rules:
  - Model is loaded ONCE and reused — never reload per request.
  - All tensor ops run on CPU by default (GPU used automatically if available).
  - predict() is synchronous (CPU-bound); routers call it inside a thread pool
    via asyncio.get_event_loop().run_in_executor() to avoid blocking the event loop.
"""

import asyncio
import logging
import math
import threading
from functools import partial
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label mapping — mirrors id2label in the model config
# ---------------------------------------------------------------------------

ID2LABEL: Dict[int, str] = {
    0: "Contract Law",
    1: "Criminal Law",
    2: "Constitutional Law",
    3: "Corporate / Company Law",
    4: "Property / Real Estate Law",
    5: "Family Law",
    6: "Labour & Employment Law",
    7: "Intellectual Property Law",
    8: "Taxation Law",
    9: "Civil Procedure / Other",
}

LABEL2ID: Dict[str, int] = {v: k for k, v in ID2LABEL.items()}
NUM_LABELS: int = len(ID2LABEL)


# ---------------------------------------------------------------------------
# ModelService
# ---------------------------------------------------------------------------

class ModelService:
    """
    Singleton wrapper around InLegalBERT.

    Usage:
        from services.model_service import model_service

        # At startup (called by main.py lifespan):
        model_service.load()

        # In a router or worker:
        result = await model_service.predict_async("Some legal text...")
    """

    def __init__(self) -> None:
        self._model: Optional[AutoModelForSequenceClassification] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # Guards the (model, tokenizer) pair so a reload can never be observed
        # half-applied — see load() and _snapshot().
        self._lock = threading.RLock()
        # Which weights are actually resident, so /health and the admin page can
        # show what is being served rather than what is merely marked active.
        self._loaded_path: Optional[str] = None
        self._loaded_version: Optional[str] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(
        self,
        model_path: Optional[str] = None,
        version_number: Optional[str] = None,
    ) -> None:
        """
        Load tokenizer and weights, replacing whatever is currently resident.

        Called at startup with no arguments (loads settings.MODEL_PATH), and
        again by the admin activate endpoint with the newly activated version's
        file_path — which is what makes an activated model actually serve
        predictions instead of only being marked active in the database.

        Both objects are built into locals FIRST and swapped in together under
        a lock. Two consequences that matter:

          * If the new weights fail to load, the exception propagates and the
            previously loaded model is still resident and still serving. A bad
            checkpoint can never leave the service with no model at all.
          * An in-flight predict() can never see the new model paired with the
            old tokenizer, because it takes both references in one locked read
            (see _snapshot()).

        Raises RuntimeError if the path is invalid or the weights won't load.
        """
        model_path = model_path or settings.MODEL_PATH
        logger.info("Loading tokenizer from %s", model_path)

        try:
            new_tokenizer = AutoTokenizer.from_pretrained(model_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load tokenizer from '{model_path}': {exc}"
            ) from exc

        logger.info("Loading model weights from %s (device=%s)", model_path, self._device)

        try:
            new_model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=NUM_LABELS,
            )
            new_model.to(self._device)
            new_model.eval()            # disable dropout for inference
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load model from '{model_path}': {exc}"
            ) from exc

        # Atomic swap — nothing above this line has touched live state.
        with self._lock:
            previous = self._loaded_version or self._loaded_path
            self._tokenizer = new_tokenizer
            self._model = new_model
            self._loaded_path = model_path
            self._loaded_version = version_number

        if previous:
            logger.info(
                "Model swapped: %s -> %s (%s) on %s.",
                previous, version_number or model_path, model_path, self._device,
            )
        else:
            logger.info("InLegalBERT loaded successfully on %s.", self._device)

    def unload(self) -> None:
        """Release model and tokenizer from memory."""
        with self._lock:
            self._model = None
            self._tokenizer = None
            self._loaded_path = None
            self._loaded_version = None
        if self._device.type == "cuda":
            torch.cuda.empty_cache()
        logger.info("Model unloaded.")

    def is_loaded(self) -> bool:
        """Return True if the model is currently in memory."""
        with self._lock:
            return self._model is not None and self._tokenizer is not None

    def _snapshot(self) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
        """
        Take a consistent (model, tokenizer) pair under the lock.

        Callers then use the returned locals for the whole forward pass, so a
        concurrent load() swapping the pair mid-inference cannot tear it.
        """
        with self._lock:
            if self._model is None or self._tokenizer is None:
                raise RuntimeError("Model is not loaded. Call model_service.load() first.")
            return self._model, self._tokenizer

    @property
    def loaded_path(self) -> Optional[str]:
        """Filesystem path of the weights currently resident."""
        return self._loaded_path

    @property
    def loaded_version(self) -> Optional[str]:
        """version_number of the resident weights, if loaded via activation."""
        return self._loaded_version

    # ------------------------------------------------------------------
    # Core inference (synchronous, CPU-bound)
    # ------------------------------------------------------------------

    def predict(self, text: str) -> Dict:
        """
        Run InLegalBERT inference on a single text string.

        Returns a dict with:
          predicted_label      — top-1 label name
          confidence           — max softmax probability
          probabilities        — {label: probability} for all 10 classes
          label_id             — integer class index of the top-1 label
          entropy              — Shannon entropy of the distribution
          margin               — difference between top-2 softmax probabilities

        Raises:
          RuntimeError: if model has not been loaded yet.
          ValueError:   if text is empty after stripping.
        """
        model, tokenizer = self._snapshot()

        text = text.strip()
        if not text:
            raise ValueError("Input text is empty.")

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        )
        # Move inputs to the same device as the model
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        logits: torch.Tensor = outputs.logits          # shape: (1, NUM_LABELS)
        probs: torch.Tensor = torch.softmax(logits, dim=-1)[0]  # shape: (NUM_LABELS,)

        # Top-1 prediction
        label_id: int = int(probs.argmax().item())
        confidence: float = float(probs[label_id].item())
        predicted_label: str = ID2LABEL[label_id]

        # Full probability distribution
        prob_list: List[float] = probs.tolist()
        probabilities: Dict[str, float] = {
            ID2LABEL[i]: round(prob_list[i], 6) for i in range(NUM_LABELS)
        }

        # Active-learning signals
        entropy: float = self._compute_entropy(prob_list)
        margin: float = self._compute_margin(prob_list)

        return {
            "predicted_label": predicted_label,
            "label_id": label_id,
            "confidence": round(confidence, 6),
            "probabilities": probabilities,
            "entropy": round(entropy, 6),
            "margin": round(margin, 6),
        }

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Run inference on a list of texts in one batched forward pass.

        Texts are truncated/padded to 512 tokens. Batch size is limited by
        available memory; for large batches use the prediction worker instead.

        Returns a list of result dicts in the same order as the input texts.
        """
        model, tokenizer = self._snapshot()
        if not texts:
            return []

        inputs = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        logits: torch.Tensor = outputs.logits           # (batch, NUM_LABELS)
        probs_batch: torch.Tensor = torch.softmax(logits, dim=-1)  # (batch, NUM_LABELS)

        results = []
        for probs in probs_batch:
            prob_list = probs.tolist()
            label_id = int(probs.argmax().item())
            confidence = float(probs[label_id].item())
            predicted_label = ID2LABEL[label_id]

            results.append({
                "predicted_label": predicted_label,
                "label_id": label_id,
                "confidence": round(confidence, 6),
                "probabilities": {
                    ID2LABEL[i]: round(prob_list[i], 6) for i in range(NUM_LABELS)
                },
                "entropy": round(self._compute_entropy(prob_list), 6),
                "margin": round(self._compute_margin(prob_list), 6),
            })

        return results

    # ------------------------------------------------------------------
    # Async wrappers (run synchronous inference in a thread pool)
    # ------------------------------------------------------------------

    async def predict_async(self, text: str) -> Dict:
        """
        Async wrapper around predict().

        Offloads the CPU-bound forward pass to a thread-pool executor so it
        doesn't block the FastAPI event loop.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, partial(self.predict, text))

    async def predict_batch_async(self, texts: List[str]) -> List[Dict]:
        """Async wrapper around predict_batch()."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, partial(self.predict_batch, texts))

    # ------------------------------------------------------------------
    # Signal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_entropy(prob_list: List[float]) -> float:
        """
        Compute Shannon entropy: H = -Σ p_i * log(p_i).

        High entropy → model is uncertain across many classes.
        Clamps probabilities to avoid log(0).
        """
        entropy = 0.0
        for p in prob_list:
            if p > 1e-12:
                entropy -= p * math.log(p)
        return entropy

    @staticmethod
    def _compute_margin(prob_list: List[float]) -> float:
        """
        Compute margin: difference between the top-2 probabilities.

        Low margin → top-2 candidates are close → model is uncertain.
        """
        sorted_probs = sorted(prob_list, reverse=True)
        if len(sorted_probs) < 2:
            return sorted_probs[0] if sorted_probs else 0.0
        return sorted_probs[0] - sorted_probs[1]

    # ------------------------------------------------------------------
    # Tokenizer access (needed by shap_service)
    # ------------------------------------------------------------------

    def get_tokenizer(self) -> AutoTokenizer:
        """
        Return the loaded tokenizer.

        Used by shap_service to tokenize text for SHAP explainability.
        Raises RuntimeError if model has not been loaded.
        """
        return self._snapshot()[1]

    def get_model(self) -> AutoModelForSequenceClassification:
        """
        Return the loaded model.

        Used by shap_service to run SHAP's forward function.
        Raises RuntimeError if model has not been loaded.
        """
        return self._snapshot()[0]

    @property
    def device(self) -> torch.device:
        """The torch device the model is running on."""
        return self._device


# Single shared instance — import this everywhere
model_service = ModelService()
