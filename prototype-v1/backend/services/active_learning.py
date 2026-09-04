"""
services/active_learning.py — Active Learning Engine.

Evaluates three uncertainty signals from the model output and produces a
routing decision that controls the downstream workflow:

  AUTO_ACCEPT       → high confidence, store prediction, no SHAP needed
  NEEDS_EXPLANATION → medium confidence, queue async SHAP job
  ROUTE_TO_REVIEWER → low confidence, hand off to human reviewer + queue SHAP

All thresholds are read from config/settings.py (ultimately from .env) so
they can be tuned without touching code.
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

from config.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

RoutingDecision = Literal["AUTO_ACCEPT", "NEEDS_EXPLANATION", "ROUTE_TO_REVIEWER"]


@dataclass(frozen=True)
class ActiveLearningResult:
    """
    Immutable result object returned by evaluate().

    Attributes:
        routing_decision  — one of AUTO_ACCEPT / NEEDS_EXPLANATION / ROUTE_TO_REVIEWER
        confidence        — max softmax probability
        entropy           — Shannon entropy of the full distribution
        margin            — gap between top-2 probabilities
        requires_shap     — True when a SHAP job should be queued
        requires_review   — True when a human reviewer should be notified
        reason            — human-readable explanation of why this decision was made
    """

    routing_decision: RoutingDecision
    confidence: float
    entropy: float
    margin: float
    requires_shap: bool
    requires_review: bool
    reason: str


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ActiveLearningEngine:
    """
    Stateless engine that maps model output signals → routing decision.

    All threshold values are read from settings at call time so that changes
    to .env take effect without restarting (settings is a module-level singleton
    that reflects env values at import time; for hot-reload support, pass
    overrides explicitly via the threshold parameters).
    """

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def evaluate(
        self,
        confidence: float,
        probabilities: Dict[str, float],
        entropy: float,
        margin: float,
        *,
        confidence_high: float | None = None,
        confidence_low: float | None = None,
    ) -> ActiveLearningResult:
        """
        Evaluate all three uncertainty signals and return a routing decision.

        Args:
            confidence:      Max softmax probability for the predicted class.
            probabilities:   Full {label: probability} dict from the model.
            entropy:         Shannon entropy already computed by model_service.
            margin:          Top-1 minus top-2 probability already computed by model_service.
            confidence_high: Override for CONFIDENCE_HIGH threshold (default: from .env).
            confidence_low:  Override for CONFIDENCE_LOW threshold (default: from .env).

        Returns:
            ActiveLearningResult with routing decision and all signal values.
        """
        high = confidence_high if confidence_high is not None else settings.CONFIDENCE_HIGH
        low  = confidence_low  if confidence_low  is not None else settings.CONFIDENCE_LOW

        routing, reason = self._route(confidence, entropy, margin, high, low)

        requires_shap   = routing != "AUTO_ACCEPT"
        requires_review = routing == "ROUTE_TO_REVIEWER"

        logger.debug(
            "ActiveLearning: conf=%.4f entropy=%.4f margin=%.4f → %s",
            confidence, entropy, margin, routing,
        )

        return ActiveLearningResult(
            routing_decision=routing,
            confidence=confidence,
            entropy=entropy,
            margin=margin,
            requires_shap=requires_shap,
            requires_review=requires_review,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # Routing logic
    # ------------------------------------------------------------------

    def _route(
        self,
        confidence: float,
        entropy: float,
        margin: float,
        high: float,
        low: float,
    ) -> Tuple[RoutingDecision, str]:
        """
        Apply threshold rules and return (decision, reason).

        Rules (applied in priority order):
          1. confidence > high AND margin is not suspiciously low
             → AUTO_ACCEPT
          2. confidence < low
             → ROUTE_TO_REVIEWER
          3. Otherwise (medium band)
             → NEEDS_EXPLANATION

        The margin guard on rule 1 catches the edge case where confidence is
        high but the top-2 gap is still very narrow (e.g. 0.86 vs 0.84),
        suggesting genuine ambiguity that warrants an explanation.
        """
        # Theoretical max entropy for NUM_LABELS=10 uniform distribution
        max_entropy = math.log(10)          # ≈ 2.303
        # Margin threshold below which AUTO_ACCEPT is overridden
        margin_low_guard = 0.10

        if confidence > high and margin >= margin_low_guard:
            return (
                "AUTO_ACCEPT",
                (
                    f"Confidence {confidence:.4f} exceeds HIGH threshold {high} "
                    f"and margin {margin:.4f} is sufficiently large. "
                    "Prediction accepted automatically."
                ),
            )

        if confidence > high and margin < margin_low_guard:
            # High raw confidence but top-2 are very close → treat as medium
            return (
                "NEEDS_EXPLANATION",
                (
                    f"Confidence {confidence:.4f} exceeds HIGH threshold {high} "
                    f"but margin {margin:.4f} < {margin_low_guard} indicates "
                    "ambiguity between top labels. SHAP explanation requested."
                ),
            )

        if confidence < low:
            return (
                "ROUTE_TO_REVIEWER",
                (
                    f"Confidence {confidence:.4f} is below LOW threshold {low}. "
                    f"Entropy {entropy:.4f} (max={max_entropy:.4f}). "
                    "Routed to human reviewer."
                ),
            )

        # Medium band: low ≤ confidence ≤ high
        return (
            "NEEDS_EXPLANATION",
            (
                f"Confidence {confidence:.4f} is in the medium band "
                f"[{low}, {high}]. SHAP explanation queued for transparency."
            ),
        )

    # ------------------------------------------------------------------
    # Utility helpers (used by tests and monitoring)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_entropy(probabilities: Dict[str, float]) -> float:
        """
        Compute Shannon entropy from a label→probability dict.

        H = -Σ p_i * log(p_i)

        Returns 0.0 for a perfectly certain distribution and log(N) for a
        perfectly uniform one.
        """
        entropy = 0.0
        for p in probabilities.values():
            if p > 1e-12:
                entropy -= p * math.log(p)
        return entropy

    @staticmethod
    def compute_margin(probabilities: Dict[str, float]) -> float:
        """
        Compute margin: difference between the two highest probabilities.

        Low margin → top candidates are close → higher uncertainty.
        """
        sorted_probs = sorted(probabilities.values(), reverse=True)
        if len(sorted_probs) < 2:
            return sorted_probs[0] if sorted_probs else 0.0
        return sorted_probs[0] - sorted_probs[1]

    @staticmethod
    def uncertainty_summary(probabilities: Dict[str, float]) -> Dict:
        """
        Return a dict of all three uncertainty signals at once.

        Convenience method for places that need all signals without running
        a full evaluate() (e.g. monitoring dashboards, unit tests).
        """
        sorted_probs = sorted(probabilities.values(), reverse=True)
        confidence = sorted_probs[0] if sorted_probs else 0.0
        entropy = ActiveLearningEngine.compute_entropy(probabilities)
        margin  = ActiveLearningEngine.compute_margin(probabilities)
        return {
            "confidence": round(confidence, 6),
            "entropy":    round(entropy,    6),
            "margin":     round(margin,     6),
        }


# ---------------------------------------------------------------------------
# Shared singleton
# ---------------------------------------------------------------------------

active_learning_engine = ActiveLearningEngine()
