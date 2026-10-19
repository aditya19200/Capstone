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

from config.domains import active_domain
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
          1. confidence > high
             → AUTO_ACCEPT
          2. confidence < low
             → ROUTE_TO_REVIEWER
          3. Otherwise (medium band)
             → NEEDS_EXPLANATION

        There was a fourth rule here, a margin guard: override AUTO_ACCEPT
        when confidence cleared `high` but the top-2 gap was under 0.10, on
        the reasoning that a near-tie deserves an explanation however
        confident the winner looks. It could never fire, and the reasoning
        does not survive being written down.

        The probabilities are a softmax, so they sum to one, so the runner-up
        is at most 1 - p1, so

            margin = p1 - p2 >= p1 - (1 - p1) = 2*p1 - 1

        At p1 > 0.85 the margin is necessarily above 0.70 — it cannot be
        under 0.10 unless p1 is under 0.55, and below 0.55 rule 2 has already
        sent the prediction to a reviewer. The guard's own example, "0.86 vs
        0.84", sums to 1.70 and is not a distribution; the unit test that
        covered it passed confidence=0.90 with margin=0.05 as two independent
        scalars, which no prediction can produce.

        Measured rather than argued, on the 435-row medical held-out set:
        of the 181 predictions above the threshold the smallest margin
        observed was 0.7727, and the guard fired zero times. See
        backend/confident_error_analysis.py and
        metrics/medical_confident_errors.json.

        Removing it changes no behaviour, because unreachable code has none.
        What it changes is the claim: there is no margin-based protection on
        auto-accept, and there never was. The protection that does exist is
        the threshold itself, plus services/ontology_check.py, which flags a
        confident prediction whose text carries vocabulary pointing at a
        different label.

        `margin` stays in the signature and in the reason string. It is real
        information about the prediction and both are stored — it is simply
        not a gate.
        """
        # Theoretical max entropy is log(num_labels) for a uniform
        # distribution, so it belongs to the active domain and not to a
        # constant. It was log(10) — correct for legal, wrong by 0.18 nats
        # for the 12-class medical pack, and the figure is printed in the
        # reason string a reviewer reads.
        max_entropy = math.log(active_domain().num_labels)

        if confidence > high:
            return (
                "AUTO_ACCEPT",
                (
                    f"Confidence {confidence:.4f} exceeds HIGH threshold {high} "
                    f"(margin {margin:.4f}). Prediction accepted automatically."
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
