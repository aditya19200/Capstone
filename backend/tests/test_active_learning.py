"""
tests/test_active_learning.py — Tests for the Active Learning Engine.

Covers:
  - All three routing decisions (AUTO_ACCEPT, NEEDS_EXPLANATION, ROUTE_TO_REVIEWER)
  - Margin guard: high confidence + low margin → NEEDS_EXPLANATION (not AUTO_ACCEPT)
  - Exact threshold boundary conditions (at, just above, just below)
  - requires_shap and requires_review flags
  - reason string is non-empty for every decision
  - Utility helpers: compute_entropy, compute_margin, uncertainty_summary
  - Custom threshold overrides
  - Uniform distribution (maximum uncertainty)
  - Single-class certainty (minimum uncertainty)
"""

import math

import pytest

from services.active_learning import ActiveLearningEngine, ActiveLearningResult, active_learning_engine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# 10 legal labels (order doesn't matter for these tests)
LABELS = [
    "Contract Law", "Criminal Law", "Constitutional Law",
    "Corporate / Company Law", "Property / Real Estate Law",
    "Family Law", "Labour & Employment Law", "Intellectual Property Law",
    "Taxation Law", "Civil Procedure / Other",
]

NUM_LABELS = len(LABELS)


def _uniform_probs() -> dict:
    """Uniform distribution — maximum entropy, minimum confidence."""
    p = 1.0 / NUM_LABELS
    return {label: p for label in LABELS}


def _certain_probs(top_label: str = "Contract Law") -> dict:
    """Near-certain distribution — all probability on one label."""
    probs = {label: 0.0 for label in LABELS}
    probs[top_label] = 1.0
    return probs


def _make_probs(top_conf: float, second_conf: float, top_label: str = "Contract Law") -> dict:
    """
    Build a realistic probability dict with controlled top-1 and top-2 values.
    Remaining probability is split evenly among the other 8 labels.
    """
    remaining = max(0.0, 1.0 - top_conf - second_conf)
    per_other = remaining / (NUM_LABELS - 2) if NUM_LABELS > 2 else 0.0
    probs = {label: per_other for label in LABELS}
    probs[top_label] = top_conf
    second_label = [l for l in LABELS if l != top_label][0]
    probs[second_label] = second_conf
    return probs


def _evaluate(confidence: float, margin: float, entropy: float = 0.5) -> ActiveLearningResult:
    """Shorthand: call evaluate() with pre-computed signals."""
    probs = _make_probs(confidence, confidence - margin)
    return active_learning_engine.evaluate(
        confidence=confidence,
        probabilities=probs,
        entropy=entropy,
        margin=margin,
    )


# ===========================================================================
# AUTO_ACCEPT
# ===========================================================================

class TestAutoAccept:

    def test_high_confidence_large_margin(self):
        """confidence > HIGH and margin ≥ 0.10 → AUTO_ACCEPT."""
        result = _evaluate(confidence=0.92, margin=0.30)
        assert result.routing_decision == "AUTO_ACCEPT"

    def test_just_above_high_threshold(self):
        """confidence = HIGH + ε → AUTO_ACCEPT."""
        result = _evaluate(confidence=0.851, margin=0.20)
        assert result.routing_decision == "AUTO_ACCEPT"

    def test_requires_shap_false(self):
        """AUTO_ACCEPT → requires_shap is False."""
        result = _evaluate(confidence=0.92, margin=0.30)
        assert result.requires_shap is False

    def test_requires_review_false(self):
        """AUTO_ACCEPT → requires_review is False."""
        result = _evaluate(confidence=0.92, margin=0.30)
        assert result.requires_review is False

    def test_reason_is_nonempty(self):
        """AUTO_ACCEPT → reason string is populated."""
        result = _evaluate(confidence=0.92, margin=0.30)
        assert isinstance(result.reason, str)
        assert len(result.reason) > 0

    def test_certain_distribution_auto_accepts(self):
        """Near-certain probability distribution → AUTO_ACCEPT."""
        probs = _certain_probs()
        result = active_learning_engine.evaluate(
            confidence=1.0,
            probabilities=probs,
            entropy=0.0,
            margin=1.0,
        )
        assert result.routing_decision == "AUTO_ACCEPT"


# ===========================================================================
# NEEDS_EXPLANATION
# ===========================================================================

class TestNeedsExplanation:

    def test_medium_confidence(self):
        """confidence in [LOW, HIGH] → NEEDS_EXPLANATION."""
        result = _evaluate(confidence=0.70, margin=0.20)
        assert result.routing_decision == "NEEDS_EXPLANATION"

    def test_just_at_low_threshold(self):
        """confidence = LOW → NEEDS_EXPLANATION (not ROUTE_TO_REVIEWER)."""
        result = _evaluate(confidence=0.55, margin=0.15)
        assert result.routing_decision == "NEEDS_EXPLANATION"

    def test_just_below_high_threshold(self):
        """confidence = HIGH - ε → NEEDS_EXPLANATION (not AUTO_ACCEPT)."""
        result = _evaluate(confidence=0.849, margin=0.20)
        assert result.routing_decision == "NEEDS_EXPLANATION"

    def test_high_confidence_low_margin_overrides_auto_accept(self):
        """confidence > HIGH but margin < 0.10 → NEEDS_EXPLANATION."""
        result = _evaluate(confidence=0.90, margin=0.05)
        assert result.routing_decision == "NEEDS_EXPLANATION"

    def test_high_confidence_exactly_at_margin_guard(self):
        """confidence > HIGH and margin == 0.10 → AUTO_ACCEPT (boundary inclusive)."""
        result = _evaluate(confidence=0.90, margin=0.10)
        assert result.routing_decision == "AUTO_ACCEPT"

    def test_requires_shap_true(self):
        """NEEDS_EXPLANATION → requires_shap is True."""
        result = _evaluate(confidence=0.70, margin=0.20)
        assert result.requires_shap is True

    def test_requires_review_false(self):
        """NEEDS_EXPLANATION → requires_review is False (no human reviewer needed)."""
        result = _evaluate(confidence=0.70, margin=0.20)
        assert result.requires_review is False

    def test_reason_mentions_shap(self):
        """NEEDS_EXPLANATION reason string references SHAP."""
        result = _evaluate(confidence=0.70, margin=0.20)
        assert "shap" in result.reason.lower() or "explanation" in result.reason.lower()


# ===========================================================================
# ROUTE_TO_REVIEWER
# ===========================================================================

class TestRouteToReviewer:

    def test_low_confidence(self):
        """confidence < LOW → ROUTE_TO_REVIEWER."""
        result = _evaluate(confidence=0.40, margin=0.10)
        assert result.routing_decision == "ROUTE_TO_REVIEWER"

    def test_just_below_low_threshold(self):
        """confidence = LOW - ε → ROUTE_TO_REVIEWER."""
        result = _evaluate(confidence=0.549, margin=0.10)
        assert result.routing_decision == "ROUTE_TO_REVIEWER"

    def test_very_low_confidence(self):
        """Near-uniform distribution → ROUTE_TO_REVIEWER."""
        probs = _uniform_probs()
        result = active_learning_engine.evaluate(
            confidence=1.0 / NUM_LABELS,
            probabilities=probs,
            entropy=math.log(NUM_LABELS),
            margin=0.0,
        )
        assert result.routing_decision == "ROUTE_TO_REVIEWER"

    def test_requires_shap_true(self):
        """ROUTE_TO_REVIEWER → requires_shap is True."""
        result = _evaluate(confidence=0.40, margin=0.10)
        assert result.requires_shap is True

    def test_requires_review_true(self):
        """ROUTE_TO_REVIEWER → requires_review is True."""
        result = _evaluate(confidence=0.40, margin=0.10)
        assert result.requires_review is True

    def test_reason_mentions_reviewer(self):
        """ROUTE_TO_REVIEWER reason string mentions human reviewer."""
        result = _evaluate(confidence=0.40, margin=0.10)
        assert "reviewer" in result.reason.lower()


# ===========================================================================
# Result dataclass fields
# ===========================================================================

class TestResultFields:

    def test_confidence_stored_correctly(self):
        """Result.confidence matches the input confidence value."""
        result = _evaluate(confidence=0.72, margin=0.15)
        assert result.confidence == pytest.approx(0.72, abs=1e-6)

    def test_entropy_stored_correctly(self):
        """Result.entropy matches the input entropy value."""
        result = _evaluate(confidence=0.72, margin=0.15, entropy=1.23)
        assert result.entropy == pytest.approx(1.23, abs=1e-6)

    def test_margin_stored_correctly(self):
        """Result.margin matches the input margin value."""
        result = _evaluate(confidence=0.72, margin=0.15)
        assert result.margin == pytest.approx(0.15, abs=1e-6)

    def test_result_is_frozen(self):
        """ActiveLearningResult is a frozen dataclass — fields are immutable."""
        result = _evaluate(confidence=0.92, margin=0.30)
        with pytest.raises((AttributeError, TypeError)):
            result.routing_decision = "AUTO_ACCEPT"  # type: ignore


# ===========================================================================
# Custom threshold overrides
# ===========================================================================

class TestCustomThresholds:

    def test_custom_high_threshold(self):
        """Supplying confidence_high=0.60 makes 0.65 trigger AUTO_ACCEPT."""
        probs = _make_probs(0.65, 0.40)
        result = active_learning_engine.evaluate(
            confidence=0.65,
            probabilities=probs,
            entropy=0.5,
            margin=0.25,
            confidence_high=0.60,
            confidence_low=0.30,
        )
        assert result.routing_decision == "AUTO_ACCEPT"

    def test_custom_low_threshold(self):
        """Supplying confidence_low=0.70 makes 0.65 trigger ROUTE_TO_REVIEWER."""
        probs = _make_probs(0.65, 0.40)
        result = active_learning_engine.evaluate(
            confidence=0.65,
            probabilities=probs,
            entropy=0.5,
            margin=0.25,
            confidence_high=0.90,
            confidence_low=0.70,
        )
        assert result.routing_decision == "ROUTE_TO_REVIEWER"

    def test_override_does_not_affect_singleton(self):
        """Per-call threshold overrides don't mutate the singleton's settings."""
        from config.settings import settings

        high_before = settings.CONFIDENCE_HIGH
        _evaluate(confidence=0.92, margin=0.30)
        assert settings.CONFIDENCE_HIGH == high_before


# ===========================================================================
# Utility helpers
# ===========================================================================

class TestUtilityHelpers:

    def test_compute_entropy_uniform(self):
        """Uniform distribution has entropy = log(N)."""
        probs = _uniform_probs()
        entropy = ActiveLearningEngine.compute_entropy(probs)
        assert entropy == pytest.approx(math.log(NUM_LABELS), abs=1e-6)

    def test_compute_entropy_certain(self):
        """Certain distribution (p=1 for one class) has entropy = 0."""
        probs = _certain_probs()
        entropy = ActiveLearningEngine.compute_entropy(probs)
        assert entropy == pytest.approx(0.0, abs=1e-6)

    def test_compute_entropy_partial(self):
        """Entropy of a non-trivial distribution is between 0 and log(N)."""
        probs = _make_probs(0.70, 0.20)
        entropy = ActiveLearningEngine.compute_entropy(probs)
        assert 0.0 < entropy < math.log(NUM_LABELS)

    def test_compute_margin_certain(self):
        """Certain distribution has margin ≈ 1.0."""
        probs = _certain_probs()
        margin = ActiveLearningEngine.compute_margin(probs)
        assert margin == pytest.approx(1.0, abs=1e-6)

    def test_compute_margin_uniform(self):
        """Uniform distribution has margin ≈ 0.0."""
        probs = _uniform_probs()
        margin = ActiveLearningEngine.compute_margin(probs)
        assert margin == pytest.approx(0.0, abs=1e-4)

    def test_compute_margin_controlled(self):
        """Margin equals top-1 minus top-2 for a controlled distribution."""
        probs = _make_probs(top_conf=0.80, second_conf=0.10)
        margin = ActiveLearningEngine.compute_margin(probs)
        assert margin == pytest.approx(0.70, abs=1e-4)

    def test_uncertainty_summary_keys(self):
        """uncertainty_summary returns all three signal keys."""
        probs = _make_probs(0.72, 0.15)
        summary = ActiveLearningEngine.uncertainty_summary(probs)
        assert set(summary.keys()) == {"confidence", "entropy", "margin"}

    def test_uncertainty_summary_values_in_range(self):
        """uncertainty_summary values are in expected ranges."""
        probs = _make_probs(0.72, 0.15)
        summary = ActiveLearningEngine.uncertainty_summary(probs)
        assert 0.0 <= summary["confidence"] <= 1.0
        assert summary["entropy"] >= 0.0
        assert 0.0 <= summary["margin"] <= 1.0

    def test_uncertainty_summary_confidence_matches_top_prob(self):
        """uncertainty_summary confidence equals the max probability."""
        probs = _make_probs(top_conf=0.80, second_conf=0.10)
        summary = ActiveLearningEngine.uncertainty_summary(probs)
        assert summary["confidence"] == pytest.approx(0.80, abs=1e-4)
