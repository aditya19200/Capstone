"""
tests/test_ontology_check.py — ontology consistency check.

The check exists to catch predictions the confidence thresholds cannot: ones
the model is confident about and wrong about. These tests pin the three
properties that matter — it fires on contradicted labels, it stays quiet on
agreeing ones, and it degrades to a no-op rather than an error when the
markers file is missing.
"""

import pytest

from services.ontology_check import OntologyCheck


@pytest.fixture
def check():
    c = OntologyCheck()
    c.load()
    if not c.is_loaded():
        pytest.skip("concept_markers.json not present — run graph/mine_concept_markers.py")
    return c


CRIMINAL_IN_DOMESTIC_SETTING = (
    "The prosecution led evidence that the accused assaulted his spouse with a "
    "blunt weapon during a domestic altercation, and he was charged under the "
    "Indian Penal Code for voluntarily causing grievous hurt."
)

GENUINE_FAMILY = (
    "The wife filed a petition for divorce citing irreconcilable differences and "
    "sought custody of the two minor children along with interim maintenance."
)


class TestConflictDetection:
    def test_flags_criminal_text_labelled_family(self, check):
        """The confusion this was built for: confident, wrong, and caught."""
        conflict = check.check(CRIMINAL_IN_DOMESTIC_SETTING, "Family Law")
        assert conflict is not None
        assert conflict["suggested_label"] == "Criminal Law"
        assert len(conflict["matched_terms"]) >= 2
        assert "Family Law" in conflict["reason"]

    def test_silent_when_label_agrees_with_vocabulary(self, check):
        """No flag when the text and the label are about the same thing."""
        assert check.check(CRIMINAL_IN_DOMESTIC_SETTING, "Criminal Law") is None

    def test_does_not_flag_genuine_family_law(self, check):
        """Family vocabulary plus a family label must not trip the check."""
        assert check.check(GENUINE_FAMILY, "Family Law") is None

    def test_no_markers_means_no_opinion(self, check):
        """Text with no legal vocabulary produces no conflict, not a guess."""
        assert check.check("The quick brown fox jumps over the lazy dog.", "Family Law") is None


class TestEvidence:
    def test_evidence_reports_terms_per_domain(self, check):
        ev = check.evidence(CRIMINAL_IN_DOMESTIC_SETTING)
        assert "Criminal Law" in ev
        assert ev["Criminal Law"]["score"] > 0
        assert ev["Criminal Law"]["terms"]

    def test_matching_respects_word_boundaries(self, check):
        """'ipc' must not fire inside a longer word — the flag is evidence."""
        ev = check.evidence("The ipcc report discusses climate policy.")
        assert "ipc" not in ev.get("Criminal Law", {}).get("terms", [])


class TestDegradation:
    def test_unloaded_check_is_a_noop_not_an_error(self):
        """
        A missing markers file must not break the review queue — the route
        falls back to confidence-only behaviour, so this has to return
        cleanly rather than raise.
        """
        c = OntologyCheck()          # never loaded
        assert c.is_loaded() is False
        assert c.evidence("Anything at all.") == {}
        assert c.check("Anything at all.", "Family Law") is None
