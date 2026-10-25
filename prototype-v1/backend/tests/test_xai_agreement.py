"""
tests/test_xai_agreement.py — Tests for services/xai_agreement.py.

The thing worth protecting here is the sign convention. A concept marker can
appear among the model's influential tokens and have pushed *against* the
label the model chose — "criminal" in a passage filed under Family Law is the
real case this was built on. Reporting that as "the model relied on Criminal
Law vocabulary" would be a confident, plausible, wrong explanation, which is
worse than no explanation at all.
"""

import pytest

from services import xai_agreement
from services.ontology_check import ontology_check


@pytest.fixture(autouse=True, scope="module")
def _markers():
    ontology_check.load()


def toks(*pairs):
    return [{"token": t, "importance": v} for t, v in pairs]


class TestGuards:
    def test_no_tokens_returns_none(self):
        assert xai_agreement.analyse("some text", "Contract Law", None) is None
        assert xai_agreement.analyse("some text", "Contract Law", []) is None

    def test_disabled_ontology_returns_none(self, monkeypatch):
        monkeypatch.setattr(ontology_check, "is_loaded", lambda: False)
        assert xai_agreement.analyse("x", "Contract Law", toks(("x", 1.0))) is None


class TestVerdicts:
    TAX = "The assessee filed a return under the Income Tax Act for the relevant year."

    def test_aligned_when_drivers_are_the_label_s_own_vocabulary(self):
        r = xai_agreement.analyse(self.TAX, "Taxation Law", toks(("tax ", 0.9), ("the ", 0.05)))
        assert r["verdict"] == "aligned"
        assert any(s["term"] == "tax" for s in r["supporting"])
        assert all(s["direction"] == "toward" for s in r["supporting"])

    def test_overruled_when_rival_vocabulary_pushed_against(self):
        text = ("The child witness gave testimony. The criminal case proceeded "
                "and the child was examined.")
        r = xai_agreement.analyse(text, "Family Law",
                                  toks(("child ", 0.6), ("criminal ", -0.4)))
        assert r["verdict"] == "overruled"
        rival = [c for c in r["contradicting"] if c["term"] == "criminal"]
        assert rival and rival[0]["direction"] == "against"

    def test_conflict_when_only_rival_vocabulary_pushed_toward(self):
        text = ("The child witness gave testimony. The criminal case proceeded "
                "and the child was examined.")
        r = xai_agreement.analyse(text, "Family Law",
                                  toks(("criminal ", 0.9), ("testimony ", 0.7)))
        assert r["verdict"] == "conflict"
        assert not [s for s in r["supporting"] if s["direction"] == "toward"]

    def test_no_overlap_when_drivers_are_not_markers(self):
        r = xai_agreement.analyse(self.TAX, "Taxation Law",
                                  toks(("susceptible ", 0.9), ("tutored ", 0.8)))
        assert r["verdict"] == "no_overlap"
        assert r["supporting"] == [] and r["contradicting"] == []

    def test_no_overlap_says_whether_the_text_has_markers_at_all(self):
        r = xai_agreement.analyse(self.TAX, "Taxation Law",
                                  toks(("susceptible ", 0.9), ("tutored ", 0.8)))
        assert "does contain concept markers" in r["summary"]
        r = xai_agreement.analyse("The quick brown fox jumps over the lazy dog.",
                                  "Taxation Law", toks(("fox ", 0.9), ("dog ", 0.5)))
        assert r["verdict"] == "no_overlap"
        assert "none of the concept markers" in r["summary"]


class TestMatching:
    def test_token_whitespace_and_hashes_are_stripped(self):
        # SHAP keeps trailing spaces and wordpiece marks; markers have neither.
        r = xai_agreement.analyse(
            "The assessee filed a return under the Income Tax Act.",
            "Taxation Law", toks(("  tax  ", 0.9), ("##tax", 0.1)),
        )
        assert any(s["term"] == "tax" for s in r["supporting"])

    def test_bigram_needs_every_word_present(self):
        text = "The assessee filed a return under the Income Tax Act for the year."
        only_one = xai_agreement.analyse(text, "Taxation Law", toks(("income ", 0.9)))
        both = xai_agreement.analyse(text, "Taxation Law",
                                     toks(("income ", 0.9), ("tax ", 0.9)))
        assert not [s for s in only_one["supporting"] if s["term"] == "income tax"]
        assert any(s["term"] == "income tax" for s in both["supporting"])

    def test_weak_tokens_are_ignored(self):
        # A marker the model barely looked at is not something it "relied on".
        r = xai_agreement.analyse(
            "The assessee filed a return under the Income Tax Act.",
            "Taxation Law", toks(("tax ", 0.001), ("unrelated ", 5.0)),
        )
        assert not [s for s in r["supporting"] if s["term"] == "tax"]


class TestShape:
    def test_record_fields_are_present(self):
        r = xai_agreement.analyse(
            "The assessee filed a return under the Income Tax Act.",
            "Taxation Law", toks(("tax ", 0.9)),
        )
        assert set(r) >= {"verdict", "summary", "supporting", "contradicting",
                          "predicted_label", "predicted_evidence",
                          "rival_label", "rival_evidence"}
        for row in r["supporting"]:
            assert set(row) == {"term", "domain", "shap_weight", "share", "direction"}
