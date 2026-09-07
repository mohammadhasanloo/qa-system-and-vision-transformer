"""Tests for the SQuAD-style scorer."""

from __future__ import annotations

import pytest

from qa_system.metrics import exact_match, normalise, score, token_f1


def test_normalise_collapses_whitespace_and_case():
    assert normalise("  The   ANSWER ") == "answer"


def test_normalise_unifies_arabic_and_persian_letterforms():
    """Same word, different code points; a prediction must not fail on spelling."""
    assert normalise("كيف") == normalise("کیف")


def test_normalise_strips_punctuation_in_both_scripts():
    assert normalise("تهران،") == normalise("تهران")
    assert normalise("Paris.") == normalise("Paris")


def test_exact_match_is_all_or_nothing():
    assert exact_match("Tehran", "tehran") == 1.0
    assert exact_match("Tehran city", "Tehran") == 0.0


def test_exact_match_takes_the_best_of_several_references():
    assert exact_match("Tehran", ["Isfahan", "Tehran"]) == 1.0


def test_f1_rewards_partial_overlap():
    # 1 shared token; precision 1/2, recall 1/1 -> F1 = 2/3
    assert token_f1("Tehran city", "Tehran") == pytest.approx(2 / 3)


def test_f1_is_zero_without_overlap():
    assert token_f1("Isfahan", "Tehran") == 0.0


def test_empty_prediction_only_matches_an_empty_reference():
    assert token_f1("", "") == 1.0
    assert token_f1("", "Tehran") == 0.0


def test_score_reports_percentages():
    result = score(["Tehran", "Isfahan"], ["Tehran", "Tehran"])
    assert result["exact_match"] == pytest.approx(50.0)
    assert result["f1"] == pytest.approx(50.0)


def test_score_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        score(["a"], ["a", "b"])
