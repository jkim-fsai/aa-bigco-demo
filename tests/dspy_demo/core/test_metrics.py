"""Tests for dspy_demo/core/metrics.py."""

from unittest.mock import MagicMock

import pytest

from dspy_demo.core.metrics import (
    gepa_boolean_metric,
    gepa_metric,
    validate_answer,
    validate_boolean_answer,
)


class TestValidateAnswer:
    """Tests for validate_answer function."""

    def test_exact_match(self, mock_example, mock_prediction):
        """Test exact answer match."""
        mock_example.answer = "Paris"
        mock_prediction.answer = "Paris"
        assert validate_answer(mock_example, mock_prediction) is True

    def test_answer_contained_in_prediction(self, mock_example, mock_prediction):
        """Test answer contained within prediction."""
        mock_example.answer = "Paris"
        mock_prediction.answer = "The capital of France is Paris."
        assert validate_answer(mock_example, mock_prediction) is True

    def test_case_insensitive(self, mock_example, mock_prediction):
        """Test case insensitive matching."""
        mock_example.answer = "PARIS"
        mock_prediction.answer = "The answer is paris."
        assert validate_answer(mock_example, mock_prediction) is True

    def test_wrong_answer(self, mock_example, mock_prediction_wrong):
        """Test wrong answer returns False."""
        mock_example.answer = "Paris"
        assert validate_answer(mock_example, mock_prediction_wrong) is False

    def test_partial_match_not_substring(self, mock_example, mock_prediction):
        """Test that partial matches that aren't substrings fail."""
        mock_example.answer = "Paris"
        mock_prediction.answer = "The answer is Par."
        assert validate_answer(mock_example, mock_prediction) is False

    def test_with_trace_parameter(self, mock_example, mock_prediction):
        """Test that trace parameter doesn't affect result."""
        mock_example.answer = "Paris"
        mock_prediction.answer = "Paris"
        assert (
            validate_answer(mock_example, mock_prediction, trace="some_trace") is True
        )


class TestGepaMetric:
    """Tests for gepa_metric function."""

    def test_correct_answer_returns_one(self, mock_example, mock_prediction):
        """Test correct answer returns 1.0."""
        mock_example.answer = "Paris"
        mock_prediction.answer = "The capital is Paris."
        result = gepa_metric(mock_example, mock_prediction)
        assert result == 1.0

    def test_wrong_answer_returns_zero(self, mock_example, mock_prediction_wrong):
        """Test wrong answer returns 0.0."""
        mock_example.answer = "Paris"
        result = gepa_metric(mock_example, mock_prediction_wrong)
        assert result == 0.0

    def test_accepts_five_parameters(self, mock_example, mock_prediction):
        """Test that function accepts all 5 GEPA-required parameters."""
        mock_example.answer = "Paris"
        mock_prediction.answer = "Paris"
        result = gepa_metric(
            mock_example,
            mock_prediction,
            trace=None,
            student_code="some_code",
            teacher_code="other_code",
        )
        assert result == 1.0

    def test_returns_float(self, mock_example, mock_prediction):
        """Test that return value is float type."""
        mock_example.answer = "Paris"
        mock_prediction.answer = "Paris"
        result = gepa_metric(mock_example, mock_prediction)
        assert isinstance(result, float)


class TestValidateBooleanAnswer:
    """Tests for validate_boolean_answer function."""

    def test_exact_yes(self, mock_boolean_example, mock_boolean_prediction):
        """Test exact 'yes' match."""
        mock_boolean_example.answer = "yes"
        mock_boolean_prediction.answer = "yes"
        assert (
            validate_boolean_answer(mock_boolean_example, mock_boolean_prediction)
            is True
        )

    def test_exact_no(self, mock_boolean_example, mock_boolean_prediction):
        """Test exact 'no' match."""
        mock_boolean_example.answer = "no"
        mock_boolean_prediction.answer = "no"
        assert (
            validate_boolean_answer(mock_boolean_example, mock_boolean_prediction)
            is True
        )

    def test_wrong_answer(self, mock_boolean_example, mock_boolean_prediction_wrong):
        """Test wrong boolean answer returns False."""
        mock_boolean_example.answer = "no"
        assert (
            validate_boolean_answer(mock_boolean_example, mock_boolean_prediction_wrong)
            is False
        )

    def test_extracts_yes_from_verbose(
        self, mock_boolean_example, mock_boolean_prediction
    ):
        """Test extraction of 'yes' from verbose prediction."""
        mock_boolean_example.answer = "yes"
        mock_boolean_prediction.answer = "Based on my reasoning, yes, this is correct."
        assert (
            validate_boolean_answer(mock_boolean_example, mock_boolean_prediction)
            is True
        )

    def test_extracts_no_from_verbose(
        self, mock_boolean_example, mock_boolean_prediction
    ):
        """Test extraction of 'no' from verbose prediction."""
        mock_boolean_example.answer = "no"
        mock_boolean_prediction.answer = (
            "After careful analysis, no, that is incorrect."
        )
        assert (
            validate_boolean_answer(mock_boolean_example, mock_boolean_prediction)
            is True
        )

    def test_no_false_positive_from_substring(
        self, mock_boolean_example, mock_boolean_prediction
    ):
        """Test that 'no' in 'not' or 'know' doesn't cause false positives."""
        mock_boolean_example.answer = "yes"
        mock_boolean_prediction.answer = "I do not know the answer."
        # Should extract "not" -> cleaned "not" is not "yes"/"no", then "know" -> not match
        # Falls through to exact match: "i do not know the answer." != "yes"
        assert (
            validate_boolean_answer(mock_boolean_example, mock_boolean_prediction)
            is False
        )

    def test_case_insensitive(self, mock_boolean_example, mock_boolean_prediction):
        """Test case insensitive matching."""
        mock_boolean_example.answer = "yes"
        mock_boolean_prediction.answer = "Yes"
        assert (
            validate_boolean_answer(mock_boolean_example, mock_boolean_prediction)
            is True
        )

    def test_with_punctuation(self, mock_boolean_example, mock_boolean_prediction):
        """Test extraction handles trailing punctuation."""
        mock_boolean_example.answer = "no"
        mock_boolean_prediction.answer = "no."
        assert (
            validate_boolean_answer(mock_boolean_example, mock_boolean_prediction)
            is True
        )

    def test_with_trace_parameter(self, mock_boolean_example, mock_boolean_prediction):
        """Test that trace parameter doesn't affect result."""
        mock_boolean_example.answer = "yes"
        mock_boolean_prediction.answer = "yes"
        assert (
            validate_boolean_answer(
                mock_boolean_example, mock_boolean_prediction, trace="trace"
            )
            is True
        )


class TestGepaBooleanMetric:
    """Tests for gepa_boolean_metric function."""

    def test_correct_returns_one(self, mock_boolean_example, mock_boolean_prediction):
        """Test correct boolean answer returns 1.0."""
        mock_boolean_example.answer = "no"
        mock_boolean_prediction.answer = "no"
        assert gepa_boolean_metric(mock_boolean_example, mock_boolean_prediction) == 1.0

    def test_wrong_returns_zero(
        self, mock_boolean_example, mock_boolean_prediction_wrong
    ):
        """Test wrong boolean answer returns 0.0."""
        mock_boolean_example.answer = "no"
        assert (
            gepa_boolean_metric(mock_boolean_example, mock_boolean_prediction_wrong)
            == 0.0
        )

    def test_accepts_five_parameters(
        self, mock_boolean_example, mock_boolean_prediction
    ):
        """Test that function accepts all 5 GEPA-required parameters."""
        mock_boolean_example.answer = "yes"
        mock_boolean_prediction.answer = "yes"
        result = gepa_boolean_metric(
            mock_boolean_example,
            mock_boolean_prediction,
            trace=None,
            student_code="code",
            teacher_code="code",
        )
        assert result == 1.0

    def test_returns_float(self, mock_boolean_example, mock_boolean_prediction):
        """Test that return value is float type."""
        mock_boolean_example.answer = "yes"
        mock_boolean_prediction.answer = "yes"
        result = gepa_boolean_metric(mock_boolean_example, mock_boolean_prediction)
        assert isinstance(result, float)
