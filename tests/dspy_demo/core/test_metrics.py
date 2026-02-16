"""Tests for dspy_demo/core/metrics.py."""

from unittest.mock import MagicMock

import pytest

from dspy_demo.core.metrics import gepa_metric, validate_answer


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
        assert validate_answer(mock_example, mock_prediction, trace="some_trace") is True


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
