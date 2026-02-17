"""Tests for log_viz/utils/instruction_grouping.py."""

import pytest
from typing import Any, Dict, List

from log_viz.utils.instruction_grouping import (
    deduplicate_instructions,
    group_by_optimizer_type,
    sort_by_iteration,
)


class TestDeduplicateInstructions:
    """Tests for deduplicate_instructions function."""

    def test_removes_duplicates(self, sample_instruction_candidates):
        """Test that duplicates are removed."""
        result = deduplicate_instructions(sample_instruction_candidates)

        # Original has 5 items with 1 duplicate
        assert len(result) == 4

    def test_preserves_order(self):
        """Test that order is preserved."""
        candidates = [
            {"index": 1, "iteration": 1, "instruction": "First"},
            {"index": 2, "iteration": 2, "instruction": "Second"},
            {"index": 3, "iteration": 3, "instruction": "Third"},
        ]
        result = deduplicate_instructions(candidates)

        assert result[0]["instruction"] == "First"
        assert result[1]["instruction"] == "Second"
        assert result[2]["instruction"] == "Third"

    def test_uses_iteration_for_key(self):
        """Test that iteration is used in dedup key."""
        candidates = [
            {"index": 1, "iteration": 1, "instruction": "Same text"},
            {
                "index": 2,
                "iteration": 2,
                "instruction": "Same text",
            },  # Different iteration
        ]
        result = deduplicate_instructions(candidates)

        # Should keep both since iterations are different
        assert len(result) == 2

    def test_falls_back_to_index(self):
        """Test fallback to index when iteration not present."""
        candidates = [
            {"index": 1, "instruction": "Text A"},
            {"index": 2, "instruction": "Text B"},
        ]
        result = deduplicate_instructions(candidates)

        assert len(result) == 2

    def test_handles_empty_list(self):
        """Test handling of empty list."""
        result = deduplicate_instructions([])
        assert result == []

    def test_uses_instruction_prefix(self):
        """Test that only first 100 chars are used for matching."""
        long_text = "A" * 150
        candidates = [
            {"index": 1, "iteration": 1, "instruction": long_text + "X"},
            {
                "index": 1,
                "iteration": 1,
                "instruction": long_text + "Y",
            },  # Same first 100 chars
        ]
        result = deduplicate_instructions(candidates)

        # Should be deduplicated since first 100 chars match
        assert len(result) == 1


class TestGroupByOptimizerType:
    """Tests for group_by_optimizer_type function."""

    def test_groups_by_type(self, sample_instruction_candidates):
        """Test grouping by optimizer type."""
        result = group_by_optimizer_type(sample_instruction_candidates)

        assert "gepa" in result
        assert "mipro" in result
        assert "other" in result

    def test_gepa_candidates_grouped(self, sample_instruction_candidates):
        """Test GEPA candidates are grouped correctly."""
        result = group_by_optimizer_type(sample_instruction_candidates)

        # 3 GEPA candidates in sample data
        assert len(result["gepa"]) == 3

    def test_mipro_candidates_grouped(self, sample_instruction_candidates):
        """Test MIPROv2 candidates are grouped correctly."""
        result = group_by_optimizer_type(sample_instruction_candidates)

        # 2 MIPRO candidates in sample data
        assert len(result["mipro"]) == 2

    def test_unknown_type_goes_to_other(self):
        """Test unknown types go to 'other' group."""
        candidates = [
            {"index": 1, "instruction": "Test", "type": "unknown_optimizer"},
        ]
        result = group_by_optimizer_type(candidates)

        assert len(result["other"]) == 1

    def test_missing_type_goes_to_other(self):
        """Test missing type goes to 'other' group."""
        candidates = [
            {"index": 1, "instruction": "Test"},  # No type field
        ]
        result = group_by_optimizer_type(candidates)

        assert len(result["other"]) == 1

    def test_case_insensitive(self):
        """Test type matching is case insensitive."""
        candidates = [
            {"index": 1, "instruction": "Test", "type": "GEPA"},
            {"index": 2, "instruction": "Test2", "type": "Mipro"},
        ]
        result = group_by_optimizer_type(candidates)

        assert len(result["gepa"]) == 1
        assert len(result["mipro"]) == 1

    def test_handles_empty_list(self):
        """Test handling of empty list."""
        result = group_by_optimizer_type([])

        assert result["gepa"] == []
        assert result["mipro"] == []
        assert result["other"] == []


class TestSortByIteration:
    """Tests for sort_by_iteration function."""

    def test_sorts_by_iteration(self):
        """Test sorting by iteration number."""
        candidates = [
            {"index": 3, "iteration": 3, "instruction": "Third"},
            {"index": 1, "iteration": 1, "instruction": "First"},
            {"index": 2, "iteration": 2, "instruction": "Second"},
        ]
        result = sort_by_iteration(candidates)

        assert result[0]["iteration"] == 1
        assert result[1]["iteration"] == 2
        assert result[2]["iteration"] == 3

    def test_falls_back_to_index(self):
        """Test fallback to index when iteration not present."""
        candidates = [
            {"index": 3, "instruction": "Third"},
            {"index": 1, "instruction": "First"},
            {"index": 2, "instruction": "Second"},
        ]
        result = sort_by_iteration(candidates)

        assert result[0]["index"] == 1
        assert result[1]["index"] == 2
        assert result[2]["index"] == 3

    def test_handles_empty_list(self):
        """Test handling of empty list."""
        result = sort_by_iteration([])
        assert result == []

    def test_does_not_modify_original(self):
        """Test that original list is not modified."""
        candidates = [
            {"index": 2, "iteration": 2, "instruction": "Second"},
            {"index": 1, "iteration": 1, "instruction": "First"},
        ]
        original_first = candidates[0]

        sort_by_iteration(candidates)

        # Original should be unchanged
        assert candidates[0] is original_first
        assert candidates[0]["iteration"] == 2
