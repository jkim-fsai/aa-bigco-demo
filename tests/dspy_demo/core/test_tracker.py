"""Tests for dspy_demo/core/tracker.py."""

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dspy_demo.core.tracker import LogPatterns, OptimizationTracker


class TestLogPatterns:
    """Tests for LogPatterns regex patterns."""

    def test_gepa_instruction_pattern(self):
        """Test GEPA instruction pattern matching."""
        msg = "Iteration 5: Proposed new text for generate_answer.predict: Given context, answer the question."
        match = LogPatterns.GEPA_INSTRUCTION.search(msg)
        assert match is not None
        assert match.group(1) == "5"
        assert "Given context" in match.group(2)

    def test_gepa_instruction_multiline(self):
        """Test GEPA instruction pattern with multiline text."""
        msg = """Iteration 10: Proposed new text for generate_answer.predict: You are given:
1. A context
2. A question

Provide the answer."""
        match = LogPatterns.GEPA_INSTRUCTION.search(msg)
        assert match is not None
        assert match.group(1) == "10"

    def test_mipro_instruction_pattern(self):
        """Test MIPROv2 instruction pattern matching."""
        msg = "0: Given the fields context and question, produce answer."
        match = LogPatterns.MIPRO_INSTRUCTION.match(msg)
        assert match is not None
        assert match.group(1) == "0"
        assert "Given the fields" in match.group(2)

    def test_mipro_score_pattern(self):
        """Test MIPROv2 score pattern matching."""
        msg = "Score: 72.5 with parameters ['instruction_1']."
        match = LogPatterns.MIPRO_SCORE.search(msg)
        assert match is not None
        assert match.group(1) == "72.5"

    def test_mipro_score_minibatch_pattern(self):
        """Test MIPROv2 score pattern with minibatch."""
        msg = "Score: 68.0 on minibatch of size 35 with parameters ['instruction_2']."
        match = LogPatterns.MIPRO_SCORE.search(msg)
        assert match is not None
        assert match.group(1) == "68.0"

    def test_gepa_iteration_pattern(self):
        """Test GEPA iteration score pattern matching."""
        msg = "Iteration 15: Valset score for new program: 0.725"
        match = LogPatterns.GEPA_ITERATION.search(msg)
        assert match is not None
        assert match.group(1) == "15"
        assert match.group(2) == "0.725"

    def test_gepa_best_score_pattern(self):
        """Test GEPA best score pattern matching."""
        msg = "Iteration 20: Best score on valset: 0.78"
        match = LogPatterns.GEPA_ITERATION.search(msg)
        assert match is not None
        assert match.group(1) == "20"
        assert match.group(2) == "0.78"

    def test_default_program_score_pattern(self):
        """Test default program score pattern matching."""
        msg = "Default program score: 66.5"
        match = LogPatterns.DEFAULT_OR_BEST.search(msg)
        assert match is not None
        assert match.group(1) == "66.5"


class TestOptimizationTracker:
    """Tests for OptimizationTracker class."""

    def test_init_creates_output_dir(self, temp_dir):
        """Test that init creates output directory."""
        output_dir = temp_dir / "runs"
        OptimizationTracker(output_dir=output_dir)
        assert output_dir.exists()

    def test_init_sets_run_id(self, temp_dir):
        """Test that init sets a run ID."""
        tracker = OptimizationTracker(output_dir=temp_dir)
        assert tracker.run_id is not None
        assert len(tracker.run_id) == 15  # YYYYMMDD_HHMMSS

    def test_reset_clears_state(self, temp_dir):
        """Test that reset clears all state."""
        tracker = OptimizationTracker(output_dir=temp_dir)
        tracker.trials = [{"trial": 1}]
        tracker.instructions = [{"index": 1}]
        tracker._written_trial_ids = {"trial_1", "trial_2"}

        tracker.reset()

        assert tracker.trials == []
        assert tracker.instructions == []
        assert tracker._written_trial_ids == set()
        # run_id is regenerated (timestamp-based, so may be same if within same second)

    def test_open_close_jsonl(self, temp_dir):
        """Test JSONL file open/close."""
        tracker = OptimizationTracker(output_dir=temp_dir)
        tracker.open_jsonl()

        assert tracker._jsonl_file is not None
        assert tracker.trials_jsonl_path.exists()

        tracker.close_jsonl()

        assert tracker._jsonl_file is None

        # Verify file has metadata entries
        with open(tracker.trials_jsonl_path) as f:
            lines = f.readlines()
        assert len(lines) >= 2  # At least start and complete metadata

        first_entry = json.loads(lines[0])
        assert first_entry["type"] == "metadata"
        assert first_entry["status"] == "started"

        last_entry = json.loads(lines[-1])
        assert last_entry["type"] == "metadata"
        assert last_entry["status"] == "completed"

    def test_emit_gepa_instruction(self, temp_dir):
        """Test emit captures GEPA instructions."""
        tracker = OptimizationTracker(output_dir=temp_dir)

        record = MagicMock()
        record.getMessage.return_value = (
            "Iteration 5: Proposed new text for generate_answer.predict: "
            "Answer the question based on context."
        )

        tracker.emit(record)

        assert len(tracker.instructions) == 1
        assert tracker.instructions[0]["type"] == "gepa"
        assert tracker.instructions[0]["iteration"] == 5
        assert "Answer the question" in tracker.instructions[0]["instruction"]

    def test_emit_gepa_iteration_score(self, temp_dir):
        """Test emit captures GEPA iteration scores."""
        tracker = OptimizationTracker(output_dir=temp_dir)
        tracker.open_jsonl()

        record = MagicMock()
        record.getMessage.return_value = (
            "Iteration 10: Valset score for new program: 0.725"
        )

        tracker.emit(record)
        tracker.close_jsonl()

        assert len(tracker.trials) == 1
        assert tracker.trials[0]["trial"] == 10
        assert tracker.trials[0]["score"] == 72.5  # Converted to percentage
        assert tracker.trials[0]["optimizer"] == "gepa"

    def test_emit_mipro_score(self, temp_dir):
        """Test emit captures MIPROv2 scores."""
        tracker = OptimizationTracker(output_dir=temp_dir)
        tracker.open_jsonl()

        record = MagicMock()
        record.getMessage.return_value = (
            "Score: 68.5 with parameters ['instruction_1']."
        )

        tracker.emit(record)
        tracker.close_jsonl()

        assert len(tracker.trials) == 1
        assert tracker.trials[0]["score"] == 68.5
        assert tracker.trials[0]["optimizer"] == "mipro"
        assert tracker.trials[0]["eval_type"] == "full"

    def test_emit_mipro_minibatch_score(self, temp_dir):
        """Test emit captures MIPROv2 minibatch scores."""
        tracker = OptimizationTracker(output_dir=temp_dir)
        tracker.open_jsonl()

        record = MagicMock()
        record.getMessage.return_value = (
            "Score: 65.0 on minibatch of size 35 with parameters ['instruction_1']."
        )

        tracker.emit(record)
        tracker.close_jsonl()

        assert len(tracker.trials) == 1
        assert tracker.trials[0]["eval_type"] == "minibatch"

    def test_trial_deduplication(self, temp_dir):
        """Test that duplicate trials are not written to JSONL."""
        tracker = OptimizationTracker(output_dir=temp_dir)
        tracker.open_jsonl()

        # Emit same score twice
        record = MagicMock()
        record.getMessage.return_value = (
            "Iteration 5: Valset score for new program: 0.70"
        )

        tracker.emit(record)
        tracker.emit(record)

        tracker.close_jsonl()

        # Read JSONL and count trial entries
        with open(tracker.trials_jsonl_path) as f:
            lines = f.readlines()
        trial_entries = [
            json.loads(l) for l in lines if json.loads(l)["type"] == "trial"
        ]

        assert len(trial_entries) == 1  # Only one, not two

    def test_get_summary(self, temp_dir):
        """Test get_summary returns correct structure."""
        tracker = OptimizationTracker(output_dir=temp_dir)

        tracker.instructions = [{"index": 1, "instruction": "test"}]
        tracker.trials = [
            {"trial": 1, "score": 65.0},
            {"trial": 2, "score": 70.0},
        ]

        summary = tracker.get_summary()

        assert "instructions_proposed" in summary
        assert "trials" in summary
        assert "best_trial" in summary
        assert summary["best_trial"]["score"] == 70.0

    def test_get_summary_empty(self, temp_dir):
        """Test get_summary with no data."""
        tracker = OptimizationTracker(output_dir=temp_dir)
        summary = tracker.get_summary()

        assert summary["instructions_proposed"] == []
        assert summary["trials"] == []
        assert summary["best_trial"] is None

    def test_is_logging_handler(self, temp_dir):
        """Test that tracker is a logging.Handler subclass."""
        tracker = OptimizationTracker(output_dir=temp_dir)
        assert isinstance(tracker, logging.Handler)
