"""Tests for log_viz/data_loader.py."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# We need to mock streamlit before importing data_loader
import sys

# Create a proper mock for streamlit that handles cache_data decorator
mock_streamlit = MagicMock()
# Make cache_data a decorator that returns the function unchanged
mock_streamlit.cache_data = lambda **kwargs: lambda func: func
sys.modules["streamlit"] = mock_streamlit

from log_viz.data_loader import TrialDataLoader


class TestTrialDataLoader:
    """Tests for TrialDataLoader class."""

    def test_init(self):
        """Test TrialDataLoader initialization."""
        loader = TrialDataLoader()
        assert loader.last_position == {}

    def test_load_jsonl_incremental_file_not_exists(self, temp_dir):
        """Test loading non-existent file returns empty DataFrame."""
        with patch("log_viz.data_loader.RUNS_DIR", temp_dir):
            loader = TrialDataLoader()
            df = loader.load_jsonl_incremental("nonexistent_run")

            assert df.empty

    def test_load_jsonl_incremental(self, temp_jsonl_file, temp_dir):
        """Test incremental loading of JSONL file."""
        with patch("log_viz.data_loader.RUNS_DIR", temp_dir):
            loader = TrialDataLoader()
            df = loader.load_jsonl_incremental("trials_20260212_100000")

            # Should have 5 trial entries (from fixture)
            assert len(df) == 5

            # Should have correct columns
            assert "trial" in df.columns
            assert "score" in df.columns

    def test_load_jsonl_incremental_tracks_position(self, temp_jsonl_file, temp_dir):
        """Test that position is tracked between reads."""
        with patch("log_viz.data_loader.RUNS_DIR", temp_dir):
            loader = TrialDataLoader()

            # First read
            loader.load_jsonl_incremental("trials_20260212_100000")
            first_position = loader.last_position.get("trials_20260212_100000", 0)

            # Position should be set
            assert first_position > 0

            # Second read should start from last position
            df = loader.load_jsonl_incremental("trials_20260212_100000")

            # No new data, so empty
            assert len(df) == 0

    def test_load_jsonl_full_resets_position(self, temp_jsonl_file, temp_dir):
        """Test that full load resets position."""
        with patch("log_viz.data_loader.RUNS_DIR", temp_dir):
            loader = TrialDataLoader()

            # First incremental read
            loader.load_jsonl_incremental("trials_20260212_100000")

            # Full load should reset and read everything
            df = loader.load_jsonl_full("trials_20260212_100000")

            assert len(df) == 5

    def test_load_historical_results(self, temp_results_file, temp_dir):
        """Test loading historical results JSON."""
        with patch("log_viz.data_loader.HISTORICAL_RESULTS", temp_results_file):
            loader = TrialDataLoader()
            result = loader.load_historical_results()

            assert result is not None
            assert result["optimizer"] == "gepa"
            assert result["baseline_accuracy"] == 65.0

    def test_load_historical_results_not_exists(self, temp_dir):
        """Test loading non-existent historical results."""
        with patch("log_viz.data_loader.HISTORICAL_RESULTS", temp_dir / "nonexistent.json"):
            loader = TrialDataLoader()
            result = loader.load_historical_results()

            assert result is None

    def test_load_result_file(self, temp_results_file):
        """Test loading a specific result file."""
        loader = TrialDataLoader()
        result = loader.load_result_file(temp_results_file)

        assert result is not None
        assert result["optimizer"] == "gepa"

    def test_load_result_file_not_exists(self, temp_dir):
        """Test loading non-existent result file."""
        loader = TrialDataLoader()
        result = loader.load_result_file(temp_dir / "nonexistent.json")

        assert result is None

    def test_get_run_metadata(self, temp_jsonl_file, temp_dir):
        """Test extracting metadata from run."""
        with patch("log_viz.data_loader.RUNS_DIR", temp_dir):
            loader = TrialDataLoader()
            metadata = loader.get_run_metadata("trials_20260212_100000")

            assert metadata["type"] == "metadata"
            assert metadata["run_id"] == "20260212_100000"
            assert metadata["status"] == "started"

    def test_get_run_metadata_not_exists(self, temp_dir):
        """Test metadata extraction for non-existent run."""
        with patch("log_viz.data_loader.RUNS_DIR", temp_dir):
            loader = TrialDataLoader()
            metadata = loader.get_run_metadata("nonexistent_run")

            assert metadata == {}

    def test_handles_malformed_json(self, temp_dir):
        """Test handling of malformed JSON lines."""
        # Create file with malformed line
        jsonl_path = temp_dir / "trials_malformed.jsonl"
        with open(jsonl_path, "w") as f:
            f.write('{"type": "metadata", "run_id": "test", "status": "started"}\n')
            f.write("not valid json\n")  # Malformed
            f.write('{"type": "trial", "trial": 1, "score": 70.0}\n')

        with patch("log_viz.data_loader.RUNS_DIR", temp_dir):
            loader = TrialDataLoader()
            df = loader.load_jsonl_incremental("trials_malformed")

            # Should have 1 trial (skipped malformed line)
            assert len(df) == 1
