"""Tests for log_viz/utils/config.py."""

import pytest
from pathlib import Path


class TestLogVizConfig:
    """Tests for log_viz configuration constants."""

    def test_paths_are_path_objects(self):
        """Test that path constants are Path objects."""
        from log_viz.utils.config import BASE_DIR, RUNS_DIR, HISTORICAL_RESULTS

        assert isinstance(BASE_DIR, Path)
        assert isinstance(RUNS_DIR, Path)
        assert isinstance(HISTORICAL_RESULTS, Path)

    def test_runs_dir_under_base_dir(self):
        """Test that RUNS_DIR is under BASE_DIR."""
        from log_viz.utils.config import BASE_DIR, RUNS_DIR

        assert str(RUNS_DIR).startswith(str(BASE_DIR))

    def test_ui_config_values(self):
        """Test UI configuration values are reasonable."""
        from log_viz.utils.config import (
            REFRESH_INTERVAL_MS,
            MAX_DISPLAY_TRIALS,
            MAX_INSTRUCTION_DISPLAY,
            PLOT_HEIGHT,
            CARD_HEIGHT,
        )

        assert REFRESH_INTERVAL_MS > 0
        assert MAX_DISPLAY_TRIALS > 0
        assert MAX_INSTRUCTION_DISPLAY > 0
        assert PLOT_HEIGHT > 0
        assert CARD_HEIGHT > 0

    def test_cache_ttl_values(self):
        """Test cache TTL values are reasonable."""
        from log_viz.utils.config import (
            CACHE_TTL_RUNS,
            CACHE_TTL_HISTORICAL,
            CACHE_TTL_JSONL,
        )

        assert CACHE_TTL_RUNS > 0
        assert CACHE_TTL_HISTORICAL > 0
        assert CACHE_TTL_JSONL > 0

    def test_colors_defined(self):
        """Test that all expected colors are defined."""
        from log_viz.utils.config import COLORS

        expected_colors = [
            "primary",
            "secondary",
            "success",
            "warning",
            "error",
            "live_run",
            "historical",
        ]

        for color in expected_colors:
            assert color in COLORS
            assert COLORS[color].startswith("#")
