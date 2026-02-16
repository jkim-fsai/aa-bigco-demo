"""Tests for dspy_demo/config.py."""

from pathlib import Path

import pytest

from dspy_demo.config import (
    COLORS,
    DATASET_CONFIG,
    MODEL_CONFIG,
    OPTIMIZER_CONFIG,
    PATH_CONFIG,
    PROCESSING_CONFIG,
    DatasetConfig,
    ModelConfig,
    OptimizerConfig,
    PathConfig,
    ProcessingConfig,
)


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ModelConfig()
        assert config.default_model == "openai/gpt-4.1-nano"
        assert config.reflection_temperature == 1.0
        assert config.summary_temperature == 0.3
        assert config.async_max == 50

    def test_immutability(self):
        """Test that frozen dataclass is immutable."""
        config = ModelConfig()
        with pytest.raises(AttributeError):
            config.default_model = "other-model"

    def test_global_instance(self):
        """Test global MODEL_CONFIG instance."""
        assert MODEL_CONFIG.default_model == "openai/gpt-4.1-nano"


class TestDatasetConfig:
    """Tests for DatasetConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DatasetConfig()
        assert config.dataset_name == "hotpotqa/hotpot_qa"
        assert config.dataset_config == "distractor"
        assert config.train_slice == "train[:9045]"
        assert config.val_slice == "validation[:370]"
        assert config.test_slice == "validation[370:740]"

    def test_global_instance(self):
        """Test global DATASET_CONFIG instance."""
        assert DATASET_CONFIG.train_slice == "train[:9045]"


class TestOptimizerConfig:
    """Tests for OptimizerConfig dataclass."""

    def test_gepa_defaults(self):
        """Test GEPA optimizer defaults."""
        config = OptimizerConfig()
        assert config.gepa_auto == "medium"
        assert config.gepa_num_threads == 10

    def test_mipro_defaults(self):
        """Test MIPROv2 optimizer defaults."""
        config = OptimizerConfig()
        assert config.mipro_auto == "light"
        assert config.mipro_num_threads == 10
        assert config.mipro_num_trials == 30
        assert config.mipro_max_bootstrapped_demos == 3
        assert config.mipro_max_labeled_demos == 3

    def test_global_instance(self):
        """Test global OPTIMIZER_CONFIG instance."""
        assert OPTIMIZER_CONFIG.gepa_auto == "medium"


class TestProcessingConfig:
    """Tests for ProcessingConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ProcessingConfig()
        assert config.max_instruction_chars == 400_000
        assert config.instruction_truncation_limit == 2000
        assert config.demo_question_limit == 100
        assert config.demo_reasoning_limit == 200

    def test_global_instance(self):
        """Test global PROCESSING_CONFIG instance."""
        assert PROCESSING_CONFIG.max_instruction_chars == 400_000


class TestPathConfig:
    """Tests for PathConfig dataclass."""

    def test_default_paths_are_paths(self):
        """Test that path fields are Path objects."""
        config = PathConfig()
        assert isinstance(config.base_dir, Path)
        assert isinstance(config.runs_dir, Path)
        assert isinstance(config.results_file, Path)

    def test_results_file_for_optimizer(self):
        """Test optimizer-specific results file path generation."""
        config = PathConfig()
        gepa_path = config.results_file_for_optimizer("gepa")
        assert gepa_path == Path("optimization_results_gepa.json")

        mipro_path = config.results_file_for_optimizer("MIPRO")
        assert mipro_path == Path("optimization_results_mipro.json")

    def test_global_instance(self):
        """Test global PATH_CONFIG instance."""
        assert isinstance(PATH_CONFIG.base_dir, Path)


class TestColors:
    """Tests for color scheme constants."""

    def test_all_colors_defined(self):
        """Test that all expected colors are defined."""
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

    def test_colors_are_hex(self):
        """Test that all colors are valid hex codes."""
        for name, color in COLORS.items():
            assert color.startswith("#"), f"{name} color should start with #"
            assert len(color) == 7, f"{name} color should be 7 chars (#RRGGBB)"
