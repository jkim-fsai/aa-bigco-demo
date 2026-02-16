"""Tests for dspy_demo/pipeline.py."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from dspy_demo.pipeline import (
    OptimizationPipeline,
    OptimizationResult,
    OptimizerType,
)


class TestOptimizerType:
    """Tests for OptimizerType enum."""

    def test_gepa_value(self):
        """Test GEPA enum value."""
        assert OptimizerType.GEPA.value == "gepa"

    def test_mipro_value(self):
        """Test MIPRO enum value."""
        assert OptimizerType.MIPRO.value == "mipro"

    def test_miprov2_value(self):
        """Test MIPROV2 enum value."""
        assert OptimizerType.MIPROV2.value == "miprov2"


class TestOptimizationResult:
    """Tests for OptimizationResult dataclass."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample optimization result."""
        return OptimizationResult(
            optimizer="gepa",
            baseline_accuracy=65.0,
            optimized_accuracy=72.5,
            improvement=7.5,
            instruction="Answer the question based on context.",
            demos=[{"question": "Q", "answer": "A"}],
            trials=[{"trial": 1, "score": 70.0}],
            instruction_candidates=[{"index": 1, "instruction": "test"}],
            evolution_summary="Optimization improved accuracy.",
        )

    def test_to_dict(self, sample_result):
        """Test to_dict returns correct structure."""
        result_dict = sample_result.to_dict()

        assert result_dict["optimizer"] == "gepa"
        assert result_dict["baseline_accuracy"] == 65.0
        assert result_dict["optimized_accuracy"] == 72.5
        assert result_dict["improvement"] == 7.5
        assert result_dict["instruction"] == "Answer the question based on context."
        assert len(result_dict["demos"]) == 1
        assert len(result_dict["optimization_trials"]) == 1
        assert len(result_dict["instruction_candidates"]) == 1
        assert "timestamp" in result_dict
        assert result_dict["evolution_summary"] == "Optimization improved accuracy."

    def test_save(self, sample_result, temp_dir):
        """Test save writes JSON file."""
        path = temp_dir / "results.json"
        result_path = sample_result.save(path)

        assert result_path == path
        assert path.exists()

        with open(path) as f:
            saved_data = json.load(f)

        assert saved_data["optimizer"] == "gepa"
        assert saved_data["baseline_accuracy"] == 65.0

    def test_save_default_path(self, sample_result):
        """Test save uses default path when none provided."""
        with patch.object(Path, "write_text") as mock_write:
            path = sample_result.save()
            assert path == Path("optimization_results.json")

    def test_timestamp_auto_generated(self):
        """Test timestamp is auto-generated."""
        result = OptimizationResult(
            optimizer="gepa",
            baseline_accuracy=65.0,
            optimized_accuracy=70.0,
            improvement=5.0,
            instruction=None,
            demos=[],
            trials=[],
            instruction_candidates=[],
        )
        assert result.timestamp is not None
        assert len(result.timestamp) > 0

    def test_evolution_summary_optional(self):
        """Test evolution_summary defaults to None."""
        result = OptimizationResult(
            optimizer="gepa",
            baseline_accuracy=65.0,
            optimized_accuracy=70.0,
            improvement=5.0,
            instruction=None,
            demos=[],
            trials=[],
            instruction_candidates=[],
        )
        assert result.evolution_summary is None


class TestOptimizationPipeline:
    """Tests for OptimizationPipeline class."""

    def test_init_no_side_effects(self):
        """Test that init doesn't configure DSPy or load data."""
        with patch("dspy_demo.pipeline.DataLoader") as MockLoader:
            MockLoader.return_value = MagicMock()
            pipeline = OptimizationPipeline()

            assert pipeline._configured is False
            # DataLoader should be created but not used
            MockLoader.assert_called_once()

    def test_init_accepts_custom_model(self):
        """Test that custom model can be provided."""
        with patch("dspy_demo.pipeline.DataLoader"):
            pipeline = OptimizationPipeline(model="custom/model")
            assert pipeline.model_name == "custom/model"

    def test_init_accepts_custom_async_max(self):
        """Test that custom async_max can be provided."""
        with patch("dspy_demo.pipeline.DataLoader"):
            pipeline = OptimizationPipeline(async_max=100)
            assert pipeline.async_max == 100

    def test_init_accepts_custom_data_loader(self):
        """Test that custom data loader can be provided."""
        custom_loader = MagicMock()
        pipeline = OptimizationPipeline(data_loader=custom_loader)
        assert pipeline.data_loader is custom_loader

    @patch("dspy_demo.pipeline.dspy")
    @patch("dspy_demo.pipeline.logging")
    def test_configure_sets_up_dspy(self, mock_logging, mock_dspy):
        """Test that configure sets up DSPy."""
        with patch("dspy_demo.pipeline.DataLoader"):
            pipeline = OptimizationPipeline()
            result = pipeline.configure()

            # Should return self for chaining
            assert result is pipeline

            # Should be configured
            assert pipeline._configured is True

            # Should configure DSPy
            mock_dspy.LM.assert_called()
            mock_dspy.configure.assert_called()

    @patch("dspy_demo.pipeline.dspy")
    @patch("dspy_demo.pipeline.logging")
    def test_configure_only_once(self, mock_logging, mock_dspy):
        """Test that configure only runs once."""
        with patch("dspy_demo.pipeline.DataLoader"):
            pipeline = OptimizationPipeline()
            pipeline.configure()
            pipeline.configure()

            # dspy.configure should only be called once
            assert mock_dspy.configure.call_count == 1

    @patch("dspy_demo.pipeline.dspy")
    def test_create_optimizer_gepa(self, mock_dspy):
        """Test GEPA optimizer creation."""
        with patch("dspy_demo.pipeline.DataLoader"):
            pipeline = OptimizationPipeline()
            mock_dspy.GEPA.return_value = MagicMock()

            optimizer = pipeline._create_optimizer(OptimizerType.GEPA)

            mock_dspy.GEPA.assert_called_once()
            assert optimizer is not None

    @patch("dspy_demo.pipeline.dspy")
    def test_create_optimizer_mipro(self, mock_dspy):
        """Test MIPROv2 optimizer creation."""
        with patch("dspy_demo.pipeline.DataLoader"):
            pipeline = OptimizationPipeline()
            mock_dspy.MIPROv2.return_value = MagicMock()

            optimizer = pipeline._create_optimizer(OptimizerType.MIPRO)

            mock_dspy.MIPROv2.assert_called_once()

    def test_extract_prompt_info_structure(self):
        """Test _extract_prompt_info returns correct structure."""
        with patch("dspy_demo.pipeline.DataLoader"):
            pipeline = OptimizationPipeline()

            mock_module = MagicMock()
            mock_module.generate_answer = MagicMock()
            mock_module.generate_answer.demos = []

            result = pipeline._extract_prompt_info(mock_module)

            assert "instruction" in result
            assert "demos" in result
            assert isinstance(result["demos"], list)

    def test_extract_prompt_info_with_demos(self):
        """Test _extract_prompt_info extracts demos."""
        with patch("dspy_demo.pipeline.DataLoader"):
            pipeline = OptimizationPipeline()

            mock_demo = MagicMock()
            mock_demo.question = "What is X?"
            mock_demo.answer = "Y"

            mock_module = MagicMock()
            mock_module.generate_answer.demos = [mock_demo]

            result = pipeline._extract_prompt_info(mock_module)

            assert len(result["demos"]) == 1
            assert result["demos"][0]["question"] == "What is X?"
            assert result["demos"][0]["answer"] == "Y"

    def test_generate_summary_deduplicates(self):
        """Test _generate_summary deduplicates candidates."""
        with patch("dspy_demo.pipeline.DataLoader"):
            with patch("dspy_demo.pipeline.dspy") as mock_dspy:
                mock_dspy.LM.return_value = MagicMock(return_value="Summary")

                pipeline = OptimizationPipeline()

                result = OptimizationResult(
                    optimizer="gepa",
                    baseline_accuracy=65.0,
                    optimized_accuracy=70.0,
                    improvement=5.0,
                    instruction=None,
                    demos=[],
                    trials=[{"trial": 1, "score": 70.0}],
                    instruction_candidates=[
                        {"index": 1, "iteration": 1, "instruction": "Test instruction"},
                        {"index": 1, "iteration": 1, "instruction": "Test instruction"},  # Duplicate
                    ],
                )

                summary = pipeline._generate_summary(result)

                # Should have generated a summary
                assert summary is not None
