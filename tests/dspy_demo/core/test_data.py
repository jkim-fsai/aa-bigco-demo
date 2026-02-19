"""Tests for dspy_demo/core/data.py."""

from unittest.mock import MagicMock, patch

import pytest

from dspy_demo.config import STRATEGYQA_DATASET_CONFIG
from dspy_demo.core.data import DataLoader, format_context


class TestFormatContext:
    """Tests for format_context function."""

    def test_single_paragraph(self):
        """Test formatting single paragraph."""
        ctx = {
            "title": ["Paris"],
            "sentences": [
                ["Paris is the capital of France.", " It has the Eiffel Tower."]
            ],
        }
        result = format_context(ctx)
        assert "[Paris]" in result
        assert "Paris is the capital of France." in result
        assert "It has the Eiffel Tower." in result

    def test_multiple_paragraphs(self):
        """Test formatting multiple paragraphs."""
        ctx = {
            "title": ["Paris", "London"],
            "sentences": [
                ["Paris is in France."],
                ["London is in England."],
            ],
        }
        result = format_context(ctx)
        assert "[Paris]" in result
        assert "[London]" in result
        assert "Paris is in France." in result
        assert "London is in England." in result

    def test_paragraphs_separated(self):
        """Test that paragraphs are separated by double newlines."""
        ctx = {
            "title": ["A", "B"],
            "sentences": [["Text A."], ["Text B."]],
        }
        result = format_context(ctx)
        assert "\n\n" in result

    def test_empty_context(self):
        """Test formatting empty context."""
        ctx = {"title": [], "sentences": []}
        result = format_context(ctx)
        assert result == ""


class TestDataLoader:
    """Tests for DataLoader class."""

    def test_init_no_side_effects(self):
        """Test that DataLoader init doesn't load data."""
        loader = DataLoader()
        assert loader._trainset is None
        assert loader._valset is None
        assert loader._testset is None

    def test_uses_default_config(self):
        """Test that default config is used."""
        loader = DataLoader()
        assert loader._config.dataset_name == "hotpotqa/hotpot_qa"

    def test_accepts_custom_config(self):
        """Test that custom config can be provided."""
        from dspy_demo.config import DatasetConfig

        custom_config = DatasetConfig(
            dataset_name="custom/dataset",
            train_slice="train[:10]",
        )
        loader = DataLoader(config=custom_config)
        assert loader._config.dataset_name == "custom/dataset"
        assert loader._config.train_slice == "train[:10]"

    @patch("dspy_demo.core.data.load_dataset")
    def test_trainset_lazy_loads(self, mock_load_dataset):
        """Test that trainset property triggers lazy loading."""
        mock_ds = [
            {
                "question": "What is X?",
                "context": {"title": ["Test"], "sentences": [["Test sentence."]]},
                "answer": "Y",
            }
        ]
        mock_load_dataset.return_value = mock_ds

        loader = DataLoader()

        # Access trainset
        result = loader.trainset

        # Verify dataset was loaded
        mock_load_dataset.assert_called_once()
        assert len(result) == 1

    @patch("dspy_demo.core.data.load_dataset")
    def test_trainset_cached(self, mock_load_dataset):
        """Test that trainset is cached after first access."""
        mock_ds = [
            {
                "question": "Q",
                "context": {"title": ["T"], "sentences": [["S"]]},
                "answer": "A",
            }
        ]
        mock_load_dataset.return_value = mock_ds

        loader = DataLoader()

        # Access trainset twice
        _ = loader.trainset
        _ = loader.trainset

        # Should only load once
        assert mock_load_dataset.call_count == 1

    @patch("dspy_demo.core.data.load_dataset")
    def test_preload_all(self, mock_load_dataset):
        """Test preload_all loads all datasets."""
        mock_ds = [
            {
                "question": "Q",
                "context": {"title": ["T"], "sentences": [["S"]]},
                "answer": "A",
            }
        ]
        mock_load_dataset.return_value = mock_ds

        loader = DataLoader()
        result = loader.preload_all()

        # Should return self for chaining
        assert result is loader

        # Should have called load_dataset 3 times (train, val, test)
        assert mock_load_dataset.call_count == 3

    @patch("dspy_demo.core.data.load_dataset")
    def test_examples_have_correct_structure(self, mock_load_dataset):
        """Test that loaded examples have correct structure."""
        mock_ds = [
            {
                "question": "What is the capital?",
                "context": {
                    "title": ["France"],
                    "sentences": [["Paris is the capital."]],
                },
                "answer": "Paris",
            }
        ]
        mock_load_dataset.return_value = mock_ds

        loader = DataLoader()
        examples = loader.trainset

        assert len(examples) == 1
        example = examples[0]

        assert hasattr(example, "question")
        assert hasattr(example, "context")
        assert hasattr(example, "answer")
        assert example.question == "What is the capital?"
        assert "Paris is the capital." in example.context
        assert example.answer == "Paris"


class TestDataLoaderStrategyQA:
    """Tests for DataLoader with StrategyQA dataset."""

    @patch("dspy_demo.core.data.load_dataset")
    def test_loads_strategyqa_format(self, mock_load_dataset):
        """Test that StrategyQA examples are loaded correctly."""
        mock_ds = [
            {"question": "Did Aristotle use a laptop?", "answer": False},
            {"question": "Would a monocle suit a cyclops?", "answer": True},
        ]
        mock_load_dataset.return_value = mock_ds

        loader = DataLoader(config=STRATEGYQA_DATASET_CONFIG)
        examples = loader.trainset

        assert len(examples) == 2
        assert examples[0].question == "Did Aristotle use a laptop?"
        assert examples[0].answer == "no"
        assert examples[1].answer == "yes"

    @patch("dspy_demo.core.data.load_dataset")
    def test_no_context_field(self, mock_load_dataset):
        """Test that StrategyQA examples have no context field."""
        mock_ds = [
            {"question": "Can one spot helium?", "answer": False},
        ]
        mock_load_dataset.return_value = mock_ds

        loader = DataLoader(config=STRATEGYQA_DATASET_CONFIG)
        examples = loader.trainset

        assert hasattr(examples[0], "question")
        assert hasattr(examples[0], "answer")
        assert not hasattr(examples[0], "context")

    @patch("dspy_demo.core.data.load_dataset")
    def test_boolean_true_converts_to_yes(self, mock_load_dataset):
        """Test that boolean True converts to 'yes'."""
        mock_ds = [{"question": "Q?", "answer": True}]
        mock_load_dataset.return_value = mock_ds

        loader = DataLoader(config=STRATEGYQA_DATASET_CONFIG)
        assert loader.trainset[0].answer == "yes"

    @patch("dspy_demo.core.data.load_dataset")
    def test_boolean_false_converts_to_no(self, mock_load_dataset):
        """Test that boolean False converts to 'no'."""
        mock_ds = [{"question": "Q?", "answer": False}]
        mock_load_dataset.return_value = mock_ds

        loader = DataLoader(config=STRATEGYQA_DATASET_CONFIG)
        assert loader.trainset[0].answer == "no"

    @patch("dspy_demo.core.data.load_dataset")
    def test_no_dataset_config_passed(self, mock_load_dataset):
        """Test that load_dataset is called without config for StrategyQA."""
        mock_ds = [{"question": "Q?", "answer": True}]
        mock_load_dataset.return_value = mock_ds

        loader = DataLoader(config=STRATEGYQA_DATASET_CONFIG)
        _ = loader.trainset

        # Should be called with just dataset_name (no config arg)
        mock_load_dataset.assert_called_once_with(
            "ChilleD/StrategyQA",
            split="train[:1280]",
        )
