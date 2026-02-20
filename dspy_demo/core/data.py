"""Dataset loading utilities with lazy initialization.

This module provides lazy-loading dataset functionality - no network calls
occur until data is actually accessed.
"""

from typing import Dict, List, Optional

from datasets import load_dataset
from dspy.primitives.example import Example

from ..config import DATASET_CONFIG, DatasetConfig


def format_context(ctx: Dict) -> str:
    """Format context titles and sentences into a readable string.

    Args:
        ctx: Dictionary with 'title' and 'sentences' keys.

    Returns:
        Formatted string with titled paragraphs.
    """
    paragraphs = []
    for title, sentences in zip(ctx["title"], ctx["sentences"]):
        text = "".join(sentences)
        paragraphs.append(f"[{title}]\n{text}")
    return "\n\n".join(paragraphs)


class DataLoader:
    """Lazy-loading dataset manager.

    Datasets are only loaded when first accessed, avoiding network calls on import.

    Example:
        loader = DataLoader()
        # No network calls yet
        train_data = loader.trainset  # Now loads from HuggingFace
    """

    def __init__(self, config: Optional[DatasetConfig] = None) -> None:
        """Initialize the data loader.

        Args:
            config: Dataset configuration. Defaults to DATASET_CONFIG.
        """
        self._config = config or DATASET_CONFIG
        self._trainset: Optional[List[Example]] = None
        self._valset: Optional[List[Example]] = None
        self._testset: Optional[List[Example]] = None

    def _load_split(self, split: str) -> List[Example]:
        """Load examples from the configured dataset.

        Routes to dataset-specific loading based on config.dataset_name.

        Args:
            split: Dataset split string (e.g., "train[:200]").

        Returns:
            List of Example objects with appropriate input fields.
        """
        load_args = [self._config.dataset_name]
        if self._config.dataset_config is not None:
            load_args.append(self._config.dataset_config)
        ds = load_dataset(*load_args, split=split)

        if self._config.dataset_name == "ChilleD/StrategyQA":
            return self._format_strategyqa(ds)
        return self._format_hotpotqa(ds)

    def _format_hotpotqa(self, ds) -> List[Example]:
        """Format HotPotQA dataset items into DSPy Examples.

        Args:
            ds: HuggingFace dataset split.

        Returns:
            List of Example objects with question, context, and answer.
        """
        examples = []
        for item in ds:
            ex = Example(
                question=item["question"],
                context=format_context(item["context"]),
                answer=item["answer"],
            ).with_inputs("question", "context")
            examples.append(ex)
        return examples

    def _format_strategyqa(self, ds) -> List[Example]:
        """Format StrategyQA dataset items into DSPy Examples.

        Converts boolean answers to "yes"/"no" strings.

        Args:
            ds: HuggingFace dataset split.

        Returns:
            List of Example objects with question and answer (no context).
        """
        examples = []
        for item in ds:
            answer = "yes" if item["answer"] else "no"
            ex = Example(
                question=item["question"],
                answer=answer,
            ).with_inputs("question")
            examples.append(ex)
        return examples

    @property
    def dataset_name(self) -> str:
        """Human-readable dataset name derived from config."""
        name = self._config.dataset_name
        # Map HuggingFace identifiers to short display names
        name_map = {
            "hotpotqa/hotpot_qa": "HotPotQA",
            "ChilleD/StrategyQA": "StrategyQA",
        }
        return name_map.get(name, name)

    @property
    def trainset(self) -> List[Example]:
        """Training dataset (lazy-loaded)."""
        if self._trainset is None:
            self._trainset = self._load_split(self._config.train_slice)
        return self._trainset

    @property
    def valset(self) -> List[Example]:
        """Validation dataset (lazy-loaded)."""
        if self._valset is None:
            self._valset = self._load_split(self._config.val_slice)
        return self._valset

    @property
    def testset(self) -> List[Example]:
        """Test dataset (lazy-loaded)."""
        if self._testset is None:
            self._testset = self._load_split(self._config.test_slice)
        return self._testset

    def preload_all(self) -> "DataLoader":
        """Preload all datasets.

        Returns:
            Self for method chaining.
        """
        _ = self.trainset
        _ = self.valset
        _ = self.testset
        return self
