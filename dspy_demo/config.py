"""Centralized configuration for the DSPy optimization demo.

This module contains all configuration constants organized into frozen dataclasses.
No side effects occur on import - only data structures are defined.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass(frozen=True)
class ModelConfig:
    """Language model configuration."""

    default_model: str = "openai/gpt-4.1-nano"
    reflection_model: str = "openai/gpt-4.1-nano"
    reflection_temperature: float = 1.0
    summary_temperature: float = 0.3
    async_max: int = 100


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset split configuration.

    HotPotQA (90,447 train / 7,405 dev):
    - Train: 90,447 examples
    - Val: 3,700 examples (first half of dev)
    - Test: 3,705 examples (second half of dev, held out)

    StrategyQA (1,600 train / 687 test):
    - Train: 1,280 examples (first 80% of train)
    - Val: 320 examples (last 20% of train)
    - Test: 687 examples (held out)
    """

    dataset_name: str = "hotpotqa/hotpot_qa"
    dataset_config: Optional[str] = "distractor"
    train_slice: str = "train"
    val_slice: str = "validation[:3700]"
    test_slice: str = "validation[3700:]"


@dataclass(frozen=True)
class OptimizerConfig:
    """Optimizer default parameters."""

    gepa_auto: str = "heavy"  # heavy+nano is fastest at +2.1%
    gepa_num_threads: int = 25
    gepa_reflection_minibatch_size: int = 3
    mipro_auto: str = "heavy"
    mipro_num_threads: int = 10
    mipro_num_trials: int = 30
    mipro_max_bootstrapped_demos: int = 3
    mipro_max_labeled_demos: int = 3


@dataclass(frozen=True)
class ProcessingConfig:
    """Text processing limits."""

    max_instruction_chars: int = 400_000
    instruction_truncation_limit: int = 2000
    demo_question_limit: int = 100
    demo_reasoning_limit: int = 200


@dataclass
class PathConfig:
    """File and directory paths."""

    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    runs_dir: Path = field(
        default_factory=lambda: Path(__file__).parent.parent / "log_viz" / "runs"
    )
    results_file: Path = field(
        default_factory=lambda: Path("optimization_results.json")
    )

    def results_file_for_optimizer(self, optimizer: str) -> Path:
        """Get results file path for a specific optimizer."""
        return Path(f"optimization_results_{optimizer.lower()}.json")


# Color scheme (TensorBoard-inspired)
COLORS: Dict[str, str] = {
    "primary": "#FF6F00",
    "secondary": "#0091EA",
    "success": "#00C853",
    "warning": "#FFD600",
    "error": "#D50000",
    "live_run": "#FF6F00",
    "historical": "#757575",
}

# Global config instances (immutable where possible)
MODEL_CONFIG = ModelConfig()
DATASET_CONFIG = DatasetConfig()
STRATEGYQA_DATASET_CONFIG = DatasetConfig(
    dataset_name="ChilleD/StrategyQA",
    dataset_config=None,
    train_slice="train[:1280]",
    val_slice="train[1280:]",
    test_slice="test",
)
OPTIMIZER_CONFIG = OptimizerConfig()
PROCESSING_CONFIG = ProcessingConfig()
PATH_CONFIG = PathConfig()
