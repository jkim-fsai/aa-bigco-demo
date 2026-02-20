"""DSPy optimization demo package.

This package provides a unified interface for running DSPy optimization
experiments with various optimizers (GEPA, MIPROv2).

Usage:
    from dspy_demo import OptimizationPipeline, OptimizerType

    pipeline = OptimizationPipeline()
    result = await pipeline.run(OptimizerType.GEPA)
    result.save()
"""

from .config import (
    ARC_DATASET_CONFIG,
    DATASET_CONFIG,
    MODEL_CONFIG,
    OPTIMIZER_CONFIG,
    PATH_CONFIG,
    PROCESSING_CONFIG,
    STRATEGYQA_DATASET_CONFIG,
    DatasetConfig,
    ModelConfig,
    OptimizerConfig,
    PathConfig,
    ProcessingConfig,
)
from .core import (
    BasicQA,
    BooleanQA,
    MultipleChoiceQA,
    gepa_boolean_metric,
    gepa_metric,
    gepa_multiple_choice_metric,
    validate_answer,
    validate_boolean_answer,
    validate_multiple_choice,
)
from .pipeline import OptimizationPipeline, OptimizationResult, OptimizerType

__all__ = [
    # Pipeline
    "OptimizationPipeline",
    "OptimizationResult",
    "OptimizerType",
    # Modules
    "BasicQA",
    "BooleanQA",
    "MultipleChoiceQA",
    # Metrics
    "validate_answer",
    "validate_boolean_answer",
    "validate_multiple_choice",
    "gepa_metric",
    "gepa_boolean_metric",
    "gepa_multiple_choice_metric",
    # Config classes
    "ModelConfig",
    "DatasetConfig",
    "OptimizerConfig",
    "ProcessingConfig",
    "PathConfig",
    # Config instances
    "MODEL_CONFIG",
    "DATASET_CONFIG",
    "STRATEGYQA_DATASET_CONFIG",
    "ARC_DATASET_CONFIG",
    "OPTIMIZER_CONFIG",
    "PROCESSING_CONFIG",
    "PATH_CONFIG",
]
