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
from .pipeline import OptimizationPipeline, OptimizationResult, OptimizerType

__all__ = [
    # Pipeline
    "OptimizationPipeline",
    "OptimizationResult",
    "OptimizerType",
    # Config classes
    "ModelConfig",
    "DatasetConfig",
    "OptimizerConfig",
    "ProcessingConfig",
    "PathConfig",
    # Config instances
    "MODEL_CONFIG",
    "DATASET_CONFIG",
    "OPTIMIZER_CONFIG",
    "PROCESSING_CONFIG",
    "PATH_CONFIG",
]
