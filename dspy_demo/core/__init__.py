"""Core DSPy demo components."""

from .data import DataLoader, format_context
from .metrics import (
    gepa_boolean_metric,
    gepa_metric,
    validate_answer,
    validate_boolean_answer,
)
from .modules import BasicQA, BooleanQA
from .tracker import OptimizationTracker

__all__ = [
    "DataLoader",
    "format_context",
    "validate_answer",
    "validate_boolean_answer",
    "gepa_metric",
    "gepa_boolean_metric",
    "BasicQA",
    "BooleanQA",
    "OptimizationTracker",
]
