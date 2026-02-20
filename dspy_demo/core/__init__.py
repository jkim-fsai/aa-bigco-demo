"""Core DSPy demo components."""

from .data import DataLoader, format_context
from .metrics import (
    gepa_boolean_metric,
    gepa_metric,
    gepa_multiple_choice_metric,
    validate_answer,
    validate_boolean_answer,
    validate_multiple_choice,
)
from .modules import BasicQA, BooleanQA, MultipleChoiceQA
from .tracker import OptimizationTracker

__all__ = [
    "DataLoader",
    "format_context",
    "validate_answer",
    "validate_boolean_answer",
    "gepa_metric",
    "gepa_boolean_metric",
    "gepa_multiple_choice_metric",
    "validate_multiple_choice",
    "BasicQA",
    "BooleanQA",
    "MultipleChoiceQA",
    "OptimizationTracker",
]
