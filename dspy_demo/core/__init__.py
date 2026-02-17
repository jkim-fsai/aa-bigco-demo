"""Core DSPy demo components."""

from .data import DataLoader, format_context
from .metrics import gepa_metric, validate_answer
from .modules import BasicQA
from .tracker import OptimizationTracker

__all__ = [
    "DataLoader",
    "format_context",
    "validate_answer",
    "gepa_metric",
    "BasicQA",
    "OptimizationTracker",
]
