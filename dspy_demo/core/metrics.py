"""Evaluation metrics for DSPy optimization."""

from typing import Any, Optional

from dspy.primitives.example import Example
from dspy.primitives.prediction import Prediction


def validate_answer(
    example: Example,
    pred: Prediction,
    trace: Optional[Any] = None,  # noqa: ARG001 - required by DSPy metric interface
) -> bool:
    """Check if gold answer appears in predicted answer (case-insensitive).

    Args:
        example: Ground truth example with answer field.
        pred: Model prediction with answer field.
        trace: Optional trace object (unused, for DSPy compatibility).

    Returns:
        True if gold answer is contained in predicted answer.
    """
    return example.answer.lower() in pred.answer.lower()


def gepa_metric(
    example: Example,
    pred: Prediction,
    trace: Optional[Any] = None,  # noqa: ARG001 - required by GEPA interface
    student_code: Optional[Any] = None,  # noqa: ARG001 - required by GEPA interface
    teacher_code: Optional[Any] = None,  # noqa: ARG001 - required by GEPA interface
) -> float:
    """GEPA-compatible metric wrapper.

    GEPA requires a metric that returns float and accepts 5 parameters.

    Args:
        example: Ground truth example with answer field.
        pred: Model prediction with answer field.
        trace: Optional trace object (unused).
        student_code: Optional student code (unused).
        teacher_code: Optional teacher code (unused).

    Returns:
        1.0 if correct, 0.0 otherwise.
    """
    correct = example.answer.lower() in pred.answer.lower()
    return 1.0 if correct else 0.0
