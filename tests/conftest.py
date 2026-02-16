"""Shared pytest fixtures for the test suite."""

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

# Mock dspy module and its submodules before any dspy_demo imports
# This must happen before pytest discovers and imports test modules
if "dspy" not in sys.modules:
    # Mock the Example class from dspy.primitives.example
    class MockExample:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

        def with_inputs(self, *args):
            return self

    class MockPrediction:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    # Create proper mock modules
    mock_primitives = MagicMock()
    mock_primitives.example = MagicMock()
    mock_primitives.example.Example = MockExample
    mock_primitives.prediction = MagicMock()
    mock_primitives.prediction.Prediction = MockPrediction

    mock_dspy = MagicMock()
    mock_dspy.primitives = mock_primitives
    mock_dspy.Example = MockExample
    mock_dspy.Prediction = MockPrediction

    # Mock commonly used dspy components
    mock_dspy.LM = MagicMock
    mock_dspy.GEPA = MagicMock
    mock_dspy.MIPROv2 = MagicMock
    mock_dspy.configure = MagicMock()
    mock_dspy.Signature = MagicMock
    mock_dspy.InputField = MagicMock
    mock_dspy.OutputField = MagicMock
    mock_dspy.Module = MagicMock
    mock_dspy.ChainOfThought = MagicMock
    mock_dspy.Evaluate = MagicMock

    # Register the mock dspy module and its submodules
    sys.modules["dspy"] = mock_dspy
    sys.modules["dspy.primitives"] = mock_primitives
    sys.modules["dspy.primitives.example"] = mock_primitives.example
    sys.modules["dspy.primitives.prediction"] = mock_primitives.prediction

import pytest

# Add log_viz to sys.path so its relative imports work when running tests from root
_log_viz_path = str(Path(__file__).parent.parent / "log_viz")
if _log_viz_path not in sys.path:
    sys.path.insert(0, _log_viz_path)


@pytest.fixture
def sample_instruction_candidates() -> List[Dict[str, Any]]:
    """Sample instruction candidates for testing."""
    return [
        {
            "index": 1,
            "instruction": "Given context and question, provide answer.",
            "type": "gepa",
            "iteration": 1,
        },
        {
            "index": 2,
            "instruction": "Read the context carefully and answer the question.",
            "type": "gepa",
            "iteration": 2,
        },
        {
            "index": 2,
            "instruction": "Read the context carefully and answer the question.",
            "type": "gepa",
            "iteration": 2,
        },  # Duplicate
        {
            "index": 3,
            "instruction": "Extract the answer from the provided context.",
            "type": "mipro",
        },
        {
            "index": 4,
            "instruction": "Answer based on facts in the context only.",
            "type": "mipro",
        },
    ]


@pytest.fixture
def sample_trials() -> List[Dict[str, Any]]:
    """Sample trial data for testing."""
    return [
        {"trial": 1, "score": 65.0, "parameters": "GEPA Iteration 1", "optimizer": "gepa"},
        {"trial": 2, "score": 68.5, "parameters": "GEPA Iteration 2", "optimizer": "gepa"},
        {"trial": 3, "score": 70.0, "parameters": "GEPA Iteration 3", "optimizer": "gepa"},
        {"trial": 4, "score": 72.5, "parameters": "GEPA Iteration 4", "optimizer": "gepa"},
        {"trial": 5, "score": 71.0, "parameters": "GEPA Iteration 5", "optimizer": "gepa"},
    ]


@pytest.fixture
def sample_historical_data(
    sample_instruction_candidates: List[Dict[str, Any]],
    sample_trials: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Sample historical data matching optimization_results.json structure."""
    return {
        "timestamp": "2026-02-12T10:00:00.000000",
        "optimizer": "gepa",
        "instruction": "Given context and question, provide a precise answer.",
        "demos": [
            {"question": "What is X?", "answer": "X is Y"},
            {"question": "Who did Z?", "answer": "Person A"},
        ],
        "optimization_trials": sample_trials,
        "instruction_candidates": sample_instruction_candidates,
        "baseline_accuracy": 65.0,
        "optimized_accuracy": 72.5,
        "improvement": 7.5,
        "evolution_summary": "The optimization improved accuracy by 7.5%.",
    }


@pytest.fixture
def temp_dir() -> Path:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_jsonl_file(temp_dir: Path, sample_trials: List[Dict[str, Any]]) -> Path:
    """Create a temporary JSONL file with sample trial data."""
    jsonl_path = temp_dir / "trials_20260212_100000.jsonl"

    with open(jsonl_path, "w") as f:
        # Write metadata
        metadata = {
            "type": "metadata",
            "run_id": "20260212_100000",
            "timestamp": "2026-02-12T10:00:00.000000",
            "status": "started",
        }
        f.write(json.dumps(metadata) + "\n")

        # Write trials
        for trial in sample_trials:
            entry = {"type": "trial", "run_id": "20260212_100000", **trial}
            f.write(json.dumps(entry) + "\n")

        # Write completion metadata
        completion = {
            "type": "metadata",
            "run_id": "20260212_100000",
            "timestamp": "2026-02-12T10:05:00.000000",
            "status": "completed",
            "total_trials": len(sample_trials),
        }
        f.write(json.dumps(completion) + "\n")

    return jsonl_path


@pytest.fixture
def temp_results_file(temp_dir: Path, sample_historical_data: Dict[str, Any]) -> Path:
    """Create a temporary optimization results JSON file."""
    results_path = temp_dir / "optimization_results.json"
    results_path.write_text(json.dumps(sample_historical_data, indent=2))
    return results_path


@pytest.fixture
def mock_example() -> MagicMock:
    """Create a mock DSPy Example object."""
    example = MagicMock()
    example.question = "What is the capital of France?"
    example.context = "France is a country in Europe. Its capital is Paris."
    example.answer = "Paris"
    return example


@pytest.fixture
def mock_prediction() -> MagicMock:
    """Create a mock DSPy Prediction object."""
    pred = MagicMock()
    pred.answer = "The capital of France is Paris."
    return pred


@pytest.fixture
def mock_prediction_wrong() -> MagicMock:
    """Create a mock DSPy Prediction object with wrong answer."""
    pred = MagicMock()
    pred.answer = "The capital is London."
    return pred
