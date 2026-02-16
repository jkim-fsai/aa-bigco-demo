"""Optimization tracking and logging handler.

This module provides a logging handler that captures optimization metrics
from DSPy optimizer log output and writes them to JSONL files for visualization.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, TextIO

from ..config import PATH_CONFIG


class LogPatterns:
    """Compiled regex patterns for parsing optimizer log messages."""

    # GEPA instruction proposal pattern
    GEPA_INSTRUCTION = re.compile(
        r"Iteration (\d+): Proposed new text for generate_answer\.predict:\s*(.+)",
        re.DOTALL,
    )

    # MIPROv2 instruction candidate pattern (e.g., "0: Given the fields...")
    MIPRO_INSTRUCTION = re.compile(r"^(\d+):\s*(.+)$")

    # MIPROv2 score pattern with optional minibatch info
    MIPRO_SCORE = re.compile(
        r"Score:\s*([\d.]+)(?:\s+on minibatch of size \d+)?\s+with parameters\s+(\[.+?\])"
    )

    # Default or best program score pattern
    DEFAULT_OR_BEST = re.compile(
        r"(?:Default program score|best full eval score.*Score):\s*([\d.]+)"
    )

    # GEPA iteration score pattern
    GEPA_ITERATION = re.compile(
        r"Iteration (\d+): (?:Valset score for new program|Best score on valset): ([\d.]+)"
    )


class OptimizationTracker(logging.Handler):
    """Custom logging handler to capture per-iteration optimization metrics.

    Attaches to DSPy optimizer loggers and extracts:
    - Instruction proposals (both GEPA and MIPROv2 formats)
    - Trial scores and parameters
    - Best score markers

    Writes real-time trial data to JSONL files for dashboard visualization.

    Example:
        tracker = OptimizationTracker()
        logging.getLogger("dspy.teleprompt.gepa").addHandler(tracker)

        tracker.open_jsonl()
        # ... run optimization ...
        tracker.close_jsonl()

        summary = tracker.get_summary()
    """

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        """Initialize the optimization tracker.

        Args:
            output_dir: Directory for JSONL output files. Defaults to PATH_CONFIG.runs_dir.
        """
        super().__init__()
        self.trials: List[Dict[str, Any]] = []
        self.instructions: List[Dict[str, Any]] = []
        self.current_instruction_idx: int = 0

        self.output_dir = output_dir or PATH_CONFIG.runs_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.run_id: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.trials_jsonl_path: Path = self.output_dir / f"trials_{self.run_id}.jsonl"
        self._jsonl_file: Optional[TextIO] = None
        self._written_trial_ids: Set[str] = set()

    def reset(self) -> None:
        """Reset tracker state for a new optimization run."""
        self.trials = []
        self.instructions = []
        self.current_instruction_idx = 0
        self._written_trial_ids = set()
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.trials_jsonl_path = self.output_dir / f"trials_{self.run_id}.jsonl"

    def open_jsonl(self) -> None:
        """Open JSONL file for writing trial data."""
        if self._jsonl_file is None:
            self._jsonl_file = open(self.trials_jsonl_path, "a", buffering=1)
            self._write_metadata("started")

    def close_jsonl(self) -> None:
        """Close JSONL file and write completion metadata."""
        if self._jsonl_file:
            self._write_metadata("completed", total_trials=len(self.trials))
            self._jsonl_file.close()
            self._jsonl_file = None

    def _write_metadata(self, status: str, **kwargs: Any) -> None:
        """Write metadata entry to JSONL file."""
        metadata = {
            "type": "metadata",
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "status": status,
            **kwargs,
        }
        if self._jsonl_file:
            self._jsonl_file.write(json.dumps(metadata) + "\n")
            self._jsonl_file.flush()

    def _write_trial(self, trial: Dict[str, Any]) -> None:
        """Write trial entry to JSONL file with deduplication."""
        trial_id = f"{trial['trial']}_{trial['score']}_{trial.get('eval_type', '')}"
        if trial_id in self._written_trial_ids:
            return

        self._written_trial_ids.add(trial_id)

        if self._jsonl_file:
            entry = {
                "type": "trial",
                "run_id": self.run_id,
                "timestamp": datetime.now().isoformat(),
                **trial,
            }
            try:
                self._jsonl_file.write(json.dumps(entry) + "\n")
                self._jsonl_file.flush()
            except Exception as e:
                logging.error(f"Failed to write trial to JSONL: {e}")

    def emit(self, record: logging.LogRecord) -> None:
        """Process a log record and extract optimization metrics.

        Args:
            record: Log record from DSPy optimizer.
        """
        msg = record.getMessage()

        self._handle_mipro_instruction_header(msg)
        self._handle_mipro_instruction(msg)
        self._handle_gepa_instruction(msg)
        self._handle_mipro_score(msg)
        self._handle_default_or_best_score(msg)
        self._handle_gepa_iteration_score(msg)
        self._handle_best_flag(msg)

    def _handle_mipro_instruction_header(self, msg: str) -> None:
        """Handle MIPROv2 instruction proposal header."""
        if "Proposed Instructions for Predictor" in msg:
            self.current_instruction_idx = 0

    def _handle_mipro_instruction(self, msg: str) -> None:
        """Handle MIPROv2 instruction candidate lines."""
        if "Iteration" in msg:
            return
        match = LogPatterns.MIPRO_INSTRUCTION.match(msg)
        if match:
            idx, instruction = match.groups()
            self.instructions.append({
                "index": int(idx),
                "instruction": instruction.strip(),
                "type": "mipro",
            })

    def _handle_gepa_instruction(self, msg: str) -> None:
        """Handle GEPA instruction proposal messages."""
        match = LogPatterns.GEPA_INSTRUCTION.search(msg)
        if match:
            iteration = int(match.group(1))
            instruction_text = match.group(2).strip()
            self.instructions.append({
                "index": iteration,
                "instruction": instruction_text,
                "type": "gepa",
                "iteration": iteration,
            })

    def _handle_mipro_score(self, msg: str) -> None:
        """Handle MIPROv2 score messages."""
        match = LogPatterns.MIPRO_SCORE.search(msg)
        if match:
            score = float(match.group(1))
            params = match.group(2)
            is_minibatch = "minibatch" in msg
            trial = {
                "trial": len(self.trials) + 1,
                "score": score,
                "parameters": params,
                "optimizer": "mipro",
                "eval_type": "minibatch" if is_minibatch else "full",
            }
            self.trials.append(trial)
            self._write_trial(trial)

    def _handle_default_or_best_score(self, msg: str) -> None:
        """Handle default program or best score messages."""
        if LogPatterns.MIPRO_SCORE.search(msg):
            return  # Already handled by _handle_mipro_score
        match = LogPatterns.DEFAULT_OR_BEST.search(msg)
        if match:
            score = float(match.group(1))
            is_best = "best" in msg.lower()
            is_default = "Default" in msg
            trial = {
                "trial": len(self.trials) + 1,
                "score": score,
                "parameters": "Default program" if is_default else "Best program",
                "optimizer": "mipro",
                "eval_type": "full",
                "is_best": is_best,
            }
            self.trials.append(trial)
            self._write_trial(trial)

    def _handle_gepa_iteration_score(self, msg: str) -> None:
        """Handle GEPA iteration score messages."""
        match = LogPatterns.GEPA_ITERATION.search(msg)
        if match:
            iteration = int(match.group(1))
            score = float(match.group(2)) * 100  # Convert to percentage
            trial = {
                "trial": iteration,
                "score": score,
                "parameters": f"GEPA Iteration {iteration}",
                "optimizer": "gepa",
            }
            self.trials.append(trial)
            self._write_trial(trial)

    def _handle_best_flag(self, msg: str) -> None:
        """Handle best score marker messages."""
        if "Best full score so far!" in msg and self.trials:
            self.trials[-1]["is_best"] = True

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of captured optimization data.

        Returns:
            Dictionary with instructions_proposed, trials, and best_trial.
        """
        return {
            "instructions_proposed": self.instructions,
            "trials": self.trials,
            "best_trial": max(self.trials, key=lambda x: x["score"]) if self.trials else None,
        }
