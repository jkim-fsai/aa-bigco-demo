"""Data loading utilities for trial data."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from utils.config import (
    CACHE_TTL_HISTORICAL,
    CACHE_TTL_JSONL,
    CACHE_TTL_RUNS,
    HISTORICAL_RESULTS,
    RUNS_DIR,
)


class TrialDataLoader:
    """Handles loading trial data from JSONL and historical JSON."""

    def __init__(self):
        self.last_position = {}  # Track file read positions per run

    @st.cache_data(ttl=CACHE_TTL_RUNS)
    def get_available_runs(_self) -> List[str]:
        """Get list of available trial run files."""
        if not RUNS_DIR.exists():
            return []
        jsonl_files = sorted(RUNS_DIR.glob("trials_*.jsonl"), reverse=True)
        return [f.stem for f in jsonl_files]

    def load_jsonl_incremental(self, run_id: str) -> pd.DataFrame:
        """Load only new lines from JSONL file since last read."""
        jsonl_path = RUNS_DIR / f"{run_id}.jsonl"

        if not jsonl_path.exists():
            return pd.DataFrame()

        # Track last read position
        if run_id not in self.last_position:
            self.last_position[run_id] = 0

        trials = []
        metadata = {}

        try:
            with open(jsonl_path, "r") as f:
                # Seek to last position
                f.seek(self.last_position[run_id])

                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if entry.get("type") == "trial":
                            trials.append(entry)
                        elif entry.get("type") == "metadata":
                            metadata = entry
                    except json.JSONDecodeError:
                        continue  # Skip malformed lines

                # Update position
                self.last_position[run_id] = f.tell()

        except Exception as e:
            st.error(f"Error reading {jsonl_path}: {e}")
            return pd.DataFrame()

        df = pd.DataFrame(trials) if trials else pd.DataFrame()

        # Add metadata to dataframe
        if not df.empty and metadata:
            df["run_status"] = metadata.get("status", "unknown")

        return df

    def load_jsonl_full(self, run_id: str) -> pd.DataFrame:
        """Load complete JSONL file (for initial load or refresh)."""
        self.last_position[run_id] = 0  # Reset position
        return self.load_jsonl_incremental(run_id)

    def load_historical_results(self, optimizer: str = "") -> Optional[Dict[str, Any]]:
        """Load optimization results JSON for a specific optimizer.

        Checks for optimizer-specific file first (e.g. optimization_results_gepa.json),
        then falls back to the generic optimization_results.json.

        Args:
            optimizer: Optimizer name (e.g. "gepa", "mipro"). Empty string
                       loads the generic file.
        """
        base_dir = HISTORICAL_RESULTS.parent

        # Try optimizer-specific file first
        if optimizer:
            specific = base_dir / f"optimization_results_{optimizer}.json"
            if specific.exists():
                try:
                    with open(specific, "r") as f:
                        return json.load(f)
                except Exception as e:
                    st.error(f"Error loading {specific}: {e}")

        # Fall back to generic file
        if not HISTORICAL_RESULTS.exists():
            return None

        try:
            with open(HISTORICAL_RESULTS, "r") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading historical results: {e}")
            return None

    @st.cache_data(ttl=CACHE_TTL_JSONL)
    def get_all_result_files(_self) -> Dict[str, Path]:
        """Get all optimization result JSON files."""
        results = {}
        base_dir = Path(".")

        # Check for main optimization_results.json
        if HISTORICAL_RESULTS.exists():
            results["main"] = HISTORICAL_RESULTS

        # Check for optimizer-specific results
        for optimizer in ["gepa", "mipro", "miprov2"]:
            result_file = base_dir / f"optimization_results_{optimizer}.json"
            if result_file.exists():
                results[optimizer] = result_file

        return results

    def load_result_file(self, file_path: Path) -> Optional[Dict]:
        """Load a specific result file."""
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading {file_path}: {e}")
            return None

    def get_run_metadata(self, run_id: str) -> Dict:
        """Extract metadata from a run."""
        jsonl_path = RUNS_DIR / f"{run_id}.jsonl"

        if not jsonl_path.exists():
            return {}

        metadata = {}
        try:
            with open(jsonl_path, "r") as f:
                first_line = f.readline()
                if first_line:
                    entry = json.loads(first_line.strip())
                    if entry.get("type") == "metadata":
                        metadata = entry
        except Exception:
            pass

        return metadata
