"""Data loading utilities for trial data."""
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from utils.config import HISTORICAL_RESULTS, RUNS_DIR


class TrialDataLoader:
    """Handles loading trial data from JSONL and historical JSON."""

    def __init__(self):
        self.last_position = {}  # Track file read positions per run

    @st.cache_data(ttl=2)  # Cache for 2 seconds
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

    @st.cache_data(ttl=10)
    def load_historical_results(_self) -> Optional[Dict]:
        """Load optimization_results.json for historical comparison."""
        if not HISTORICAL_RESULTS.exists():
            return None

        try:
            with open(HISTORICAL_RESULTS, "r") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading historical results: {e}")
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
