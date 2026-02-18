"""Shared sidebar component for run selection across pages."""

from typing import Any, Dict, Optional

import streamlit as st

from data_loader import TrialDataLoader
from utils.config import RUNS_DIR

_METADATA_KEYS = {
    "type",
    "run_id",
    "timestamp",
    "status",
    "trainset_size",
    "valset_size",
    "testset_size",
    "optimizer",
}

_HYPERPARAM_LABELS = {
    "model": "Inference Model",
    "reflection_model": "Reflection Model",
    "auto": "Auto Budget",
    "num_threads": "Threads",
    "reflection_minibatch_size": "Reflection Minibatch",
    "num_trials": "Num Trials",
    "max_bootstrapped_demos": "Max Bootstrap Demos",
    "max_labeled_demos": "Max Labeled Demos",
}


def _display_hyperparams(metadata: Optional[Dict[str, Any]]) -> None:
    """Display optimizer hyperparameters from run metadata."""
    if not metadata:
        st.caption("No hyperparameter info available")
        return

    hyperparams = {k: v for k, v in metadata.items() if k not in _METADATA_KEYS}
    if not hyperparams:
        st.caption("No hyperparameter info available for this run")
        return

    for key, value in hyperparams.items():
        label = _HYPERPARAM_LABELS.get(key, key.replace("_", " ").title())
        st.caption(f"**{label}:** {value}")


def render_sidebar(loader: TrialDataLoader) -> Dict[str, Any]:
    """Render shared sidebar controls and return sidebar state.

    Args:
        loader: TrialDataLoader instance for fetching run data.

    Returns:
        Dictionary with keys:
            - selected_run: str or None (selected run ID)
            - auto_refresh: bool
            - refresh_interval: int (seconds)
            - show_historical: bool
    """
    with st.sidebar:
        st.header("⚙️ Controls")

        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto-refresh", value=True)

        refresh_interval = 2
        if auto_refresh:
            refresh_interval = st.slider(
                "Refresh interval (seconds)", min_value=1, max_value=10, value=2
            )

        st.divider()

        # Run selection
        st.subheader("Select Run")
        available_runs = loader.get_available_runs()

        if not available_runs:
            st.warning("No optimization runs found.")
            st.info(
                f"Run demo.py to generate trial data.\nData will appear in: {RUNS_DIR}"
            )
            return {
                "selected_run": None,
                "auto_refresh": auto_refresh,
                "refresh_interval": refresh_interval,
                "show_historical": False,
            }

        # Use session_state key binding for cross-page persistence
        if "selected_run" not in st.session_state:
            st.session_state.selected_run = available_runs[0]

        if st.session_state.selected_run not in available_runs:
            st.session_state.selected_run = available_runs[0]

        selected_run = st.selectbox(
            "Run ID",
            options=available_runs,
            format_func=lambda x: x.replace("trials_", ""),
            key="selected_run",
        )

        # Run metadata
        metadata = loader.get_run_metadata(selected_run)
        if metadata:
            st.caption(f"Started: {metadata.get('timestamp', 'N/A')}")
            st.caption(f"Status: {metadata.get('status', 'unknown')}")

        st.divider()

        # Dataset info (derived from run metadata)
        st.subheader("📚 Dataset Splits")
        train_size = metadata.get("trainset_size")
        val_size = metadata.get("valset_size")
        test_size = metadata.get("testset_size")
        run_optimizer = metadata.get("optimizer", "")

        if train_size is not None:
            st.caption(f"**Training Set:** {train_size} examples")
            if val_size is not None:
                st.caption(f"**Validation Set:** {val_size} examples")
            st.caption(
                f"Used by {run_optimizer.upper() or 'optimizer'} for optimization"
            )
            st.caption("")
            if test_size is not None:
                st.caption(f"**Test Set:** {test_size} examples")
                st.caption("Held-out for final evaluation")
        else:
            st.caption("Dataset split info not available for this run")

        st.divider()

        # Hyperparameters (derived from run metadata)
        st.subheader("🔧 Hyperparameters")
        _display_hyperparams(metadata)

        st.divider()

        # Historical comparison toggle
        show_historical = st.checkbox("Compare with historical run", value=False)

        st.divider()

        # Refresh button
        if st.button("🔄 Force Refresh"):
            st.cache_data.clear()
            st.rerun()

    return {
        "selected_run": selected_run,
        "auto_refresh": auto_refresh,
        "refresh_interval": refresh_interval,
        "show_historical": show_historical,
    }
