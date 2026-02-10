"""Main Streamlit dashboard for DSPy optimization visualization."""
import time

import pandas as pd
import streamlit as st

from components.instruction_viewer import display_instruction_evolution
from components.metrics_cards import display_metrics_cards
from components.trials_table import display_trials_table
from data_loader import TrialDataLoader
from plots import (
    create_eval_type_comparison,
    create_running_max_plot,
    create_score_distribution_plot,
    create_score_over_time_plot,
)
from utils.config import RUNS_DIR

# Page config
st.set_page_config(
    page_title="DSPy Optimization Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("📊 DSPy Optimization Dashboard")
st.markdown("*Real-time visualization of DSPy optimization trials*")

# Initialize data loader
@st.cache_resource
def get_data_loader():
    return TrialDataLoader()

loader = get_data_loader()

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Controls")

    # Auto-refresh toggle
    auto_refresh = st.checkbox("Auto-refresh", value=True)

    if auto_refresh:
        refresh_interval = st.slider(
            "Refresh interval (seconds)",
            min_value=1,
            max_value=10,
            value=2
        )

    st.divider()

    # Run selection
    st.subheader("Select Run")
    available_runs = loader.get_available_runs()

    if not available_runs:
        st.warning("No optimization runs found.")
        st.info(f"Run demo.py to generate trial data.\nData will appear in: {RUNS_DIR}")
        st.stop()

    selected_run = st.selectbox(
        "Run ID",
        options=available_runs,
        format_func=lambda x: x.replace("trials_", "")
    )

    # Run metadata
    metadata = loader.get_run_metadata(selected_run)
    if metadata:
        st.caption(f"Started: {metadata.get('timestamp', 'N/A')}")
        st.caption(f"Status: {metadata.get('status', 'unknown')}")

    st.divider()

    # Historical comparison toggle
    show_historical = st.checkbox("Compare with historical run", value=False)

    st.divider()

    # Refresh button
    if st.button("🔄 Force Refresh"):
        st.cache_data.clear()
        st.rerun()

# Load current run data
df = loader.load_jsonl_full(selected_run)

# Load historical data if requested
historical_df = None
historical_data = None
if show_historical:
    historical_data = loader.load_historical_results()
    if historical_data and "optimization_trials" in historical_data:
        historical_df = pd.DataFrame(historical_data["optimization_trials"])

# Main content area
if df.empty:
    st.info("⏳ Waiting for optimization to start...")
    st.markdown("The dashboard will automatically update once trials begin.")
else:
    # Metrics cards
    display_metrics_cards(df)

    st.divider()

    # Main plots
    col1, col2 = st.columns(2)

    with col1:
        # Score over time
        fig_score = create_score_over_time_plot(df, historical_df)
        st.plotly_chart(fig_score, use_container_width=True)

    with col2:
        # Running maximum
        fig_max = create_running_max_plot(df)
        st.plotly_chart(fig_max, use_container_width=True)

    # Secondary plots
    col3, col4 = st.columns(2)

    with col3:
        # Score distribution
        fig_dist = create_score_distribution_plot(df)
        st.plotly_chart(fig_dist, use_container_width=True)

    with col4:
        # Eval type comparison
        fig_eval = create_eval_type_comparison(df)
        st.plotly_chart(fig_eval, use_container_width=True)

    st.divider()

    # Trials table
    display_trials_table(df)

    st.divider()

    # Instruction evolution (only if historical data available)
    if historical_data:
        display_instruction_evolution(historical_data)

# Auto-refresh logic
if auto_refresh and not df.empty:
    time.sleep(refresh_interval)
    st.rerun()
