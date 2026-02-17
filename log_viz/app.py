"""Main Streamlit dashboard for DSPy optimization visualization."""

import time

import pandas as pd
import streamlit as st

from components.instruction_viewer import display_instruction_evolution
from components.metrics_cards import display_metrics_cards
from components.sidebar import render_sidebar
from components.trials_table import display_trials_table
from data_loader import TrialDataLoader
from plots import (
    create_eval_type_comparison,
    create_running_max_plot,
    create_score_distribution_plot,
    create_score_over_time_plot,
)

# Page config
st.set_page_config(
    page_title="DSPy Optimization Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title
st.title("📊 DSPy Optimization Dashboard")
st.markdown(
    "*Real-time visualization of [GEPA](https://arxiv.org/abs/2507.19457) optimization trials*"
)
st.caption(
    "GEPA: Reflective Prompt Evolution via Genetic-Pareto Search (Agrawal et al., 2025)"
)


# Initialize data loader
@st.cache_resource
def get_data_loader():
    return TrialDataLoader()


loader = get_data_loader()

# Sidebar controls (shared across pages)
sidebar_state = render_sidebar(loader)
selected_run = sidebar_state["selected_run"]
auto_refresh = sidebar_state["auto_refresh"]
refresh_interval = sidebar_state["refresh_interval"]
show_historical = sidebar_state["show_historical"]

if selected_run is None:
    st.stop()

# Load current run data
df = loader.load_jsonl_full(selected_run)

# Always load historical data (for test set results)
historical_data = loader.load_historical_results()

# Load historical data for comparison if requested
historical_df = None
if show_historical:
    if historical_data and "optimization_trials" in historical_data:
        historical_df = pd.DataFrame(historical_data["optimization_trials"])

# Main content area
if df.empty:
    st.info("⏳ Waiting for optimization to start...")
    st.markdown("The dashboard will automatically update once trials begin.")
else:
    # Metrics cards
    display_metrics_cards(df)

    # Display held-out test set results if available
    if historical_data and "baseline_accuracy" in historical_data:
        st.divider()
        st.subheader("🎯 Held-Out Test Set Results")
        st.caption("Final evaluation on completely unseen test set (100 examples)")

        col1, col2, col3 = st.columns(3)

        with col1:
            baseline = historical_data["baseline_accuracy"]
            st.metric(
                label="Baseline Test Accuracy",
                value=f"{baseline:.1f}%",
                delta=None,
                help="Performance of unoptimized model on test set",
            )

        with col2:
            optimized = historical_data["optimized_accuracy"]
            improvement = historical_data.get("improvement", optimized - baseline)
            st.metric(
                label="Optimized Test Accuracy",
                value=f"{optimized:.1f}%",
                delta=f"{improvement:+.1f}%",
                delta_color="normal",
                help="Performance of GEPA-optimized model on test set",
            )

        with col3:
            # Show overfitting indicator
            train_best = df["score"].max() if not df.empty else 0
            generalization_gap = train_best - optimized
            st.metric(
                label="Generalization Gap",
                value=f"{generalization_gap:+.1f}%",
                delta=f"{generalization_gap:+.1f}%",
                delta_color="inverse",
                help="Difference between best training score and test score (positive = overfitting)",
            )

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
        st.caption(
            """**Score Distribution** shows the frequency of different performance levels achieved across all optimization trials.
        Scores represent the percentage of training examples where GPT-4.1-nano's generated answer contains the gold answer (exact string match, case-insensitive).
        A narrow distribution indicates consistent performance, while a wide spread suggests high variance in GEPA's evolutionary exploration of the prompt space.
        Per [Agrawal et al. 2025](https://arxiv.org/abs/2507.19457), GEPA maintains a Pareto front of high-performing candidates during optimization."""
        )

    with col4:
        # Eval type comparison (only for optimizers that log eval_type like MIPROv2)
        if "eval_type" in df.columns and df["eval_type"].notna().any():
            fig_eval = create_eval_type_comparison(df)
            st.plotly_chart(fig_eval, use_container_width=True)
        else:
            # Alternative visualization for GEPA: Show optimizer info
            st.markdown("### Optimizer Details")
            st.markdown(f"""
            **Optimizer**: {df.iloc[0].get('optimizer', 'Unknown').upper()}

            **Total Iterations Logged**: {len(df)}

            **Score Range**: {df['score'].min():.1f}% - {df['score'].max():.1f}%

            **Unique Iterations**: {df['trial'].nunique()}

            ---

            *Note: GEPA does not distinguish between minibatch and full evaluations in the same way as MIPROv2.
            GEPA uses evolutionary search with a Pareto front to balance multiple objectives during optimization.*
            """)
            st.info(
                "💡 The 'Eval Type Comparison' chart is available when using MIPROv2 optimizer, which logs minibatch vs full evaluation scores."
            )

    st.divider()

    # Trials table
    display_trials_table(df)

    st.divider()

    # Instruction evolution (select and view any optimizer's results)
    display_instruction_evolution(loader)

# Auto-refresh logic
if auto_refresh and not df.empty:
    time.sleep(refresh_interval)
    st.rerun()
