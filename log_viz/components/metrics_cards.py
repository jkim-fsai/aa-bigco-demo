"""Metrics cards component for displaying summary statistics."""
import pandas as pd
import streamlit as st


def display_metrics_cards(df: pd.DataFrame):
    """Display summary metrics in card format."""

    if df.empty:
        st.warning("No trial data available yet.")
        return

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Trials",
            value=len(df),
            delta=None
        )

    with col2:
        current_best = df["score"].max()
        st.metric(
            label="Best Score",
            value=f"{current_best:.1f}%",
            delta=None
        )

    with col3:
        latest_score = df.iloc[-1]["score"] if not df.empty else 0
        st.metric(
            label="Latest Score",
            value=f"{latest_score:.1f}%",
            delta=None
        )

    with col4:
        mean_score = df["score"].mean()
        st.metric(
            label="Mean Score",
            value=f"{mean_score:.1f}%",
            delta=None
        )

    # Additional row with more details
    col5, col6, col7, col8 = st.columns(4)

    with col5:
        std_score = df["score"].std()
        st.metric(
            label="Std Dev",
            value=f"{std_score:.1f}%"
        )

    with col6:
        min_score = df["score"].min()
        st.metric(
            label="Min Score",
            value=f"{min_score:.1f}%"
        )

    with col7:
        # Calculate improvement from first to best
        if len(df) > 1:
            improvement = df["score"].max() - df.iloc[0]["score"]
            st.metric(
                label="Improvement",
                value=f"{improvement:+.1f}%",
                delta=f"{improvement:+.1f}%"
            )
        else:
            st.metric(label="Improvement", value="N/A")

    with col8:
        # Check run status
        status = df.iloc[-1].get("run_status", "running")
        status_color = "🟢" if status == "completed" else "🟡"
        st.metric(
            label="Status",
            value=f"{status_color} {status.title()}"
        )
