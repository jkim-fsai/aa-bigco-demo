"""Metrics cards component for displaying summary statistics."""

import pandas as pd
import streamlit as st

OPTIMIZER_DESCRIPTIONS = {
    "GEPA": """
        **GEPA (Genetic-Pareto)** is a reflective prompt optimizer from [Agrawal et al. (2025)](https://arxiv.org/abs/2507.19457)
        that uses evolutionary search to discover optimal instructions.

        **How Scoring Works:**
        - **Answer Generation**: GPT-4.1-nano generates answers to questions given context
        - **Evaluation Metric**: Exact string match (case-insensitive) - checks if the gold answer appears in the predicted answer
        - **Score Scale**: 0-100% (averaged from binary 0.0/1.0 per example)
        - **Reflection LLM**: GPT-4.1-nano (temp=1.0) proposes new instruction candidates based on performance feedback

        **Training Set**: HotPotQA examples used for optimization (see sidebar for dataset split sizes)
        **Process**: GEPA runs evolutionary search, evaluating different prompt instructions and tracking the best performers via a Pareto front.

        *Scores shown here are training set performance during optimization, not held-out test performance.*
        """,
    "MIPROv2": """
        **MIPROv2 (Multi-prompt Instruction Proposal Optimizer)** from [Opsahl-Ong et al. (2024)](https://arxiv.org/abs/2406.11695)
        uses Bayesian optimization to search over instruction and few-shot demo combinations.

        **How Scoring Works:**
        - **Answer Generation**: GPT-4.1-nano generates answers to questions given context
        - **Evaluation Metric**: Exact string match (case-insensitive) - checks if the gold answer appears in the predicted answer
        - **Score Scale**: 0-100% (averaged from binary 0.0/1.0 per example)

        **Training Set**: HotPotQA examples used for optimization (see sidebar for dataset split sizes)
        **Process**: MIPROv2 proposes instruction candidates and few-shot demo sets, evaluating via minibatch and full evaluations.

        *Scores shown here are training set performance during optimization, not held-out test performance.*
        """,
}


def display_metrics_cards(df: pd.DataFrame, optimizer_label: str = ""):
    """Display summary metrics in card format.

    Args:
        df: Trial data DataFrame.
        optimizer_label: Display name of the optimizer (e.g. "GEPA", "MIPROv2").
    """

    if df.empty:
        st.warning("No trial data available yet.")
        return

    label = optimizer_label or "Unknown"
    st.subheader(f"📊 Training Set Metrics ({label} Optimization)")

    with st.expander("ℹ️ About the Scoring System", expanded=False):
        description = OPTIMIZER_DESCRIPTIONS.get(
            label,
            f"""
        **{label}** optimizer.

        **How Scoring Works:**
        - **Answer Generation**: GPT-4.1-nano generates answers to questions given context
        - **Evaluation Metric**: Exact string match (case-insensitive) - checks if the gold answer appears in the predicted answer
        - **Score Scale**: 0-100% (averaged from binary 0.0/1.0 per example)

        **Training Set**: HotPotQA examples used for optimization (see sidebar for dataset split sizes)

        *Scores shown here are training set performance during optimization, not held-out test performance.*
        """,
        )
        st.markdown(description)

    st.caption(
        f"Scores below are from {label}'s internal evaluation on the training set during optimization"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label="Total Trials", value=len(df), delta=None)

    with col2:
        current_best = df["score"].max()
        st.metric(label="Best Score", value=f"{current_best:.1f}%", delta=None)

    with col3:
        latest_score = df.iloc[-1]["score"] if not df.empty else 0
        st.metric(label="Latest Score", value=f"{latest_score:.1f}%", delta=None)

    with col4:
        mean_score = df["score"].mean()
        st.metric(label="Mean Score", value=f"{mean_score:.1f}%", delta=None)

    # Additional row with more details
    col5, col6, col7, col8 = st.columns(4)

    with col5:
        std_score = df["score"].std()
        st.metric(label="Std Dev", value=f"{std_score:.1f}%")

    with col6:
        min_score = df["score"].min()
        st.metric(label="Min Score", value=f"{min_score:.1f}%")

    with col7:
        # Calculate improvement from first to best
        if len(df) > 1:
            improvement = df["score"].max() - df.iloc[0]["score"]
            st.metric(
                label="Improvement",
                value=f"{improvement:+.1f}%",
                delta=f"{improvement:+.1f}%",
            )
        else:
            st.metric(label="Improvement", value="N/A")

    with col8:
        # Check run status
        status = df.iloc[-1].get("run_status", "running")
        status_color = "🟢" if status == "completed" else "🟡"
        st.metric(label="Status", value=f"{status_color} {status.title()}")
