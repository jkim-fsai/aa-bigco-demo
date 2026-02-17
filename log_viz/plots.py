"""Plotly visualization functions for optimization trials."""

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.config import COLORS, PLOT_HEIGHT


def create_score_over_time_plot(
    df: pd.DataFrame,
    historical_df: Optional[pd.DataFrame] = None,
    title: str = "Optimization Progress (Training Set)",
) -> go.Figure:
    """Create main score vs iteration plot with live and historical data."""

    fig = go.Figure()

    # Plot live run
    if not df.empty:
        fig.add_trace(
            go.Scatter(
                x=df["trial"],
                y=df["score"],
                mode="lines+markers",
                name="Current Run",
                line=dict(color=COLORS["live_run"], width=2),
                marker=dict(size=6),
                hovertemplate="<b>Trial %{x}</b><br>Score: %{y:.1f}%<extra></extra>",
            )
        )

    # Plot historical run for comparison
    if historical_df is not None and not historical_df.empty:
        fig.add_trace(
            go.Scatter(
                x=historical_df["trial"],
                y=historical_df["score"],
                mode="lines+markers",
                name="Historical Run",
                line=dict(color=COLORS["historical"], width=2, dash="dash"),
                marker=dict(size=4),
                hovertemplate="<b>Trial %{x}</b><br>Score: %{y:.1f}%<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Trial Number",
        yaxis_title="Score (%)",
        height=PLOT_HEIGHT,
        hovermode="x unified",
        showlegend=True,
        template="plotly_white",
    )

    return fig


def create_running_max_plot(df: pd.DataFrame) -> go.Figure:
    """Create running maximum score curve (best score so far)."""

    if df.empty:
        return go.Figure()

    # Calculate running maximum
    df_sorted = df.sort_values("trial")
    running_max = df_sorted["score"].cummax()

    fig = go.Figure()

    # Running max line
    fig.add_trace(
        go.Scatter(
            x=df_sorted["trial"],
            y=running_max,
            mode="lines",
            name="Best Score So Far",
            line=dict(color=COLORS["success"], width=3),
            fill="tozeroy",
            fillcolor="rgba(0, 200, 83, 0.1)",
            hovertemplate="<b>Trial %{x}</b><br>Best: %{y:.1f}%<extra></extra>",
        )
    )

    # Individual trials as scatter
    fig.add_trace(
        go.Scatter(
            x=df_sorted["trial"],
            y=df_sorted["score"],
            mode="markers",
            name="Individual Trials",
            marker=dict(size=4, color=COLORS["primary"], opacity=0.5),
            hovertemplate="<b>Trial %{x}</b><br>Score: %{y:.1f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title="Best Score Tracker (Training Set)",
        xaxis_title="Trial Number",
        yaxis_title="Score (%)",
        height=PLOT_HEIGHT,
        template="plotly_white",
        showlegend=True,
    )

    return fig


def create_score_distribution_plot(df: pd.DataFrame) -> go.Figure:
    """Create histogram of score distribution."""

    if df.empty:
        return go.Figure()

    fig = px.histogram(
        df,
        x="score",
        nbins=20,
        title="Score Distribution (Training Set)",
        labels={"score": "Score (%)", "count": "Frequency"},
        color_discrete_sequence=[COLORS["secondary"]],
    )

    fig.update_layout(height=PLOT_HEIGHT, template="plotly_white", showlegend=False)

    return fig


def create_eval_type_comparison(df: pd.DataFrame) -> go.Figure:
    """Compare scores by evaluation type (minibatch vs full)."""

    if df.empty or "eval_type" not in df.columns:
        return go.Figure()

    fig = px.box(
        df,
        x="eval_type",
        y="score",
        title="Score by Evaluation Type",
        labels={"eval_type": "Evaluation Type", "score": "Score (%)"},
        color="eval_type",
        color_discrete_map={"minibatch": COLORS["warning"], "full": COLORS["success"]},
    )

    fig.update_layout(height=PLOT_HEIGHT, template="plotly_white", showlegend=False)

    return fig
