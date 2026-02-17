"""Compare multiple optimizer runs side-by-side."""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.sidebar import render_sidebar
from data_loader import TrialDataLoader

st.set_page_config(page_title="Optimizer Comparison", page_icon="⚖️", layout="wide")


# Initialize data loader and shared sidebar
@st.cache_resource
def get_data_loader():
    return TrialDataLoader()


loader = get_data_loader()
sidebar_state = render_sidebar(loader)
selected_run = sidebar_state["selected_run"]

st.title("⚖️ Optimizer Comparison")
st.markdown("*Compare GEPA vs MIPROv2 performance side-by-side*")

if selected_run:
    st.info(f"Viewing run: **{selected_run.replace('trials_', '')}**")

# Find all result files
result_files = {}
for optimizer in ["gepa", "mipro", "miprov2"]:
    result_file = Path(f"optimization_results_{optimizer}.json")
    if result_file.exists():
        result_files[optimizer.upper()] = result_file

# Also check main results file
main_result = Path("optimization_results.json")
if main_result.exists():
    result_files["Latest"] = main_result

if not result_files:
    st.warning("No optimization results found. Run demo_compare.py first.")
    st.code("""
# Run GEPA
python demo_compare.py gepa

# Run MIPROv2
python demo_compare.py mipro
    """)
    st.stop()

# Load all results
results = {}
for name, path in result_files.items():
    try:
        with open(path) as f:
            data = json.load(f)
            data["_name"] = name
            results[name] = data
    except Exception as e:
        st.error(f"Error loading {path}: {e}")

st.divider()

# Performance comparison
st.subheader("📊 Performance Comparison")

comparison_data = []
for name, data in results.items():
    if "baseline_accuracy" in data and "optimized_accuracy" in data:
        comparison_data.append(
            {
                "Optimizer": data.get("optimizer", name).upper(),
                "Baseline": data["baseline_accuracy"],
                "Optimized": data["optimized_accuracy"],
                "Improvement": data.get(
                    "improvement",
                    data["optimized_accuracy"] - data["baseline_accuracy"],
                ),
            }
        )

if comparison_data:
    df_comparison = pd.DataFrame(comparison_data)

    col1, col2 = st.columns([2, 1])

    with col1:
        # Bar chart comparison
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                name="Baseline",
                x=df_comparison["Optimizer"],
                y=df_comparison["Baseline"],
                marker_color="lightblue",
            )
        )

        fig.add_trace(
            go.Bar(
                name="Optimized",
                x=df_comparison["Optimizer"],
                y=df_comparison["Optimized"],
                marker_color="darkblue",
            )
        )

        fig.update_layout(
            title="Test Set Accuracy Comparison",
            xaxis_title="Optimizer",
            yaxis_title="Accuracy (%)",
            barmode="group",
            height=400,
            template="plotly_white",
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.dataframe(
            df_comparison.style.format(
                {
                    "Baseline": "{:.1f}%",
                    "Optimized": "{:.1f}%",
                    "Improvement": "{:+.1f}%",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
