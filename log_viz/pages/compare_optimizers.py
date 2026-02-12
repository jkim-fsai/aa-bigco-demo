"""Compare multiple optimizer runs side-by-side."""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Optimizer Comparison",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Optimizer Comparison")
st.markdown("*Compare GEPA vs MIPROv2 instruction evolution and performance*")

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
        comparison_data.append({
            "Optimizer": data.get("optimizer", name).upper(),
            "Baseline": data["baseline_accuracy"],
            "Optimized": data["optimized_accuracy"],
            "Improvement": data.get("improvement", data["optimized_accuracy"] - data["baseline_accuracy"])
        })

if comparison_data:
    df_comparison = pd.DataFrame(comparison_data)

    col1, col2 = st.columns([2, 1])

    with col1:
        # Bar chart comparison
        fig = go.Figure()

        fig.add_trace(go.Bar(
            name="Baseline",
            x=df_comparison["Optimizer"],
            y=df_comparison["Baseline"],
            marker_color="lightblue"
        ))

        fig.add_trace(go.Bar(
            name="Optimized",
            x=df_comparison["Optimizer"],
            y=df_comparison["Optimized"],
            marker_color="darkblue"
        ))

        fig.update_layout(
            title="Test Set Accuracy Comparison",
            xaxis_title="Optimizer",
            yaxis_title="Accuracy (%)",
            barmode="group",
            height=400,
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.dataframe(
            df_comparison.style.format({
                "Baseline": "{:.1f}%",
                "Optimized": "{:.1f}%",
                "Improvement": "{:+.1f}%"
            }),
            hide_index=True,
            use_container_width=True
        )

st.divider()

# Instruction evolution comparison
st.subheader("📝 Instruction Evolution Comparison")

tabs = st.tabs(list(results.keys()))

for tab, (name, data) in zip(tabs, results.items()):
    with tab:
        st.markdown(f"### {name} - {data.get('optimizer', 'Unknown').upper()}")

        # Show instruction candidates
        if "instruction_candidates" in data and data["instruction_candidates"]:
            candidates = data["instruction_candidates"]
            optimizer_type = data.get("optimizer", "unknown")

            st.markdown(f"**Captured {len(candidates)} instruction proposals**")

            # Group by type if available
            gepa_cands = [c for c in candidates if c.get("type") == "gepa"]
            mipro_cands = [c for c in candidates if c.get("type") == "mipro"]

            if gepa_cands:
                st.markdown("#### GEPA Evolutionary Proposals")
                for i, cand in enumerate(gepa_cands[:5], 1):
                    with st.expander(f"Iteration {cand.get('iteration', cand['index'])}"):
                        st.markdown(cand["instruction"])
                if len(gepa_cands) > 5:
                    st.info(f"Showing 5 of {len(gepa_cands)} GEPA proposals")

            if mipro_cands:
                st.markdown("#### MIPROv2 Proposals")
                for cand in mipro_cands:
                    with st.expander(f"Candidate {cand['index']}"):
                        st.markdown(cand["instruction"])

            if not gepa_cands and not mipro_cands:
                # Show all if no type specified
                for cand in candidates[:5]:
                    with st.expander(f"Candidate {cand['index']}"):
                        st.markdown(cand["instruction"])
                if len(candidates) > 5:
                    st.info(f"Showing 5 of {len(candidates)} proposals")

        else:
            st.warning("No instruction candidates captured for this run")

        # Show final instruction if available
        if "instruction" in data and data["instruction"]:
            st.markdown("#### Final Optimized Instruction")
            st.success(data["instruction"])

        # Show few-shot demos if available
        if "demos" in data and data["demos"]:
            st.markdown(f"#### Few-Shot Demonstrations ({len(data['demos'])} examples)")
            for i, demo in enumerate(data["demos"][:3], 1):
                with st.expander(f"Demo {i}"):
                    st.markdown(f"**Q:** {demo['question']}")
                    st.markdown(f"**A:** {demo['answer']}")
                    if "reasoning" in demo:
                        st.markdown(f"**Reasoning:** {demo['reasoning']}")

st.divider()

# Key insights
st.subheader("💡 Key Insights")

if len(results) >= 2:
    st.markdown("""
    **GEPA vs MIPROv2:**

    - **GEPA** uses evolutionary search to evolve instructions via reflection
    - **MIPROv2** optimizes both instructions and few-shot demonstrations
    - **GEPA** typically proposes more instruction variants (~20-30 per run)
    - **MIPROv2** focuses on instruction quality + example selection

    **What to look for:**
    - Which optimizer achieves better test set performance?
    - How do instruction styles differ between optimizers?
    - Does GEPA's evolutionary approach lead to more creative instructions?
    - Does MIPROv2's few-shot selection provide better generalization?
    """)
else:
    st.info("Run both optimizers to see a comparison. Use: `python demo_compare.py gepa` and `python demo_compare.py mipro`")
