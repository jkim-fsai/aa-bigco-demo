"""Instruction evolution viewer component."""
from typing import Dict

import streamlit as st


def display_instruction_evolution(historical_data: Dict):
    """Display instruction candidates and evolution."""

    st.subheader("Instruction Evolution")

    if not historical_data:
        st.info("No historical data available.")
        return

    # Display instruction candidates
    if "instruction_candidates" in historical_data:
        candidates = historical_data["instruction_candidates"]

        if candidates:
            st.markdown("#### Proposed Instructions")
            for cand in candidates:
                with st.expander(f"Candidate {cand['index']}"):
                    st.markdown(cand["instruction"])
        else:
            st.info("No instruction candidates captured.")

    # Display final optimized instruction
    if "instruction" in historical_data and historical_data["instruction"]:
        st.markdown("#### Final Optimized Instruction")
        st.success(historical_data["instruction"])

    # Display few-shot demonstrations
    if "demos" in historical_data and historical_data["demos"]:
        st.markdown("#### Few-Shot Demonstrations")
        st.write(f"**{len(historical_data['demos'])} examples**")

        for i, demo in enumerate(historical_data["demos"], 1):
            with st.expander(f"Demo {i}"):
                st.markdown(f"**Question:** {demo['question']}")
                st.markdown(f"**Answer:** {demo['answer']}")
                if "reasoning" in demo:
                    st.markdown(f"**Reasoning:** {demo['reasoning']}")
