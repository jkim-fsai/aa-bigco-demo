"""Instruction evolution viewer component."""

from typing import Any, Dict

import streamlit as st

from utils import MAX_INSTRUCTION_DISPLAY, deduplicate_instructions, group_by_optimizer_type


def display_instruction_evolution(historical_data: Dict[str, Any]) -> None:
    """Display instruction candidates and evolution.

    Args:
        historical_data: Dictionary containing optimization results.
    """
    st.subheader("📝 Instruction Evolution")

    if not historical_data:
        st.info("No historical data available.")
        return

    has_content = False

    # Display instruction candidates
    if "instruction_candidates" in historical_data:
        candidates = historical_data["instruction_candidates"]

        if candidates:
            has_content = True

            # Group by optimizer type using shared utility
            groups = group_by_optimizer_type(candidates)
            gepa_candidates = groups["gepa"]
            mipro_candidates = groups["mipro"]

            if gepa_candidates:
                # Deduplicate using shared utility
                unique_candidates = deduplicate_instructions(gepa_candidates)

                st.markdown("#### 🧬 GEPA Proposed Instructions")
                st.caption(f"Evolutionary search proposed {len(unique_candidates)} instruction variants")

                for cand in unique_candidates[:MAX_INSTRUCTION_DISPLAY]:
                    iteration = cand.get("iteration", cand["index"])
                    with st.expander(f"Iteration {iteration} - Instruction Proposal"):
                        st.markdown(cand["instruction"])

                if len(unique_candidates) > MAX_INSTRUCTION_DISPLAY:
                    st.info(
                        f"Showing {MAX_INSTRUCTION_DISPLAY} of {len(unique_candidates)} proposals. "
                        "Check optimization_results.json for all."
                    )

            if mipro_candidates:
                st.markdown("#### 🔍 MIPROv2 Proposed Instructions")
                st.caption(f"Proposed {len(mipro_candidates)} instruction candidates")
                for cand in mipro_candidates:
                    with st.expander(f"Candidate {cand['index']}"):
                        st.markdown(cand["instruction"])

            if not gepa_candidates and not mipro_candidates:
                # Legacy format without type field
                st.markdown("#### Proposed Instructions During Optimization")
                st.caption(f"{len(candidates)} instruction candidates")
                for cand in candidates:
                    with st.expander(f"Candidate {cand['index']}"):
                        st.markdown(cand["instruction"])

    # Display LLM-generated evolution summary
    if "evolution_summary" in historical_data and historical_data["evolution_summary"]:
        has_content = True
        st.markdown("#### 🧠 LLM Analysis of Evolution")
        st.caption("AI-generated summary of the optimization process")
        with st.container(border=True):
            st.markdown(historical_data["evolution_summary"])
        st.divider()

    # Display final optimized instruction
    if "instruction" in historical_data and historical_data["instruction"]:
        has_content = True
        st.markdown("#### 🎯 Final Optimized Instruction")
        st.success(historical_data["instruction"])
    elif not has_content:
        st.warning("""**Instruction extraction unavailable**

        GEPA generates detailed instructions during optimization (visible in console logs), but they were not successfully extracted from the compiled module.

        **Why this happens:**
        - GEPA stores optimized instructions in the compiled module's internal structure
        - The extraction logic needs to be updated to handle GEPA's specific format
        - Instructions are being used by the model but aren't accessible via standard DSPy attributes

        **What you can see:**
        - The optimized model IS using the evolved instructions (reflected in the test scores)
        - The console logs show GEPA proposing instructions during optimization
        - Run the demo again to see instruction proposals in real-time

        **Baseline Instruction (for reference):**
        ```
        Given the fields `context`, `question`, produce the fields `answer`.
        ```
        """)

    # Display few-shot demonstrations
    if "demos" in historical_data and historical_data["demos"]:
        has_content = True
        st.markdown("#### Few-Shot Demonstrations")
        st.write(f"**{len(historical_data['demos'])} examples**")

        for i, demo in enumerate(historical_data["demos"], 1):
            with st.expander(f"Demo {i}"):
                st.markdown(f"**Question:** {demo['question']}")
                st.markdown(f"**Answer:** {demo['answer']}")
                if "reasoning" in demo:
                    st.markdown(f"**Reasoning:** {demo['reasoning']}")
