"""Interactive trials table component."""

import pandas as pd
import streamlit as st


def display_trials_table(df: pd.DataFrame, max_rows: int = 100):
    """Display sortable, filterable trials table."""

    if df.empty:
        st.info("No trials to display.")
        return

    st.subheader("Trial Details")

    # Filters
    col1, col2 = st.columns(2)

    with col1:
        # Filter by evaluation type
        if "eval_type" in df.columns:
            eval_types = ["All"] + df["eval_type"].dropna().unique().tolist()
            selected_eval = st.selectbox("Evaluation Type", eval_types)

            if selected_eval != "All":
                df = df[df["eval_type"] == selected_eval]

    with col2:
        # Filter by optimizer
        if "optimizer" in df.columns:
            optimizers = ["All"] + df["optimizer"].dropna().unique().tolist()
            selected_optimizer = st.selectbox("Optimizer", optimizers)

            if selected_optimizer != "All":
                df = df[df["optimizer"] == selected_optimizer]

    # Score range filter
    if not df.empty:
        min_score, max_score = st.slider(
            "Score Range",
            min_value=float(df["score"].min()),
            max_value=float(df["score"].max()),
            value=(float(df["score"].min()), float(df["score"].max())),
        )

        df = df[(df["score"] >= min_score) & (df["score"] <= max_score)]

    # Display table with formatting
    if df.empty:
        st.info("No trials match the selected filters.")
        return

    display_df = df[["trial", "score", "parameters", "optimizer"]].copy()
    if "eval_type" in df.columns:
        display_df["eval_type"] = df["eval_type"]

    display_df["score"] = display_df["score"].apply(lambda x: f"{x:.1f}%")

    # Limit rows for performance
    if len(display_df) > max_rows:
        st.warning(f"Showing most recent {max_rows} of {len(display_df)} trials")
        display_df = display_df.tail(max_rows)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "trial": st.column_config.NumberColumn("Trial #", width="small"),
            "score": st.column_config.TextColumn("Score", width="small"),
            "parameters": st.column_config.TextColumn("Parameters", width="large"),
            "optimizer": st.column_config.TextColumn("Optimizer", width="small"),
            "eval_type": st.column_config.TextColumn("Eval Type", width="small"),
        },
    )

    # Download button
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="Download as CSV", data=csv, file_name="trials.csv", mime="text/csv"
    )
