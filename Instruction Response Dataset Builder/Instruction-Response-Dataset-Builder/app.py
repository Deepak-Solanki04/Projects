import streamlit as st
import pandas as pd
from builder import (
    generate_response, save_entry,
    load_dataset, export_json,
    get_stats, QUALITY_LABELS
)

st.set_page_config(page_title="Instruction-Response Dataset Builder", page_icon="🗂️")

st.title(" Instruction-Response Dataset Builder")
st.markdown("Build high-quality annotated datasets for LLM fine-tuning and RLHF pipelines.")

# --- TABS ---
tab1, tab2, tab3 = st.tabs([" Annotate", " Dataset Viewer", " Export"])

# ========================
# TAB 1 — ANNOTATE
# ========================
with tab1:
    st.subheader("Step 1 — Enter an Instruction")
    instruction = st.text_area("Instruction:", height=100,
                                placeholder="e.g. Explain the concept of machine learning in simple terms")

    if st.button("Generate Response"):
        if not instruction.strip():
            st.warning("Please enter an instruction first.")
        else:
            st.session_state["instruction"] = instruction
            st.session_state["response"] = generate_response(instruction)

    if "response" in st.session_state:
        st.markdown("---")
        st.subheader("Step 2 — Review Generated Response")
        response = st.text_area("Response (you can edit this):",
                                 value=st.session_state["response"], height=150)
        st.session_state["response"] = response

        st.markdown("---")
        st.subheader("Step 3 — Annotate the Response")

        col1, col2 = st.columns(2)
        with col1:
            rating = st.slider("Quality Rating:", min_value=1, max_value=5, value=3)
        with col2:
            label = st.selectbox("Quality Label:", QUALITY_LABELS)

        notes = st.text_input("Annotator Notes (optional):",
                               placeholder="e.g. Response is too vague, needs more detail")

        st.markdown("---")
        if st.button(" Save to Dataset"):
            df = save_entry(
                st.session_state["instruction"],
                st.session_state["response"],
                rating, label, notes
            )
            st.success(f" Entry saved! Dataset now has {len(df)} entries.")
            del st.session_state["instruction"]
            del st.session_state["response"]

# ========================
# TAB 2 — DATASET VIEWER
# ========================
with tab2:
    st.subheader(" Dataset Overview")
    df = load_dataset()

    if df.empty:
        st.info("No entries yet. Start annotating in the ➕ Annotate tab!")
    else:
        stats = get_stats(df)
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Entries", stats["total_entries"])
        col2.metric("Average Rating", f"{stats['avg_rating']}/5")
        col3.metric("Unique Labels", len(stats["label_distribution"]))

        st.markdown("---")
        st.subheader("Label Distribution")
        label_df = pd.DataFrame(
            list(stats["label_distribution"].items()),
            columns=["Label", "Count"]
        )
        st.bar_chart(label_df.set_index("Label"))

        st.markdown("---")
        st.subheader("All Entries")
        st.dataframe(df, use_container_width=True)

# ========================
# TAB 3 — EXPORT
# ========================
with tab3:
    st.subheader(" Export Dataset")
    df = load_dataset()

    if df.empty:
        st.info("No data to export yet!")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                label=" Download as CSV",
                data=df.to_csv(index=False),
                file_name="instruction_response_dataset.csv",
                mime="text/csv"
            )

        with col2:
            st.download_button(
                label=" Download as JSON",
                data=export_json(),
                file_name="instruction_response_dataset.json",
                mime="application/json"
            )

        st.markdown("---")
        st.subheader("Dataset Preview")
        st.json(export_json())

st.markdown("---")
st.caption("Built with Python & Streamlit | LLM Dataset Annotation Tool | Created by Deepak Singh Solanki")