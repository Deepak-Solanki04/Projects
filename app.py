import streamlit as st
from evaluator import evaluate_prompt

st.set_page_config(page_title="Prompt Quality Evaluator", page_icon="🧠")

st.title(" Prompt Quality Evaluator")
st.markdown("Evaluate how effective your AI prompt is before sending it to an LLM.")

prompt_input = st.text_area("Enter your prompt below:", height=150,
                             placeholder="e.g. Explain the concept of overfitting in machine learning with examples.")

if st.button("Evaluate Prompt"):
    if not prompt_input.strip():
        st.warning("Please enter a prompt first.")
    else:
        scores, feedback, verdict = evaluate_prompt(prompt_input)

        st.markdown("---")
        st.subheader(" Evaluation Result")
        st.markdown(f"### {verdict}")

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Clarity", f"{scores['Clarity']}/10")
        col2.metric("Specificity", f"{scores['Specificity']}/10")
        col3.metric("Context", f"{scores['Context']}/10")
        col4.metric("Tone", f"{scores['Tone']}/10")
        col5.metric("Overall", f"{scores['Overall']}/10")

        st.markdown("---")
        st.subheader(" Detailed Feedback")
        for f in feedback:
            st.markdown(f"- {f}")

st.markdown("---")
st.caption("Built with Python & Streamlit | Prompt Engineering Tool | Created by Deepak Singh Solanki")