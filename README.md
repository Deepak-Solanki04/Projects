#  Prompt Quality Evaluator

A Python-based tool to evaluate the quality of prompts given to Large Language Models (LLMs).

##  What it evaluates
- **Clarity** – Detects vague or ambiguous language
- **Specificity** – Checks prompt length and detail level
- **Context** – Looks for instructional keywords
- **Tone** – Analyzes subjectivity and neutrality

##  How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

##  Tech Stack
- Python, Streamlit, TextBlob, NLTK

##  Use Case
Useful in LLM post-training pipelines to filter and score human-written prompts
before using them as training data.