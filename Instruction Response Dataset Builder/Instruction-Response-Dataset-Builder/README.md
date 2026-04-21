# Instruction-Response Dataset Builder

A Python-based annotation tool to build high-quality instruction-response datasets for LLM fine-tuning and RLHF pipelines.

## What it does
- Enter any instruction and generate a simulated AI response
- Annotate responses with quality ratings (1-5) and labels
- Export the final dataset as CSV or JSON for model training

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tech Stack
- Python, Streamlit, Pandas

## Use Case
Replicates real-world human annotation workflows used in post-training
of Large Language Models (RLHF, instruction tuning pipelines).