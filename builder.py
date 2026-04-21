import pandas as pd
import os
import json
from datetime import datetime

DATASET_FILE = "dataset.csv"

QUALITY_LABELS = [
    "Perfect - use as is",
    "Good - minor edits needed",
    "Average - major edits needed",
    "Poor - should be rejected",
    "Toxic/Harmful - reject immediately"
]

def generate_response(instruction):
    """
    Simulates an AI response based on instruction keywords.
    In a real pipeline this would call an LLM API.
    """
    instruction_lower = instruction.lower()

    if any(w in instruction_lower for w in ["explain", "what is", "define", "describe"]):
        return f"[AI Response]: This is an explanation of '{instruction}'. The concept involves multiple aspects that can be broken down step by step for better understanding."
    elif any(w in instruction_lower for w in ["list", "give me", "provide", "name"]):
        return f"[AI Response]: Here are key points about '{instruction}':\n1. First important aspect\n2. Second relevant point\n3. Third consideration\n4. Additional context"
    elif any(w in instruction_lower for w in ["how to", "steps", "guide", "tutorial"]):
        return f"[AI Response]: Step-by-step guide for '{instruction}':\nStep 1: Begin with the basics\nStep 2: Apply the core concept\nStep 3: Verify and refine your approach"
    elif any(w in instruction_lower for w in ["compare", "difference", "vs", "versus"]):
        return f"[AI Response]: Comparing aspects of '{instruction}':\n- Similarity: Both share common ground\n- Difference: They diverge in key ways\n- Use case: Choose based on your specific need"
    else:
        return f"[AI Response]: Regarding '{instruction}' — this is a generated response that addresses the instruction with relevant context and information."

def save_entry(instruction, response, rating, label, notes=""):
    """Save a single annotation entry to CSV."""
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "instruction": instruction,
        "response": response,
        "rating": rating,
        "quality_label": label,
        "annotator_notes": notes
    }

    if os.path.exists(DATASET_FILE):
        df = pd.read_csv(DATASET_FILE)
        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    else:
        df = pd.DataFrame([entry])

    df.to_csv(DATASET_FILE, index=False)
    return df

def load_dataset():
    """Load existing dataset."""
    if os.path.exists(DATASET_FILE):
        return pd.read_csv(DATASET_FILE)
    return pd.DataFrame()

def export_json():
    """Export dataset as JSON."""
    if os.path.exists(DATASET_FILE):
        df = pd.read_csv(DATASET_FILE)
        return df.to_json(orient="records", indent=2)
    return "[]"

def get_stats(df):
    """Return basic dataset statistics."""
    if df.empty:
        return None
    stats = {
        "total_entries": len(df),
        "avg_rating": round(df["rating"].mean(), 2),
        "label_distribution": df["quality_label"].value_counts().to_dict()
    }
    return stats