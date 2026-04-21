import nltk
import re
from textblob import TextBlob
from nltk.corpus import stopwords, words as nltk_words
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('words', quiet=True)

stop_words = set(stopwords.words('english'))
english_vocab = set(w.lower() for w in nltk_words.words())

CONTEXT_KEYWORDS = [
    "explain", "describe", "summarize", "compare", "analyze",
    "list", "generate", "write", "classify", "evaluate",
    "step by step", "in detail", "example", "because", "so that",
    "what is", "how to", "why does", "define", "create", "suggest"
]

VAGUE_WORDS = [
    "thing", "stuff", "something", "anything", "whatever",
    "good", "bad", "nice", "do it", "make it", "fix it"
]

def real_word_ratio(tokens):
    """Returns ratio of real English words to total alpha tokens."""
    alpha_tokens = [w.lower() for w in tokens if w.isalpha()]
    if not alpha_tokens:
        return 0
    real = [w for w in alpha_tokens if w in english_vocab or w in stop_words]
    return len(real) / len(alpha_tokens)

def evaluate_prompt(prompt):
    scores = {}
    feedback = []

    tokens = word_tokenize(prompt)
    alpha_tokens = [w.lower() for w in tokens if w.isalpha()]
    word_count = len(alpha_tokens)

    # --- GIBBERISH DETECTION (runs first) ---
    ratio = real_word_ratio(tokens)
    is_gibberish = ratio < 0.5  # less than 50% real words = gibberish

    # --- 1. CLARITY ---
    if is_gibberish:
        scores['Clarity'] = 1
        feedback.append(" Clarity: Input appears to be gibberish or random text.")
    else:
        vague_found = [w for w in VAGUE_WORDS if w in prompt.lower()]
        clarity_score = 10
        if vague_found:
            clarity_score -= len(vague_found) * 2
            feedback.append(f" Clarity: Vague words found → {vague_found}. Be more specific.")
        else:
            feedback.append(" Clarity: No vague words detected. Good!")
        scores['Clarity'] = max(clarity_score, 2)

    # --- 2. SPECIFICITY ---
    if is_gibberish:
        scores['Specificity'] = 1
        feedback.append(" Specificity: No meaningful content detected.")
    elif word_count < 5:
        scores['Specificity'] = 2
        feedback.append(" Specificity: Prompt is too short. Add more detail.")
    elif word_count < 15:
        scores['Specificity'] = 5
        feedback.append(" Specificity: Prompt is a bit short. Consider adding context.")
    elif word_count < 40:
        scores['Specificity'] = 8
        feedback.append(" Specificity: Good length and detail.")
    else:
        scores['Specificity'] = 10
        feedback.append(" Specificity: Very detailed prompt!")

    # --- 3. CONTEXT ---
    if is_gibberish:
        scores['Context'] = 1
        feedback.append(" Context: No valid context or instruction detected.")
    else:
        context_found = [kw for kw in CONTEXT_KEYWORDS if kw in prompt.lower()]
        if len(context_found) == 0:
            scores['Context'] = 3
            feedback.append(" Context: No instructional keywords found. Try adding words like 'explain', 'list', 'step by step'.")
        elif len(context_found) == 1:
            scores['Context'] = 6
            feedback.append(f" Context: Some context found → {context_found}. Could be stronger.")
        else:
            scores['Context'] = 10
            feedback.append(f" Context: Strong contextual keywords found → {context_found}")

    # --- 4. TONE ---
    if is_gibberish:
        scores['Tone'] = 1
        feedback.append(" Tone: Unable to analyze tone — input is not valid text.")
    else:
        blob = TextBlob(prompt)
        subjectivity = blob.sentiment.subjectivity
        if subjectivity > 0.6:
            scores['Tone'] = 4
            feedback.append(" Tone: Prompt seems too subjective. Keep it neutral and objective.")
        else:
            scores['Tone'] = 9
            feedback.append(" Tone: Neutral and objective tone detected.")

    # --- OVERALL ---
    total = sum(scores.values()) / len(scores)
    scores['Overall'] = round(total, 1)

    if total >= 8:
        verdict = " Excellent Prompt!"
    elif total >= 5:
        verdict = " Average Prompt — needs improvement"
    else:
        verdict = " Poor Prompt — significant improvements needed"

    return scores, feedback, verdict