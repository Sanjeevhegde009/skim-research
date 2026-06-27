"""
Scoring for the PageIndex + RAG eval runs (LoCoMo / LongMemEval).

Token-level F1 (SQuAD-style, comparable to published LoCoMo numbers) plus a
frontier-model 0-2 quality judge. Adversarial / NOT_ANSWERABLE questions are
scored deterministically — a clean refusal is correct (2), any substantive
answer is a fabrication (0) — so the LLM judge adds no run-to-run noise there.

Consumed by run_pageindex.py and run_longmemeval.py via score_answer / token_f1.
"""

import json
import re
import string
from collections import Counter

from llm import compiler_call


# ─────────────────────────────────────────────
# Token-level F1 (SQuAD-style, comparable to published LoCoMo results)
# ─────────────────────────────────────────────

def normalize_answer(text: str) -> str:
    """Lowercase, remove articles, punctuation, extra whitespace."""
    text = str(text).lower()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text


# A clean refusal IS the correct answer to a NOT_ANSWERABLE (adversarial) question — detect it
# lexically and score it deterministically. The LLM judge only adds run-to-run noise on adversarial,
# re-grading identical correct refusals 2 vs 0 between runs. Shared by token_f1 and score_answer.
REFUSAL_WORDS = ["not answerable", "not mentioned", "don't know", "cannot",
                 "no information", "not discussed", "not available", "unanswerable",
                 "not contain", "not provide", "no mention", "doesn't mention",
                 "does not mention", "not in", "no specific"]


def _refused(actual):
    a = (actual or "").lower()
    return any(w in a for w in REFUSAL_WORDS)


def token_f1(expected: str, actual: str, is_adversarial: bool = False) -> dict:
    """Compute token-level F1, precision, recall. Standard SQuAD metric.
    This is the metric published LoCoMo results use — directly comparable."""
    if is_adversarial:
        refused = _refused(actual)
        return {"token_f1": 1.0 if refused else 0.0,
                "token_precision": 1.0 if refused else 0.0,
                "token_recall": 1.0 if refused else 0.0}

    exp_norm = normalize_answer(expected)
    act_norm = normalize_answer(actual)
    exp_tokens = exp_norm.split()
    act_tokens = act_norm.split()

    if not exp_tokens or not act_tokens:
        return {"token_f1": 0.0, "token_precision": 0.0, "token_recall": 0.0}

    exp_counts = Counter(exp_tokens)
    act_counts = Counter(act_tokens)
    overlap = sum(min(count, act_counts.get(token, 0)) for token, count in exp_counts.items())

    precision = overlap / len(act_tokens) if act_tokens else 0.0
    recall = overlap / len(exp_tokens) if exp_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"token_f1": f1, "token_precision": precision, "token_recall": recall}


def score_answer(question, expected, actual, category_name):
    """Use frontier model to judge answer quality. Returns dict with score and F1."""
    if not actual:
        return {"score": 0, "f1": 0.0, "judgment": "empty", "hallucination": False}

    is_adversarial = (expected == "NOT_ANSWERABLE")
    if is_adversarial:
        # Binary + deterministic: a clean refusal is correct (2), any substantive answer is a
        # fabrication (0). Skip the LLM judge — on adversarial it only adds noise (re-grading
        # identical correct refusals 2 vs 0 across runs) and costs a call.
        refused = _refused(actual)
        return {"score": 2 if refused else 0, "f1": 1.0 if refused else 0.0,
                "judgment": "refused" if refused else "fabricated",
                "hallucination": not refused}

    prompt = f"""Judge this answer strictly.

QUESTION: {question}
EXPECTED ANSWER: {expected}
ACTUAL ANSWER: {actual}
QUESTION TYPE: {category_name}

{"This is an ADVERSARIAL question — the correct behavior is to say the information is not available or cannot be determined. Score 2 ONLY if the model correctly refuses to answer or says it doesn't know. Score 0 if it makes up an answer." if is_adversarial else ""}

Return ONLY a JSON object:
{{"score": 0 or 1 or 2, "f1": 0.0 to 1.0, "judgment": "one line", "hallucination": true/false}}

Scoring:
  2 = correct (captures key information)
  1 = partially correct (some right, missing key details)
  0 = wrong or hallucinated
  f1 = word-overlap F1 between expected and actual (0.0-1.0)

JSON only:"""

    raw = compiler_call(prompt, temperature=0.0)
    try:
        clean = raw.strip()
        if clean.startswith("```"): clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
        result = json.loads(clean)
        return {
            "score": result.get("score", 0),
            "f1": result.get("f1", 0.0),
            "judgment": result.get("judgment", ""),
            "hallucination": result.get("hallucination", False),
        }
    except (json.JSONDecodeError, KeyError):
        # Fallback: simple token F1
        exp_tokens = set(expected.lower().split())
        act_tokens = set(actual.lower().split())
        if is_adversarial:
            refused = any(w in actual.lower() for w in
                         ["not answerable", "not mentioned", "don't know", "cannot",
                          "no information", "not discussed", "not available", "unanswerable"])
            return {"score": 2 if refused else 0, "f1": 1.0 if refused else 0.0,
                    "judgment": "fallback", "hallucination": not refused}
        if not exp_tokens or not act_tokens:
            return {"score": 0, "f1": 0.0, "judgment": "fallback", "hallucination": False}
        overlap = exp_tokens & act_tokens
        p = len(overlap) / len(act_tokens) if act_tokens else 0
        r = len(overlap) / len(exp_tokens) if exp_tokens else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        score = 2 if f1 > 0.5 else (1 if f1 > 0.2 else 0)
        return {"score": score, "f1": f1, "judgment": "token-f1 fallback", "hallucination": False}
