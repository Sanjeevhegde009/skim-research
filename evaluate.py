"""
LoCoMo Evaluation — HLMA vs baselines on real long-conversation data.
Uses frontier model (Claude API) for compilation + scoring.
Uses SLM (Ollama) for all querying (HLMA and baselines).
"""

import json
import time
import os
import sys
import re
import string
import hashlib
from pathlib import Path
from collections import Counter
from hlma import HLMAMemory, compiler_call, estimate_tokens
from baselines import baseline_full_history, baseline_summary_only, baseline_sliding_window
from locomo_loader import load_conversations
import config


def _compilation_hash(conv) -> str:
    """Stable hash of everything that affects wiki structure.

    Covers: raw turns content + compiler model.
    Does NOT cover query-side prompts — those don't affect the wiki.
    If you change compiler prompts in hlma.py, delete the wiki dir manually
    or set HLMA_FORCE_RECOMPILE=1 to trigger a fresh compile.
    """
    h = hashlib.sha256()
    for session in conv["sessions"]:
        for t in session["turns"]:
            h.update(f"{t.get('dia_id','')}|{t['speaker']}|{t['text']}\n".encode())
        h.update(f"SESSION:{session.get('key','')}|{session.get('date_time','')}\n".encode())
    h.update(config.COMPILER_MODEL.encode())
    return h.hexdigest()[:16]


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


def generate_plain_summary(all_turns):
    """Generate a plain summary (no pointers) for the summary-only baseline."""
    # Chunk turns to fit compiler context — take evenly spaced samples if too long
    text = "\n".join(f"[{t.get('dia_id','')}] {t['speaker']}: {t['text']}" for t in all_turns)
    tokens = estimate_tokens(text)

    if tokens > 15000:
        # Sample every Nth turn to fit
        step = max(1, len(all_turns) // 300)
        sampled = all_turns[::step]
        text = "\n".join(f"[{t.get('dia_id','')}] {t['speaker']}: {t['text']}" for t in sampled)

    prompt = f"""Summarize this long conversation. Capture all important facts, events,
decisions, preferences, names, dates, relationships, and plans. Be comprehensive.
Under 400 words.

CONVERSATION:
{text}

Summary:"""

    return compiler_call(prompt)


def run_hlma_on_conversation(conv):
    """Ingest all sessions, generate summary, lint, then answer QA."""
    import shutil
    wiki_dir = f"wiki_eval_{conv['sample_id']}"
    cache_file = Path(wiki_dir) / "_cache_hash.txt"
    force = os.environ.get("HLMA_FORCE_RECOMPILE")

    expected_hash = _compilation_hash(conv)
    cached_hash   = cache_file.read_text().strip() if cache_file.exists() else ""
    wiki_complete = (Path(wiki_dir) / "summary.md").exists()

    if not force and wiki_complete and cached_hash == expected_hash:
        print(f"  [CACHE HIT] Reusing wiki at {wiki_dir} (hash {expected_hash})")
        mem = HLMAMemory(wiki_dir=wiki_dir)
    else:
        if os.path.exists(wiki_dir):
            reason = "forced" if force else ("hash mismatch" if cached_hash else "no cache")
            print(f"  [RECOMPILE] {wiki_dir} ({reason})")
            shutil.rmtree(wiki_dir)

        mem = HLMAMemory(wiki_dir=wiki_dir)

        # Copy schema.md into wiki dir
        schema_src = Path("schema.md")
        if schema_src.exists():
            (Path(wiki_dir) / "schema.md").write_text(schema_src.read_text(encoding="utf-8"), encoding="utf-8")

        # Ingest session by session (summary regenerated after each)
        import hlma as _hlma
        _fails_before = _hlma.COMPILER_FAILURES
        for session in conv["sessions"]:
            mem.ingest_session(session["turns"],
                               session_label=f"{session['key']} ({session['date_time']})",
                               session_date=session["date_time"])

        # Write cache hash so next run can skip compilation — but only if every
        # compiler call succeeded. A failed call means facts were silently dropped;
        # caching that wiki as complete would bake the holes in permanently.
        _fails = _hlma.COMPILER_FAILURES - _fails_before
        if _fails == 0:
            cache_file.write_text(expected_hash)
        else:
            print(f"  [WARN] {_fails} compiler call(s) failed during compilation — "
                  f"cache NOT written; the next run will recompile this conversation.")

    summary = mem.wiki.get_summary()
    pages = mem.wiki.get_all_pages()
    print(f"  HLMA: {len(pages)} pages, summary ~{estimate_tokens(summary)} tok")

    # Answer questions
    results = []
    total = len(conv["qa"])
    for i, qa in enumerate(conv["qa"]):
        print(f"    HLMA query {i+1}/{total}: {qa['question'][:50]}...", end=" ", flush=True)
        trace = mem.query(qa["question"])
        hops = f"{trace.hops}hop" + (f" → {trace.pages_retrieved}" if trace.pages_retrieved else "")
        print(f"[{hops}]")
        # Per-tier ReAct trail — tier, candidate answer, and clear/vague verdict
        for s in getattr(trace, "steps", []):
            tier = s.get("tier", "?")
            src = s.get("source", "")
            a = (s.get("answer", "") or "")[:70].replace("\n", " ")
            v = s.get("verdict", "")
            tok = s.get("tokens", 0)
            tok_str = f" ~{tok}tok" if tok else ""
            print(f"        tier {tier} [{src}]{tok_str} → \"{a}\" → {v}")
        results.append({
            "question": qa["question"],
            "expected": qa["answer"],
            "category": qa["category_name"],
            "answer": trace.answer,
            "tokens_est": trace.tokens_est,
            "tokens_t1": getattr(trace, "tokens_t1", 0),
            "tokens_t2_pick": getattr(trace, "tokens_t2_pick", 0),
            "tokens_t2_ans": getattr(trace, "tokens_t2_ans", 0),
            "tokens_t3": getattr(trace, "tokens_t3", 0),
            "hops": trace.hops,
            "pages_retrieved": trace.pages_retrieved,
            "reasoning": trace.reasoning,
            "steps": getattr(trace, "steps", []),
            "t3_debug": getattr(trace, "t3_debug", {}),
        })

    # Cleanup — preserve the wiki for inspection when HLMA_KEEP_WIKI is set, so we can
    # read what was actually compiled (pages + summary) vs what the queries needed.
    if os.path.exists(wiki_dir) and not os.environ.get("HLMA_KEEP_WIKI"):
        shutil.rmtree(wiki_dir)
    elif os.environ.get("HLMA_KEEP_WIKI"):
        print(f"  [KEEP_WIKI] wiki preserved at: {wiki_dir}")
    return results, summary, pages


def run_evaluation(conv_indices=None, max_qa=None, with_baselines=False):
    """Run evaluation on selected LoCoMo conversations.
    HLMA-only by default; pass with_baselines=True to include comparison baselines."""
    convs = load_conversations()

    if conv_indices is None:
        conv_indices = [0]  # Default: first conversation
    if isinstance(conv_indices, int):
        conv_indices = [conv_indices]

    print("=" * 70)
    print("HLMA EVALUATION ON LOCOMO")
    print(f"  Compiler: {config.COMPILER_MODEL}")
    print(f"  Query:    {config.QUERY_MODEL}")
    print(f"  Baselines: {'included' if with_baselines else 'skipped (HLMA only)'}")
    print("=" * 70)

    for ci in conv_indices:
        conv = convs[ci]
        qa_list = conv["qa"][:max_qa] if max_qa else conv["qa"]
        conv_copy = {**conv, "qa": qa_list}

        print(f"\n{'='*70}")
        print(f"Conv {ci}: {conv['speakers']}")
        print(f"  {len(conv['sessions'])} sessions, {len(conv['all_turns'])} turns, {len(qa_list)} QA")
        raw_tok = sum(estimate_tokens(t["text"]) for t in conv["all_turns"])
        print(f"  Raw tokens: ~{raw_tok}")

        all_results = {}

        # --- HLMA ---
        print(f"\n--- HLMA ---")
        hlma_results, hlma_summary, hlma_pages = run_hlma_on_conversation(conv_copy)
        all_results["hlma"] = hlma_results

        # --- Baselines (optional) ---
        if with_baselines:
            print(f"\n--- Full History (SLM) ---")
            fh_results = []
            for i, qa in enumerate(qa_list):
                print(f"    FH {i+1}/{len(qa_list)}", end=" ", flush=True)
                r = baseline_full_history(conv["all_turns"], qa["question"])
                print("✓")
                fh_results.append({
                    "question": qa["question"], "expected": qa["answer"],
                    "category": qa["category_name"], "answer": r["answer"],
                    "tokens_est": r["tokens_est"],
                })
            all_results["full_history"] = fh_results

            print(f"\n--- Summary Only (SLM) ---")
            print(f"  Generating plain summary...")
            plain_summary = generate_plain_summary(conv["all_turns"])
            print(f"  Plain summary: ~{estimate_tokens(plain_summary)} tok")
            so_results = []
            for i, qa in enumerate(qa_list):
                print(f"    SO {i+1}/{len(qa_list)}", end=" ", flush=True)
                r = baseline_summary_only(conv["all_turns"], qa["question"], plain_summary)
                print("✓")
                so_results.append({
                    "question": qa["question"], "expected": qa["answer"],
                    "category": qa["category_name"], "answer": r["answer"],
                    "tokens_est": r["tokens_est"],
                })
            all_results["summary_only"] = so_results

            print(f"\n--- Sliding Window (SLM) ---")
            sw_results = []
            for i, qa in enumerate(qa_list):
                print(f"    SW {i+1}/{len(qa_list)}", end=" ", flush=True)
                r = baseline_sliding_window(conv["all_turns"], qa["question"])
                print("✓")
                sw_results.append({
                    "question": qa["question"], "expected": qa["answer"],
                    "category": qa["category_name"], "answer": r["answer"],
                    "tokens_est": r["tokens_est"],
                })
            all_results["sliding_window"] = sw_results

        # --- Score and compare ---
        print(f"\n--- Scoring ---")
        scored = {}
        for method, results in all_results.items():
            scored[method] = []
            for i, r in enumerate(results):
                print(f"    Scoring {method} {i+1}/{len(results)}", end=" ", flush=True)
                # LLM judge score
                s = score_answer(r["question"], r["expected"], r["answer"], r["category"])
                # Token F1 (deterministic, no LLM call)
                is_adv = (r["expected"] == "NOT_ANSWERABLE")
                tf1 = token_f1(r["expected"], r["answer"], is_adversarial=is_adv)
                print("✓")
                scored[method].append({**r, **s, **tf1})

        print_results(scored, ci, conv)

        # Save
        out_file = f"eval_conv{ci}_results.json"
        with open(out_file, "w") as f:
            json.dump(scored, f, indent=2, default=str)
        print(f"\nSaved to {out_file}")


def print_results(scored, conv_idx, conv):
    methods = list(scored.keys())

    print(f"\n{'='*70}")
    print(f"RESULTS — Conv {conv_idx}: {conv['speakers']}")
    print(f"{'='*70}")

    # Aggregate
    print(f"\n{'Metric':<25}", end="")
    for m in methods: print(f"{m:<18}", end="")
    print()
    print("-" * (25 + 18 * len(methods)))

    for label, fn in [
        ("Avg Score (0-2)", lambda rs: f"{sum(r['score'] for r in rs)/max(len(rs),1):.2f}"),
        ("Avg F1 (LLM judge)", lambda rs: f"{sum(r['f1'] for r in rs)/max(len(rs),1):.3f}"),
        ("Avg F1 (token)", lambda rs: f"{sum(r.get('token_f1',0) for r in rs)/max(len(rs),1):.3f}"),
        ("Fully Correct", lambda rs: f"{sum(1 for r in rs if r['score']==2)}/{len(rs)}"),
        ("Hallucinations", lambda rs: f"{sum(1 for r in rs if r.get('hallucination'))}/{len(rs)}"),
        ("Avg Tok/Query", lambda rs: f"{sum(r['tokens_est'] for r in rs)/max(len(rs),1):.0f}"),
        ("  T1 (summary)", lambda rs: f"{sum(r.get('tokens_t1',0) for r in rs)/max(len(rs),1):.0f}"),
        ("  T2 pick", lambda rs: f"{sum(r.get('tokens_t2_pick',0) for r in rs)/max(len(rs),1):.0f}"),
        ("  T2 answer", lambda rs: f"{sum(r.get('tokens_t2_ans',0) for r in rs)/max(len(rs),1):.0f}"),
        ("  T3 (raw)", lambda rs: f"{sum(r.get('tokens_t3',0) for r in rs)/max(len(rs),1):.0f}"),
    ]:
        print(f"{label:<25}", end="")
        for m in methods: print(f"{fn(scored[m]):<18}", end="")
        print()

    # By category
    cats = sorted(set(r["category"] for r in scored[methods[0]]))
    print(f"\nBY CATEGORY (avg score / token F1):")
    print(f"{'Category':<18}", end="")
    for m in methods: print(f"{m:<18}", end="")
    print()
    print("-" * (18 + 18 * len(methods)))

    for cat in cats:
        print(f"  {cat:<16}", end="")
        for m in methods:
            rs = [r for r in scored[m] if r["category"] == cat]
            if rs:
                avg_s = sum(r["score"] for r in rs) / len(rs)
                avg_tf = sum(r.get("token_f1", 0) for r in rs) / len(rs)
                print(f"{avg_s:.2f} / {avg_tf:.3f}    ", end="")
            else:
                print(f"{'n/a':<18}", end="")
        print()

    # Sample detailed results (first 3 per category)
    print(f"\nSAMPLE RESULTS (first 2 per category):")
    for cat in cats:
        samples = [r for r in scored[methods[0]] if r["category"] == cat][:2]
        for s in samples:
            q = s["question"][:65]
            print(f"\n  [{cat}] {q}")
            print(f"  Expected: {str(s['expected'])[:70]}")
            for m in methods:
                r = next(x for x in scored[m] if x["question"] == s["question"])
                sym = ["✗", "◐", "✓"][r["score"]]
                h = " ⚠" if r.get("hallucination") else ""
                a = r["answer"][:70].replace("\n", " ")
                print(f"    {sym} {m:<16} {a}{h}")


if __name__ == "__main__":
    # Usage: python evaluate.py [conv_index] [max_qa]
    conv_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    max_qa = int(sys.argv[2]) if len(sys.argv) > 2 else None
    run_evaluation(conv_indices=[conv_idx], max_qa=max_qa)
