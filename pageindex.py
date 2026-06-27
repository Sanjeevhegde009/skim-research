"""
PageIndex-style VECTORLESS RAG over the RAW conversation (experiment branch).

No embeddings, no HLMA compilation. The pipeline is index-then-navigate:
  1. BUILD a tree index — one node per session, each with an LLM-written "table of contents"
     summary of what that session covers (cached to pageindex_<sample_id>.json).
  2. At query time, the LLM NAVIGATES by reasoning — reads the session summaries and picks the
     session(s) most likely to hold the answer (no similarity math).
  3. READ the chosen sessions' raw turns and ANSWER with a conservative, date-aware,
     refuse-if-absent prompt (mirrors HLMA's discipline so the head-to-head is fair).

Reuses hlma's LLM plumbing (compiler_call / query_call) + config. Kept fully separate from
hlma.py so the stable framework is untouched.
"""
import json
import os
import re
from pathlib import Path

from hlma import compiler_call, query_call, estimate_tokens

# Bounded, type-gated reasoning in the ANSWER step (default off; base path unchanged).
# Only questions that need SERIAL computation get a scratchpad; the rest stay terse.
# The trigger is language-general (count/compare/duration/state/intersection) — NOT tuned to
# any dataset. Reasoning is bounded to a few steps in one capped call (it cannot run forever),
# and the final answer is extracted terse so token-F1 isn't penalised. Toggle: HLMA_PI_REASON=1
PI_REASON = os.environ.get("HLMA_PI_REASON", "").lower() in ("1", "true", "yes")

# RAG escalation on the refusal RESIDUE (default off; stable base path unchanged). When the base
# navigate->answer refuses, hand the question to pi_rag: decompose into 3-5 sub-questions, semantic-
# retrieve raw turns per sub-question, strict-synthesize (answer or refuse). Toggle: PI_RAG=1
PI_RAG = os.environ.get("PI_RAG", "").strip().lower() in ("1", "true", "yes")

_REASON_RE = re.compile(
    r'\bhow many\b|\bhow often\b|\bhow frequently\b'                       # counting
    r'|\bhow long\b|\bhow old\b'                                           # duration / age
    r'|\bbetween\b[^?]*\band\b'                                            # interval
    r'|\b(?:the most|the least|more than|fewer than|greater than|'
    r'biggest|largest|smallest|longest|shortest|'
    r'bigger|larger|smaller|older|younger|longer|shorter|taller|'
    r'higher|lower|closer|nearer|better|worse|greater|fewer)\b'           # comparison / superlative
    r'|\bin common\b|\bcompare\b|\bboth\b'                                 # intersection
    r'|\bas of\b|\bby (?:the )?(?:time|end)\b',                            # state-at-a-time
    re.IGNORECASE)


def needs_reasoning(question: str) -> bool:
    """True for questions that require serial computation (count, compare, duration, latest
    state, intersection) — language-general markers, not dataset-specific."""
    return bool(_REASON_RE.search(question))


def _turns_block(turns, dated=False):
    out = []
    for t in turns:
        tag = f"[{t.get('date','')}] " if dated else ""
        out.append(f"{tag}{t['speaker']}: {t['text']}")
    return "\n".join(out)


def build_index(conv, force=False):
    """One LLM-summarized node per session. Cached; ~one call per session (lighter than
    HLMA's multi-pass compile). Returns {'sample_id', 'nodes': [{key,date,summary,turns}]}."""
    path = Path(f"pageindex_{conv['sample_id']}.json")
    if path.exists() and not force:
        return json.loads(path.read_text())
    nodes = []
    for s in conv["sessions"]:
        body = _turns_block(s["turns"])
        prompt = (
            f"SESSION {s['key']} (date: {s['date_time']}):\n{body}\n\n"
            "Summarize what this session covers, for a navigation index that a question-"
            "answering model will read to decide whether to open this session. Capture the "
            "topics, events, key facts, names, and dates in 2-4 dense, specific sentences.")
        summary = compiler_call(prompt, temperature=0.0)
        nodes.append({
            "key": s["key"], "date": s["date_time"], "summary": summary.strip(),
            "turns": [{"dia_id": t.get("dia_id", ""), "speaker": t["speaker"], "text": t["text"]}
                      for t in s["turns"]],
        })
        print(f"  indexed {s['key']} ({len(s['turns'])} turns)")
    idx = {"sample_id": conv["sample_id"], "nodes": nodes}
    path.write_text(json.dumps(idx))
    return idx


NAV_MAX_SESSIONS = 3        # default: precision (open the best session(s))
NAV_BROAD_SESSIONS = 8      # aggregation: recall (open every session with a relevant instance)


def _navigate(question, nodes, broad=False, trace=None):
    """Reasoning-based node selection — the LLM picks relevant session(s) from the index.
    broad=True (for count/compare/aggregate questions): recall over precision — gather EVERY
    session that could hold a relevant instance, because the answer is scattered across the
    conversation and reasoning needs the complete set, not the single best session."""
    cap = NAV_BROAD_SESSIONS if broad else NAV_MAX_SESSIONS
    toc = "\n".join(f"SESSION {n['key']} ({n['date']}): {n['summary']}" for n in nodes)
    if broad:
        sys = (
            "You navigate a conversation by its session index. This question needs to COUNT, "
            "COMPARE, or AGGREGATE across the whole conversation, so COMPLETENESS matters more "
            "than precision: select EVERY session whose summary could contain a relevant "
            "instance — do NOT narrow to just the best one; missing a session means an "
            "undercount. Each line is 'SESSION <key> (<date>): <summary>'. Reply with the "
            f"session keys exactly as shown (up to {cap}), comma-separated, or NONE if none "
            "are relevant.")
    else:
        sys = (
            "You navigate a conversation by its session index to find where a question is "
            "answered. Each line is 'SESSION <key> (<date>): <summary>'. Pick the session(s) "
            f"whose summary most likely contains the answer — up to {cap}. Reply with the "
            "session keys exactly as shown (e.g. session_4), comma-separated, or NONE if no "
            "session is relevant.")
    usr = f"QUESTION: {question}\n\nSESSION INDEX:\n{toc}\n\nSession key(s):"
    out = query_call([{"role": "system", "content": sys},
                      {"role": "user", "content": usr}], temperature=0.0)
    if trace is not None:
        trace["nav_tokens"] = estimate_tokens(sys) + estimate_tokens(usr) + estimate_tokens(out)
        trace["nav_raw"] = out.strip()
        trace["nav_broad"] = broad
    keys = re.findall(r'session_\d+', out.lower())
    return list(dict.fromkeys(keys))[:cap]


def _answer(question, turns, trace=None):
    """Conservative, date-aware, refuse-if-absent answer — mirrors HLMA's discipline.
    For computational questions (when PI_REASON is on) the model reasons in a BOUNDED
    scratchpad first, then emits a terse final answer; otherwise it answers directly."""
    body = _turns_block(turns, dated=True)
    reasoning = PI_REASON and needs_reasoning(question)

    if reasoning:
        # Bounded serial computation: a few steps in ONE capped call, refusal preserved,
        # final answer extracted terse so the metric isn't penalised for the scratchpad.
        sys = (
            "You answer a question that needs COMPUTATION over the conversation excerpts — "
            "counting, comparing, a duration, or resolving the LATEST state when a value "
            "changed over time. Use ONLY the excerpts.\n"
            "Reason in AT MOST 4 short numbered steps (no more), then give the final answer.\n"
            "DATES: each excerpt is tagged with the date it was spoken; resolve relative "
            "wording against that date.\n"
            "If the excerpts do not support an answer, or the question assumes something they "
            "do not confirm, the final answer is exactly: This information is not available. "
            "Never guess from general knowledge.\n"
            "Format EXACTLY:\nSTEPS:\n1. ...\n2. ...\nANSWER: <short final answer>")
        usr = f"CONVERSATION EXCERPTS:\n{body}\n\nQUESTION: {question}\n\nReason, then answer:"
        out = query_call([{"role": "system", "content": sys},
                          {"role": "user", "content": usr}], temperature=0.0)
        if trace is not None:
            trace["ans_tokens"] = estimate_tokens(sys) + estimate_tokens(usr) + estimate_tokens(out)
            trace["reasoning"] = out.strip()            # captured for tracing
            trace["reasoned"] = True
        m = re.search(r'ANSWER:\s*(.+)', out, re.IGNORECASE | re.DOTALL)
        return (m.group(1).strip() if m else out.strip())

    sys = (
        "You answer a question using ONLY the provided conversation excerpts.\n"
        "Output ONLY the answer — a short phrase, name, date, or list; never restate the "
        "question, never explain.\n"
        "DATES: each excerpt is tagged with the date it was spoken; resolve relative wording "
        "('last week', 'yesterday') against that date to an absolute date.\n"
        "PREMISE CHECK: if the excerpts do not actually state or support the answer, or the "
        "question assumes an event/fact the excerpts do not confirm, output exactly: This "
        "information is not available. Never guess from general knowledge.")
    usr = f"CONVERSATION EXCERPTS:\n{body}\n\nQUESTION: {question}\n\nAnswer:"
    out = query_call([{"role": "system", "content": sys},
                      {"role": "user", "content": usr}], temperature=0.0)
    if trace is not None:
        trace["ans_tokens"] = estimate_tokens(sys) + estimate_tokens(usr) + estimate_tokens(out)
    return out.strip()


_REFUSAL = "this information is not available"


def _is_refusal(ans):
    return ans.strip().lower().startswith(_REFUSAL)


def _collect_turns(nodes, keys):
    turns = []
    for n in nodes:
        if n["key"] in keys:
            for t in n["turns"]:
                turns.append({**t, "date": n["date"]})
    return turns


def query(index, question):
    """Navigate → read chosen sessions' raw turns → answer. Refuse if nav finds nothing.

    With PI_RAG, a base REFUSAL hands the question to pi_rag (decompose → semantic-retrieve raw
    turns → strict synthesis); its answer is adopted only if it recovers one, else the refusal
    stands. Stable base path is byte-identical when PI_RAG is off. Trace records every step."""
    nodes = index["nodes"]
    trace = {}
    # Aggregation questions need BROAD navigation (scattered evidence) AND reasoning over the
    # complete set — coupled. Gated by PI_REASON, so the base path is precision-navigate only.
    broad = PI_REASON and needs_reasoning(question)
    keys = _navigate(question, nodes, broad=broad, trace=trace)
    if keys:
        turns = _collect_turns(nodes, keys)
        ans = _answer(question, turns, trace=trace)
    else:
        turns, ans = [], "This information is not available."

    # RAG escalation on the refusal residue (only fires when PI_RAG is set and the base refused).
    if PI_RAG and _is_refusal(ans):
        import pi_rag
        trace["rag_fired"] = True
        rag = pi_rag.answer(question, index, trace=trace)
        if rag and not _is_refusal(rag["answer"]):
            ans = rag["answer"]
            keys = rag.get("sessions") or keys
            turns = rag.get("turns", turns)
            trace["rag_recovered"] = True
        trace["nav_tokens"] = trace.get("nav_tokens", 0) + trace.get("rag_decomp_tokens", 0)
        trace["ans_tokens"] = trace.get("ans_tokens", 0) + trace.get("rag_ans_tokens", 0)

    trace["sessions"] = keys
    trace["excerpts"] = [{"dia_id": t.get("dia_id", ""), "speaker": t["speaker"],
                          "text": t["text"], "date": t.get("date", "")} for t in turns]
    return {"answer": ans, "sessions": keys, "trace": trace}
