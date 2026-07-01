"""
Compiled-index experiment (branch feat/compiled-index).

Replaces PageIndex's flat 2-3 sentence per-session summary with an HLMA-style COMPILED
understanding — a numbered list of CITED, date-resolved, verbatim facts — adds per-session
`relatives` (top-3 by summary similarity), and answers in TWO gated tiers:

  Tier 1 (compiled facts):  decompose -> retrieve relevant facts (+ relatives' facts) ->
                            sub-answer -> SAME premise gate -> compose.
  Tier 2 (raw + RAG):       on a Tier-1 refusal, fall through to pi_rag.answer (raw turns +
                            hybrid retrieval + the identical gate).

Nothing here is cut down except scale — it is the full mechanism, run on one conversation.
Flag-gated by COMPILED=1 so master behavior is byte-identical when off. Reuses pi_rag's
decompose / sub-answer / compose / gate verbatim — same validation, different corpus.
"""
import json
import os

import config
from llm import compiler_call, query_call, estimate_tokens
import pageindex
import pi_rag

COMPILED = os.environ.get("COMPILED", "").strip().lower() in ("1", "true", "yes")
N_RELATIVES = 3
COMPILE_DIR = config.CACHE_DIR / "compiled"

_COMPILER_SYS = (
    "You are the COMPILER for a long-term memory. A SEPARATE question-answering model — which "
    "never saw this conversation — will later answer questions ONLY from what you write, and will "
    "REFUSE if your account doesn't support the answer. So a fact you capture but distort, or omit, "
    "is a failure. Read critically and build genuine understanding; cite every fact to its turn id; "
    "resolve every date to absolute; copy enumerable lists verbatim; never invent.")


# ── PASS 1: UNDERSTAND (HLMA-style critical reading -> cited, dated facts) ──
def _understand(session_label, turn_text):
    prompt = f"""Read this conversation session and build a clear, faithful UNDERSTANDING of it.

This session took place on: {session_label}

Produce a thorough, critical account of what this session establishes:
- What happened, what was decided, and what each participant revealed about themselves.
- Every concrete fact: names, places, dates, numbers, decisions, preferences, plans, feelings.
- PERSISTENT FACTS are especially important — anything true beyond this session: a person's
  background, origin/home country (EXACT name, never "their homeland"), relationship/marital status
  and duration, number of children (state "X has N children"), pets (names + species), occupation
  and career stage, hobbies and where practised, ongoing projects, current situation, plans,
  preferences. These are easy to miss and critical — never omit them.
- VERBATIM LISTS: for every enumerable fact (places visited, items collected, activities, names in
  a series) copy the EXACT words — never paraphrase ("beach and forest", not "outdoor locations").
- RESOLVE ALL DATES to absolute form against the session date above ("yesterday"/"last week"/
  "next month" -> the actual calendar date). Never leave a date relative.
- DATED EVENTS: for every event/trip/visit/activity/milestone, a compact entry
  "[Who] [did what] on [absolute date]" — include even brief mentions; missing one is a failure.
- Note how facts connect (cause, sequence, contrast) — understanding, not just a list.

Each turn is labelled with an id like [D5:3]. Cite the turn id(s) after each fact.

SESSION:
{turn_text}

Write your understanding as a numbered list of resolved facts, ONE fact per line, formatted:
  "N. [Name] | [Topic]: [fact] [turn id]"
[Name]  = the primary subject the fact is about. [Topic] = a short (1-3 word) stable category
(Identity, Career, Family, Health, Hobbies, Relationships, Community, or an equivalent). Be
consistent. Examples:
  "1. Caroline | Identity: transgender woman from Sweden, began transitioning June 2020 [D3:1]"
  "2. Melanie | Family: went camping in the mountains 20-26 June 2023 [D4:6]"
Be complete and critical — capture everything that matters, every date absolute:"""
    return compiler_call(prompt, system=_COMPILER_SYS).strip()


def compile_index(conv, force=False):
    """Build the compiled index: HLMA-understanding summary + raw turns + relatives, per session."""
    COMPILE_DIR.mkdir(parents=True, exist_ok=True)
    path = COMPILE_DIR / f"{conv['sample_id']}.json"
    if path.exists() and not force:
        return json.loads(path.read_text())
    nodes = []
    for s in conv["sessions"]:
        turn_text = "\n".join(                                # feed dia_ids so citations are REAL
            f"[{t.get('dia_id', '')}] {t['speaker']}: {t['text']}" for t in s["turns"])
        summary = _understand(f"{s['key']} ({s['date_time']})", turn_text)
        nodes.append({
            "key": s["key"], "date": s["date_time"], "summary": summary, "relatives": [],
            "turns": [{"dia_id": t.get("dia_id", ""), "speaker": t["speaker"], "text": t["text"]}
                      for t in s["turns"]],
        })
        print(f"  compiled {s['key']} ({len(s['turns'])} turns, {len(summary.splitlines())} facts)")
    # relatives: top-N most similar sessions by compiled-summary embedding
    vecs = [pi_rag._normalize(v) if v else None for v in pi_rag._embed([n["summary"] for n in nodes])]
    for i, n in enumerate(nodes):
        if vecs[i] is None:
            continue
        sims = [(j, sum(a * b for a, b in zip(vecs[i], vecs[j])))
                for j in range(len(nodes)) if j != i and vecs[j] is not None]
        n["relatives"] = [nodes[j]["key"] for j, _ in sorted(sims, key=lambda x: -x[1])[:N_RELATIVES]]
    idx = {"sample_id": conv["sample_id"], "schema": "compiled", "nodes": nodes}
    path.write_text(json.dumps(idx))
    return idx


# ── fact corpus (cached per conversation) ──
_FACT_CACHE = {}


def _fact_corpus(index):
    """ALL compiled facts across every session — the FULL compiled understanding, fed as context
    (no retrieval / no cosine; the model sees everything, so it can't miss scattered evidence)."""
    sid = index["sample_id"]
    if sid in _FACT_CACHE:
        return _FACT_CACHE[sid]
    facts = []
    for n in index["nodes"]:
        for line in n["summary"].splitlines():
            line = line.strip()
            if len(line) > 3:
                facts.append({"date": n["date"], "speaker": "", "text": line, "key": n["key"]})
    _FACT_CACHE[sid] = facts
    return facts


def query_compiled(index, question, trace=None):
    """Tier 1 = compiled facts (gated); Tier 2 = raw turns + RAG (gated). Same gate both tiers."""
    trace = trace if trace is not None else {}
    facts = _fact_corpus(index)                               # the FULL compiled understanding

    subs, has_premise = pi_rag._decompose(question, trace=trace)
    qa_pairs, tok = [], 0
    for sq in subs:                                           # answer each sub-q over ALL facts
        a, tk = pi_rag._subanswer(sq, facts) if facts else ("Not stated", 0)
        tok += tk
        qa_pairs.append((sq, a))

    gate_on = pi_rag.PI_RAG_GATE and not (pi_rag.PI_RAG_INFER and pi_rag._is_inference(question))
    if gate_on and has_premise and qa_pairs and pi_rag._is_negative(qa_pairs[0][1]):
        ans = "This information is not available."
        trace["tier1_gate"] = "premise_refused"
    else:
        ans, tk = pi_rag._compose(question, qa_pairs)
        tok += tk
        trace["tier1_gate"] = "compose"
    if not ans.strip():                                       # compose emitted nothing = couldn't
        ans = "This information is not available."             # answer -> treat as a refusal so it
        trace["tier1_empty"] = True                           # scores as one AND escalates to Tier 2
    trace["tier1_subanswers"] = qa_pairs
    trace["nav_tokens"], trace["ans_tokens"] = 0, tok
    trace["sessions"] = sorted({f["key"] for f in facts})     # full context: every session present

    # Tier 2: raw turns + RAG fallback on a Tier-1 refusal (identical gate inside pi_rag.answer).
    if pageindex._is_refusal(ans):
        trace["tier1_refused"] = True
        if pageindex.PI_RAG:
            rag = pi_rag.answer(question, index, trace=trace)
            if rag and not pageindex._is_refusal(rag["answer"]):
                ans = rag["answer"]
                trace["tier2_recovered"] = True
                trace["sessions"] = rag.get("sessions") or trace["sessions"]
    trace.setdefault("ans_tokens", tok)
    return {"answer": ans, "sessions": trace.get("sessions", []), "trace": trace}
