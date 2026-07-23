"""
Run the PageIndex + RAG framework on LongMemEval_S — our SECOND benchmark.

Why a separate runner: LongMemEval gives each QUESTION its own haystack of ~50-80 sessions
(~115K tokens, designed to OVERFLOW the context window), so we build ONE index PER QUESTION
rather than one per conversation. The base framework is reused verbatim — `pageindex.query`
(navigate -> read -> answer, with the PI_RAG escalation hook) and `pi_rag.answer` are UNCHANGED.

READER SWAP (token-economics / reader-controlled comparison): set LME_READER_MODEL to swap the
query model (e.g. gpt-4o) WITHOUT editing config.py. Indexes (compiler) and embeddings
(text-embedding-3-small) are reader-INDEPENDENT and cached, so swapping the reader re-runs ONLY the
per-query calls — cheap. Outputs are suffixed by the reader so a non-default run never clobbers the
gpt-4.1-mini results. Actual reader-token usage is metered (real `usage` from the API) and priced.

BEST-CONFIG GUARD: aborts unless PI_RAG / HYBRID / INFER are all ON, so a weaker config can't be
mistaken for the real one. pageindex.py / pi_rag.py / evaluate.py / llm.py stay byte-identical.

Run (Windows cmd):
  set OPENAI_API_KEY=sk-...
  set PI_RAG=1 & set PI_RAG_HYBRID=1 & set PI_RAG_INFER=1
  python run_longmemeval.py 150                    # default reader (gpt-4.1-mini)
  set LME_READER_MODEL=gpt-4o & python run_longmemeval.py 150   # reader-controlled run (gpt-4o)

Resumable: per-question indexes (longmemeval_index/), the shared summary cache, and partial
results flush every 10 instances.
"""
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import requests

import config
import llm
import longmemeval_loader
from evaluate import score_answer, token_f1
import pageindex
import pi_rag

# ── reader override — swap the query model/provider without editing config.py ──
# LME_READER_MODEL    : e.g. gpt-4o, or a local Ollama tag like gemma4:e4b
# LME_READER_PROVIDER : "openai" (default) or "ollama" for a local model
# LME_OLLAMA_NUM_CTX  : Ollama context window — MUST be big enough for the ToC nav prompt (~4.6K tok),
#                       or Ollama silently truncates it and navigation breaks.
DEFAULT_READER = config.QUERY_MODEL
DEFAULT_PROVIDER = config.QUERY_PROVIDER
READER = os.environ.get("LME_READER_MODEL", DEFAULT_READER).strip()      # .strip(): Windows `set X=v &`
PROVIDER = os.environ.get("LME_READER_PROVIDER", DEFAULT_PROVIDER).strip()  # keeps the trailing space
config.QUERY_MODEL = READER
config.QUERY_PROVIDER = PROVIDER
OLLAMA_NUM_CTX = int(os.environ.get("LME_OLLAMA_NUM_CTX", "8192"))
_tag = "" if READER == DEFAULT_READER else "_" + re.sub(r"[^a-z0-9]+", "", READER.lower())
if pi_rag.PI_RAG_HYDE:                                # HyDE A/B writes to its OWN results/summary
    _tag += "_hyde"                                   # so the validated baseline files stay intact
if pageindex.PI_RAG_FORCE_AGG:                        # forced-aggregate-escalation A/B: own files too
    _tag += "_aggforce"
if pageindex.PI_RAG_VERIFY:                           # grounding-verify gate A/B: own files too
    _tag += "_verify"
if pageindex.PI_NAV_BROAD:                            # broad-nav A/B: own files too
    _tag += "_navbroad"
if pageindex.PI_REASON:                               # reasoning answer-step A/B: own files too
    _tag += "_reason"
if pageindex.PI_DATEMATH:                             # deterministic date-math A/B: own files too
    _tag += "_datemath"
if pageindex.PI_RECENCY:                              # value-history/recency A/B: own files too
    _tag += "_recency"

config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
_LME_RESULTS = config.RESULTS_DIR / "longmemeval"
_LME_RESULTS.mkdir(parents=True, exist_ok=True)
INDEX_DIR = config.CACHE_DIR / "longmemeval_index"   # per-question index caches (reader-independent; shared)
SUMMARY_CACHE = config.CACHE_DIR / "longmemeval_session_summaries.json"
RESULTS_PATH = _LME_RESULTS / f"results{_tag}.json"
SUMMARY_PATH = _LME_RESULTS / f"summary{_tag}.txt"

# ── token economics ──────────────────────────────────────────────────────────────
# $/1M tokens (input, output). VERIFY against current OpenAI pricing — these change over time.
PRICES = {
    "gpt-4o":       (2.50, 10.00),
    "gpt-4.1-mini": (0.40,  1.60),
    "gpt-4o-mini":  (0.15,  0.60),
}
_TOK = {"in": 0, "out": 0, "calls": 0}         # cumulative ACTUAL reader usage
_Q = {"in": 0, "out": 0, "calls": 0}           # current-question reader usage


def _cost(model, tin, tout):
    pin, pout = PRICES.get(model, (0.0, 0.0))
    return tin / 1e6 * pin + tout / 1e6 * pout


def _tracking_query_call(messages, temperature=0.1):
    """Mirror of llm.query_call (openai/ollama) that ALSO records actual token usage — and, for
    Ollama, sets num_ctx so the long navigation prompt isn't silently truncated. Installed onto
    pageindex/pi_rag's query_call binding so every reader call is metered; llm.py stays untouched."""
    if config.QUERY_PROVIDER == "ollama":
        def _do():
            resp = requests.post(config.OLLAMA_URL,
                json={"model": config.QUERY_MODEL, "messages": messages, "stream": False,
                      "think": False,
                      "options": {"temperature": temperature, "num_ctx": OLLAMA_NUM_CTX}},
                timeout=600)
            resp.raise_for_status()
            j = resp.json()
            pin, pout = j.get("prompt_eval_count", 0), j.get("eval_count", 0)
            _TOK["in"] += pin; _TOK["out"] += pout; _TOK["calls"] += 1
            _Q["in"] += pin; _Q["out"] += pout; _Q["calls"] += 1
            return j["message"]["content"].strip()
    else:
        key = llm._get_api_key(config.QUERY_API_KEY_ENV)
        if not key:
            return ""
        def _do():
            body = {"model": config.QUERY_MODEL, "temperature": temperature,
                    "max_tokens": 1024, "messages": messages}
            resp = requests.post("https://api.openai.com/v1/chat/completions",
                                 headers={"Authorization": f"Bearer {key}",
                                          "Content-Type": "application/json"}, json=body, timeout=120)
            resp.raise_for_status()
            j = resp.json()
            u = j.get("usage", {}) or {}
            pin, pout = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
            _TOK["in"] += pin; _TOK["out"] += pout; _TOK["calls"] += 1
            _Q["in"] += pin; _Q["out"] += pout; _Q["calls"] += 1
            return j["choices"][0]["message"]["content"].strip()
    try:
        return llm._api_call_with_retry(_do)
    except Exception as e:
        print(f"  [QUERY ERROR] {e}")
        return ""


if config.QUERY_PROVIDER in ("openai", "ollama"):   # meter reader calls (pageindex + pi_rag bindings)
    pageindex.query_call = _tracking_query_call
    pi_rag.query_call = _tracking_query_call

# LongMemEval has very long single turns (e.g. an assistant "list of 100 items") that exceed the
# embedding model's 8192-token per-input cap -> pi_rag._embed 400s and stalls on retries. Truncate
# each embedding input to a safe char budget (retrieval only; the FULL turn text is still used to
# answer). Non-invasive wrapper on pi_rag._embed; pi_rag.py stays byte-identical.
_EMB_CHAR_CAP = 20000          # ~5-6K tokens, safely under the 8192-token embedding limit
_orig_embed = pi_rag._embed
def _capped_embed(texts):
    return _orig_embed([t[:_EMB_CHAR_CAP] for t in texts])
pi_rag._embed = _capped_embed


def _session_summary(body, key, date, cache):
    """Per-session ToC summary, cached by session CONTENT so a session reused across questions is
    summarized once. The prompt mirrors pageindex.build_index — keep the two in sync."""
    h = hashlib.sha1(body.encode("utf-8")).hexdigest()
    s = cache.get(h)
    if s is None:
        prompt = (
            f"SESSION {key} (date: {date}):\n{body}\n\n"
            "Summarize what this session covers, for a navigation index that a question-"
            "answering model will read to decide whether to open this session. Capture the "
            "topics, events, key facts, names, and dates in 2-4 dense, specific sentences.")
        s = pageindex.compiler_call(prompt, temperature=0.0).strip()
        cache[h] = s
    return s


def build_index_cached(conv, cache):
    """One index per question, written in the EXACT pageindex_<sample_id>.json schema that
    pageindex.query / pi_rag read — so the stable framework is untouched. Re-summarizes only
    sessions not already in the content cache."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    path = INDEX_DIR / f"{conv['sample_id']}.json"
    if path.exists():
        return json.loads(path.read_text())
    nodes = []
    for s in conv["sessions"]:
        body = pageindex._turns_block(s["turns"])
        summary = _session_summary(body, s["key"], s["date_time"], cache)
        nodes.append({
            "key": s["key"], "date": s["date_time"], "summary": summary,
            "turns": [{"dia_id": t.get("dia_id", ""), "speaker": t["speaker"], "text": t["text"]}
                      for t in s["turns"]],
        })
    idx = {"sample_id": conv["sample_id"], "nodes": nodes}
    path.write_text(json.dumps(idx))
    return idx


def _flush(results, cache):
    SUMMARY_CACHE.write_text(json.dumps(cache))
    RESULTS_PATH.write_text(json.dumps(results, indent=1))


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None

    # ── header + best-config guard ───────────────────────────────────────────────
    flags = {"PI_RAG": pageindex.PI_RAG, "HYBRID": pi_rag.PI_RAG_HYBRID,
             "INFER": pi_rag.PI_RAG_INFER, "GATE": pi_rag.PI_RAG_GATE, "HYDE": pi_rag.PI_RAG_HYDE,
             "FORCE_AGG": pageindex.PI_RAG_FORCE_AGG, "VERIFY": pageindex.PI_RAG_VERIFY,
             "NAV_BROAD": pageindex.PI_NAV_BROAD, "REASON": pageindex.PI_REASON,
             "DATEMATH": pageindex.PI_DATEMATH, "RECENCY": pageindex.PI_RECENCY}
    rd = f"{config.QUERY_PROVIDER}/{READER}" + (
        f"   [overridden from {DEFAULT_READER}]" if READER != DEFAULT_READER else "")
    print("=" * 70)
    print("LongMemEval — PageIndex + RAG")
    print(f"  reader (query): {rd}")
    if config.QUERY_PROVIDER == "ollama":
        print(f"  ollama num_ctx: {OLLAMA_NUM_CTX}  (raise via LME_OLLAMA_NUM_CTX if nav looks truncated)")
    print(f"  index/judge:    {config.COMPILER_PROVIDER}/{config.COMPILER_MODEL}")
    print(f"  PI_RAG={flags['PI_RAG']}  hybrid={flags['HYBRID']}  infer={flags['INFER']}  "
          f"gate={flags['GATE']}  hyde={flags['HYDE']}  force_agg={flags['FORCE_AGG']}  "
          f"verify={flags['VERIFY']}  nav_broad={flags['NAV_BROAD']}  reason={flags['REASON']}  "
          f"datemath={flags['DATEMATH']}  recency={flags['RECENCY']}  tau={pi_rag.TAU}")
    print(f"  outputs: {RESULTS_PATH}  +  {SUMMARY_PATH}")
    print("=" * 70)
    missing = [k for k in ("PI_RAG", "HYBRID", "INFER") if not flags[k]]
    if missing:
        print(f"\nABORT: best-config flags not all ON (missing: {', '.join(missing)}).")
        print("This run would use a WEAKER config than our best LoCoMo result (tag rag-allconv).")
        print("Set all three first:  set PI_RAG=1 & set PI_RAG_HYBRID=1 & set PI_RAG_INFER=1")
        sys.exit(1)

    convs = longmemeval_loader.load(limit=limit)
    only_type = os.environ.get("LME_ONLY_TYPE", "").strip()   # comma-sep type(s) -> run just those
    if only_type:
        types = {t.strip() for t in only_type.split(",") if t.strip()}
        convs = [c for c in convs if c["qa"][0]["category_name"] in types]
        print(f"\n[filter] LME_ONLY_TYPE={sorted(types)} -> {len(convs)} instances (of the stratified {limit})")
    comp = Counter(c["qa"][0]["category_name"] for c in convs)
    n_abs = sum(c["qa"][0]["abstention"] for c in convs)
    kind = f"stratified subset of {limit}" if (limit and limit < 500) else "full file"
    print(f"\nloaded {len(convs)} LongMemEval instances ({kind})")
    print(f"  by type: {dict(comp)}")
    print(f"  abstention: {n_abs}\n")

    cache = json.loads(SUMMARY_CACHE.read_text()) if SUMMARY_CACHE.exists() else {}

    # resume: reuse already-scored questions from a prior (possibly interrupted) run of THIS reader,
    # so a restart doesn't re-pay for finished GPT-4o queries. Delete RESULTS_PATH to force fresh.
    done = {}
    if RESULTS_PATH.exists():
        try:
            done = {r["question_id"]: r for r in json.loads(RESULTS_PATH.read_text())}
        except Exception:
            done = {}
    if done:
        print(f"resuming: {len(done)} questions already in {RESULTS_PATH} will be reused\n")

    results = []
    for i, conv in enumerate(convs):
        q = conv["qa"][0]
        if conv["sample_id"] in done:                     # already scored -> reuse, keep economics whole
            rec = done[conv["sample_id"]]
            results.append(rec)
            _TOK["in"] += rec.get("tok_in", 0); _TOK["out"] += rec.get("tok_out", 0)
            _TOK["calls"] += rec.get("reader_calls", 0)
            print(f"  {i+1}/{len(convs)} [cached] {q['question'][:44]}")
            continue
        is_abs = (q["answer"] == "NOT_ANSWERABLE")
        idx = build_index_cached(conv, cache)
        _Q["in"] = 0; _Q["out"] = 0; _Q["calls"] = 0      # meter THIS question's reader usage
        r = pageindex.query(idx, q["question"], question_date=conv.get("question_date", ""))
        qin, qout, qcalls = _Q["in"], _Q["out"], _Q["calls"]
        ans = r["answer"]
        sc = score_answer(q["question"], q["answer"], ans, q["category_name"])
        tf = token_f1(q["answer"], ans, is_adversarial=is_abs)
        tr = r.get("trace", {})
        results.append({
            "question_id": conv["sample_id"],
            "question": q["question"],
            "expected": q["answer"],
            "question_type": q["category_name"],
            "abstention": is_abs,
            "answer": ans,
            "score": sc["score"],
            "correct": int(sc["score"] == 2),
            "f1": sc.get("f1", 0.0),
            "token_f1": tf["token_f1"],
            "hallucination": bool(sc.get("hallucination")),
            "sessions": r["sessions"],
            "n_sessions": len(idx["nodes"]),
            "rag_fired": tr.get("rag_fired", False),
            "rag_forced": tr.get("rag_forced", False),
            "rag_verify": tr.get("rag_verify", False),
            "nav_broad": tr.get("nav_broad", False),
            "datemath": tr.get("datemath", ""),
            "datemath_events": tr.get("datemath_events", ""),
            "datemath_table": tr.get("datemath_table", ""),
            "recency": tr.get("recency", ""),
            "recency_retrieved": tr.get("recency_retrieved", 0),
            "recency_states": tr.get("recency_states", ""),
            "recency_table": tr.get("recency_table", ""),
            "verify_verdict": tr.get("verify_verdict", ""),
            "base_answer": tr.get("base_answer", ""),
            "rag_recovered": tr.get("rag_recovered", False),
            "rag_gate": tr.get("rag_gate", ""),
            "reader": READER,
            "tok_in": qin, "tok_out": qout, "reader_calls": qcalls,
            "cost_reader_usd": round(_cost(READER, qin, qout), 6),
            "est_nav_tok": tr.get("nav_tokens", 0), "est_ans_tok": tr.get("ans_tokens", 0),
        })
        mark = "OK " if sc["score"] == 2 else ("hit" if (is_abs and sc["score"]) else "  X")
        rag = ("rag+" if tr.get("rag_recovered") else "rag.") if tr.get("rag_fired") else "    "
        print(f"  {i+1}/{len(convs)} [{q['category_name'][:16]:<16}] {mark} {rag} "
              f"sess={len(idx['nodes']):>3} {qin+qout:>5}tok | {q['question'][:32]}", flush=True)
        if (i + 1) % 10 == 0:
            _flush(results, cache)

    _flush(results, cache)

    # one-time INDEX cost estimate (compiler model), from UNIQUE session bodies + their summaries
    seen, idx_in, idx_out = set(), 0, 0
    for conv in convs:
        for s in conv["sessions"]:
            body = pageindex._turns_block(s["turns"])
            h = hashlib.sha1(body.encode("utf-8")).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            idx_in += llm.estimate_tokens(body) + 50        # session body + prompt boilerplate
            idx_out += llm.estimate_tokens(cache.get(h, ""))  # the cached summary
    _report(results, idx_in, idx_out)


def _report(results, idx_in=0, idx_out=0):
    n = len(results)
    if not n:
        print("no results"); return
    ans = [r for r in results if not r["abstention"]]          # answerable
    abst = [r for r in results if r["abstention"]]             # abstention (_abs)

    def acc(rs):
        return sum(r["correct"] for r in rs) / len(rs) if rs else 0.0

    def avgtf(rs):
        return sum(r["token_f1"] for r in rs) / len(rs) if rs else 0.0

    bytype = defaultdict(list)
    for r in ans:
        bytype[r["question_type"]].append(r)

    fired = [r for r in results if r["rag_fired"]]
    rec = [r for r in fired if r["rag_recovered"]]

    L = []
    L.append("=" * 70)
    L.append(f"LongMemEval — PageIndex + RAG   ({datetime.now():%Y-%m-%d %H:%M})")
    L.append(f"reader={READER}  index/judge={config.COMPILER_MODEL}  instances={n}")
    L.append("NOTE: accuracy here = strict gpt-4o-mini score==2. For the OFFICIAL comparable number,")
    L.append(f"      re-score with:  python rescore_longmemeval.py {RESULTS_PATH}")
    L.append("=" * 70)
    L.append("")
    L.append(f"OVERALL accuracy : {acc(results):.3f}  ({sum(r['correct'] for r in results)}/{n})")
    L.append(f"  answerable     : {acc(ans):.3f}  ({sum(r['correct'] for r in ans)}/{len(ans)})   "
             f"token-F1 {avgtf(ans):.3f}")
    L.append(f"  abstention     : {acc(abst):.3f}  ({sum(r['correct'] for r in abst)}/{len(abst)})"
             f"   (correct = refused)")
    L.append(f"  hallucinations : {sum(1 for r in results if r['hallucination'])}/{n}")
    L.append("")
    L.append(f"{'question_type (answerable)':<28}{'n':>5}{'acc':>9}{'tokF1':>9}")
    L.append("-" * 51)
    for t in sorted(bytype):
        v = bytype[t]
        L.append(f"{t:<28}{len(v):>5}{acc(v):>9.3f}{avgtf(v):>9.3f}")
    L.append(f"{'abstention':<28}{len(abst):>5}{acc(abst):>9.3f}{avgtf(abst):>9.3f}")
    L.append("")
    L.append(f"RAG escalation fired on {len(fired)}/{n} base refusals; recovered {len(rec)}.")
    L.append(f"avg haystack: {sum(r['n_sessions'] for r in results) / n:.0f} sessions/instance.")

    # ── token economics (ACTUAL reader usage from the API) ──
    tin, tout, ncalls = _TOK["in"], _TOK["out"], _TOK["calls"]
    L.append("")
    L.append(f"TOKEN ECONOMICS  (reader={READER}; actual API usage — VERIFY prices in PRICES)")
    L.append("-" * 51)
    if ncalls:
        L.append(f"  reader: {ncalls} calls,  {tin:,} in + {tout:,} out = {tin + tout:,} tok")
        L.append(f"          {(tin + tout) / n:,.0f} tok/question  "
                 f"({tin / n:,.0f} in / {tout / n:,.0f} out)")
        rc = _cost(READER, tin, tout)
        if READER in PRICES:
            L.append(f"  reader cost @ {READER:<13}: ${rc:>8.3f}   (${rc / n:.4f}/question)")
        else:
            L.append(f"  reader cost @ {READER:<13}: $   0.000   (LOCAL model — no API cost)")
        for alt in ("gpt-4o", "gpt-4.1-mini"):       # what this token volume would cost on the cloud
            if alt != READER:
                L.append(f"  same volume @ {alt:<13}: ${_cost(alt, tin, tout):>8.3f}")
    L.append(f"  one-time INDEX (est, {config.COMPILER_MODEL}): "
             f"${_cost(config.COMPILER_MODEL, idx_in, idx_out):.3f}  "
             f"({idx_in + idx_out:,} tok; built once, reused across reader runs)")

    out = "\n".join(L)
    print("\n" + out)
    SUMMARY_PATH.write_text(out + "\n", encoding="utf-8")
    print(f"\nwrote {RESULTS_PATH} + {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
