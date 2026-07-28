"""
RAG escalation for PageIndex (default off; flag PI_RAG=1). Fires ONLY on the residue — the
questions the base navigate->answer path refused. One bounded pipeline handles counting,
multi-hop, and adversarial alike:

  1. DECOMPOSE the question into 3-5 thoughtful sub-questions — a PREMISE PROBE (is the thing the
     question assumes even true?) plus one per fact/instance/link the answer needs.
  2. For each sub-question, SEMANTIC-retrieve the most similar RAW turns (embeddings + cosine over
     the whole conversation; turn vectors cached per conversation, built lazily on first use).
  3. STRICT SYNTHESIS: answer ONLY if the gathered evidence directly supports it; if a premise
     probe found nothing, or any needed fact is missing / weakly / tangentially evidenced, refuse.

Bounded by construction — fixed sub-questions, one retrieval each, one synthesis, no loop. Reuses
llm's request plumbing for the embeddings call and query_call for decompose/synthesis, so stable
PageIndex stays untouched.
"""
import json
import math
import os
import re
from collections import Counter

import requests

import config
from llm import _get_api_key, _api_call_with_retry, query_call, estimate_tokens

EMB_MODEL = "text-embedding-3-small"
EMB_DIM = 512        # reduced dims (3-small supports it): smaller cache, faster pure-python cosine
N_SUBQ = 5           # max sub-questions
TOPK = 6             # raw turns retrieved per sub-question
# Relevance threshold: a retrieved turn counts as evidence ONLY if cosine >= TAU. This gives
# retrieval a "nothing found" state — without it, top-k always returns the nearest turns and a
# false premise gets plausible-but-tangential "evidence" it can fabricate from. Tunable: PI_RAG_TAU.
TAU = float(os.environ.get("PI_RAG_TAU", "0.45"))   # start STRICT; calibrate down from logged sims
# Deterministic premise gate (default ON). Set PI_RAG_GATE=0 to A/B the pure two-stage (compose over
# sub-answers, no gate): recovers a bit more non-adversarial but fabricates more adversarial.
PI_RAG_GATE = os.environ.get("PI_RAG_GATE", "1").strip().lower() not in ("0", "false", "no")
# Hybrid retrieval: fuse semantic cosine with lexical BM25 (default off). Cosine misses facts
# stated once with low embedding similarity (a place "Sweden", a date, a named entity); BM25
# surfaces them by literal term overlap. Recovers query-term-overlap misses, not zero-overlap ones
# (a nickname "Jo" shares no token with the query). Toggle: PI_RAG_HYBRID=1
PI_RAG_HYBRID = os.environ.get("PI_RAG_HYBRID", "").strip().lower() in ("1", "true", "yes")
# Inference questions ("would/likely/might X...") have no factual premise to check — the answer is
# DERIVED from supporting facts, not stated — so the existence-gate wrongly refuses them. When on,
# such questions BYPASS the gate and are answered by compose from the details (compose still refuses
# if the basis isn't there). General English markers, not dataset-specific. Toggle: PI_RAG_INFER=1
PI_RAG_INFER = os.environ.get("PI_RAG_INFER", "").strip().lower() in ("1", "true", "yes")
_INFER_RE = re.compile(r'\b(would|likely|might|probably)\b', re.IGNORECASE)


def _is_inference(question):
    return bool(_INFER_RE.search(question or ""))


_REFUSAL = "this information is not available"


# ── embeddings ──────────────────────────────────────────────────────────────────

def _embed(texts):
    """Embed a list of texts via OpenAI embeddings; returns a list of vectors (lists of float)."""
    key = _get_api_key(config.QUERY_API_KEY_ENV)
    if not key:
        return [[] for _ in texts]

    def _do():
        resp = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": EMB_MODEL, "input": texts, "dimensions": EMB_DIM}, timeout=120)
        resp.raise_for_status()
        return [d["embedding"] for d in resp.json()["data"]]

    try:
        return _api_call_with_retry(_do)
    except Exception as e:
        print(f"  [RAG EMBED ERROR] {e}")
        return [[] for _ in texts]


def _normalize(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def turn_embeddings(index, force=False):
    """Embed every raw turn of the conversation once; cache normalized vectors to disk. Returns
    (turns, vecs) — turns[i] = {dia_id, speaker, text, date}, vecs[i] its unit vector. Built lazily,
    so only conversations that actually escalate pay for it."""
    sid = index["sample_id"]
    config.EMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = config.EMB_CACHE_DIR / f"{sid}.json"
    turns = []
    for n in index["nodes"]:
        for t in n["turns"]:
            turns.append({"dia_id": t.get("dia_id", ""), "speaker": t["speaker"],
                          "text": t["text"], "date": n["date"]})
    if path.exists() and not force:
        cache = json.loads(path.read_text())
        if cache.get("n") == len(turns) and cache.get("dim") == EMB_DIM:
            return turns, cache["vecs"]
    texts = [f"{t['speaker']}: {t['text']}" for t in turns]
    vecs = []
    for i in range(0, len(texts), 256):                       # batch the embedding calls
        vecs.extend(_embed(texts[i:i + 256]))
    if not vecs or not vecs[0]:
        return turns, []                                      # embeddings unavailable
    vecs = [_normalize(v) for v in vecs]
    path.write_text(json.dumps({"n": len(turns), "dim": EMB_DIM, "vecs": vecs}))
    return turns, vecs


def _cosine_topk(qvec, vecs, k, tau=None):
    """Top-k turns by cosine, but ONLY those clearing the relevance threshold τ. vecs are
    pre-normalized, so cosine == dot product. Returns [] when nothing is relevant enough —
    that 'nothing found' state is what lets a false premise be refused instead of fabricated."""
    if tau is None:
        tau = TAU
    qv = _normalize(qvec)
    sims = [sum(a * b for a, b in zip(qv, v)) for v in vecs]
    order = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
    return [(i, sims[i]) for i in order if sims[i] >= tau][:k]


# ── lexical retrieval (BM25) for hybrid search ─────────────────────────────────────
_TOK = re.compile(r"[a-z0-9]+")


def _toks(text):
    out = []
    for t in _TOK.findall(text.lower()):
        for suf in ("ing", "ed", "ly", "es", "s"):          # light stemming: move/moved, year/years
            if t.endswith(suf) and len(t) - len(suf) >= 3:
                t = t[:-len(suf)]
                break
        out.append(t)
    return out


class _BM25:
    """Pure-Python BM25 over the conversation's turns (no deps). Surfaces turns by literal term
    overlap, catching once-said facts cosine ranks too low."""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        toked = [_toks(d) for d in docs]
        self.N = len(toked)
        self.dl = [len(d) for d in toked]
        self.avgdl = (sum(self.dl) / self.N) if self.N else 1.0
        self.tf = [Counter(d) for d in toked]
        df = Counter()
        for d in toked:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def scores(self, query):
        q = [t for t in _toks(query) if t in self.idf]
        out = [0.0] * self.N
        for i in range(self.N):
            tf, dl = self.tf[i], self.dl[i]
            tot = 0.0
            for t in q:
                f = tf.get(t, 0)
                if f:
                    tot += self.idf[t] * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out[i] = tot
        return out


_BM25_CACHE = {}


def _bm25_for(sample_id, turns):
    if sample_id not in _BM25_CACHE:
        _BM25_CACHE[sample_id] = _BM25([f"{t['speaker']}: {t['text']}" for t in turns])
    return _BM25_CACHE[sample_id]


def _hybrid_hits(query, sims, cos_order, bm25, k, tau):
    """Reciprocal-Rank-Fusion of cosine (sims/cos_order) and BM25. Returns [(idx, cosine_sim)] for
    the top-k fused turns — a turn surfaces if it's semantically near OR lexically matching."""
    cos_top = [i for i in cos_order if sims[i] >= tau][:k]
    bm = bm25.scores(query)
    bm_top = [i for i in sorted(range(len(bm)), key=lambda i: bm[i], reverse=True) if bm[i] > 0][:k]
    rrf = {}
    for r, i in enumerate(cos_top):
        rrf[i] = rrf.get(i, 0.0) + 1.0 / (60 + r)
    for r, i in enumerate(bm_top):
        rrf[i] = rrf.get(i, 0.0) + 1.0 / (60 + r)
    fused = sorted(rrf, key=lambda i: rrf[i], reverse=True)[:k]
    return [(i, sims[i]) for i in fused]


# ── decomposition ───────────────────────────────────────────────────────────────

def _decompose(question, trace=None):
    """Returns (subs, has_premise). The PREMISE check is emitted FIRST and tagged, so the caller
    can gate on its sub-answer deterministically (a negative premise -> refuse before composing)."""
    sys = (
        "You break a hard question into sub-questions so a search system can VERIFY it against a "
        "conversation transcript, then either ANSWER it or prove it CANNOT be answered.\n"
        "FIRST, on its own line, output the PREMISE CHECK prefixed EXACTLY with 'PREMISE:' — a "
        "yes/no question testing whether the thing the question ASSUMES is even true (e.g. 'When "
        "did X visit Japan?' -> 'PREMISE: Is there any evidence X went to Japan?'; 'What does X "
        "love about having turtles?' -> 'PREMISE: Does X own turtles?').\n"
        "THEN 2-4 more sub-questions, one per line, for the facts/instances/links the answer needs "
        "(for counting, ask for each occurrence; for multi-hop, ask for each step).\n"
        "Output ONLY these lines. Never answer them.")
    usr = f"QUESTION: {question}\n\nSub-questions:"
    out = query_call([{"role": "system", "content": sys},
                      {"role": "user", "content": usr}], temperature=0.0)
    premise, others = None, []
    for ln in out.splitlines():
        clean = re.sub(r'^\s*[-*\d.)]+\s*', '', ln).strip()
        if re.match(r'(?i)premise\s*:', clean):
            p = re.sub(r'(?i)^premise\s*:\s*', '', clean).strip()
            if len(p) > 3:
                premise = p
        elif len(clean) > 5:
            others.append(clean)
    subs = ([premise] + others if premise else others)[:N_SUBQ]
    if trace is not None:
        trace["rag_decomp_tokens"] = estimate_tokens(sys) + estimate_tokens(usr) + estimate_tokens(out)
        trace["rag_subqueries"] = subs
        trace["rag_has_premise"] = premise is not None
    return subs, premise is not None


# ── two-stage answering: answer each sub-question, then compose ────────────────────
# A single synthesis over all the raw chunks overloads the model — it rubber-stamps a match
# without resolving 'whose X'. Instead, answer EACH sub-question in isolation over only its own
# chunks (a narrow, focused, ownership-aware judgment), then COMPOSE the final answer from those
# sub-answers alone (no raw turns), so the composer just reads 'premise: Not stated' and refuses.

def _subanswer(subq, hits):
    """Answer ONE narrow sub-question over only its own retrieved turns. Strict and ownership-aware;
    replies 'Not stated' when the turns don't confirm the EXACT subject. Returns (answer, tokens)."""
    body = "\n".join(f"  - [{t['date']}] {t['speaker']}: {t['text']}" for t in hits)
    sys = (
        "Answer ONE narrow sub-question using ONLY the turns below.\n"
        "Be literal about the SUBJECT and the pronouns: a turn '<Speaker>: ... your X ...' means X "
        "belongs to the LISTENER, not the speaker; a turn where someone ASKS about X is not evidence "
        "that they HAVE or DID X.\n"
        "If the turns clearly state the answer ABOUT THE EXACT SUBJECT the sub-question names, give "
        "it in a short phrase (for a yes/no sub-question, start with Yes or No). If they do NOT, "
        "reply EXACTLY: Not stated. Never guess from general knowledge.\n"
        "DATES: each turn is tagged [date]. ALWAYS resolve relative wording ('last week', "
        "'yesterday', 'next month') to an ABSOLUTE date against that tag, and give the absolute date "
        "(e.g. 'last week' on a turn dated 6 July 2023 -> 'the week before 6 July 2023'; 'next month' "
        "on a turn dated 25 May 2023 -> 'June 2023'). Never answer with the bare relative phrase.")
    usr = f"SUB-QUESTION: {subq}\n\nTURNS:\n{body or '  (none)'}\n\nAnswer:"
    out = query_call([{"role": "system", "content": sys},
                      {"role": "user", "content": usr}], temperature=0.0)
    return out.strip(), estimate_tokens(sys) + estimate_tokens(usr) + estimate_tokens(out)


def _compose(question, qa_pairs):
    """Compose the final answer from the verified sub-answers ONLY (no raw turns). Returns
    (answer, tokens). Refuses if the premise sub-answer (or any needed fact) is 'Not stated'."""
    body = "\n".join(f"- {q}\n    -> {a}" for q, a in qa_pairs)
    sys = (
        "You answer the ORIGINAL question using ONLY the verified sub-answers below — each is the "
        "result of checking one fact against the conversation. Do NOT add anything not in them.\n"
        "One sub-answer checks the question's PREMISE (whether the thing it assumes is even true). "
        "If that premise sub-answer is 'Not stated' or No, or any fact the answer needs is 'Not "
        "stated', the answer is exactly: This information is not available.\n"
        "Pull the answer from ANY sub-answer that contains it — INCLUDING the premise sub-answer's "
        "supporting detail (e.g. a date inside 'Yes, ... on 25 May, 2023, ... next month').\n"
        "If the answer is a relative date, resolve it to an ABSOLUTE date using any timestamp in the "
        "sub-answers (give 'June 2023', not 'next month'; 'the week before 6 July 2023', not 'last "
        "week').\n"
        "Otherwise give the short final answer — a phrase, name, date, or list; never restate the "
        "question, never explain.")
    usr = f"ORIGINAL QUESTION: {question}\n\nVERIFIED SUB-ANSWERS:\n{body}\n\nFinal answer:"
    out = query_call([{"role": "system", "content": sys},
                      {"role": "user", "content": usr}], temperature=0.0)
    return out.strip(), estimate_tokens(sys) + estimate_tokens(usr) + estimate_tokens(out)


def _session_of(dia_id):
    m = re.match(r'[A-Za-z]+(\d+):', dia_id or "")
    return f"session_{m.group(1)}" if m else ""


def _is_negative(atom):
    """A premise sub-answer counts as negative if it starts with 'No' or 'Not stated'."""
    return re.match(r'\s*(not stated|no)\b', str(atom).strip(), re.IGNORECASE) is not None


# ── orchestration ────────────────────────────────────────────────────────────────

def answer(question, index, trace=None):
    """Decompose -> per-sub-question semantic retrieval over raw turns -> strict synthesis.
    Returns {answer, sessions, turns}, or None on hard failure (caller keeps the base refusal)."""
    turns, vecs = turn_embeddings(index)
    if not vecs:
        return None                                           # embeddings unavailable
    subs, has_premise = _decompose(question, trace=trace)
    if not subs:
        return None
    subq_vecs = _embed(subs)
    bm25 = _bm25_for(index["sample_id"], turns) if PI_RAG_HYBRID else None
    packs, picked, ev_scores, best_raw = [], {}, {}, 0.0
    for sq, qv in zip(subs, subq_vecs):
        if not qv:
            packs.append((sq, [])); ev_scores[sq] = []; continue
        qn = _normalize(qv)
        sims = [sum(a * b for a, b in zip(qn, v)) for v in vecs]
        order = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
        best_raw = max(best_raw, sims[order[0]])              # UNFILTERED nearest — for τ tuning
        if bm25 is not None:                                  # hybrid: fuse cosine + BM25
            hits = _hybrid_hits(sq, sims, order, bm25, TOPK, TAU)
        else:
            hits = [(i, sims[i]) for i in order if sims[i] >= TAU][:TOPK]
        hit_turns = [turns[i] for i, _ in hits]
        packs.append((sq, hit_turns))
        ev_scores[sq] = [(turns[i]["dia_id"], round(s, 3)) for i, s in hits]
        for t in hit_turns:
            picked[t["dia_id"]] = t
    if trace is not None:
        trace["rag_evidence"] = ev_scores
        trace["rag_best_sim"] = round(best_raw, 3)            # true nearest sim, independent of τ
        trace["rag_tau"] = TAU
    # Deterministic gate (before any answer): nothing cleared τ anywhere -> refuse, skip the rest.
    if not picked:
        if trace is not None:
            trace["rag_gate"] = "empty"
        return {"answer": "This information is not available.", "sessions": [], "turns": []}
    # Two-stage: answer each sub-question over its OWN chunks (premise is qa_pairs[0]).
    qa_pairs, ans_tok = [], 0
    for sq, hits in packs:
        a, tk = _subanswer(sq, hits) if hits else ("Not stated", 0)
        ans_tok += tk
        qa_pairs.append((sq, a))
    # Deterministic PREMISE GATE: a negative premise atom -> refuse, never reach the composer
    # (this is what stops the composer reaching past a failed premise to grab a tangential 'Yes').
    # Inference questions bypass the gate (no factual premise to check) -> compose infers from
    # the details, and still refuses if the basis is 'Not stated'.
    gate_on = PI_RAG_GATE and not (PI_RAG_INFER and _is_inference(question))
    if trace is not None:
        trace["rag_inference_bypass"] = (PI_RAG_INFER and _is_inference(question))
    if gate_on and has_premise and qa_pairs and _is_negative(qa_pairs[0][1]):
        if trace is not None:
            trace["rag_subanswers"] = qa_pairs
            trace["rag_ans_tokens"] = ans_tok
            trace["rag_gate"] = "premise_refused"
        return {"answer": "This information is not available.", "sessions": [], "turns": []}
    ans, tk = _compose(question, qa_pairs)
    ans_tok += tk
    if trace is not None:
        trace["rag_subanswers"] = qa_pairs
        trace["rag_ans_tokens"] = ans_tok
        trace["rag_gate"] = "compose"
    sessions = sorted({_session_of(t["dia_id"]) for t in picked.values()} - {""})
    return {"answer": ans, "sessions": sessions, "turns": list(picked.values())}
