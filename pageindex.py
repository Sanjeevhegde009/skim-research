"""
PageIndex-style VECTORLESS RAG over the RAW conversation (experiment branch).

No embeddings, no wiki compilation. The pipeline is index-then-navigate:
  1. BUILD a tree index — one node per session, each with an LLM-written "table of contents"
     summary of what that session covers (cached to pageindex_<sample_id>.json).
  2. At query time, the LLM NAVIGATES by reasoning — reads the session summaries and picks the
     session(s) most likely to hold the answer (no similarity math).
  3. READ the chosen sessions' raw turns and ANSWER with a conservative, date-aware,
     refuse-if-absent prompt (conservative, cite-or-refuse discipline).

Reuses llm.py's plumbing (compiler_call / query_call) + config.
"""
import json
import os
import re
from datetime import date as _date
from pathlib import Path

import config
from llm import compiler_call, query_call, estimate_tokens

# Bounded, type-gated reasoning in the ANSWER step (default off; base path unchanged).
# Only questions that need SERIAL computation get a scratchpad; the rest stay terse.
# The trigger is language-general (count/compare/duration/state/intersection) — NOT tuned to
# any dataset. Reasoning is bounded to a few steps in one capped call (it cannot run forever),
# and the final answer is extracted terse so token-F1 isn't penalised. Toggle: PI_REASON=1
PI_REASON = os.environ.get("PI_REASON", "").lower() in ("1", "true", "yes")

# RAG escalation on the refusal RESIDUE (default off; stable base path unchanged). When the base
# navigate->answer refuses, hand the question to pi_rag: decompose into 3-5 sub-questions, semantic-
# retrieve raw turns per sub-question, strict-synthesize (answer or refuse). Toggle: PI_RAG=1
PI_RAG = os.environ.get("PI_RAG", "").strip().lower() in ("1", "true", "yes")

# Force RAG escalation on count/compare/aggregate questions EVEN WHEN the base did not refuse
# (default off). The base path navigates a partial view (<=3 sessions) and systematically
# UNDERCOUNTS aggregations, answering confidently instead of refusing — so RAG never fires. When on,
# such questions are handed to RAG's global (whole-haystack) retrieval and its non-refusal answer is
# PREFERRED over the base's. Targets the largest error bucket. Toggle: PI_RAG_FORCE_AGG=1
PI_RAG_FORCE_AGG = os.environ.get("PI_RAG_FORCE_AGG", "").strip().lower() in ("1", "true", "yes")

# Grounding-verify gate (default off). The refusal gate only escalates when the reader ADMITS it
# can't answer; it is blind to CONFIDENT-WRONG answers (~half our errors — non-refusal, non-
# hallucination, just wrong). When on, a non-refusal base answer is checked against the evidence it
# was drawn from; if UNSUPPORTED (a fact not present, wrong date/count arithmetic, or a superseded
# value on a 'most recent' question) it is escalated like a refusal. One extra cheap reader call,
# only on answers that would not otherwise escalate. Toggle: PI_RAG_VERIFY=1
PI_RAG_VERIFY = os.environ.get("PI_RAG_VERIFY", "").strip().lower() in ("1", "true", "yes")

# Broad navigation as its OWN flag (default off). The best config runs precision nav
# (NAV_MAX_SESSIONS=3), so a count scattered over >=4 sessions is an undercount BY CONSTRUCTION —
# the base path cannot open enough sessions (verified: 'health devices' gold=4, cap=3, answered 3
# with the 4th sitting in the ToC). Broad nav (cap 8) exists but is welded to PI_REASON, which ALSO
# swaps the answer prompt — two variables. This flag enables broad nav ALONE on count/aggregate
# questions; the answer step stays byte-identical. Toggle: PI_NAV_BROAD=1
PI_NAV_BROAD = os.environ.get("PI_NAV_BROAD", "").strip().lower() in ("1", "true", "yes")

# Deterministic date-math layer (default off). Temporal questions fail because the reader EYEBALLS
# date arithmetic (proven: even gpt-4o answered a 2-event ordering backwards) and because the
# pipeline never told it WHEN the question was asked (question_date was dropped by the runner, so
# every "how many months ago...?" had no reference point). When on, temporal-computation questions
# take a 3-step path: (1) LLM EXTRACTS the relevant events with ABSOLUTE dates (resolving relative
# wording against each turn's date tag — extraction, which LLMs do well); (2) PYTHON computes the
# timeline, pairwise differences, and distances from the question date — exactly; (3) the LLM answers
# by COPYING the computed number, forbidden from doing its own arithmetic. Falls back to the normal
# answer path when no dated events can be extracted. Toggle: PI_DATEMATH=1
PI_DATEMATH = os.environ.get("PI_DATEMATH", "").strip().lower() in ("1", "true", "yes")

# Recency / value-history layer (default off). Knowledge-update questions fail because the corpus
# holds a DATED SEQUENCE of values for one attribute (Rachel moved to Chicago May 24 -> back to the
# suburbs May 27) and the reader returns whichever value navigation surfaced most directly, blind to
# which is newest. Datemath's sibling, same extract->compute->copy shape: (1) LLM extracts every
# dated statement of the asked attribute (actual states only — never plans/considerations, which are
# lures); (2) PYTHON sorts the history and marks the LATEST entry as superseding; (3) the LLM answers
# from the history — latest by default, second-latest for "previous ..." questions, combined for
# "so far" totals. Trigger = language-general state markers, NOT the dataset's type label. Recency
# questions also navigate BROAD (the value history is scattered by nature). Toggle: PI_RECENCY=1
PI_RECENCY = os.environ.get("PI_RECENCY", "").strip().lower() in ("1", "true", "yes")

_RECENCY_RE = re.compile(
    r'\brecent(?:ly)?\b|\bcurrently\b|\bcurrent\b|\blatest\b|\bnewest\b'        # explicit latest
    r'|\bstill\b|\banymore\b|\bthese days\b|\bnowadays\b|\busually\b'           # present state/habit
    r'|\bso far\b|\bup to now\b|\bin total now\b'                               # running totals
    # the value BEFORE last — but NOT "our previous chat/conversation" (that's conversational
    # framing, typical of single-session-assistant recall questions, not a changed value):
    r'|\bprevious(?:ly)?\b(?!\s+(?:chat|conversation|discussion|session|talk))'
    r'|\bbefore (?:the|my|that)\b'
    r'|\bagain\b|\bswitched\b|\bchanged\b|\bupdated\b',                         # change markers
    re.IGNORECASE)


def is_recency(question: str) -> bool:
    """True for questions about the CURRENT/LATEST (or explicitly previous) state of something that
    may have changed over time — the class the value-history layer handles."""
    return bool(_RECENCY_RE.search(question or ""))

_DATEMATH_RE = re.compile(
    r'\bhow long\b|\bhow old\b'                                             # duration / age
    r'|\bhow many (?:days|weeks|months|years)\b'                            # counted spans
    r'|\b(?:days|weeks|months|years)\s+(?:ago|passed|apart|since|before|after|later)\b'
    r'|\bago\b|\bhow much (?:older|younger)\b'                              # elapsed / age gap
    r'|\border of\b|\bearliest\b|\blatest\b|\bchronolog'                    # chronological ordering
    r'|\b(?:which|who|what)\b[^?]{0,60}\bfirst\b|\bhappened first\b',       # which-came-first
    re.IGNORECASE)


def is_temporal_math(question: str) -> bool:
    """True for questions whose answer is a date computation (duration, elapsed time, age gap,
    chronological order) — the class the date-math layer handles."""
    return bool(_DATEMATH_RE.search(question or ""))

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


# Aggregate/count/compare trigger for FORCED escalation. Broader than _REASON_RE (adds total/
# average/combined and comparatives) and kept SEPARATE so it never perturbs the PI_REASON base path.
_AGG_RE = re.compile(
    r'\bhow many\b|\bhow much\b|\bhow often\b|\bhow frequently\b|\bhow many times\b'      # count
    r'|\btotal\b|\bin total\b|\baltogether\b|\bcombined\b|\baverage\b|\bsum\b|\bcount\b'  # aggregate
    r'|\bmore\b|\bfewer\b|\bless\b|\bolder\b|\byounger\b|\blonger\b|\bshorter\b|'
    r'\bfaster\b|\bslower\b|\bbigger\b|\bsmaller\b|\bgreater\b|\bhigher\b|\blower\b'       # compare
    r'|\bcompare\b|\bdifference\b|\bin common\b',
    re.IGNORECASE)


def is_aggregate(question: str) -> bool:
    """True for count / aggregate / compare questions — the ones the partial-view base path
    undercounts and should hand to RAG's global retrieval when PI_RAG_FORCE_AGG is on."""
    return bool(_AGG_RE.search(question or ""))


def _turns_block(turns, dated=False):
    out = []
    for t in turns:
        tag = f"[{t.get('date','')}] " if dated else ""
        out.append(f"{tag}{t['speaker']}: {t['text']}")
    return "\n".join(out)


def build_index(conv, force=False):
    """One LLM-summarized node per session. Cached; ~one call per session.
    Returns {'sample_id', 'nodes': [{key,date,summary,turns}]}."""
    if os.environ.get("COMPILED", "").strip().lower() in ("1", "true", "yes"):
        import compiled                                        # branch: HLMA-compiled index
        return compiled.compile_index(conv, force)
    config.INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = config.INDEX_CACHE_DIR / f"{conv['sample_id']}.json"
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


def _navigate(question, nodes, broad=False, recency=False, trace=None):
    """Reasoning-based node selection — the LLM picks relevant session(s) from the index.
    broad=True (for count/compare/aggregate questions): recall over precision — gather EVERY
    session that could hold a relevant instance, because the answer is scattered across the
    conversation and reasoning needs the complete set, not the single best session.
    recency=True (for current/latest-state questions): the same completeness discipline, but
    framed for VALUE CHANGES — prioritise later-dated sessions, because a newer mention
    supersedes older ones and missing it means a stale answer."""
    cap = NAV_BROAD_SESSIONS if broad else NAV_MAX_SESSIONS
    toc = "\n".join(f"SESSION {n['key']} ({n['date']}): {n['summary']}" for n in nodes)
    if recency and broad:
        sys = (
            "You navigate a conversation by its session index. The question asks about the "
            "CURRENT or LATEST state of something that may have CHANGED over time, so "
            "COMPLETENESS ACROSS TIME matters more than precision: select EVERY session whose "
            "summary could mention this entity or attribute — especially LATER-dated sessions, "
            "since a newer mention supersedes older ones and missing it means answering with a "
            "STALE value. Each line is 'SESSION <key> (<date>): <summary>'. Reply with the "
            f"session keys exactly as shown (up to {cap}), comma-separated, or NONE if none "
            "are relevant.")
    elif broad:
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
    """Conservative, date-aware, refuse-if-absent answer — cite-or-refuse discipline.
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


def _parse_qdate(s):
    """Parse a question/session timestamp like '2023/05/30 (Tue) 23:38' to a date, else None."""
    m = re.search(r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})', s or "")
    try:
        return _date(int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None
    except ValueError:
        return None


def _cal_months(a, b):
    """Whole calendar months from a to b (a <= b)."""
    return (b.year - a.year) * 12 + (b.month - a.month) - (1 if b.day < a.day else 0)


def _fmt_span(a, b):
    """Exact span from a to b in every unit the question might ask for."""
    dd = (b - a).days
    w, r = divmod(dd, 7)
    parts = [f"{dd} days", f"{dd + 1} days counting both end days",
             f"{w} weeks" + (f" {r} days" if r else "")]
    if dd >= 28:
        parts.append(f"{_cal_months(a, b)} calendar months")
    if dd >= 365:
        parts.append(f"~{dd / 365.25:.1f} years")
    return " = ".join(parts[:2]) + "; " + "; ".join(parts[2:])


def _compute_block(events, qdate):
    """Deterministic timeline + differences over the extracted events. events = [(desc, date,
    approx)]. Every number here is computed in Python — the final LLM call only copies them."""
    evs = sorted(events, key=lambda e: e[1])
    L = []
    if qdate:
        L.append(f"TODAY (the date the question is asked): {qdate.isoformat()}")
    L.append("TIMELINE (earliest -> latest; ~ = day approximate):")
    for i, (desc, dt, ap) in enumerate(evs, 1):
        rel = ""
        if qdate:
            dd = (qdate - dt).days
            side = "before today" if dd >= 0 else "after today"
            rel = (f"   [{abs(dd)} days {side}; {abs(dd)//7} weeks; "
                   f"{_cal_months(*sorted((dt, qdate)))} calendar months]")
        L.append(f"  E{i}. {dt.isoformat()}{'~' if ap else ''} — {desc}{rel}")
    if len(evs) > 1:
        L.append("DIFFERENCES (exact, precomputed — copy, never recompute):")
        idx = range(len(evs))
        pairs = ([(i, j) for i in idx for j in idx if i < j] if len(evs) <= 5
                 else [(i, i + 1) for i in idx[:-1]])
        for i, j in pairs:
            L.append(f"  E{i+1} -> E{j+1}: {_fmt_span(evs[i][1], evs[j][1])}")
    return "\n".join(L)


def _answer_datemath(question, turns, qdate, trace=None):
    """3-step temporal answer: LLM extracts dated events -> Python computes -> LLM copies the
    computed number. Returns None (caller falls back to the normal path) if extraction finds no
    usable dated events."""
    body = _turns_block(turns, dated=True)
    sys1 = (
        "You extract DATED EVENTS from conversation excerpts so that exact date arithmetic can be "
        "done in code. List every event RELEVANT to the question, one per line, EXACTLY:\n"
        "EVENT: <short description> || DATE: <YYYY-MM-DD>\n"
        "- Each excerpt is tagged [date] = when it was SPOKEN. Resolve relative wording against "
        "that tag ('yesterday' on a turn tagged 2023/05/30 -> 2023-05-29; 'three weeks ago' -> "
        "that date minus 21 days; 'last Saturday' -> the Saturday before it).\n"
        "- Date each event by WHEN THE EVENT HAPPENED, never by when it was mentioned (a trip "
        "recalled today still gets the trip's own date).\n"
        "- If only the month is known, write DATE: <YYYY-MM>; only the year, DATE: <YYYY>. If "
        "undatable, DATE: unknown.\n"
        "- Include the event the question anchors on AND every event it asks about; nothing else.\n"
        "- Output ONLY EVENT lines, no commentary.")
    usr1 = f"QUESTION (for relevance only): {question}\n\nEXCERPTS:\n{body}\n\nEVENT lines:"
    out1 = query_call([{"role": "system", "content": sys1},
                       {"role": "user", "content": usr1}], temperature=0.0)
    tok = estimate_tokens(sys1) + estimate_tokens(usr1) + estimate_tokens(out1)

    events = []
    for ln in out1.splitlines():
        m = re.match(r'\s*EVENT:\s*(.+?)\s*\|\|\s*DATE:\s*(\S+)', ln)
        if not m:
            continue
        desc, ds = m.group(1), m.group(2).strip().rstrip('.')
        md = re.match(r'(\d{4})(?:-(\d{1,2}))?(?:-(\d{1,2}))?$', ds)
        if not md:
            continue                                        # 'unknown' or malformed -> skip
        try:                                                # bare YYYY -> mid-year approx; never DROP
            events.append((desc, _date(int(md.group(1)), int(md.group(2) or 7),
                                       int(md.group(3) or 15)),
                           md.group(3) is None))            # a partially-dated event (losing it broke
        except ValueError:                                  # 'which trip first': Thailand DATE: 2022
            continue                                        # was discarded -> only Europe remained)
    if trace is not None:
        trace["datemath_events"] = out1.strip()
    if not events:
        if trace is not None:
            trace["datemath"] = "no_events"
        return None, tok                                    # fall back to the normal answer path

    computed = _compute_block(events, qdate)
    if trace is not None:
        trace["datemath"] = "computed"
        trace["datemath_table"] = computed
    sys2 = (
        "Answer the question using ONLY the events and the COMPUTED table below. Every difference "
        "and duration in the table was computed EXACTLY in code.\n"
        "- COPY the number that answers the question; NEVER do your own date arithmetic.\n"
        "- Match the unit the question asks for (days / weeks / months / years); for 'how many X "
        "ago' or 'how long since', use the distance-from-TODAY values.\n"
        "- For order/which-first questions, give the events IN TIMELINE ORDER — always by their "
        "descriptions (names), NEVER by the E-numbers (E1/E2 mean nothing to the asker).\n"
        "- Only if NO event in the table is relevant to the question, answer exactly: This "
        "information is not available. Otherwise answer from the events you have — the evidence "
        "was already vetted upstream; do not second-guess the question's wording.\n"
        "Output ONLY the short final answer.")
    usr2 = f"QUESTION: {question}\n\n{computed}\n\nAnswer:"
    out2 = query_call([{"role": "system", "content": sys2},
                       {"role": "user", "content": usr2}], temperature=0.0)
    tok += estimate_tokens(sys2) + estimate_tokens(usr2) + estimate_tokens(out2)
    return out2.strip(), tok


RECENCY_TOPK = 12    # deeper than RAG's answering k: gathering a value HISTORY is a recall task,
                     # and extraction filters non-states, so noise in the candidate turns is cheap


def _recency_retrieve(question, index, nav_turns, trace=None):
    """Augment the recency layer's evidence with the RAG retrieval pass (cosine+BM25 over ALL raw
    turns). Value updates are usually lexically findable ('Rachel ... moved ... suburbs') even when
    their session's nav summary ERASED them — retrieval reads turns, not summaries, so it reaches
    what navigation structurally cannot. Returns (merged_turns, n_added); every turn carries its
    session date, which the extraction step needs. Falls back to nav_turns alone on any failure."""
    import pi_rag
    try:
        turns, vecs = pi_rag.turn_embeddings(index)
        if not vecs:
            return nav_turns, 0
        qv = pi_rag._embed([question])[0]
        if not qv:
            return nav_turns, 0
        qn = pi_rag._normalize(qv)
        sims = [sum(a * b for a, b in zip(qn, v)) for v in vecs]
        order = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
        if pi_rag.PI_RAG_HYBRID:
            bm25 = pi_rag._bm25_for(index["sample_id"], turns)
            hits = pi_rag._hybrid_hits(question, sims, order, bm25, RECENCY_TOPK, pi_rag.TAU)
        else:
            hits = [(i, sims[i]) for i in order if sims[i] >= pi_rag.TAU][:RECENCY_TOPK]
    except Exception as e:
        print(f"  [RECENCY RETRIEVE ERROR] {e}")
        return nav_turns, 0
    seen = {t.get("dia_id") for t in nav_turns if t.get("dia_id")}
    merged, added = list(nav_turns), 0
    for i, _ in hits:
        t = turns[i]
        if t.get("dia_id") not in seen:
            merged.append(t)
            seen.add(t.get("dia_id"))
            added += 1
    if trace is not None:
        trace["recency_retrieved"] = added
    return merged, added


def _answer_recency(question, turns, qdate, trace=None):
    """3-step knowledge-update answer: LLM extracts the dated VALUE HISTORY of the asked attribute
    -> Python sorts it and marks the latest -> LLM answers from the history (latest by default).
    Returns (None, tok) when no dated states can be extracted (caller falls back)."""
    body = _turns_block(turns, dated=True)
    sys1 = (
        "You extract the DATED VALUE HISTORY of one attribute from conversation excerpts, so the "
        "LATEST value can be selected in code. The question asks about something that may have "
        "CHANGED over time. List every statement of its value/state, one per line, EXACTLY:\n"
        "STATE: <the value or state as stated> || DATE: <YYYY-MM-DD>\n"
        "- Each excerpt is tagged [date] = when it was SPOKEN. Date each state by when it became "
        "true if stated ('three weeks ago' resolved against the tag), else by the tag itself.\n"
        "- ONLY actual states/events. NEVER include plans, considerations, or wishes ('thinking "
        "about getting a wide-angle lens' is NOT a purchase).\n"
        "- Include statements that UPDATE or REVERSE earlier ones ('moved back again', 'switched "
        "to', 'another 2') — the history is the point.\n"
        "- If only the month is known, DATE: <YYYY-MM>; only the year, DATE: <YYYY>.\n"
        "- Output ONLY STATE lines, no commentary.")
    usr1 = f"QUESTION (defines the attribute): {question}\n\nEXCERPTS:\n{body}\n\nSTATE lines:"
    out1 = query_call([{"role": "system", "content": sys1},
                       {"role": "user", "content": usr1}], temperature=0.0)
    tok = estimate_tokens(sys1) + estimate_tokens(usr1) + estimate_tokens(out1)

    states = []
    for ln in out1.splitlines():
        m = re.match(r'\s*STATE:\s*(.+?)\s*\|\|\s*DATE:\s*(\S+)', ln)
        if not m:
            continue
        desc, ds = m.group(1), m.group(2).strip().rstrip('.')
        md = re.match(r'(\d{4})(?:-(\d{1,2}))?(?:-(\d{1,2}))?$', ds)
        if not md:
            continue
        try:
            states.append((desc, _date(int(md.group(1)), int(md.group(2) or 7),
                                       int(md.group(3) or 15))))
        except ValueError:
            continue
    if trace is not None:
        trace["recency_states"] = out1.strip()
    if not states:
        if trace is not None:
            trace["recency"] = "no_states"
        return None, tok

    states.sort(key=lambda s: s[1])
    L = []
    if qdate:
        L.append(f"TODAY (the date the question is asked): {qdate.isoformat()}")
    L.append("VALUE HISTORY (oldest -> newest, sorted in code):")
    for i, (desc, dt) in enumerate(states, 1):
        tag = "   <== LATEST (supersedes all earlier values)" if i == len(states) else ""
        L.append(f"  {i}. {dt.isoformat()} — {desc}{tag}")
    table = "\n".join(L)
    if trace is not None:
        trace["recency"] = "computed"
        trace["recency_table"] = table

    sys2 = (
        "Answer the question using ONLY the VALUE HISTORY below (sorted oldest -> newest in code).\n"
        "- By default the answer is the LATEST entry — earlier values are superseded.\n"
        "- If the question asks for the PREVIOUS value (the one before the change), use the entry "
        "just before the latest.\n"
        "- If the question asks for a running total ('so far'), combine the entries.\n"
        "- Copy the value's wording from the entry; do not add anything.\n"
        "- Only if no entry is relevant to the question, answer exactly: This information is not "
        "available.\n"
        "Output ONLY the short final answer.")
    usr2 = f"QUESTION: {question}\n\n{table}\n\nAnswer:"
    out2 = query_call([{"role": "system", "content": sys2},
                       {"role": "user", "content": usr2}], temperature=0.0)
    tok += estimate_tokens(sys2) + estimate_tokens(usr2) + estimate_tokens(out2)
    return out2.strip(), tok


def _is_grounded(question, answer, turns, trace=None):
    """Cheap grounding check: is ANSWER actually supported by the EVIDENCE it was drawn from?
    A genuine verification (fact present, arithmetic/date-math recomputed, recency respected) — not a
    surface match. Returns True if supported; defaults True on an ambiguous verdict (fail-safe: don't
    over-escalate a possibly-correct answer)."""
    body = _turns_block(turns, dated=True)
    sys = (
        "You VERIFY a proposed answer against conversation evidence. Decide whether the ANSWER is "
        "fully and correctly supported by the EXCERPTS.\n"
        "Check ALL of:\n"
        "- Every fact in the answer actually appears in the excerpts (not guessed from general "
        "knowledge).\n"
        "- Any counting, arithmetic, or DATE math is correct given the excerpts — recompute it "
        "yourself before judging.\n"
        "- If the question asks for the CURRENT / MOST RECENT / LATEST value, the answer reflects the "
        "latest-dated excerpt, not a superseded earlier one.\n"
        "Reply on ONE line, EXACTLY 'SUPPORTED' if it holds, otherwise 'UNSUPPORTED: <brief reason>'. "
        "Be strict, but do not demand more than the question actually asks.")
    usr = f"QUESTION: {question}\n\nPROPOSED ANSWER: {answer}\n\nEXCERPTS:\n{body}\n\nVerdict:"
    out = query_call([{"role": "system", "content": sys},
                      {"role": "user", "content": usr}], temperature=0.0)
    low = out.strip().lower()
    if trace is not None:
        trace["verify_verdict"] = out.strip()[:140]
        trace["verify_tokens"] = estimate_tokens(sys) + estimate_tokens(usr) + estimate_tokens(out)
    return not (low.startswith("unsupported") or "unsupported" in low or "not supported" in low)


def query(index, question, question_date=None):
    """Navigate → read chosen sessions' raw turns → answer. Refuse if nav finds nothing.

    With PI_RAG, a base REFUSAL hands the question to pi_rag (decompose → semantic-retrieve raw
    turns → strict synthesis); its answer is adopted only if it recovers one, else the refusal
    stands. Stable base path is byte-identical when PI_RAG is off. Trace records every step."""
    if os.environ.get("COMPILED", "").strip().lower() in ("1", "true", "yes"):
        import compiled                                        # branch: compiled two-tier gated query
        return compiled.query_compiled(index, question)
    nodes = index["nodes"]
    trace = {}
    # Aggregation questions need BROAD navigation (scattered evidence) AND reasoning over the
    # complete set — coupled. Gated by PI_REASON, so the base path is precision-navigate only.
    # PI_NAV_BROAD decouples them: broad nav alone (wider trigger: needs_reasoning OR is_aggregate,
    # so total/average questions qualify too), answer step unchanged. Date-math questions also
    # navigate broad — an ordering/duration needs the events from EVERY involved session.
    datemath_q = PI_DATEMATH and is_temporal_math(question)
    # Recency questions also navigate broad: the value history is scattered by nature (each update
    # lives in a different session). Datemath keeps precedence — duration questions need arithmetic
    # over the history, not a pick from it.
    recency_q = PI_RECENCY and is_recency(question) and not datemath_q
    broad = (PI_REASON and needs_reasoning(question)) or \
            (PI_NAV_BROAD and (needs_reasoning(question) or is_aggregate(question))) or \
            datemath_q or recency_q
    keys = _navigate(question, nodes, broad=broad, recency=recency_q, trace=trace)
    if keys:
        turns = _collect_turns(nodes, keys)
        ans, dm_tok = None, 0
        if datemath_q:                                    # extract dates -> compute in code -> copy
            qd = _parse_qdate(question_date)
            ans, dm_tok = _answer_datemath(question, turns, qd, trace=trace)
        elif recency_q:                                   # extract value history -> pick latest
            qd = _parse_qdate(question_date)
            ev_turns = turns
            if PI_RAG:                                    # RAG retrieval augmentation: reach value
                ev_turns, _ = _recency_retrieve(          # updates whose sessions nav can't see
                    question, index, turns, trace=trace)  # (summary erased them; turns are visible)
            ans, dm_tok = _answer_recency(question, ev_turns, qd, trace=trace)
        if ans is None:                                   # no trigger, or extraction found nothing
            ans = _answer(question, turns, trace=trace)   # (_answer assigns ans_tokens)
        trace["ans_tokens"] = trace.get("ans_tokens", 0) + dm_tok   # add layer cost (0 if unused)
    else:
        turns, ans = [], "This information is not available."

    # RAG escalation. Normally fires only on a base REFUSAL (recover the residue). With
    # PI_RAG_FORCE_AGG it ALSO fires on count/aggregate/compare questions the base answered
    # confidently — because the base saw a partial view and likely undercounted — and PREFERS RAG's
    # non-refusal answer (global count supersedes the partial one). base_answer is kept for analysis.
    base_ans = ans
    forced = PI_RAG and PI_RAG_FORCE_AGG and is_aggregate(question) and not _is_refusal(ans)
    # Grounding verify: a CONFIDENT (non-refusal) base answer that is not actually supported by the
    # evidence it was drawn from is treated like a refusal, so escalation also fires on confident-
    # WRONG answers. Only run when nothing else already escalates (saves the extra call).
    verify_fail = False
    if PI_RAG and PI_RAG_VERIFY and keys and not _is_refusal(ans) and not forced:
        verify_fail = not _is_grounded(question, ans, turns, trace=trace)
        trace["verify_fail"] = verify_fail
    if PI_RAG and (_is_refusal(ans) or forced or verify_fail):
        import pi_rag
        trace["rag_fired"] = True
        trace["rag_forced"] = forced
        trace["rag_verify"] = verify_fail
        trace["base_answer"] = base_ans
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
