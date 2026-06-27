"""
LongMemEval data loader — parses longmemeval_s.json into the SAME per-conversation structure the
PageIndex pipeline already consumes, with ONE crucial difference from LoCoMo:

  LoCoMo  = 10 conversations, each with MANY QA  -> one index per conversation.
  LongMemEval = ~500 instances, each ONE question with ITS OWN haystack of ~50-80 sessions
                -> one index PER QUESTION.

So load() returns one `conv` dict per instance, shaped exactly like locomo_loader's output so that
pageindex.build_index / pageindex.query / pi_rag / evaluate work unchanged:

    {sample_id, sessions:[{key,date_time,session_id,turns}], all_turns, qa:[ONE], speakers,
     question_date}

Field mapping (LongMemEval -> ours):
  question_id            -> sample_id   (one index/embedding cache per question)
  question_type          -> qa[0].category / category_name
  question_id ".._abs"   -> abstention question: answer "NOT_ANSWERABLE", so
                            evaluate.score_answer's deterministic refusal scoring grades it
  haystack_sessions[i]   -> session_i ; each turn {role,content} -> {speaker,text} with a
    (list of {role,content}) synthesized dia_id "S{i}:{j}" so pi_rag._session_of resolves a session
  haystack_dates[i]      -> session_i date_time
  haystack_session_ids[i]-> session_i session_id (stable id, used for cross-question summary reuse)
  answer_session_ids     -> qa[0].evidence (provenance only)

No API keys, no network — pure parse. Required fields are accessed directly so a format mismatch
fails loudly rather than silently degrading.
"""
import json
import random
from collections import defaultdict

import config


def _stratified_sample(data, k, seed=0):
    """Deterministic, REPRESENTATIVE subset of k instances. The file is grouped by question_type
    (and the 30 abstention questions sit mid-file), so a naive data[:k] omits whole categories and
    most/all abstention. We give each (question_type, abstention) cell a proportional quota via the
    largest-remainder method (sums to exactly k), sample within each cell with a seeded RNG, then
    interleave — so the subset spans every type AND includes abstention in proportion, reproducibly."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for d in data:
        groups[(d["question_type"], d["question_id"].endswith("_abs"))].append(d)
    total = len(data)
    base, rem = {}, {}
    for key, items in groups.items():
        exact = k * len(items) / total
        base[key], rem[key] = int(exact), exact - int(exact)
    for key in sorted(rem, key=lambda x: rem[x], reverse=True)[:k - sum(base.values())]:
        base[key] += 1
    picked = []
    for key, items in groups.items():
        pool = items[:]
        rng.shuffle(pool)
        picked.extend(pool[:base[key]])
    rng.shuffle(picked)                      # interleave types so a partial run isn't blocky/biased
    return picked


def load(path=None, limit=None, sample=True, seed=0):
    """Load LongMemEval instances as per-question `conv` dicts.
    With `limit` and sample=True (default): a deterministic, type+abstention-stratified subset
    (the file is grouped by type, so 'first N' would omit whole categories). sample=False keeps
    file order (first N), e.g. for quick debugging."""
    path = path or getattr(config, "LONGMEMEVAL_PATH", "longmemeval_s.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if limit and limit < len(data):
        data = _stratified_sample(data, limit, seed) if sample else data[:limit]

    convs = []
    for inst in data:
        qid = inst["question_id"]
        is_abs = qid.endswith("_abs")

        haystack = inst.get("haystack_sessions", [])
        dates = inst.get("haystack_dates", [])
        sess_ids = inst.get("haystack_session_ids", [])

        sessions, all_turns = [], []
        for i, sess in enumerate(haystack):
            turns = [{
                "dia_id": f"S{i}:{j}",                 # -> pi_rag._session_of -> session_{i}
                "speaker": t.get("role", ""),
                "text": t.get("content", ""),
                "session": f"session_{i}",
                "has_answer": t.get("has_answer", False),
            } for j, t in enumerate(sess)]
            sessions.append({
                "key": f"session_{i}",
                "date_time": dates[i] if i < len(dates) else "",
                "session_id": sess_ids[i] if i < len(sess_ids) else f"session_{i}",
                "turns": turns,
            })
            all_turns.extend(turns)

        qa = [{
            "question": inst["question"],
            "answer": "NOT_ANSWERABLE" if is_abs else str(inst.get("answer", "")),
            "category": inst["question_type"],
            "category_name": inst["question_type"],
            "abstention": is_abs,
            "evidence": inst.get("answer_session_ids", []),
        }]

        convs.append({
            "speakers": "user & assistant",
            "sessions": sessions,
            "all_turns": all_turns,
            "qa": qa,
            "sample_id": qid,
            "question_date": inst.get("question_date", ""),
        })
    return convs


if __name__ == "__main__":
    convs = load(limit=10)
    print(f"Loaded {len(convs)} LongMemEval instances (first 10)")
    from collections import Counter
    types = Counter(c["qa"][0]["category_name"] for c in convs)
    n_abs = sum(c["qa"][0]["abstention"] for c in convs)
    for i, c in enumerate(convs):
        q = c["qa"][0]
        print(f"  {i}: {c['sample_id']:<28} {len(c['sessions']):>3} sessions, "
              f"{len(c['all_turns']):>4} turns | {q['category_name']:<22} "
              f"abs={q['abstention']} | ans={str(q['answer'])[:30]}")
    print(f"\n  types: {dict(types)}")
    print(f"  abstention: {n_abs}/{len(convs)}")
