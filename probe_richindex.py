"""
Phase-0 feasibility probe for the rich navigation index (branch feat/rich-index).

The make-or-break question, tested on KNOWN erasure cases BEFORE building the full machinery:
  (A) CAPTURE  — does the rich compile record the incidental instances the 2-4 sentence summary
                 erases (Cedar Creek property, Oakwood bungalow, ...)?
  (B) RETRIEVE — do those de-disguised facts EMBED near the question, so retrieve-over-ledger opens
                 the erased sessions among the full haystack?

If both hold on the properties case, the mechanism works and the cost is justified. If the compile
STILL drops the incidental instances, we stop here for ~$0.20.

Run:  python probe_richindex.py        (needs OPENAI_API_KEY; ~one compile per haystack session, cached)
"""
import json

import config
import pi_rag
import rich_index

# question_id -> substrings of the instances the CURRENT summary erased (capture check)
TESTS = {
    "gpt4_7fce9456": ["cedar creek", "bungalow", "oakwood", "1-bedroom", "condo"],  # 4 viewings
}
TOPK = 12


def _cos(a, b):
    return sum(x * y for x, y in zip(a, b))


def main():
    data = json.load(open(getattr(config, "LONGMEMEVAL_PATH", "data/longmemeval_s.json"),
                         encoding="utf-8"))
    for qid, needles in TESTS.items():
        inst = next(d for d in data if d["question_id"] == qid)
        Q = inst["question"]
        hsid = inst["haystack_session_ids"]
        gold_ix = {hsid.index(sid) for sid in inst["answer_session_ids"] if sid in hsid}
        print("=" * 96)
        print("Q:", Q)
        print(f"gold evidence sessions: {sorted('session_%d' % i for i in gold_ix)}")
        print(f"haystack size: {len(inst['haystack_sessions'])} sessions\n")

        # ── compile a rich ledger for EVERY haystack session (the real retrieve-over-ledger setting)
        all_facts, fact_src = [], []
        gold_ledgers = {}
        n = len(inst["haystack_sessions"])
        print(f"  compiling {n} session ledgers (cached after first run)...", flush=True)
        for i, sess in enumerate(inst["haystack_sessions"]):
            turns = [{"dia_id": f"S{i}:{j}", "speaker": t["role"], "text": t["content"]}
                     for j, t in enumerate(sess)]
            ledger = rich_index.rich_compile(turns, f"session_{i}", inst["haystack_dates"][i])
            nf = len(rich_index.ledger_facts(ledger))
            print(f"    {i+1}/{n} session_{i}: {nf} facts{'  <- GOLD' if i in gold_ix else ''}", flush=True)
            if i in gold_ix:
                gold_ledgers[i] = ledger
            for f in rich_index.ledger_facts(ledger):
                all_facts.append(f)
                fact_src.append(i)

        # ── (A) CAPTURE: are the erased instances now in the gold-session ledgers?
        print("--- (A) CAPTURE: rich ledgers of the gold sessions ---")
        joined = " || ".join(gold_ledgers.values()).lower()
        for n in needles:
            print(f"    needle {n!r:22} present in ledgers: {n in joined}")
        for i, led in sorted(gold_ledgers.items()):
            print(f"  session_{i}:")
            for f in rich_index.ledger_facts(led):
                print("     -", f[:120])
        print()

        # ── (B) RETRIEVE: embed all facts + question, rank, which sessions surface in top-K
        vecs = [pi_rag._normalize(v) if v else None for v in pi_rag._embed(all_facts)]
        qv = pi_rag._embed([Q])[0]
        if not qv or not vecs or vecs[0] is None:
            print("  [embed unavailable — set OPENAI_API_KEY]"); continue
        qn = pi_rag._normalize(qv)
        sims = [(_cos(qn, v) if v else -1.0, k) for k, v in enumerate(vecs)]
        sims.sort(reverse=True)
        top_sessions, seen = [], set()
        print("--- (B) RETRIEVE-OVER-LEDGER: top facts by cosine to the question ---")
        for s, k in sims[:TOPK]:
            i = fact_src[k]
            mark = "GOLD" if i in gold_ix else "    "
            if i not in seen:
                top_sessions.append(i); seen.add(i)
            print(f"    [{mark}] {s:.3f}  session_{i}  {all_facts[k][:80]}")
        covered = gold_ix & seen
        print(f"\n  gold sessions surfaced in top-{TOPK}: {sorted('session_%d' % i for i in covered)}")
        print(f"  VERDICT: captured {sum(1 for n in needles if n in joined)}/{len(needles)} needles;"
              f" retrieved {len(covered)}/{len(gold_ix)} gold sessions")
        print()


if __name__ == "__main__":
    main()
