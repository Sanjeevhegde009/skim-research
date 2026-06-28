"""
Diagnostic probe — localize where LongMemEval points are lost (read-only, no API).

For each ANSWERABLE question it cross-tabs, per question_type:
  RETRIEVAL  : did the navigator open a gold `answer_session`?   (hit / miss)
  CORRECTNESS: the saved judge label                              (correct / wrong)

That 2x2 separates NAVIGATION failures (missed the evidence) from REASONING/GATE
failures (had the evidence, still answered wrong) — so we fix the right stage.
Abstention is reported separately (no gold evidence; correct = refusal).

Usage:
  python probe_longmemeval.py <results.json> [raw=data/longmemeval_s.json] [correct_key=correct_gpt4o]
"""
import json
import re
import sys
from collections import defaultdict, Counter

RESULTS = sys.argv[1]
RAW = sys.argv[2] if len(sys.argv) > 2 else "data/longmemeval_s.json"
CKEY = sys.argv[3] if len(sys.argv) > 3 else "correct_gpt4o"

raw = {x["question_id"]: x for x in json.load(open(RAW, encoding="utf-8"))}
res = json.load(open(RESULTS, encoding="utf-8"))
if res and CKEY not in res[0]:
    CKEY = "correct"   # fall back to whatever correctness flag exists
print(f"results: {RESULTS}  ({len(res)} records)  |  correctness key: {CKEY}\n")


def opened_ids(rec, inst):
    """Map the navigator's opened keys (session_<i>) -> their session_ids."""
    ids, hs = set(), inst["haystack_session_ids"]
    for k in rec.get("sessions", []):
        m = re.search(r"(\d+)$", str(k))
        if m and int(m.group(1)) < len(hs):
            ids.add(hs[int(m.group(1))])
    return ids


cat = defaultdict(Counter)   # question_type -> Counter[(hit|miss, correct|wrong)]
abst = Counter()
nojoin = 0
for r in res:
    correct = bool(r.get(CKEY, r.get("correct", False)))
    if r.get("abstention"):
        abst["correct" if correct else "wrong"] += 1
        continue
    inst = raw.get(r["question_id"])
    if not inst:
        nojoin += 1
        continue
    gold = set(inst.get("answer_session_ids", []))
    hit = bool(opened_ids(r, inst) & gold)
    cat[r["question_type"]][("hit" if hit else "miss", "correct" if correct else "wrong")] += 1

order = ["temporal-reasoning", "multi-session", "knowledge-update",
         "single-session-user", "single-session-assistant", "single-session-preference"]
cats = [c for c in order if c in cat] + [c for c in cat if c not in order]

hdr = f"{'category':<26}{'n':>4}{'acc':>6}{'recall':>8}   hit_ok hit_x miss_ok miss_x"
print(hdr)
print("-" * len(hdr))
TOT = Counter()
for c in cats:
    cc = cat[c]
    TOT.update(cc)
    hc, hw = cc[("hit", "correct")], cc[("hit", "wrong")]
    mc, mw = cc[("miss", "correct")], cc[("miss", "wrong")]
    n = hc + hw + mc + mw
    acc = (hc + mc) / n if n else 0
    rec = (hc + hw) / n if n else 0
    print(f"{c:<26}{n:>4}{acc:>6.2f}{rec:>8.2f}   {hc:>5} {hw:>5} {mc:>6} {mw:>6}")

hc, hw = TOT[("hit", "correct")], TOT[("hit", "wrong")]
mc, mw = TOT[("miss", "correct")], TOT[("miss", "wrong")]
n = hc + hw + mc + mw
print("-" * len(hdr))
print(f"{'ANSWERABLE TOTAL':<26}{n:>4}{(hc+mc)/n:>6.2f}{(hc+hw)/n:>8.2f}   {hc:>5} {hw:>5} {mc:>6} {mw:>6}")
print(f"\nabstention: {abst['correct']}/{abst['correct']+abst['wrong']} correct refusals")
if nojoin:
    print(f"(unjoined question_ids: {nojoin})")

print("\n--- of the WRONG answerable answers: navigation vs reasoning ---")
for c in cats:
    cc = cat[c]
    hw, mw = cc[("hit", "wrong")], cc[("miss", "wrong")]
    if hw + mw:
        print(f"  {c:<26} wrong={hw+mw:>3}   reasoning/gate(had evidence)={hw:>3}   navigation(missed)={mw:>3}")
