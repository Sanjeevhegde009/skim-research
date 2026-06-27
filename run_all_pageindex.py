"""
Run PageIndex-over-raw across ALL LoCoMo conversations with FULL per-question tracing.

Everything is written under pageindex_runs/ and prefixed pageindex_ — HLMA's files
(eval_*, *_run.log, eval_all_summary.*) are NEVER written or replaced; they are only read
(read-only) to print a side-by-side comparison. Indexes are cached at root as
pageindex_<sample_id>.json and reused.

Per conversation it writes:
  pageindex_runs/pageindex_conv{N}_results.json   compact scored results
  pageindex_runs/pageindex_conv{N}_trace.jsonl    FULL trace per question — navigator raw
                                                  output, sessions opened, EXACT excerpts the
                                                  answerer saw, gold evidence ids, judge verdict
  pageindex_runs/pageindex_conv{N}.log            human-readable streaming log
And after every conversation (crash-safe):
  pageindex_runs/pageindex_all_summary.txt / .json   cross-conv table + HLMA comparison

Run:  set OPENAI_API_KEY=sk-...
      python run_all_pageindex.py            # all convs
      python run_all_pageindex.py 0 3        # convs 0,1,2 (end exclusive)
"""
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from locomo_loader import load_conversations
from evaluate import score_answer, token_f1
import pageindex

# When the RAG escalation is on, write to a SEPARATE dir so the committed base results in
# pageindex_runs/ (the baseline the probe and comparisons read) are never overwritten.
OUT = Path("pageindex_rag_runs" if pageindex.PI_RAG else "pageindex_runs")
OUT.mkdir(exist_ok=True)
CATS = ["adversarial", "multi-hop", "open-domain", "single-hop", "temporal"]


def aggregate(results):
    n = len(results)
    by = defaultdict(lambda: [0, 0.0, 0.0])
    for r in results:
        c = r["category"]
        by[c][0] += 1; by[c][1] += r["score"]; by[c][2] += r["token_f1"]
    return {
        "n": n,
        "avg_score": round(sum(r["score"] for r in results) / n, 3),
        "avg_llm_f1": round(sum(r["f1"] for r in results) / n, 3),
        "avg_token_f1": round(sum(r["token_f1"] for r in results) / n, 3),
        "fully_correct": sum(1 for r in results if r["score"] == 2),
        "hallucinations": sum(1 for r in results if r["hallucination"]),
        "avg_tok": round(sum(r["tok_nav"] + r["tok_ans"] for r in results) / n),
        "avg_tok_nav": round(sum(r["tok_nav"] for r in results) / n),
        "avg_tok_ans": round(sum(r["tok_ans"] for r in results) / n),
        "categories": {c: {"score": round(by[c][1] / by[c][0], 3),
                           "token_f1": round(by[c][2] / by[c][0], 3)} for c in by},
    }


def hlma_baseline(cid):
    """Read-only: stable HLMA per-conv results for comparison (never written)."""
    p = Path(f"eval_conv{cid}_results.json")
    if not p.exists():
        return None
    h = json.loads(p.read_text()); h = h["hlma"] if isinstance(h, dict) else h
    return aggregate([{"category": x["category"], "score": x.get("score", 0),
                       "f1": x.get("f1", 0.0), "token_f1": x.get("token_f1", 0),
                       "hallucination": x.get("hallucination"),
                       "tok_nav": 0, "tok_ans": x.get("tokens_est", 0)} for x in h])


def run_conv(cid, conv):
    logp = OUT / f"pageindex_conv{cid}.log"
    tracep = OUT / f"pageindex_conv{cid}_trace.jsonl"
    with open(logp, "w", encoding="utf-8") as logf, open(tracep, "w", encoding="utf-8") as tracef:
        def log(s):
            print(s); logf.write(s + "\n"); logf.flush()

        log(f"{'='*64}\nCONV {cid} — {conv['speakers']} ({conv['sample_id']})  "
            f"started {datetime.now().isoformat()}\n{'='*64}")
        idx = pageindex.build_index(conv)
        log(f"index: {len(idx['nodes'])} session nodes\n")
        results = []
        for i, q in enumerate(conv["qa"]):
            question, expected, cat = q["question"], q["answer"], q["category_name"]
            r = pageindex.query(idx, question)
            ans, tr = r["answer"], r.get("trace", {})
            is_adv = (expected == "NOT_ANSWERABLE")
            sc = score_answer(question, expected, ans, cat)
            tf = token_f1(expected, ans, is_adversarial=is_adv)
            rec = {"question": question, "expected": expected, "category": cat, "answer": ans,
                   "score": sc["score"], "f1": sc.get("f1", 0.0), "token_f1": tf["token_f1"],
                   "hallucination": sc.get("hallucination"), "judgment": sc.get("judgment", ""),
                   "sessions": r.get("sessions", []),
                   "tok_nav": tr.get("nav_tokens", 0), "tok_ans": tr.get("ans_tokens", 0)}
            results.append(rec)
            # FULL trace: everything to reconstruct why/what/how later
            tracef.write(json.dumps({**rec,
                                     "nav_raw": tr.get("nav_raw", ""),
                                     "reasoned": tr.get("reasoned", False),
                                     "reasoning": tr.get("reasoning", ""),
                                     "gold_evidence": q.get("evidence", []),
                                     "excerpts": tr.get("excerpts", [])}, ensure_ascii=False) + "\n")
            log(f"  {i+1}/{len(conv['qa'])} [{cat[:4]}] s={sc['score']} f1={tf['token_f1']:.2f} "
                f"sess={r.get('sessions')} | {question[:46]}")
    (OUT / f"pageindex_conv{cid}_results.json").write_text(json.dumps(results, indent=1))
    return results


def _conv_block(cid, s, hb):
    """The full per-conversation RESULTS block — identical layout to run_pageindex.py's
    single-conv print (metrics + navigate/answer token split + by-category + the side-by-side
    with stable HLMA). hb is the read-only HLMA aggregate (or None)."""
    n = s["n"]
    L = ["=" * 70,
         f"RESULTS - Conv {cid}: {s.get('speakers', '')}  (PageIndex over raw)",
         "=" * 70, "",
         f"{'Metric':<28} pageindex",
         "-" * 43,
         f"{'Avg Score (0-2)':<28} {s['avg_score']:.2f}",
         f"{'Avg F1 (LLM judge)':<28} {s['avg_llm_f1']:.3f}",
         f"{'Avg F1 (token)':<28} {s['avg_token_f1']:.3f}",
         f"{'Fully Correct':<28} {s['fully_correct']}/{n}",
         f"{'Hallucinations':<28} {s['hallucinations']}/{n}",
         f"{'Avg Tok/Query':<28} {s['avg_tok']}",
         f"{'  navigate':<28} {s.get('avg_tok_nav', 0)}",
         f"{'  answer':<28} {s.get('avg_tok_ans', 0)}",
         "",
         "BY CATEGORY (avg score / token F1):",
         f"{'Category':<18} pageindex",
         "-" * 36]
    for c in CATS:
        if c in s["categories"]:
            cc = s["categories"][c]
            L.append(f"  {c:<16} {cc['score']:.2f} / {cc['token_f1']:.3f}")
    # side-by-side with stable HLMA (read-only)
    L += ["", "=" * 70,
          f"conv {cid}     {'PAGEINDEX (judge/F1)':>22}   {'HLMA stable (judge/F1)':>24}"]
    for c in sorted(set(list(s["categories"]) + (list(hb["categories"]) if hb else []))):
        pc = s["categories"].get(c)
        hc = hb["categories"].get(c) if hb else None
        pis = f"{pc['score']:.2f}/{pc['token_f1']:.3f}" if pc else "-"
        hls = f"{hc['score']:.2f}/{hc['token_f1']:.3f}" if hc else "-"
        L.append(f"  {c:<12} {pis:>22}   {hls:>24}")
    L.append("-" * 70)
    pis = f"{s['avg_score']:.2f}/{s['avg_token_f1']:.3f}"
    hls = f"{hb['avg_score']:.2f}/{hb['avg_token_f1']:.3f}" if hb else "-"
    L.append(f"  {'OVERALL':<12} {pis:>22}   {hls:>24}")
    hh = str(hb["hallucinations"]) if hb else "-"
    L.append(f"  {'hallucinations':<12} {str(s['hallucinations']):>22}   {hh:>24}")
    L.append("")
    return L


def write_summary(summary):
    # JSON (machine-readable, full per-conv aggregates)
    (OUT / "pageindex_all_summary.json").write_text(json.dumps(
        {"generated": datetime.now().isoformat(), "results": summary}, indent=2),
        encoding="utf-8")
    # TXT: a FULL RESULTS block per conversation, then a cross-conv roll-up
    lines = [f"PageIndex-over-raw - all-conv results "
             f"({datetime.now().strftime('%Y-%m-%d %H:%M')})", ""]
    done = []
    for cid in sorted(summary, key=int):
        s = summary[cid]
        hb = hlma_baseline(cid)
        lines += _conv_block(cid, s, hb)
        done.append((s, hb))
    if len(done) > 1:
        lines += ["", "=" * 60, "CROSS-CONVERSATION ROLL-UP", "=" * 60,
                  f"{'conv':<6}{'PI score/F1':>16}{'HLMA score/F1':>18}"
                  f"{'PI hall':>9}{'PI tok/q':>10}", "-" * 60]
        for cid in sorted(summary, key=int):
            s = summary[cid]
            hb = hlma_baseline(cid)
            pi = f"{s['avg_score']:.2f}/{s['avg_token_f1']:.3f}"
            hl = f"{hb['avg_score']:.2f}/{hb['avg_token_f1']:.3f}" if hb else "n/a"
            lines.append(f"{cid:<6}{pi:>16}{hl:>18}{s['hallucinations']:>9}{s['avg_tok']:>10}")
        lines.append("-" * 60)
        msp = sum(s["avg_score"] for s, _ in done) / len(done)
        msf = sum(s["avg_token_f1"] for s, _ in done) / len(done)
        hs = [h for _, h in done if h]
        mhp = sum(h["avg_score"] for h in hs) / len(hs) if hs else 0
        mhf = sum(h["avg_token_f1"] for h in hs) / len(hs) if hs else 0
        mean_pi = f"{msp:.2f}/{msf:.3f}"
        mean_hl = f"{mhp:.2f}/{mhf:.3f}"
        lines.append(f"{'MEAN':<6}{mean_pi:>16}{mean_hl:>18}")
        lines += ["", "PageIndex category macro (score / token F1):"]
        for c in CATS:
            sv = [s["categories"][c]["score"] for s, _ in done if c in s["categories"]]
            fv = [s["categories"][c]["token_f1"] for s, _ in done if c in s["categories"]]
            if sv:
                lines.append(f"  {c:<14}{sum(sv) / len(sv):.3f} / {sum(fv) / len(fv):.3f}")
    (OUT / "pageindex_all_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"  [pageindex_all_summary.txt updated - {len(summary)} conv block(s)]")


def main():
    args = [a for a in sys.argv[1:]]
    convs = load_conversations()
    total = len(convs)
    if len(args) >= 2:
        rng = range(int(args[0]), min(int(args[1]), total))
    elif len(args) == 1:
        rng = range(int(args[0]), total)
    else:
        rng = range(total)
    import pi_rag
    print(f"PageIndex-over-raw — convs {list(rng)}  → all output under {OUT}/  (HLMA untouched)")
    print(f"PI_RAG={pageindex.PI_RAG}  gate={pi_rag.PI_RAG_GATE}  hybrid={pi_rag.PI_RAG_HYBRID}  "
          f"infer={pi_rag.PI_RAG_INFER}  tau={pi_rag.TAU}\n")
    summary = {}
    # Pre-load any already-scored convs OUTSIDE this run's range, so even a partial run writes a
    # COMPLETE summary (every conv that has a results file). Convs in rng are (re)scored below.
    for cid in range(total):
        p = OUT / f"pageindex_conv{cid}_results.json"
        if cid not in rng and p.exists():
            s = aggregate(json.loads(p.read_text()))
            s["speakers"] = convs[cid]["speakers"]
            summary[cid] = s
    for cid in rng:
        results = run_conv(cid, convs[cid])
        s = aggregate(results)
        s["speakers"] = convs[cid]["speakers"]
        summary[cid] = s
        write_summary(summary)   # crash-safe: rewrite after every conv
    print(f"\nDone. Full traces in {OUT}/pageindex_conv*_trace.jsonl")


if __name__ == "__main__":
    main()
