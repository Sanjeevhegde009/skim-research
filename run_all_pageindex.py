"""
Run PageIndex-over-raw across ALL LoCoMo conversations with FULL per-question tracing.

Outputs go under results/locomo/<mode>/ where <mode> is "rag" (PI_RAG on) or "base".
Indexes are cached under cache/pageindex/ and reused.

Per conversation it writes (under results/locomo/<mode>/):
  conv{N}_results.json   compact scored results
  conv{N}_trace.jsonl    FULL trace per question — navigator raw output, sessions opened,
                         EXACT excerpts the answerer saw, gold evidence ids, judge verdict
  conv{N}.log            human-readable streaming log
And after every conversation (crash-safe):
  all_summary.txt / .json   cross-conv roll-up table

Run:  set OPENAI_API_KEY=sk-...
      python run_all_pageindex.py            # all convs
      python run_all_pageindex.py 0 3        # convs 0,1,2 (end exclusive)
"""
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

from locomo_loader import load_conversations
from evaluate import score_answer, token_f1
import config
import pageindex

# Model overrides — swap any of the three models via env, no config.py edit needed.
#   READER_*   (-> QUERY_*)  : navigation + answering
#   COMPILER_*               : index building
#   JUDGE_*                  : answer scoring (keep on an API for fair, comparable grading)
# Provider is "openai" (default) or "ollama"; model is e.g. gpt-4o or a tag like gemma4:e4b.
# Unset = config.py defaults, so the all-API setup needs no change.
for _env, _attr in [("READER_PROVIDER", "QUERY_PROVIDER"), ("READER_MODEL", "QUERY_MODEL"),
                    ("COMPILER_PROVIDER", "COMPILER_PROVIDER"), ("COMPILER_MODEL", "COMPILER_MODEL"),
                    ("JUDGE_PROVIDER", "JUDGE_PROVIDER"), ("JUDGE_MODEL", "JUDGE_MODEL")]:
    _v = os.environ.get(_env, "").strip()
    if _v:
        setattr(config, _attr, _v)

# RAG and base runs write to separate subdirs so they never overwrite each other.
OUT = config.RESULTS_DIR / "locomo" / ("rag" if pageindex.PI_RAG else "base")
OUT.mkdir(parents=True, exist_ok=True)
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


def run_conv(cid, conv):
    logp = OUT / f"conv{cid}.log"
    tracep = OUT / f"conv{cid}_trace.jsonl"
    with open(logp, "w", encoding="utf-8") as logf, open(tracep, "w", encoding="utf-8") as tracef:
        def log(s):
            print(s); logf.write(s + "\n"); logf.flush()

        log(f"{'='*64}\nCONV {cid} — {conv['speakers']} ({conv['sample_id']})  "
            f"started {datetime.now().isoformat()}\n{'='*64}")
        idx = pageindex.build_index(conv)
        # LoCoMo questions are asked after the whole conversation, so the reference "question date"
        # is the latest session's date (nodes are chronological). The datemath/recency layers need it
        # to resolve "how many months/days ago ..." against a real "now".
        qdate = idx["nodes"][-1]["date"] if idx["nodes"] else None
        log(f"index: {len(idx['nodes'])} session nodes  (question_date={qdate})\n")
        results = []
        for i, q in enumerate(conv["qa"]):
            question, expected, cat = q["question"], q["answer"], q["category_name"]
            r = pageindex.query(idx, question, qdate)
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
    (OUT / f"conv{cid}_results.json").write_text(json.dumps(results, indent=1))
    return results


def _conv_block(cid, s):
    """The full per-conversation RESULTS block — metrics + navigate/answer token split +
    by-category + an overall summary."""
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
    L.append("-" * 36)
    L.append(f"  {'OVERALL':<16} {s['avg_score']:.2f} / {s['avg_token_f1']:.3f}")
    L.append(f"  {'hallucinations':<16} {s['hallucinations']}")
    L.append("")
    return L


def write_summary(summary):
    # JSON (machine-readable, full per-conv aggregates)
    (OUT / "all_summary.json").write_text(json.dumps(
        {"generated": datetime.now().isoformat(), "results": summary}, indent=2),
        encoding="utf-8")
    # TXT: a FULL RESULTS block per conversation, then a cross-conv roll-up
    lines = [f"PageIndex-over-raw - all-conv results "
             f"({datetime.now().strftime('%Y-%m-%d %H:%M')})", ""]
    done = []
    for cid in sorted(summary, key=int):
        s = summary[cid]
        lines += _conv_block(cid, s)
        done.append(s)
    if len(done) > 1:
        lines += ["", "=" * 60, "CROSS-CONVERSATION ROLL-UP", "=" * 60,
                  f"{'conv':<6}{'PI score/F1':>16}{'PI hall':>9}{'PI tok/q':>10}", "-" * 60]
        for cid in sorted(summary, key=int):
            s = summary[cid]
            pi = f"{s['avg_score']:.2f}/{s['avg_token_f1']:.3f}"
            lines.append(f"{cid:<6}{pi:>16}{s['hallucinations']:>9}{s['avg_tok']:>10}")
        lines.append("-" * 60)
        msp = sum(s["avg_score"] for s in done) / len(done)
        msf = sum(s["avg_token_f1"] for s in done) / len(done)
        mean_pi = f"{msp:.2f}/{msf:.3f}"
        lines.append(f"{'MEAN':<6}{mean_pi:>16}")
        lines += ["", "PageIndex category macro (score / token F1):"]
        for c in CATS:
            sv = [s["categories"][c]["score"] for s in done if c in s["categories"]]
            fv = [s["categories"][c]["token_f1"] for s in done if c in s["categories"]]
            if sv:
                lines.append(f"  {c:<14}{sum(sv) / len(sv):.3f} / {sum(fv) / len(fv):.3f}")
    (OUT / "all_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"  [all_summary.txt updated - {len(summary)} conv block(s)]")


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
    print(f"PageIndex-over-raw — convs {list(rng)}  → all output under {OUT}/")
    print(f"PI_RAG={pageindex.PI_RAG} hybrid={pi_rag.PI_RAG_HYBRID} infer={pi_rag.PI_RAG_INFER} "
          f"gate={pi_rag.PI_RAG_GATE} nav_broad={pageindex.PI_NAV_BROAD} reason={pageindex.PI_REASON} "
          f"datemath={pageindex.PI_DATEMATH} recency={pageindex.PI_RECENCY} evunion={pageindex.PI_EVUNION} "
          f"rich={pageindex.PI_RICHINDEX} density={pageindex.PI_DENSITY} scope={pageindex.PI_DENSITY_SCOPE} "
          f"tau={pi_rag.TAU}\n")
    summary = {}
    # Pre-load any already-scored convs OUTSIDE this run's range, so even a partial run writes a
    # COMPLETE summary (every conv that has a results file). Convs in rng are (re)scored below.
    for cid in range(total):
        p = OUT / f"conv{cid}_results.json"
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
    print(f"\nDone. Full traces in {OUT}/conv*_trace.jsonl")


if __name__ == "__main__":
    main()
