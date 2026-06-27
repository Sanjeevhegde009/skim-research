"""
Run the PageIndex-over-raw experiment and compare head-to-head with stable HLMA.

Builds the session index (cached), answers all QA via navigate-then-read, scores with the
SAME harness HLMA uses (score_answer judge 0-2 + token_f1), and prints PageIndex vs the
stable HLMA results (eval_conv{N}_results.json, which on this branch is the e0a8868 baseline)
side by side — per category and overall, with adversarial as the kill-switch.

Run:  set OPENAI_API_KEY=sk-...
      python run_pageindex.py 0          # full conv 0
      python run_pageindex.py 0 30       # first 30 (cheap probe first)
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

from locomo_loader import load_conversations
from evaluate import score_answer, token_f1
import config
import pageindex


def _agg(items):
    d = defaultdict(lambda: [0, 0.0, 0.0, 0])
    for it in items:
        c = it["category"]
        d[c][0] += 1
        d[c][1] += it.get("score", 0)
        d[c][2] += it.get("token_f1", 0)
        d[c][3] += bool(it.get("hallucination"))
    return d


def main():
    cid = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    conv = load_conversations()[cid]
    print(f"PageIndex-over-raw — conv {cid} ({conv['sample_id']}, {conv['speakers']})")
    import pi_rag
    print(f"query model: {config.QUERY_PROVIDER}/{config.QUERY_MODEL}  |  "
          f"compiler (index): {config.COMPILER_PROVIDER}/{config.COMPILER_MODEL}")
    print(f"PI_RAG={pageindex.PI_RAG}  gate={pi_rag.PI_RAG_GATE}  hybrid={pi_rag.PI_RAG_HYBRID}  "
          f"tau={pi_rag.TAU}  PI_REASON={pageindex.PI_REASON}")
    idx = pageindex.build_index(conv)
    print(f"index ready: {len(idx['nodes'])} session nodes\n")

    qa = conv["qa"][:limit] if limit else conv["qa"]
    results = []
    for i, q in enumerate(qa):
        question, expected = q["question"], q["answer"]
        cat = q["category_name"]
        r = pageindex.query(idx, question)
        ans = r["answer"]
        is_adv = (expected == "NOT_ANSWERABLE")
        sc = score_answer(question, expected, ans, cat)
        tf = token_f1(expected, ans, is_adversarial=is_adv)
        tr = r.get("trace", {})
        results.append({"question": question, "expected": expected, "category": cat,
                        "answer": ans, "score": sc["score"], "f1": sc.get("f1", 0.0),
                        "token_f1": tf["token_f1"], "hallucination": sc.get("hallucination"),
                        "sessions": r["sessions"], "reasoned": tr.get("reasoned", False),
                        "reasoning": tr.get("reasoning", ""),
                        "rag_fired": tr.get("rag_fired", False),
                        "rag_recovered": tr.get("rag_recovered", False),
                        "rag_gate": tr.get("rag_gate", ""),
                        "rag_best_sim": tr.get("rag_best_sim", 0.0),
                        "subqueries": tr.get("rag_subqueries", []),
                        "subanswers": tr.get("rag_subanswers", []),
                        "tok_nav": tr.get("nav_tokens", 0), "tok_ans": tr.get("ans_tokens", 0)})
        tag = (" rag✓RECOVERED" if tr.get("rag_recovered") else " rag·refused") if tr.get("rag_fired") else ""
        print(f"  {i+1}/{len(qa)} [{cat[:4]}] s={sc['score']} sess={r['sessions']}{tag} | {question[:42]}", flush=True)

    Path(f"pageindex_conv{cid}_results.json").write_text(json.dumps(results, indent=1))

    # ---- Full HLMA-style RESULTS block ----
    n = len(results)
    avg = lambda f: sum(f(r) for r in results) / n if n else 0
    print("\n" + "=" * 70)
    print(f"RESULTS — Conv {cid}: {conv['speakers']}  (PageIndex over raw)")
    print("=" * 70 + "\n")
    print(f"{'Metric':<28} pageindex")
    print("-" * 43)
    print(f"{'Avg Score (0-2)':<28} {avg(lambda r: r['score']):.2f}")
    print(f"{'Avg F1 (LLM judge)':<28} {avg(lambda r: r['f1']):.3f}")
    print(f"{'Avg F1 (token)':<28} {avg(lambda r: r['token_f1']):.3f}")
    print(f"{'Fully Correct':<28} {sum(1 for r in results if r['score'] == 2)}/{n}")
    print(f"{'Hallucinations':<28} {sum(1 for r in results if r['hallucination'])}/{n}")
    print(f"{'Avg Tok/Query':<28} {avg(lambda r: r['tok_nav'] + r['tok_ans']):.0f}")
    print(f"{'  navigate':<28} {avg(lambda r: r['tok_nav']):.0f}")
    print(f"{'  answer':<28} {avg(lambda r: r['tok_ans']):.0f}")

    print("\nBY CATEGORY (avg score / token F1):")
    print(f"{'Category':<18} pageindex")
    print("-" * 36)
    for c in sorted(_agg(results)):
        a = _agg(results)[c]
        print(f"  {c:<16} {a[1]/a[0]:.2f} / {a[2]/a[0]:.3f}")

    # Side-by-side with stable HLMA (eval_conv{cid}_results.json == e0a8868 on this branch)
    pi = _agg(results)
    hl = None
    p = Path(f"eval_conv{cid}_results.json")
    if p.exists():
        h = json.loads(p.read_text()); h = h["hlma"] if isinstance(h, dict) else h
        # restrict HLMA to the same questions when limited
        if limit:
            qset = {r["question"] for r in results}
            h = [x for x in h if x["question"] in qset]
        hl = _agg(h)

    print("\n" + "=" * 70)
    print(f"conv {cid}     {'PAGEINDEX (judge/F1)':>22}   {'HLMA stable (judge/F1)':>24}")
    cats = sorted(set(list(pi) + (list(hl) if hl else [])))
    tpi = [0, 0.0, 0.0, 0]; thl = [0, 0.0, 0.0, 0]
    for c in cats:
        a = pi.get(c, [0, 0.0, 0.0, 0]); b = hl.get(c, [0, 0.0, 0.0, 0]) if hl else [0, 0.0, 0.0, 0]
        for k in range(4):
            tpi[k] += a[k]; thl[k] += b[k]
        pis = f"{a[1]/a[0]:.2f}/{a[2]/a[0]:.3f}" if a[0] else "-"
        hls = f"{b[1]/b[0]:.2f}/{b[2]/b[0]:.3f}" if b[0] else "-"
        print(f"  {c:<12} {pis:>22}   {hls:>24}")
    print("-" * 70)
    pis = f"{tpi[1]/tpi[0]:.2f}/{tpi[2]/tpi[0]:.3f}" if tpi[0] else "-"
    hls = f"{thl[1]/thl[0]:.2f}/{thl[2]/thl[0]:.3f}" if thl[0] else "-"
    print(f"  {'OVERALL':<12} {pis:>22}   {hls:>24}")
    print(f"  {'hallucinations':<12} {tpi[3]:>22}   {thl[3]:>24}")
    print("\nVERDICT: does PageIndex-over-raw match HLMA overall AND hold adversarial?")


if __name__ == "__main__":
    main()
