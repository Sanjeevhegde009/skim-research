"""
Re-score the saved LongMemEval answers with LongMemEval's OFFICIAL GPT-4o judge.

Our run (run_longmemeval.py) graded with a strict gpt-4o-mini 0-2 judge (correct = score==2).
LongMemEval's leaderboard uses a GPT-4o, yes/no, TYPE-AWARE judge — get_anscheck_prompt, copied
VERBATIM below from the repo's src/evaluation/evaluate_qa.py. This re-grades the EXISTING answers
in longmemeval_results.json only — no pipeline re-run; indexes/embeddings/answers untouched — so the
number becomes comparable to published results and we can measure how much the strict mini-judge
understated us.

Abstention (_abs, expected == NOT_ANSWERABLE) is kept on our DETERMINISTIC refusal rule (a clean
refusal is correct): the loader discarded the explanation text, and the official abstention judge
only checks "did the model identify it as unanswerable", which the deterministic rule captures.

Run (Windows):
  set OPENAI_API_KEY=sk-...
  python rescore_longmemeval.py             # judge = gpt-4o  (override: set LME_JUDGE_MODEL=...)
Reads longmemeval_results.json -> writes longmemeval_rescored_gpt4o.json + a side-by-side summary.
Framework files (pageindex.py / pi_rag.py / evaluate.py / llm.py) stay byte-identical.
"""
import json
import os
import sys
from collections import defaultdict

import config
from llm import _call_openai_compat, _api_call_with_retry
from evaluate import _refused

JUDGE_MODEL = os.environ.get("LME_JUDGE_MODEL", "gpt-4o")
RESULTS = sys.argv[1] if len(sys.argv) > 1 else "longmemeval_results.json"
_stem = RESULTS[:-5] if RESULTS.endswith(".json") else RESULTS
OUT = _stem + "_official.json"
SUMMARY = _stem + "_official_summary.txt"


# ── LongMemEval's official judge prompt — copied VERBATIM from src/evaluation/evaluate_qa.py ──
def get_anscheck_prompt(task, question, answer, response, abstention=False):
    if not abstention:
        if task in ['single-session-user', 'single-session-assistant', 'multi-session']:
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'temporal-reasoning':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'knowledge-update':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'single-session-preference':
            template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        else:
            raise NotImplementedError
    else:
        template = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
        prompt = template.format(question, answer, response)
    return prompt


def judge(prompt):
    """Official judge call: single user message, temp 0, parse `'yes' in response.lower()`."""
    out = _api_call_with_retry(lambda: _call_openai_compat(
        prompt, "", 0.0, JUDGE_MODEL, config.COMPILER_API_KEY_ENV, "openai", ""))
    return "yes" in (out or "").strip().lower()


def main():
    rows = json.load(open(RESULTS))
    print(f"re-scoring {len(rows)} saved answers with judge={JUDGE_MODEL}\n")
    out = []
    for i, x in enumerate(rows, 1):
        is_abs = (x["expected"] == "NOT_ANSWERABLE")
        if is_abs:
            new_correct = _refused(x["answer"])               # deterministic: clean refusal = correct
        else:
            p = get_anscheck_prompt(x["question_type"], x["question"], x["expected"], x["answer"])
            new_correct = judge(p)
        rec = {**x, "correct_mini": int(x["correct"]), "correct_gpt4o": int(new_correct)}
        out.append(rec)
        flip = "" if rec["correct_mini"] == rec["correct_gpt4o"] else (
            "  ^FLIP+" if new_correct else "  ^FLIP-")
        print(f"  {i}/{len(rows)} [{x['question_type'][:16]:<16}] mini={rec['correct_mini']} "
              f"gpt4o={rec['correct_gpt4o']}{flip} | {x['question'][:36]}", flush=True)

    json.dump(out, open(OUT, "w"), indent=1)
    _report(out)


def _report(rows):
    n = len(rows)
    bytype = defaultdict(lambda: [0, 0, 0])                    # n, mini_correct, gpt4o_correct
    abst = [0, 0, 0]
    for x in rows:
        if x["expected"] == "NOT_ANSWERABLE":
            d = abst
        else:
            d = bytype[x["question_type"]]
        d[0] += 1; d[1] += x["correct_mini"]; d[2] += x["correct_gpt4o"]
    mini = sum(x["correct_mini"] for x in rows)
    g4 = sum(x["correct_gpt4o"] for x in rows)

    reader = rows[0].get("reader", "gpt-4.1-mini") if rows else "?"   # new runs tag each record
    L = ["=" * 60,
         "LongMemEval re-score: gpt-4o-mini (strict 0-2) vs GPT-4o (official yes/no)",
         f"judge={JUDGE_MODEL}  reader={reader}  instances={n}",
         "=" * 60,
         "",
         f"{'type (answerable)':<26}{'n':>4}{'mini':>8}{'gpt4o':>8}{'Δ':>8}",
         "-" * 54]
    for t in sorted(bytype):
        a = bytype[t]
        L.append(f"{t:<26}{a[0]:>4}{a[1]/a[0]:>8.3f}{a[2]/a[0]:>8.3f}{(a[2]-a[1])/a[0]:>+8.3f}")
    if abst[0]:
        L.append(f"{'abstention':<26}{abst[0]:>4}{abst[1]/abst[0]:>8.3f}{abst[2]/abst[0]:>8.3f}"
                 f"{(abst[2]-abst[1])/abst[0]:>+8.3f}")
    L += ["-" * 54,
          f"{'OVERALL':<26}{n:>4}{mini/n:>8.3f}{g4/n:>8.3f}{(g4-mini)/n:>+8.3f}"]
    up = sum(1 for x in rows if not x["correct_mini"] and x["correct_gpt4o"])
    dn = sum(1 for x in rows if x["correct_mini"] and not x["correct_gpt4o"])
    L += ["",
          f"flips: +{up} (mini wrong -> gpt4o right),  -{dn} (mini right -> gpt4o wrong)"]
    text = "\n".join(L)
    print("\n" + text)
    open(SUMMARY, "w", encoding="utf-8").write(text + "\n")
    print(f"\nwrote {OUT} + {SUMMARY}")


if __name__ == "__main__":
    main()
