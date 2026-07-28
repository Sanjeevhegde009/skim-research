# skim

*Read the index, not the whole book — and admit when you don't know.*

> **Status — research repo.** This is the framework and its benchmark validation (LongMemEval_S,
> LoCoMo). A packaged, dataset-agnostic library — `skim.ingest(conversations)` →
> `skim.answer(question)`, cite-or-refuse on **your** data — is in progress.

A low-context, **cite-or-refuse** long-term memory system for LLM conversations. The reader never
ingests the full history: it navigates a per-session table-of-contents index, opens only the
sessions it needs, answers from their raw turns — or refuses. Two models do the work: a **compiler**
(`gpt-4o-mini`) builds the index once and caches it; a cheap **reader** (`gpt-4.1-mini`) answers
every query.

## How a query is answered

1. **Navigate** — pick which sessions to open. Two navigators: *summary-nav* (the reader reads
   LLM-written per-session summaries; no embeddings) and *rich-nav* (cosine over a de-disguised
   per-session fact ledger, which surfaces incidental mentions a summary drops — e.g. *"that one was
   out of my league"* → *a property viewing*).
2. **Assemble evidence** — for single-answer questions, hand the reader a retrieval-ranked,
   density-capped window of raw turns (not whole transcripts); for counts and orderings, keep
   whole-session completeness. Route-scoped, because a capped window drops instances a count needs.
3. **Compute in code** — dates, counts, and "latest value" are done in Python, never in the model's
   head: the LLM extracts structured facts, code computes, the LLM copies the result.
4. **Escalate on refusal only** — if the first pass refuses, RAG runs: decompose → **premise gate**
   → hybrid (BM25 + cosine) retrieval → compose. The premise gate refuses false-premise questions
   instead of fabricating an answer.

Answers always come from **raw turns**; the summaries and the fact ledger are navigation aids only.

## Results

Cheap reader (`gpt-4.1-mini`), official judges, best config (`rich + density + scope`):

| Benchmark | Result |
|---|---|
| **LongMemEval_S** (full 500) | **0.766** official GPT-4o judge — beats GPT-4o full-context (0.606) at ~1/100th the reader cost, 1.8% fabrication. See [Benchmark comparison](#benchmark-comparison-longmemeval_s). |
| **LoCoMo** (all 10 convs) | **1.352** macro — see the per-category table below |

LoCoMo, all 10 convs — `gpt-4.1-mini` reader / `gpt-4o-mini` index+judge (judge score is 0–2):

| Category | judge score | token-F1 |
|---|---|---|
| open-domain | 1.50 | 0.62 |
| adversarial | 1.43 | 0.72 |
| temporal | 1.27 | 0.63 |
| single-hop | 1.07 | 0.39 |
| multi-hop | 0.79 | 0.23 |
| **MEAN** | **1.35** | **0.59** |

Adversarial is scored **deterministically** (a clean refusal = correct), which holds fabrication near
1%. All numbers come from live temperature-0 calls, so totals move by ~±0.02 run to run.

Full config ladder (0.580 → 0.654 → 0.722 → **0.766**) and per-mechanism detail:
**[RESULTS.md](RESULTS.md)**. Query-flow diagrams: **[PIPELINE_FLOW.md](PIPELINE_FLOW.md)**.

## Setup
```bash
pip install -r requirements.txt          # only dependency: requests
export OPENAI_API_KEY=sk-...             # read from the environment; never stored in a file
./download_data.sh                       # -> data/locomo10.json (2.7M) + data/longmemeval_s.json (265M)
```

## Run

Behavior is controlled entirely by env flags; every flag defaults **off**. The LongMemEval runner
aborts unless at least `PI_RAG PI_RAG_HYBRID PI_RAG_INFER` are set. The **0.766 best config** turns
on all mechanisms:

```bash
export PI_RAG=1 PI_RAG_HYBRID=1 PI_RAG_INFER=1 \
       PI_NAV_BROAD=1 PI_REASON=1 PI_DATEMATH=1 PI_RECENCY=1 PI_EVUNION=1 \
       PI_RICHINDEX=1 PI_DENSITY=1 PI_DENSITY_SCOPE=1

python run_longmemeval.py            # full 500 (append e.g. `150` for a stratified subset)
python rescore_longmemeval.py results/longmemeval/results_navbroad_reason_datemath_recency_evunion_rich_density_scoped.json
```

`run_longmemeval.py` writes strict inline scores; `rescore_longmemeval.py` re-judges the saved
answers with LongMemEval's official GPT-4o yes/no judge (the comparable number). Both are resumable
and cache-backed — a re-run only recomputes what changed.

**LoCoMo** (validated on the base config):
```bash
export PI_RAG=1 PI_RAG_HYBRID=1 PI_RAG_INFER=1
python run_pageindex.py 0            # one conversation (2nd arg caps QA, e.g. `0 20`)
python run_all_pageindex.py          # all 10 -> results/locomo/rag/all_summary.txt
```

Every flag is documented inline where it is defined in `pageindex.py` / `pi_rag.py`.

## Benchmark comparison (LongMemEval_S)

Best config (`rich + density + scope`, full 500, official GPT-4o judge). The one **judge-comparable**
comparison — same judge, same 500 questions — is against the standard strong baseline, GPT-4o reading
the full ~115K-token history:

| axis | **ours** | GPT-4o full-context |
|---|---|---|
| accuracy (official) | **0.766** | 0.606 |
| reader model | `gpt-4.1-mini` | `gpt-4o` |
| tokens / query | 7,514 | ~115,000 |
| reader $ / query | **$0.0031** | ~$0.29 |
| reader $ / correct answer | **$0.0040** | ~$0.47 (**~118×**) |
| fabrication rate | **1.8%** | high (answers false-premise Qs) |
| abstention (correct refusals) | **0.700** | — |

Against the baseline everyone anchors on, this wins all three axes at once: **+0.16 accuracy, ~118×
cheaper per correct answer, and far less fabrication.**

**Where it sits in the wider field — directional, verify before quoting.** Cross-system LongMemEval
numbers are *not* reliable: reader model, judge, subset, and prompt all differ between reports. Top
vendor / paper systems (GPT-4o-class readers plus multi-call extraction / knowledge-graph pipelines)
report the **~0.70–0.80** band and likely edge us on **raw** accuracy. This project is not chasing
that crown — the operating point is deliberately different:

> **~0.77 accuracy at ~1% of the reader cost, fabrication under 2%, with calibrated refusal.**

For cost-at-scale or trustworthy abstention (production use, not a leaderboard screenshot), that
frontier is the more useful one.

## Repository layout
```
pageindex.py            navigation (summary + rich), route classification, density assembly,
                        compute layers (datemath / recency / scratchpad), and query()
rich_index.py           de-disguised per-session fact ledger + retrieve-over-ledger navigation
pi_rag.py               RAG escalation: decompose -> premise gate -> hybrid BM25+cosine -> compose
evaluate.py             scoring: token-F1 + 0-2 GPT judge (adversarial scored deterministically)
llm.py                  provider-agnostic call layer (openai / anthropic / openai_compatible / ollama)
config.py               model + path configuration
longmemeval_loader.py   \  parse each benchmark's format into the shared session shape
locomo_loader.py        /
run_longmemeval.py      LongMemEval runner (reader-controlled, token-metered, resumable)
rescore_longmemeval.py  re-judge saved answers with LongMemEval's official GPT-4o judge
run_pageindex.py        LoCoMo: one conversation
run_all_pageindex.py    LoCoMo: all conversations + cross-conv roll-up
baselines.py            naive comparison strategies (library only)
probe_*.py              standalone diagnostics (retrieval-vs-reasoning; rich-index feasibility)
download_data.sh        fetch both datasets from their official sources

RESULTS.md              config ladder, best-config numbers, reproduce steps
PIPELINE_FLOW.md        query-flow diagrams (mermaid)
FAILURE_ANALYSIS.md     the 0.580-baseline failure map that motivated the mechanisms

data/                   datasets (gitignored)
cache/                  index summaries, turn embeddings, fact ledgers (gitignored, regenerated)
results/                per-run results + official re-scores (gitignored)
```

## Notes
- **Models** are set in `config.py` (default `gpt-4o-mini` compiler/judge + `gpt-4.1-mini` reader).
  Swap providers there, or override the reader per-run via `LME_READER_MODEL` / `LME_READER_PROVIDER`
  (Ollama supported for a fully-local reader).
- **Reproducibility:** `download_data.sh` pins the official datasets — LoCoMo from
  `snap-research/locomo`, LongMemEval from `xiaowu0162/longmemeval-cleaned`. Index summaries, turn
  embeddings, and fact ledgers are cached under `cache/` and reused across runs.
- **No heavy deps:** cosine similarity is pure-Python — no numpy, torch, or vector database.
- **Determinism:** all calls run at temperature 0; residual API nondeterminism moves totals by
  ~±0.02 on 500 questions, so treat sub-0.02 differences as noise.

## License
MIT — see [LICENSE](LICENSE).
