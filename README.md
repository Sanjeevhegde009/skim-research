# pageindex-rag

A low-context, **cite-or-refuse** memory-retrieval stack for long conversations:
**PageIndex-over-raw** (vectorless table-of-contents navigation) + **RAG escalation**
(fires only on refusal residue) + a **premise gate** (deterministic refusal, ~1% fabrication).

The reader never sees the full history. It navigates a per-session table-of-contents index by
**reasoning** (no embeddings for navigation), reads only the chosen sessions, and answers under a
conservative cite-or-refuse prompt. RAG escalates **only** when the first pass refuses.

**Validated (cheap reader, `gpt-4.1-mini`):**

| Benchmark | Result |
|---|---|
| **LoCoMo** (all 10 convs) | 1.275 → **1.352** macro (PageIndex base → +RAG), ~1% hallucination |
| **LongMemEval_S** (full 500) | **0.766** official GPT-4o judge (best config) — beats GPT-4o full-context (0.606) at ~1/100th the reader cost, 1.8% fabrication |

See **[Benchmark comparison](#benchmark-comparison-longmemeval_s)** for the cost / honesty / accuracy breakdown, and **[RESULTS.md](RESULTS.md)** for the config ladder and reproduce steps.

Only the provider-agnostic LLM-call plumbing is shared infrastructure (`llm.py`) — no wiki
compiler, no embeddings for navigation.

## Repository layout
```
pageindex.py            vectorless ToC index + reasoning navigation
pi_rag.py               RAG escalation: decompose -> hybrid BM25+cosine -> premise gate -> compose
evaluate.py             scoring: token-F1 + 0–2 frontier judge (deterministic refusal scoring)
llm.py                  provider-agnostic call layer (openai / anthropic / openai_compatible / ollama)
config.py               model + path configuration
locomo_loader.py        \\  map each benchmark into the shared `conv` shape
longmemeval_loader.py   /
run_pageindex.py        LoCoMo: one conversation
run_all_pageindex.py    LoCoMo: all conversations + cross-conv roll-up
run_longmemeval.py      LongMemEval runner (reader-controlled, token-metered)
rescore_longmemeval.py  re-judge saved answers with LongMemEval's official GPT-4o judge
baselines.py            naive comparison strategies (library only)
download_data.sh        fetch both datasets from their official sources

data/                   datasets (gitignored; filled by download_data.sh)
cache/                  regenerated index ToC + embeddings + LongMemEval summaries (gitignored)
results/
  locomo/{rag,base}/    per-conv results, traces, logs, all_summary.{txt,json}
  longmemeval/          results, summaries, official re-score
```

## Setup
```bash
pip install -r requirements.txt          # only dependency is `requests`
export OPENAI_API_KEY=sk-...             # keys live in env vars only, never in files
./download_data.sh                       # -> data/locomo10.json (2.7M) + data/longmemeval_s.json (265M)
```

## Run
The full **PageIndex + RAG + gating** config is switched on by three env flags (the premise gate
is **on by default**); without them you get the weaker base PageIndex.
```bash
export PI_RAG=1 PI_RAG_HYBRID=1 PI_RAG_INFER=1
```

**LoCoMo**
```bash
python run_pageindex.py 0           # one conversation (add a 2nd arg to cap QA, e.g. `0 20`)
python run_all_pageindex.py         # all 10 convs -> results/locomo/rag/all_summary.txt
```

**LongMemEval** (per-question ~115k-token haystacks; the runner aborts unless all three flags are set)
```bash
python run_longmemeval.py 150       # 150-question stratified subset -> results/longmemeval/
python rescore_longmemeval.py       # re-judge with the official GPT-4o yes/no judge
```

## Results (reference)
LoCoMo, all 10 convs — `gpt-4.1-mini` reader / `gpt-4o-mini` index+judge:

| Category | score / token-F1 |
|---|---|
| adversarial | 1.43 / 0.72 |
| open-domain | 1.50 / 0.62 |
| temporal | 1.27 / 0.63 |
| single-hop | 1.07 / 0.39 |
| multi-hop | 0.79 / 0.23 |
| **MEAN** | **1.35 / 0.59** |

Every number comes from live temperature-0 LLM calls, so it varies slightly run to run.
Adversarial is scored **deterministically** (a clean refusal = correct), which keeps fabrication ~1%.

## Benchmark comparison (LongMemEval_S)

Best config (`rich + density + scope`, full 500, official GPT-4o judge). The one **judge-comparable**
comparison — same judge, same 500 questions — is against the standard strong baseline, GPT-4o reading
the full ~115k-token history:

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
report the **~0.70–0.80** band and likely edge us on **raw** accuracy. We are not chasing that crown —
the operating point here is deliberately different:

> **~0.77 accuracy at ~1% of the reader cost, fabrication under 2%, with calibrated refusal.**

For cost-at-scale or trustworthy abstention (production use, not a leaderboard screenshot), that
frontier is the more useful one. The full ladder (0.580 → 0.654 → `holy` 0.722 → **0.766**), the three
flag-gated mechanisms, and exact reproduce steps are in **[RESULTS.md](RESULTS.md)**.

## Notes
- **Models** are set in `config.py` (default `gpt-4o-mini` compiler/judge + `gpt-4.1-mini` reader). Swap providers there, or for LongMemEval via `LME_READER_MODEL` / `LME_READER_PROVIDER`.
- **Reproducibility:** `download_data.sh` pins the official datasets — LoCoMo from `snap-research/locomo`, LongMemEval from `xiaowu0162/longmemeval-cleaned`. Indexes and embeddings are cached under `cache/` and reused across runs.
- Cosine similarity is pure-Python; there is no numpy/torch dependency.

## License
MIT — see [LICENSE](LICENSE).
