# pageindex-rag

A low-context, **cite-or-refuse** memory-retrieval stack for long conversations:
**PageIndex-over-raw** (vectorless table-of-contents navigation) + **RAG escalation**
(fires only on refusal residue) + a **premise gate** (deterministic refusal, ~1% fabrication).

Extracted as a standalone project from the HLMA research repo (stable commit `5758625`).
The HLMA wiki-compiler is **not** part of this project — only its provider-agnostic
LLM-call plumbing was kept, now in `llm.py`.

**Validated:**
- **LoCoMo** — 1.275 → **1.352** macro (PageIndex base → +RAG), on a cheap reader.
- **LongMemEval** — **0.440** strict / **0.587** official GPT-4o judge, ~1.3% fabrication.

## Layout
| File | Role |
|------|------|
| `pageindex.py` | Vectorless ToC index + reasoning navigation; RAG fires on refusal residue. |
| `pi_rag.py` | RAG escalation: decompose → hybrid BM25+cosine retrieval → premise gate → compose. |
| `evaluate.py` | Scoring only: token-F1 + 0–2 frontier judge; deterministic refusal scoring for adversarial. |
| `llm.py` | Provider-agnostic call layer (`anthropic` / `openai` / `openai_compatible` / `ollama`). |
| `config.py` | Compiler & query model selection; `LONGMEMEVAL_PATH`. |
| `locomo_loader.py`, `longmemeval_loader.py` | Map each benchmark into the shared `conv` shape. |
| `run_pageindex.py`, `run_all_pageindex.py` | LoCoMo runners (single conv / all convs). |
| `run_longmemeval.py`, `rescore_longmemeval.py` | LongMemEval runner + official GPT-4o re-judge. |
| `baselines.py` | Naive comparison strategies (full-history / summary-only / sliding-window). Library only — not wired into a runner yet. |

## Setup
```bash
pip install requests
export OPENAI_API_KEY=sk-...        # keys live in env vars only, never in files
```
Datasets are **not** committed (large) — place `locomo10.json` and `longmemeval_s.json`
at the repo root.

## Run
```bash
# LoCoMo — conv 0 (optional second arg caps the QA count)
python run_pageindex.py 0
python run_pageindex.py 0 20

# LongMemEval — best config (the runner aborts unless all three are set)
export PI_RAG=1 PI_RAG_HYBRID=1 PI_RAG_INFER=1
python run_longmemeval.py 150

# Re-judge the saved LongMemEval answers with the official GPT-4o judge
python rescore_longmemeval.py
```
