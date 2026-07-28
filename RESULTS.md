# LongMemEval_S — Results

Reader: **gpt-4.1-mini**. Judge: official LongMemEval GPT-4o yes/no (via `rescore_longmemeval.py`).
All configs are compared on the **same 500 questions**.

## skim, full config = **0.766** (full 500)

The comparison point is **the same framework without the three added mechanisms — the "0.722
config"** (the full config adds ledger navigation, a focused evidence window, and count-safe routing;
see the ladder below). Against it, the full config **Pareto-dominates: +0.044 accuracy AND 40% fewer
reader tokens, with fewer hallucinations.** +0.044 on 500 is above the ±0.02 drift floor (+46 gains /
−24 regressions, +22 net).

| type | n | 0.722 config | skim (full) | Δ |
|------|---|-----------|-----------|---|
| single-session-user | 70 | 0.829 | 0.929 | +0.100 |
| single-session-preference | 30 | 0.200 | 0.367 | +0.167 |
| multi-session | 133 | 0.617 | 0.662 | +0.045 |
| temporal-reasoning | 133 | 0.744 | 0.782 | +0.038 |
| knowledge-update | 78 | 0.795 | 0.821 | +0.026 |
| single-session-assistant | 56 | 0.964 | 0.911 | −0.054 |
| **OVERALL** | **500** | **0.722** | **0.766** | **+0.044** |

Economy & honesty (same 500, full config vs the 0.722 config): reader **7,514 tok/q vs 12,522
(60%)**, $0.0031/q vs $0.0051; hallucinations **9/500 vs 13**; abstention **0.700 vs 0.667**.

The one regression (single-session-assistant, a *lookup* route the density window never touches) is
because ledger navigation is slightly worse than summary navigation on "what did you tell me in that
chat" recall — isolable; the next step is nav-source routing (summary nav for single-session lookups,
ledger nav for compute).

## Ladder (official, full 500)

Each step adds mechanisms to the one above; the ablation is what shows 0.766 isn't one lucky trick.

| step | adds (flags) | plain-English | official |
|------|--------------|---------------|----------|
| minimal | `PI_RAG + HYBRID + INFER` | navigate + RAG escalation + premise gate | 0.580 |
| + compute | `+ NAV_BROAD + REASON + DATEMATH` | broad nav + scratchpad + date math in code | 0.654 |
| + recency/union | `+ RECENCY + EVUNION` | latest-value tracking + retrieval-augmented evidence | **0.722** |
| + nav/evidence | `+ RICHINDEX + DENSITY + DENSITY_SCOPE` | ledger navigation + focused window + count-safe routing | **0.766** |

The 0.722 step is tagged `baseline-0.722`; the full config is tagged `best-0.766`.

## The three added mechanisms, in detail (flag-gated, off by default; the 0.722 config is byte-identical when unset)

- **Ledger navigation** (`PI_RICHINDEX`, `rich_index.py`): a de-disguised per-session fact ledger is
  embedded; navigation ranks sessions by cosine over facts, breaching the lexical disguise summary-nav
  can't ("out of my league" = a property viewing). Nav-only; answers still come from raw turns.
- **Focused evidence window** (`PI_DENSITY`, `_assemble_density`): on compute routes the reader gets a
  retrieval-ranked, capped window (global hybrid top-12 + ≤6 per-nav-session injection) instead of
  whole navigated sessions, so a pivot fact stays salient rather than diluted.
- **Count-safe routing** (`PI_DENSITY_SCOPE`): the cap applies only to single-pivot compute (datemath
  single-value, recency); counts and orderings keep whole-session completeness, because a capped
  window drops instances a count needs.

## Reproduce

```bash
export OPENAI_API_KEY=sk-...
# the 0.722 config (8 flags) + the 3 added mechanisms
export PI_RAG=1 PI_RAG_HYBRID=1 PI_RAG_INFER=1 PI_NAV_BROAD=1 PI_REASON=1 \
       PI_DATEMATH=1 PI_RECENCY=1 PI_EVUNION=1 \
       PI_RICHINDEX=1 PI_DENSITY=1 PI_DENSITY_SCOPE=1
python run_longmemeval.py            # full 500 (omit arg); writes results_..._rich_density_scoped.json
python rescore_longmemeval.py results/longmemeval/results_navbroad_reason_datemath_recency_evunion_rich_density_scoped.json
```

Escape hatches: `PI_DENSITY_SCOPE=` off → the density cap applies to every compute route (counts
included); all three mechanisms off → the 0.722 config.

## Git references

- `baseline-0.722` — the stable 0.722 config, i.e. skim without the three added mechanisms (git tag, commit `bd13855`).
- `best-0.766` — skim's full config: ledger navigation + focused window + count-safe routing (git tag).
- All on `main`.
