# LongMemEval_S — Results

Reader: **gpt-4.1-mini**. Judge: official LongMemEval GPT-4o yes/no (via `rescore_longmemeval.py`).
Baselines and configs are compared on the **same 500 questions**.

## Best config — `rich + density + scope` = **0.766** (full 500)

The current best. **Pareto-dominates the stable `holy` baseline (0.722): +0.044 accuracy AND 40% fewer
reader tokens, with fewer hallucinations.** +0.044 on 500 is above the ±0.02 drift floor (+46 gains /
−24 regressions, +22 net).

| type | n | holy | rich+density+scope | Δ |
|------|---|------|--------------------|---|
| single-session-user | 70 | 0.829 | 0.929 | +0.100 |
| single-session-preference | 30 | 0.200 | 0.367 | +0.167 |
| multi-session | 133 | 0.617 | 0.662 | +0.045 |
| temporal-reasoning | 133 | 0.744 | 0.782 | +0.038 |
| knowledge-update | 78 | 0.795 | 0.821 | +0.026 |
| single-session-assistant | 56 | 0.964 | 0.911 | −0.054 |
| **OVERALL** | **500** | **0.722** | **0.766** | **+0.044** |

Economy & honesty (same 500): reader **7,514 tok/q (60% of holy's 12,522)**, $0.0031/q vs $0.0051;
hallucinations **9/500** (holy 13); abstention **0.700** (holy 0.667).

The one regression (single-session-assistant, a *lookup* route density/scope never touch) is caused by
rich's fact-nav being slightly worse than summary-nav on "what did you tell me in that chat" recall —
isolable; the next step is nav-source routing (summary-nav for single-session lookups, rich-nav for
compute).

## Ladder (official, full 500)

| config | official | notes |
|--------|----------|-------|
| baseline | 0.580 | PI_RAG + HYBRID + INFER |
| +NAV_BROAD +REASON +DATEMATH | 0.654 | beats GPT-4o full-context (0.606) |
| **holy** = +RECENCY +EVUNION | **0.722** | stable baseline, git tag `holy` |
| **+RICHINDEX +DENSITY +DENSITY_SCOPE** | **0.766** | git tag `best-0.766` |

## Mechanisms (all flag-gated, off by default → `holy` byte-identical when unset)

- **PI_RICHINDEX** — retrieve-over-ledger nav (`rich_index.py`): a de-disguised per-session fact
  ledger is embedded; navigation ranks sessions by cosine over facts, breaching the lexical disguise
  summary-nav can't ("out of my league" = a property viewing). Nav-only; answers from raw turns.
- **PI_DENSITY** — density-preserving assembly (`_assemble_density`): on compute routes the reader
  gets a retrieval-ranked, capped window (global hybrid top-12 + ≤6 per-nav-session injection) instead
  of whole navigated sessions, so a pivot fact stays salient rather than diluted.
- **PI_DENSITY_SCOPE** — route-scoped: the cap applies only to bounded-pivot compute (datemath
  single-value, recency); counts and orderings keep whole-session + union completeness.

## Reproduce

```bash
export OPENAI_API_KEY=sk-...
# holy's 8 flags + the 3 mechanisms
export PI_RAG=1 PI_RAG_HYBRID=1 PI_RAG_INFER=1 PI_NAV_BROAD=1 PI_REASON=1 \
       PI_DATEMATH=1 PI_RECENCY=1 PI_EVUNION=1 \
       PI_RICHINDEX=1 PI_DENSITY=1 PI_DENSITY_SCOPE=1
python run_longmemeval.py            # full 500 (omit arg); writes results_..._rich_density_scoped.json
python rescore_longmemeval.py results/longmemeval/results_navbroad_reason_datemath_recency_evunion_rich_density_scoped.json
```

Escape hatches: `PI_DENSITY_SCOPE=` off → blanket rich+density; all three off → `holy` (0.722).

## Git references

- `holy` — the stable 0.722 baseline (git tag, commit `bd13855`).
- `best-0.766` — this config, rich + density + scope (git tag).
- All on `main`.
