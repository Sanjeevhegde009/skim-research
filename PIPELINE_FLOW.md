# skim — query-flow diagrams (0.766 best config)

**Config:** `PI_RAG + HYBRID + INFER + NAV_BROAD + REASON + DATEMATH + RECENCY + EVUNION + RICHINDEX + DENSITY + DENSITY_SCOPE`
**Result:** LongMemEval_S full-500 official **0.766**, reader = gpt-4.1-mini, fabrication 1.8%.

Two models: a **compiler** (gpt-4o-mini) builds the index once; a cheap **reader** (gpt-4.1-mini)
answers every query. The guiding principle: **the LLM extracts, Python computes** — arithmetic,
sorting, and date math are done in code, never in the model's head.

---

## 1. Offline — build once, cached

```mermaid
flowchart LR
    S["Haystack:<br/>~50-80 sessions<br/>(~115K tokens)"] --> IDX["INDEX BUILD (compiler, cached)<br/>one node per session +<br/>LLM table-of-contents summary"]
    S --> RICH["RICH LEDGER (compiler, cached)<br/>de-disguised fact ledger per session,<br/>facts embedded"]
    S --> EMB["TURN EMBEDDINGS + BM25<br/>(built lazily, cached)"]
    IDX -.feeds.-> NAVS["summary-nav<br/>(reader reads summaries)"]
    RICH -.feeds.-> NAVR["rich-nav<br/>(cosine over facts)"]
    EMB -.feeds.-> RETI["retrieval / density window<br/>(reads raw turns)"]
```

---

## 2. Query — the main flow

```mermaid
flowchart TD
    Q["QUESTION (+ question_date)"] --> CL{"Classify by<br/>regex trigger"}
    CL -->|"is_temporal_math"| R1["route = DATEMATH"]
    CL -->|"is_recency"| R2["route = RECENCY"]
    CL -->|"is_aggregate OR needs_reasoning"| R3["route = AGGREGATE"]
    CL -->|"none"| R4["route = LOOKUP"]

    R1 --> NB["NAVIGATE — broad, cap 8"]
    R2 --> NB
    R3 --> NB
    R4 --> NP["NAVIGATE — precision, cap 3"]

    NB --> NAV["NAVIGATE -> session keys<br/>rich-nav (cosine over fact ledger, PI_RICHINDEX)<br/>or summary-nav (reader reads summaries)"]
    NP --> NAV
    NAV --> COL{"assemble evidence<br/>(route-scoped, PI_DENSITY)"}

    COL -->|"single-pivot<br/>(datemath / recency)"| DEN["DENSITY WINDOW:<br/>global hybrid top-12 +<br/>bounded per-nav-session injection<br/>(capped, ~18 turns)"]
    COL -->|"counts / orderings"| WHOLE["COMPLETENESS:<br/>whole nav sessions +<br/>evidence-union retrieval"]
    COL -->|"plain lookup"| LK["nav sessions'<br/>raw turns"]

    DEN --> T{"any turns?"}
    WHOLE --> T
    LK --> T
    T -->|"no"| REF["REFUSE:<br/>'This information is not available'"]
    T -->|"yes"| DISP{"dispatch by route"}

    DISP -->|"datemath"| CDM["_answer_datemath<br/>(extract → compute → copy)"]
    DISP -->|"recency"| CRC["_answer_recency<br/>(extract → sort → pick latest)"]
    DISP -->|"agg / lookup"| CAN["_answer<br/>(scratchpad reason, or direct)"]

    CDM -->|"extraction empty"| CAN
    CRC -->|"extraction empty"| CAN

    CDM --> ANS["candidate answer"]
    CRC --> ANS
    CAN --> ANS
    REF --> ANS

    ANS --> ESC{"is it a refusal?"}
    ESC -->|"no"| FIN["FINAL ANSWER"]
    ESC -->|"yes (PI_RAG)"| RAG["RAG ESCALATION — pi_rag.answer:<br/>decompose → PREMISE GATE →<br/>per-sub-question hybrid retrieval → compose"]
    RAG --> RC{"recovered a<br/>non-refusal?"}
    RC -->|"yes → adopt"| FIN
    RC -->|"no → keep refusal"| FIN
```

---

## 3. Inside a compute layer (datemath & recency share this shape)

This is the core idea — the LLM never does the arithmetic:

```mermaid
flowchart LR
    IN["retrieved turns<br/>(each tagged with its date)"] --> EX["1. EXTRACT — LLM<br/>pull structured facts:<br/>datemath → dated events<br/>recency → dated value history"]
    EX --> CO["2. COMPUTE — Python (exact)<br/>datemath → timeline + differences,<br/>distance from question_date<br/>recency → sort by date, mark LATEST"]
    CO --> CP["3. COPY — LLM<br/>state the computed value;<br/>forbidden to recompute"]
    CP --> OUT["answer<br/>(or 'not available' if no basis)"]
```

---

## Legend / notes

- **Navigation** picks *sessions*. Summary-nav (reader reads the LLM summaries, vectorless) is cheap
  but lossy — a summary drops incidental facts. **Rich-nav** (`PI_RICHINDEX`) embeds a de-disguised
  fact ledger and retrieves over it, so an incidental mention ("out of my league" = a property
  viewing) still opens its session. Answers always come from raw turns, never the ledger.
- **Density assembly** (`PI_DENSITY`) hands the reader a small retrieval-ranked window instead of
  whole sessions, keeping the pivot fact salient (~3.6× fewer reader tokens on compute questions).
  **Route scope** (`PI_DENSITY_SCOPE`) applies the cap only to single-pivot compute (datemath /
  recency); counts and orderings keep whole-session **completeness**, because a capped window drops
  instances a count needs.
- **Evidence union** feeds the completeness path: cosine+BM25 over all turns, so a value/event a
  summary erased still reaches the compute layer.
- **RAG escalation** fires only on a *refusal* — it re-answers the residue via decompose + a strict
  premise gate. This is what keeps fabrication low on answerable questions.
- **Honesty:** overall fabrication is **1.8%** (9/500); on unanswerable questions the system
  correctly refuses **70%** of the time. The residue is false-premise questions where retrieval still
  returns topically-near turns and the compute layer answers instead of refusing.

**Render:** GitHub, VS Code (Mermaid extension), or https://mermaid.live all display the diagrams.
