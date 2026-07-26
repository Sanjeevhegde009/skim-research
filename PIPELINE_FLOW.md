# PageIndex + RAG — Query Logic Flow (0.722 config)

**Config:** `PI_RAG + HYBRID + INFER + NAV_BROAD + REASON + DATEMATH + RECENCY + EVUNION`
**Result:** LongMemEval_S full-500 official **0.722**, reader = gpt-4.1-mini, fabrication 2.6%.

Two models: a **compiler** (gpt-4o-mini) builds the index once; a cheap **reader** (gpt-4.1-mini)
answers every query. The guiding principle: **the LLM extracts, Python computes** — arithmetic,
sorting, and date math are done in code, never in the model's head.

---

## 1. Offline — build once, cached

```mermaid
flowchart LR
    S["Haystack:<br/>~50-80 sessions<br/>(~115K tokens)"] --> IDX["INDEX BUILD (compiler, cached)<br/>one node per session +<br/>LLM table-of-contents summary"]
    S --> EMB["TURN EMBEDDINGS + BM25<br/>(built lazily, cached)"]
    IDX -.feeds.-> NAVI["NAVIGATION<br/>(reads summaries)"]
    EMB -.feeds.-> RETI["RETRIEVAL / union<br/>(reads raw turns)"]
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

    NB --> NAV["_navigate: LLM reads the session<br/>SUMMARIES (vectorless) and picks<br/>the session keys — no similarity math"]
    NP --> NAV
    NAV --> COL["Collect RAW TURNS<br/>of the chosen sessions"]

    COL --> UQ{"compute route?<br/>(datemath / recency / agg)"}
    UQ -->|"yes (EVUNION)"| UNI["EVIDENCE UNION:<br/>nav turns  +  top-12 hybrid retrieval<br/>(cosine + BM25 over ALL turns,<br/>reaches what summaries erased)"]
    UQ -->|"no"| ONLY["nav turns only"]

    UNI --> T{"any turns?"}
    ONLY --> T
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

- **Vectorless navigation** picks *sessions* by having the reader read the LLM-written summaries —
  cheap, but lossy (a summary can drop an incidental fact, making its session unfindable).
- **Evidence union** (EVUNION) patches that lossiness for the compute routes: it also pulls raw
  turns by cosine+BM25, so a value/event the summary erased can still reach the compute layer.
- **RAG escalation** fires only on a *refusal* — it re-answers the residue via decompose + a strict
  premise gate. It is what keeps fabrication low on answerable questions (0.6%).
- **Where the 2.6% fabrication comes from:** the `EVIDENCE UNION → compute` path also runs when
  nav returns NONE. On a *false-premise* question (e.g. "…before my job at Google" when there is no
  Google job), retrieval still returns topically-near turns and the compute layer answers instead of
  refusing — 10 of 30 abstention questions. This is the known honesty tax of the 0.722 config.

**Render:** GitHub, VS Code (Mermaid extension), or https://mermaid.live all display the diagrams.
