# LongMemEval failure walkthrough — why the framework is weak where it's weak

Six real failures from the full-500 official run (0.580), one per failure mode, traced to root cause
with the actual gold evidence turns. Method: join `results/longmemeval/results_official.json`
(answer + sessions opened) with the gold `answer_session_ids` and the raw evidence turns in
`data/longmemeval_s.json`. For each: what the answer should be, what the pipeline did at each step,
and the specific weakness it exposes.

---

## A. multi-session — partial retrieval (confident undercount)
**Q:** "How many items of clothing do I need to pick up or return from a store?" (asked 2023-02-15)
**Gold:** 3 — **Answered:** "Two pairs of boots from Zara (one to pick up, one to return)."
**Gold sessions:** `_1`, `_2`, `_3`. **Opened:** `_1`, `_3` (missed `_2`). RAG did **not** fire.

The three items are: return boots, pick up the exchanged boots, **and pick up the blazer dry-cleaning**.
- `_3`: *"I need to return some boots to Zara… exchanged them for a larger size… haven't had a chance to pick them up."*
- `_1`: *"I just exchanged a pair of boots from Zara on 2/5, and I still need to pick up the new pair."*
- `_2` (**missed**): *"I'll take a break and pick up my **dry cleaning for the navy blue blazer**."*

**What happened:** the navigator opened the two sessions obviously about "Zara boots," counted them
(pick-up + return = 2), and answered **confidently**. It missed `_2`, whose summary is about *blazer care
/ storing winter clothes* — the "pick up dry cleaning" is an incidental aside, not surfaced as a to-do.
Because the answer wasn't a refusal, **RAG never escalated** to look for more.

**Weakness:** flat session summaries don't expose incidental *instances* (a dry-cleaning pickup is an
"item," but the summary is about blazer care). Counting needs **all** instances; the system retrieves a
subset and a **confident wrong count never trips the gate**.

---

## B. temporal — reasoning (had the evidence, math catastrophically wrong)
**Q:** "How many weeks ago did I attend the friends and family sale at Nordstrom?" (asked 2022-12-01)
**Gold:** 2 — **Answered:** "**About 79 weeks ago.**" Opened the gold session (HIT, FULL). RAG recovered.

- gold `_1` (2022-11-18): *"**Yesterday**, I attended a friends and family sale…"* → sale ≈ 2022-11-17.
  2022-12-01 − 2022-11-17 = 14 days = **2 weeks**.

**What happened:** it *had the right session*, but it opened **8 sessions** total (7 distractors with
other dates), and in the answer step it anchored to the wrong date and botched the arithmetic — "79 weeks"
puts it ~1.5 years off (mid-2021). Relative-date resolution ("yesterday" → 11-17) plus cross-date
subtraction, done as **ad-hoc mental math over raw dated text amid distractors**, fell apart.

**Weakness:** no explicit temporal representation. The model does date arithmetic inline over raw turns,
and a context full of distractor dates makes it pick the wrong anchor. This is the textbook case for a
**dates-as-data** layer (cf. Zep's bi-temporal edges), not LLM mental math.

---

## C. temporal — navigation miss (never found either anchor)
**Q:** "How many days passed between my visit to MoMA and the 'Ancient Civilizations' exhibit at the Met?" (asked 2023-02-01)
**Gold:** 7–8 days — **Answered:** "This information is not available." **Opened: nothing** (MISS). RAG fired, did not recover.

- gold `_1` (2023-01-08): *"I just got back from a guided tour at the **Museum of Modern Art**…"*
- gold `_2` (2023-01-15): *"I attended the '**Ancient Civilizations**' exhibit at the Metropolitan…"*
  2023-01-15 − 2023-01-08 = **7 days**.

**What happened:** both museum visits are *incidental* mentions inside sessions whose topics are "modern
art movements" and "mummification / ancient civilizations." The navigator, hunting for "when did I visit
MoMA / the Met exhibit," couldn't identify either session and opened **nothing** → refused. It's also a
**two-anchor** question (needs both dates). At least it refused honestly rather than fabricating.

**Weakness:** same root as A (incidental events lost to topic-summaries), but here it causes a *total*
miss, made worse by needing **two** events located. No event-with-date index to answer "when did I do X."

---

## D. knowledge-update — partial/stale (answered the superseded value)
**Q:** "Where did Rachel move to after her recent relocation?" (asked 2023-06-13)
**Gold:** "the suburbs" — **Answered:** "**Chicago**." **Opened:** `_1` only (missed `_2`). RAG did **not** fire.

- gold `_1` (2023-05-24): *"Rachel… recently moved to a new apartment in the city… **She moved to Chicago.**"*
- gold `_2` (2023-05-27, **missed**): *"Rachel actually just **moved back to the suburbs again**."* ← the update

**What happened:** Rachel moved to Chicago (05-24), then **back to the suburbs** (05-27). The system found
the *first* mention, answered "Chicago" **confidently**, and never looked for a newer version. It has **no
notion that a fact can be superseded** — it retrieves a match and answers.

**Weakness:** **no recency / supersession logic.** This is exactly what Mem0's ADD/UPDATE/DELETE and Zep's
bi-temporal invalidation exist for — they record that "Rachel: Chicago" was *replaced by* "Rachel: suburbs."
Flat session-navigate has no concept of "find the latest." The clearest case for structured update memory.

---

## E. preference — gate over-refusal (couldn't apply a learned preference)
**Q:** "Can you recommend some interesting cultural events happening around me this weekend?" (asked 2023-05-30)
**Rubric:** user prefers events to **practice Spanish/French / language learning**.
**Answered:** "This information is not available." **Opened: nothing** (MISS). RAG fired, did not recover.

- gold `_1` (2023-05-29): *"cultural events… that celebrate language diversity"*, *"language learning apps…
  in French and Spanish"*, *"language exchange opportunities and cultural events… for French and Spanish."*

**What happened:** the query "recommend cultural events this weekend" has **no lexical/semantic bridge** to
a session about *language-learning apps and exchange*. The navigator found nothing → refused. Compounding:
a preference question isn't a fact lookup — the right move is *"find the user's preference (language
practice), then tailor the recommendation."* The framework does **lookup-or-refuse**, so it refuses.

**Weakness:** (1) navigation misses the preference session (semantic gap); (2) more fundamentally, the
system is **fact-retrieval, not preference-application** — it has no user-preference *profile* to apply.
This is why preference is the weakest category (0.20).

---

## F. abstention — fabrication on a *compound* false premise (gate too coarse)
**Q:** "How many plants did I initially plant for tomatoes **and chili peppers**?" (asked 2023-05-30)
**Gold:** not answerable (user mentioned tomatoes, never chili peppers) — **Answered:** "**5 tomato plants.**"
Opened a gold session (HIT). RAG fired, recovered.

- gold `_1` (2023-05-22): *"I planted **5 tomato plants** initially…"*
- gold `_2` (2023-05-29): about **cucumbers** (3 plants) — **not** chili peppers.

**What happened:** the premise is compound — *tomatoes* (true: 5) **and** *chili peppers* (false: never
mentioned). The system found the true half, answered "5 tomato plants," and **silently dropped the
unsupported half**. It didn't flag that chili peppers have no evidence. RAG "recovered" by matching the
tomato part → answered. So it's not a wild fabrication; it's an **answer-by-omission** that implicitly
accepts the false half.

**Weakness:** the gate checks "is there evidence for *an* answer," not "is *every clause* of the premise
supported." It's strong on fully-false premises (abstention 0.867) but **blind to partially-false /
compound** ones. Needs per-clause premise checking.

---

## Synthesis — six failures, six distinct weaknesses

| # | category | failure | root weakness |
|---|---|---|---|
| A | multi-session | partial retrieval, confident undercount | flat summaries lose incidental **instances**; no all-gather for counts |
| B | temporal | catastrophic date math (79 wks) | no temporal data layer; LLM does mental math over distractor dates |
| C | temporal | navigation miss (2 anchors) | incidental **events** not surfaced; no event-with-date index |
| D | knowledge-update | answered stale value | **no supersession/recency**; retrieves a match, never the latest |
| E | preference | over-refusal | no user-**preference profile**; lookup-or-refuse, can't *apply* |
| F | abstention | compound false-premise fabrication | gate checks "an answer," not **every clause** |

**The unifying theme:** four of six (A, C, D, E) fail because the **flat session-summary + navigate design
loses structure** — incidental instances, incidental events, fact supersession, user preferences. The
SOTA systems (Mem0's extracted facts, Zep's temporal graph) *win precisely by extracting these into
structured, update-able, retrievable forms.* B is the answer step doing arithmetic it shouldn't. F is the
gate being too coarse.

**The honest caveat for the "experience / learning memory" direction:** these are **declarative-memory
structure** gaps, not "the agent hasn't learned the right strategy." No amount of lesson-learning recovers
the "suburbs" update (D) or the missing chili-pepper absence (F) — the memory never *represented* them.
So if the goal is to move *these* numbers, the lever is **memory structure** (extract / index / supersede),
the same thing the SOTA did. Experiential/procedural memory ("what strategy works") is a real and more-open
axis — but it's orthogonal to these recall failures and won't lift LongMemEval.
