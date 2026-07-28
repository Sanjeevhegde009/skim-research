# LongMemEval failure walkthrough — the 0.580 baseline that motivated the mechanisms

Six real failures from the **0.580 baseline** full-500 run (config `PI_RAG + HYBRID + INFER`), one
per failure mode, traced to root cause with the actual gold evidence turns. This is the diagnosis
that drove the ladder to **0.766** — each section ends with **→ Now**, stating which mechanism
addressed it and the current per-category score. Four of six were substantially fixed (A–D); one is
partially fixed (E); one is still open (F).

Method: join `results/longmemeval/results_official.json` (answer + sessions opened) with the gold
`answer_session_ids` and the raw evidence turns in `data/longmemeval_s.json`. For each: what the
answer should be, what the pipeline did at each step, the weakness it exposed, and how it stands now.
Config ladder and current numbers: [RESULTS.md](RESULTS.md).

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

**→ Now (largely fixed):** rich-nav (`PI_RICHINDEX`) retrieves over a de-disguised fact ledger that
records incidental instances, broad nav opens every relevant session, and route-scoped completeness
keeps *all* their turns for count questions. Multi-session **0.39 → 0.66**.

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

**→ Now (fixed):** `PI_DATEMATH` — the LLM extracts dated events, **Python** computes the timeline and
differences against the question date, the LLM only copies the result (forbidden to recompute). The
"79-weeks" class of error is gone. Temporal **0.42 → 0.78**.

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

**→ Now (largely fixed):** the rich-nav fact ledger *is* the event-with-date index — a de-disguised
"visited MoMA / the Met" fact is retrievable where the topic summary was not, so both anchors open;
datemath then does the subtraction in code. Folds into the temporal **0.42 → 0.78**.

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

**→ Now (fixed):** `PI_RECENCY` — the reader extracts the attribute's **dated value history**, Python
sorts it and marks the latest, the reader answers the latest by default. "Chicago" (05-24) is correctly
superseded by "suburbs" (05-27). Knowledge-update **0.65 → 0.82**.

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

**→ Now (partially fixed, still the weakest):** rich-nav closes the semantic gap in (1) — the fact
ledger surfaces the language-learning session that summaries missed — lifting preference **0.20 →
0.37**. Weakness (2) stands: there is still no preference *profile* the reader applies to a "recommend
something" ask. This remains the clearest open gap.

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
supported." It's strong on fully-false premises but **blind to partially-false / compound** ones. Needs
per-clause premise checking.

**→ Now (open):** the premise gate is still per-question, not per-clause, so a compound false premise
("tomatoes **and** chili peppers") can still be answered by omission. Overall abstention is 0.70 and
fabrication 1.8%, but this specific class is unaddressed — per-clause premise checking is the next lever.

---

## Synthesis — six failures, and where each stands at 0.766

| # | category | baseline failure (0.580) | root weakness | status now |
|---|---|---|---|---|
| A | multi-session | confident undercount | summaries lose incidental **instances** | **fixed** — rich-nav + scoped completeness (0.39 → 0.66) |
| B | temporal | catastrophic date math | LLM mental math over distractor dates | **fixed** — datemath computes in code (0.42 → 0.78) |
| C | temporal | navigation miss (2 anchors) | incidental **events** not surfaced | **fixed** — rich ledger is the event index (→ 0.78) |
| D | knowledge-update | answered stale value | no supersession / recency | **fixed** — recency value-history (0.65 → 0.82) |
| E | preference | over-refusal | no preference **profile** to apply | **partial** — nav gap closed (0.20 → 0.37); no profile yet |
| F | abstention | compound false-premise | gate checks "an answer," not each clause | **open** — needs per-clause premise checks |

**What changed:** the baseline lost *structure* — incidental instances (A), incidental events (C), fact
supersession (D). The mechanism ladder rebuilt exactly that: a de-disguised fact ledger makes incidental
mentions retrievable (A, C), recency tracks superseded values (D), and compute-in-code replaces the
model's arithmetic (B). LongMemEval **0.580 → 0.766**.

**What's left, precisely:** preference *application* (E — retrieval is fixed, but there is no profile the
reader applies to a "recommend something" ask) and per-clause premise checking (F). Both are
declarative-memory **structure** gaps — a value the memory never represented can't be recovered by better
prompting — so the remaining levers are more memory structure, the same axis the fixes above used.
