"""
Rich navigation index — retrieve-over-ledger session navigation (flag: PI_RICHINDEX).

The 2-4 sentence session summary used for navigation DROPS incidental instances: a property viewing
mentioned inside a home-warranty chat ("that one in Cedar Creek was out of my league") vanishes, so
nav can never open that session and multi-session counts undercount — the "disguised-count wall".

This builds a COMPLETENESS-preserving fact ledger per session: every instance / entity / dated event
/ value, DE-DISGUISED (stated as WHAT IT IS), cited to its turn. Two hard rules learned the hard way:
  1. Used for NAVIGATION ONLY. Answers still come from RAW TURNS, never from the ledger — answering
     from de-disguised text loses accuracy: it invites false-premise lures and is a lossy digest of
     the raw turns (both measured, earlier, as net-negative).
  2. Nav RETRIEVES over the fact vectors; it does NOT dump all ledgers into one bloated ToC prompt.
     Because a de-disguised fact ("viewed a Cedar Creek property") is RETRIEVABLE where the raw turn
     ("out of my league") was not, retrieval over the ledger finds the erased instances cheaply, and
     the query-time nav prompt stays small.

Flag-gated (PI_RICHINDEX) and cached in its OWN dir so the base index cache is never touched.
"""
import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import config
from llm import compiler_call
import pi_rag                       # embeddings (pi_rag._embed is runtime-patched by the runner)

RICH_WORKERS = int(os.environ.get("PI_RICH_WORKERS", "8"))   # parallel session compiles (I/O-bound)

RICH_DIR = config.CACHE_DIR / "rich_index"
# A session enters navigation only if its best de-disguised fact clears this cosine to the question —
# the "nothing relevant here" floor that preserves refuse-on-absence. Tunable: PI_RICH_TAU.
RICH_TAU = float(os.environ.get("PI_RICH_TAU", "0.35"))
_VERSION = "v2-generic"   # part of the cache key — bump when the prompt changes so ledgers recompile

_SYS = (
    "You build a COMPLETE fact ledger of one conversation session, used ONLY to help a search system "
    "decide whether to open this session later. You capture every concrete thing the USER did, "
    "viewed, bought, attended, owns, or decided — especially incidental mentions — and state plainly "
    "WHAT each thing IS. Completeness of the user's own instances matters more than prose.")


def _prompt(key, date, turn_text):
    return f"""Read this session and list the USER's concrete INSTANCES and named ENTITIES — the
things a later search would look for: an event attended, a place visited or viewed, an item bought or
owned, a trip taken, a person named, a decision made, a value/measurement stated, a state that
changed. ONE per line.

RULES:
- CAPTURE INCIDENTAL MENTIONS. Things said only in passing ("by the way...", "that one...", an aside)
  count fully, EVEN IF the session's main topic is something else — these are exactly what a topic
  summary drops, and the whole reason this ledger exists.
- DE-DISGUISE each mention: state its CATEGORY plainly, in words a searcher would use, even when the
  user did not. Generic illustrations of the transformation (NOT from this session):
    "we grabbed dinner at this great Thai place"        ->  "tried a Thai restaurant [restaurant]"
    "my sister finally sent me that vinyl I wanted"     ->  "acquired a vinyl record [item]"
    "the place by the lake was lovely but out of reach" ->  "viewed a home by the lake; did not
        pursue it (too expensive) [property viewing]"
  Name the category (a property viewing, a restaurant tried, a trip taken, an item bought) so the
  searcher can match it, not the user's idiom.
- DO NOT enumerate generic advice, the ASSISTANT's lists of options, or the sub-parts of a topic
  (every provider recommended, every document type, every tip) — those are not the user's instances
  and only add noise. Ledger the user's OWN concrete things and the entities they name.
- Resolve relative dates to absolute against the session date. Cite the turn id. Never invent.

FORMAT, one per line:  - <plain fact naming its category and any date> [dia_id]

SESSION {key} (date {date}):
{turn_text}

Ledger:"""


def rich_compile(turns, key, date, force=False):
    """Compile ONE session's raw turns into a de-disguised fact ledger. Cache keyed by prompt
    VERSION + content (shared sessions compile once; a prompt change recompiles). Returns the
    ledger string."""
    turn_text = "\n".join(
        f"[{t.get('dia_id', '')}] {t['speaker']}: {t['text']}" for t in turns)
    h = hashlib.sha1((_VERSION + turn_text).encode("utf-8")).hexdigest()
    RICH_DIR.mkdir(parents=True, exist_ok=True)
    path = RICH_DIR / f"{h}.txt"
    if path.exists() and not force:
        return path.read_text(encoding="utf-8")
    ledger = compiler_call(_prompt(key, date, turn_text), system=_SYS, temperature=0.0).strip()
    path.write_text(ledger, encoding="utf-8")
    return ledger


def ledger_facts(ledger):
    """Split a ledger into individual fact lines (bullets), stripped of the leading marker."""
    facts = []
    for ln in ledger.splitlines():
        ln = ln.strip().lstrip("-*").strip()
        if len(ln) > 5:
            facts.append(ln)
    return facts


def build_rich(index, force=False):
    """Compile every session's de-disguised ledger, embed the facts, cache the fact store per
    haystack. Returns {facts, src, vecs}. Session ledgers are content-cached (shared sessions
    compile once); the per-haystack store caches the embeddings."""
    sid = index["sample_id"]
    RICH_DIR.mkdir(parents=True, exist_ok=True)
    path = RICH_DIR / f"store_{_VERSION}_{sid}.json"
    if path.exists() and not force:
        return json.loads(path.read_text())
    facts, src = [], []
    nodes = index["nodes"]
    print(f"  [rich] building fact index for {sid} ({len(nodes)} sessions, {RICH_WORKERS}-way "
          f"parallel, one-time, cached)...", flush=True)
    done, lock = [0], threading.Lock()

    def _one(n):                                             # compile ONE session (cached on disk)
        r = ledger_facts(rich_compile(n["turns"], n["key"], n["date"]))
        with lock:
            done[0] += 1
            if done[0] % 10 == 0 or done[0] == len(nodes):
                print(f"  [rich]   compiled {done[0]}/{len(nodes)} sessions", flush=True)
        return n["key"], r

    with ThreadPoolExecutor(max_workers=RICH_WORKERS) as ex:
        for key, ff in ex.map(_one, nodes):                 # ex.map preserves node order
            for f in ff:
                facts.append(f)
                src.append(key)
    vecs = pi_rag._embed(facts) if facts else []
    vecs = [pi_rag._normalize(v) if v else None for v in vecs]
    store = {"facts": facts, "src": src, "vecs": vecs}
    path.write_text(json.dumps(store))
    return store


def navigate_rich(question, index, cap, trace=None):
    """Retrieve-over-ledger navigation: rank sessions by their best de-disguised fact's cosine to
    the question; return up to `cap` DISTINCT sessions whose best fact clears RICH_TAU. Empty when
    nothing is relevant (preserves refuse-on-absence). The ledger lives in the vector store, never
    in a prompt — so query-time nav cost stays bounded regardless of ledger richness."""
    store = build_rich(index)
    qv = pi_rag._embed([question])
    if not qv or not qv[0] or not store["facts"]:
        return []
    qn = pi_rag._normalize(qv[0])
    sims = sorted(
        ((sum(a * b for a, b in zip(qn, v)) if v else -1.0, i)
         for i, v in enumerate(store["vecs"])), reverse=True)
    keys, seen, ev = [], set(), []
    for s, i in sims:
        if s < RICH_TAU or len(keys) >= cap:
            break
        k = store["src"][i]
        if k not in seen:
            keys.append(k)
            seen.add(k)
            ev.append((k, round(s, 3), store["facts"][i][:70]))
    if trace is not None:
        trace["rich_nav"] = ev
        trace["rich_facts"] = len(store["facts"])
    return keys
