"""
HLMA Core — Markdown wiki with Karpathy-style compilation.

Compiler: Frontier model — reads schema.md, creates/updates wiki pages with
          YAML frontmatter and [[wikilinks]], generates prose summary, writes log.
Query:    Any model (SLM or API) — reads summary.md, retrieves pages on demand.
Lint:     Frontier model — post-compilation health check.
"""

import json
import time
import re
import os
import requests
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from datetime import datetime, timedelta

import config

# When on, the query model wraps its Tier-1 decomposition in <thinking>...</thinking>;
# Python captures it into the trace (for debugging/monitoring) and strips it before
# parsing. Off by default — production path is unchanged. Toggle via env var.
DEBUG_REASONING = os.environ.get("HLMA_DEBUG_REASONING", "").lower() in ("1", "true", "yes")


# Shared system prompt for ALL compiler passes — the "constitution" that tells every
# stage what the wiki is, who consumes it downstream, and how to compile for retrieval.
COMPILER_SYSTEM_PROMPT = """You are the COMPILER for a persistent memory wiki built from an ongoing conversation, maintained across sessions.

The wiki has three layers: PAGES (markdown "Entity - Topic" docs, the source of truth, each claim cited with a [turn ID] like [D5:7]); a SUMMARY (prose digest with [→Page Name] pointers, read first on every query); and RAW TURNS (the original lines, cited by ID). Later, a SEPARATE query model — which never saw the conversation — answers questions by: reading the SUMMARY, then opening a PAGE BY ITS NAME, then following [turn IDs] to raw turns. A fact that is captured but not retrievable by that process is a failure.

Therefore, every pass MUST obey:
1. KEEP PAGE NAMES STABLE. Update existing pages under their exact existing names. Never rename a page, never create a variant name for a topic that already has a page.
2. STATE FACTS PLAINLY. Defining attributes must be written as standalone facts a direct question can retrieve — never buried as incidental narrative.
3. KEEP EVERY [turn ID] CITATION. They are the provenance trail; never drop them.
4. RESOLVE DATES TO ABSOLUTE. Never leave "last week"/"recently" — compute the actual date.
5. BE FAITHFUL. Only what the conversation supports. Never invent, never merge distinct entities.

You are not writing for yourself to re-read. You are building a navigable structure for a stranger to search under time pressure."""


# ─────────────────────────────────────────────
# LLM Interfaces
# ─────────────────────────────────────────────

def _get_api_key(env_var):
    key = os.environ.get(env_var, "")
    if not key:
        print(f"  [ERROR] Set {env_var} environment variable")
    return key


def _api_call_with_retry(fn, max_retries=5):
    """Retry an API call with exponential backoff on 429 / transient errors."""
    delay = 10
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            msg = str(e)
            is_431        = "431" in msg or "Request Header Fields Too Large" in msg
            is_rate_limit = "429" in msg or "Too Many Requests" in msg
            is_transient  = "500" in msg or "502" in msg or "503" in msg or "timeout" in msg.lower()
            # Connection-level failures (network blip, DNS hiccup, server hangup)
            # are as transient as a 500 — a dropped call here silently loses facts.
            is_network    = isinstance(e, (ConnectionError, OSError)) or any(
                s in msg for s in ("Connection aborted", "RemoteDisconnected",
                                   "NameResolutionError", "getaddrinfo",
                                   "Connection refused", "Connection reset",
                                   "ConnectionError", "Max retries exceeded"))
            if is_431 or is_rate_limit or is_transient or is_network:
                body = ""
                if hasattr(e, "response") and e.response is not None:
                    try:
                        body = e.response.json().get("error", {}).get("message", "")
                    except Exception:
                        body = e.response.text[:400]
                print(f"  [RATE LIMIT] {body or msg}")
                if attempt < max_retries - 1:
                    # 431 is edge flakiness, not rate limiting — a short fixed wait
                    # suffices; exponential backoff just stalls compilation.
                    wait = 2 if is_431 else delay * (2 ** attempt)
                    print(f"  [RETRY {attempt+1}/{max_retries-1}] waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise
            else:
                raise


def _call_anthropic(prompt, system, temperature, model, api_key_env):
    key = _get_api_key(api_key_env)
    if not key: return ""
    messages = [{"role": "user", "content": prompt}]
    body = {"model": model, "max_tokens": 4096, "temperature": temperature, "messages": messages}
    if system: body["system"] = system
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        json=body, timeout=300)
    resp.raise_for_status()
    return resp.json()["content"][0]["text"].strip()


def _call_openai_compat(prompt, system, temperature, model, api_key_env, provider, base_url=""):
    key = _get_api_key(api_key_env)
    if not key: return ""
    if provider == "openai":
        url = "https://api.openai.com/v1/chat/completions"
    else:
        url = base_url
    if not url:
        print("  [ERROR] Set base URL"); return ""
    messages = []
    if system: messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    body = {"model": model, "temperature": temperature, "max_tokens": 4096, "messages": messages}
    resp = requests.post(url, headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json"}, json=body, timeout=300)
    if resp.status_code == 431:
        # OpenAI's edge sporadically mislabels requests as "headers too large".
        # Log what was actually sent so the real trigger is identifiable.
        print(f"  [431 DIAG] body_bytes={len(json.dumps(body))} "
              f"key_len={len(key)} model={model} "
              f"prompt_chars={sum(len(m['content']) for m in messages)}")
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# Count of compiler calls that hard-failed (exhausted retries). evaluate.py checks
# this around compilation: any failure means the wiki has holes and must NOT be
# cached as complete.
COMPILER_FAILURES = 0


def compiler_call(prompt: str, system: str = "", temperature: float = 0.0) -> str:
    """Call compiler model. Temperature defaults to 0.0 for determinism."""
    global COMPILER_FAILURES
    try:
        if config.COMPILER_PROVIDER == "anthropic":
            return _api_call_with_retry(lambda: _call_anthropic(
                prompt, system, temperature,
                config.COMPILER_MODEL, config.COMPILER_API_KEY_ENV))
        else:
            return _api_call_with_retry(lambda: _call_openai_compat(
                prompt, system, temperature,
                config.COMPILER_MODEL, config.COMPILER_API_KEY_ENV,
                config.COMPILER_PROVIDER, config.COMPILER_BASE_URL))
    except Exception as e:
        COMPILER_FAILURES += 1
        print(f"  [COMPILER ERROR] {e}")
        return ""


def query_call(messages: list, temperature: float = 0.1) -> str:
    """Call query model — Ollama or API. Retries on 429 / transient errors."""
    def _do_call():
        if config.QUERY_PROVIDER == "ollama":
            resp = requests.post(
                config.OLLAMA_URL,
                json={"model": config.QUERY_MODEL, "messages": messages,
                      "stream": False,
                      "think": False,
                      "options": {"temperature": temperature}},
                timeout=600)
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()
        elif config.QUERY_PROVIDER == "anthropic":
            key = _get_api_key(config.QUERY_API_KEY_ENV)
            if not key: return ""
            system = ""
            user_content = ""
            for m in messages:
                if m["role"] == "system": system = m["content"]
                else: user_content += m["content"] + "\n"
            body = {"model": config.QUERY_MODEL, "max_tokens": 1024,
                    "temperature": temperature,
                    "messages": [{"role": "user", "content": user_content.strip()}]}
            if system: body["system"] = system
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                         "content-type": "application/json"},
                json=body, timeout=120)
            resp.raise_for_status()
            return resp.json()["content"][0]["text"].strip()
        else:
            key = _get_api_key(config.QUERY_API_KEY_ENV)
            if not key: return ""
            url = "https://api.openai.com/v1/chat/completions" if config.QUERY_PROVIDER == "openai" else config.QUERY_BASE_URL
            body = {"model": config.QUERY_MODEL, "temperature": temperature,
                    "max_tokens": 1024, "messages": messages}
            resp = requests.post(url, headers={"Authorization": f"Bearer {key}",
                                 "Content-Type": "application/json"}, json=body, timeout=120)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
    try:
        return _api_call_with_retry(_do_call)
    except Exception as e:
        print(f"  [QUERY ERROR] {e}")
        return ""


def estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)


_NOT_AVAILABLE = "This information is not available."

def _extract_answer(question: str, answer: str) -> str:
    """Post-process: strip filler to the minimum phrase that answers the question.

    Only fires for long answers (>8 words). Skips 'not available', short answers,
    and anything starting with 'Likely' (already handled by inference formatting).
    Uses the query model — cheap, fast, no extra compiler cost.
    """
    if not answer or answer == _NOT_AVAILABLE:
        return answer
    if answer.lower().startswith("likely"):
        return answer
    if len(answer.split()) <= 8:
        return answer
    sys = (
        "Extract only the key answer phrase from the given answer. "
        "Output 3-8 words maximum — a name, date, list, or short phrase. "
        "No explanation. No full sentence. No leading 'The answer is'."
    )
    usr = f"Question: {question}\nAnswer: {answer}\n\nExtracted key phrase:"
    extracted = query_call([{"role": "system", "content": sys},
                            {"role": "user", "content": usr}], temperature=0.0)
    # Sanity check: reject if extraction looks wrong (empty or longer than original)
    if extracted and 0 < len(extracted.split()) <= 12 and extracted != answer:
        return extracted.strip()
    return answer


# ─────────────────────────────────────────────
# Wiki Storage — Markdown files with frontmatter
# ─────────────────────────────────────────────

class WikiStorage:
    def __init__(self, wiki_dir: str = "wiki"):
        self.wiki_dir = Path(wiki_dir)
        self.wiki_dir.mkdir(parents=True, exist_ok=True)

    def _page_path(self, name: str) -> Path:
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', name)
        return self.wiki_dir / f"{safe_name}.md"

    def write_page(self, name: str, content: str):
        self._page_path(name).write_text(content, encoding="utf-8")

    def delete_page(self, name: str):
        path = self._page_path(name)
        if path.exists():
            path.unlink()
            return
        for f in self.wiki_dir.glob("*.md"):
            if f.stem.lower() == name.lower():
                f.unlink()
                return

    def read_page(self, name: str) -> Optional[str]:
        path = self._page_path(name)
        if path.exists():
            return path.read_text(encoding="utf-8")
        for f in self.wiki_dir.glob("*.md"):
            if f.stem.lower() == name.lower():
                return f.read_text(encoding="utf-8")
        return None

    def read_page_content(self, name: str) -> Optional[str]:
        """Read page content without YAML frontmatter."""
        raw = self.read_page(name)
        if not raw: return None
        if raw.startswith("---"):
            parts = raw.split("---", 2)
            if len(parts) >= 3:
                return parts[2].strip()
        return raw

    def list_pages(self) -> list[str]:
        return sorted([f.stem for f in self.wiki_dir.glob("*.md")
                       if f.stem not in ("summary", "schema", "log")])

    def get_all_pages(self) -> dict[str, str]:
        pages = {}
        for f in sorted(self.wiki_dir.glob("*.md")):
            if f.stem not in ("summary", "schema", "log"):
                pages[f.stem] = f.read_text(encoding="utf-8")
        return pages

    def find_pages(self, terms: list[str]) -> dict[str, str]:
        results = {}
        for name, content in self.get_all_pages().items():
            text = (name + " " + content).lower()
            if any(t.lower() in text for t in terms):
                results[name] = content
        return results

    def get_summary(self) -> str:
        path = self.wiki_dir / "summary.md"
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def write_summary(self, content: str):
        (self.wiki_dir / "summary.md").write_text(content, encoding="utf-8")

    def write_catalog(self, catalog: dict):
        """Write page-name → one-line description index for query-time routing."""
        (self.wiki_dir / "_catalog.json").write_text(
            json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")

    def get_catalog(self) -> dict:
        """Read the annotated catalog. Empty dict if not yet generated."""
        path = self.wiki_dir / "_catalog.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def page_count(self) -> int:
        return len(self.list_pages())

    def total_tokens(self) -> int:
        return sum(estimate_tokens(c) for c in self.get_all_pages().values())

    def get_wikilinks(self, content: str) -> list[str]:
        return re.findall(r'\[\[([^\]]+)\]\]', content)

    def reset(self):
        for f in self.wiki_dir.glob("*.md"):
            if f.stem != "schema":  # preserve schema
                f.unlink()
        for fname in ("_turns.json", "_catalog.json"):
            p = self.wiki_dir / fname
            if p.exists():
                p.unlink()

    # --- Log ---
    def append_log(self, entry: str):
        log_path = self.wiki_dir / "log.md"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        line = f"## [{timestamp}] {entry}\n"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line)

    def get_log(self) -> str:
        path = self.wiki_dir / "log.md"
        return path.read_text(encoding="utf-8") if path.exists() else ""

    # --- Schema ---
    def get_schema(self) -> str:
        """Read schema.md from wiki dir or project root."""
        wiki_schema = self.wiki_dir / "schema.md"
        if wiki_schema.exists():
            return wiki_schema.read_text(encoding="utf-8")
        root_schema = Path("schema.md")
        if root_schema.exists():
            return root_schema.read_text(encoding="utf-8")
        return ""

    # --- Raw turns (Tier 3 provenance) ---
    def store_turns(self, turns: list[dict]):
        """Append raw turns to the source store, indexed by dia_id."""
        store_path = self.wiki_dir / "_turns.json"
        existing = {}
        if store_path.exists():
            existing = json.loads(store_path.read_text(encoding="utf-8"))
        for t in turns:
            tid = t.get("dia_id", "")
            if tid:
                entry = {"speaker": t["speaker"], "text": t["text"]}
                # Session timestamp is provenance too: without it, relative wording
                # in raw turns ("last week") is unresolvable at query time.
                if t.get("date_time"):
                    entry["date"] = t["date_time"]
                existing[tid] = entry
        store_path.write_text(json.dumps(existing), encoding="utf-8")

    def get_turns(self, turn_ids: list[str]) -> list[dict]:
        """Retrieve raw turns by their dia_ids."""
        store_path = self.wiki_dir / "_turns.json"
        if not store_path.exists():
            return []
        store = json.loads(store_path.read_text(encoding="utf-8"))
        result = []
        for tid in turn_ids:
            if tid in store:
                result.append({"dia_id": tid, **store[tid]})
        return result

    def expand_with_neighbors(self, turns: list[dict], window: int = 1) -> list[dict]:
        """Add neighboring turns (±window) to each turn. Dates often sit adjacent
        to the event they describe — 'I went to Paris' / 'that was in January'."""
        store_path = self.wiki_dir / "_turns.json"
        if not store_path.exists():
            return turns
        store = json.loads(store_path.read_text(encoding="utf-8"))

        wanted_ids = []
        for t in turns:
            tid = t["dia_id"]
            wanted_ids.append(tid)
            # Parse Dx:y format
            m = re.match(r'(D\d+):(\d+)', tid)
            if m:
                session, idx = m.group(1), int(m.group(2))
                for offset in range(-window, window + 1):
                    if offset == 0:
                        continue
                    neighbor = f"{session}:{idx + offset}"
                    if neighbor in store:
                        wanted_ids.append(neighbor)

        # Dedupe preserving order, then sort by session+index for readability
        seen = set()
        ordered = []
        for tid in wanted_ids:
            if tid not in seen and tid in store:
                seen.add(tid)
                ordered.append(tid)

        def sort_key(tid):
            m = re.match(r'D(\d+):(\d+)', tid)
            return (int(m.group(1)), int(m.group(2))) if m else (0, 0)
        ordered.sort(key=sort_key)

        return [{"dia_id": tid, **store[tid]} for tid in ordered]

    def search_turns(self, keywords: list[str], max_results: int = 8) -> list[dict]:
        """Keyword search across all raw turns. Keywords are pre-stemmed by the caller;
        substring matching handles inflected forms ('camp' matches 'camping'/'camped')."""
        store_path = self.wiki_dir / "_turns.json"
        if not store_path.exists():
            return []
        store = json.loads(store_path.read_text(encoding="utf-8"))
        scored = []
        for tid, t in store.items():
            text_lower = t["text"].lower()
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scored.append((score, tid, t))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [{"dia_id": tid, **t} for _, tid, t in scored[:max_results]]


# ─────────────────────────────────────────────
# Compiler — reads schema, writes wiki pages with frontmatter
# ─────────────────────────────────────────────

class Compiler:
    def __init__(self, wiki: WikiStorage):
        self.wiki = wiki

    # ── ATOMIC INTEGRATION HELPERS ───────────────────────────────────────────

    def _split_facts(self, facts_raw: str) -> list[str]:
        """Split a numbered prose fact list into individual fact strings."""
        facts, current = [], []
        for line in facts_raw.strip().split('\n'):
            s = line.strip()
            if not s:
                continue
            if re.match(r'^\d+[.)]\s+', s):
                if current:
                    facts.append(' '.join(current))
                current = [s]
            elif current:
                current.append(s)
        if current:
            facts.append(' '.join(current))
        return [f for f in facts if f.strip()]

    def _fact_entity(self, fact: str, known_entities: list[str]) -> str:
        """Return primary entity name from 'N. EntityName: ...' formatted fact."""
        m = re.match(r'^\d+[.)]\s+([^:]{2,40}):', fact)
        if m:
            candidate = m.group(1).strip()
            # Reject literal placeholders the Understand pass sometimes emits
            if candidate.lower() in ("entity name", "entity", "name", "person"):
                pass
            else:
                for ent in known_entities:
                    if ent.lower() == candidate.lower():
                        return ent
                for ent in known_entities:
                    if candidate.lower().startswith(ent.lower()) or ent.lower().startswith(candidate.lower()):
                        return ent
                if len(candidate.split()) <= 3:
                    return candidate
        # Fallback: first known entity name found in fact text
        fact_lower = fact.lower()
        for ent in known_entities:
            if re.search(r'\b' + re.escape(ent.lower()) + r'\b', fact_lower):
                return ent
        return "General"

    def _page_excerpts(self, page_names: list[str]) -> str:
        """Return compact name+excerpt+word-count lines for routing context."""
        lines = []
        for name in page_names:
            content = self.wiki.read_page(name) or ""
            if content.strip().startswith("---"):
                parts = content.split("---", 2)
                content = parts[2].strip() if len(parts) >= 3 else content
            words = len(content.split())
            body = ' '.join(l.strip() for l in content.split('\n') if l.strip())[:250]
            size_tag = f" [{words}w]" if words > 200 else ""
            lines.append(f"  [{name}{size_tag}]: {body}")
        return '\n'.join(lines)

    _FACT_TAG_RE = re.compile(
        r'^\d+[.)]\s+([^|]{2,40}?)\s*\|\s*([^:]{2,30}?):', re.IGNORECASE)
    _TAG_PLACEHOLDERS = frozenset({"entity name", "entity", "name", "person"})

    def _parse_routing_from_facts(self, facts: list[str]) -> dict[int, str]:
        """Pure-Python routing: parse 'N. Entity | Topic: ...' → {idx: page_name}.
        Deterministic — no LLM. Returns only facts that matched the format."""
        routing = {}
        for i, fact in enumerate(facts):
            m = self._FACT_TAG_RE.match(fact)
            if not m:
                continue
            entity = m.group(1).strip()
            topic  = m.group(2).strip().title()   # normalise to Title Case
            if entity.lower() in self._TAG_PLACEHOLDERS:
                continue
            routing[i] = f"{entity} - {topic}"
        return routing

    def _route_all_facts(self, facts: list[str],
                          page_sizes: dict[str, int]) -> dict[int, str]:
        """Pass 2A: Route ALL facts in one LLM call. Returns {0-based idx: page_name}."""
        facts_text = "\n".join(facts)

        if page_sizes:
            pages_list = "\n".join(
                f"  - {name}" + (f" [{w}w — large]" if w > 150 else "")
                for name, w in sorted(page_sizes.items())
            )
            pages_section = f"EXISTING PAGES:\n{pages_list}"
        else:
            pages_section = "EXISTING PAGES: (none — first session)"

        prompt = (
            f"Route each numbered fact to exactly ONE wiki page.\n\n"
            f"{pages_section}\n\n"
            f"FACTS:\n{facts_text}\n\n"
            f"RULES:\n"
            f"- Each fact → ONE page. Never assign the same fact to two pages.\n"
            f"- Strongly prefer EXISTING pages. Only create NEW when truly necessary.\n"
            f"- NEW pages must use broad topics: 'Identity', 'Career', 'Family', 'Art', 'Health'.\n"
            f"  Never use narrow event names like 'Café Visit on Sep 9' or 'Concert'.\n"
            f"- Pages marked [large] are big — route to a more specific existing page\n"
            f"  or create NEW only if this fact is a genuinely distinct sub-topic.\n"
            f"- Near-synonyms are ONE page: 'Family'/'Parenting'/'Children'/'Home Life' → pick one.\n"
            f"- Cross-entity fact ('A said she admires B') → route to the SUBJECT\n"
            f"  (the person whose feeling/action it is — here, A), NOT the object.\n"
            f"- NEW format: 'NEW: Entity Name - Topic'\n\n"
            f"Output exactly one line per fact, nothing else:\n"
            f"N → Page Name\n\n"
            f"Routes:"
        )

        result = compiler_call(prompt, temperature=0.0)
        routing: dict[int, str] = {}

        if result:
            for line in result.strip().split('\n'):
                line = line.strip()
                if not line:
                    continue
                m = re.match(r'^(\d+)\s*[-–→>]+\s*(.+)$', line)
                if not m:
                    continue
                idx = int(m.group(1)) - 1   # 1-based → 0-based
                target = m.group(2).strip()
                if target.lower().startswith("new:"):
                    target = target[4:].strip()
                if target and target.lower() not in ("entity name", "entity", "name"):
                    routing[idx] = target

        # Fallback: unrouted facts → entity Identity page
        known_entities = list({n.split(' - ')[0].strip()
                                for n in page_sizes if ' - ' in n})
        for i, fact in enumerate(facts):
            if i not in routing:
                entity = self._fact_entity(fact, known_entities) or "General"
                routing[i] = f"{entity} - Identity"

        return routing

    def _integrate_page_facts(self, page_name: str, current_content: str,
                               facts: list[str], session_label: str) -> str:
        """Pass 2B: Integrate all facts destined for one page in a single call."""
        # Strip frontmatter — LLM only needs the prose, not YAML metadata
        if current_content.strip().startswith("---"):
            parts = current_content.split("---", 2)
            current_content = parts[2].strip() if len(parts) >= 3 else current_content

        is_new = not current_content.strip()
        facts_text = "\n".join(facts)

        if is_new:
            prompt = (
                f"Create wiki page '{page_name}'.\n\n"
                f"SESSION: {session_label}\n\n"
                f"FACTS:\n{facts_text}\n\n"
                f"Write clear prose with [turn ID] citations inline. "
                f"Exact proper nouns always (country names, pet names, places — never paraphrase). "
                f"No 'currently'/'now' — timeless form or absolute dates. "
                f"No frontmatter, no headers.\n\nContent:"
            )
        else:
            prompt = (
                f"Update wiki page '{page_name}'.\n\n"
                f"SESSION: {session_label}\n\n"
                f"CURRENT CONTENT:\n{current_content}\n\n"
                f"NEW FACTS:\n{facts_text}\n\n"
                f"RULES:\n"
                f"- Include each [turn ID] citation inline\n"
                f"- If a fact SUPERSEDES existing content, REPLACE the old statement\n"
                f"- If a fact DUPLICATES existing content exactly, skip it\n"
                f"- Keep ALL existing [turn ID] citations — never drop any\n"
                f"- Exact proper nouns always (never replace 'Sweden' with 'her home country')\n"
                f"- No 'currently'/'now' — timeless form or absolute dates\n\n"
                f"Output ONLY the complete updated prose:"
            )

        result = compiler_call(prompt, system=COMPILER_SYSTEM_PROMPT, temperature=0.0)
        if not result or len(result.strip()) < 10:
            return current_content
        r = result.strip()
        if r.startswith("---"):
            parts = r.split("---", 2)
            r = parts[2].strip() if len(parts) >= 3 else r
        return r

    def _two_pass_integrate(self, facts_raw: str, schema: str,
                             session_label: str) -> tuple[int, int, dict]:
        """Two-pass integration: one route call then one integrate call per touched page.
        Eliminates per-fact duplication and gives the router global fact visibility."""
        facts = self._split_facts(facts_raw)
        if not facts:
            return 0, 0, {}

        # Snapshot page sizes for routing context
        all_pages = self.wiki.get_all_pages()
        page_sizes = {name: len(content.split()) for name, content in all_pages.items()}

        # Pass 2A: route facts — Python-first, LLM only for untagged fallback
        routing = self._parse_routing_from_facts(facts)
        unparsed = [i for i in range(len(facts)) if i not in routing]
        if unparsed:
            llm_routing = self._route_all_facts(
                [facts[i] for i in unparsed], page_sizes)
            for local_idx, target in llm_routing.items():
                routing[unparsed[local_idx]] = target

        # Apply canonical name enforcement on every routed target
        current_names = self.wiki.list_pages()
        for idx in list(routing.keys()):
            target = routing[idx]
            canonical = self._canonical_name(target, current_names)
            if canonical and canonical != target:
                routing[idx] = canonical
                self._n_enforced = getattr(self, '_n_enforced', 0) + 1

        # Group facts by target page
        page_facts: dict[str, list[str]] = {}
        for idx, target in routing.items():
            if 0 <= idx < len(facts):
                page_facts.setdefault(target, []).append(facts[idx])

        print(f"      Two-pass: {len(facts)} facts → {len(page_facts)} pages")

        created = updated = 0
        touched: dict = {}

        # Pass 2B: integrate per page → one LLM call each
        for page_name, page_fact_list in page_facts.items():
            current = self.wiki.read_page(page_name) or ""
            new_content = self._integrate_page_facts(
                page_name, current, page_fact_list, session_label)
            if new_content.strip() and new_content.strip() != current.strip():
                self.wiki.write_page(page_name, new_content)
                touched[page_name] = ""
                if current.strip():
                    updated += 1
                else:
                    created += 1

        return created, updated, touched

    def compile_session(self, turns: list[dict], session_label: str = "") -> dict:
        if not turns:
            return {"pages_created": 0, "pages_updated": 0, "facts_extracted": 0}

        # Store raw turns for Tier 3 provenance
        self.wiki.store_turns(turns)

        # Snapshot the page set BEFORE this session, so we can report the TRUE structural
        # delta across the whole pipeline (integrate + split + consolidate + name-enforce),
        # not just the integrate-stage count which misses split/merge/enforce effects.
        pages_before = set(self.wiki.list_pages())
        # Per-session structural-op tallies (incremented by the passes below)
        self._n_enforced = 0
        self._n_split = 0
        self._n_consolidated = 0

        turn_text = "\n".join(
            f"[{t.get('dia_id','')}] {t['speaker']}: {t['text']}" for t in turns)

        # Pre-pass infrastructure exists in _pre_extract_anchors — not yet injected
        # into the prompt because it caused wiki fragmentation. Reserved for future use.
        anchors = self._pre_extract_anchors(turns, session_label)
        n_dates = len(anchors.get("dates", []))

        # ── PASS 1: UNDERSTAND (critical reading + date resolution) ──
        # Design choice: this pass THINKS about the session, it doesn't transcribe.
        # Critically: dates are resolved HERE, against the session timestamp, because
        # this is the only point where the session date is in context. Relative
        # references ("last week") become absolute dates now or never.
        understand_prompt = f"""Read this conversation session and build a clear understanding of it.

This session took place on: {session_label}

Produce a thorough, critical account of what this session establishes:
- What happened, what was decided, and what each participant revealed about themselves
- Every concrete fact: names, places, dates, numbers, decisions, preferences, plans, feelings
- PERSISTENT FACTS are especially important — capture anything that remains true beyond
  this session: a person's background, expertise, ongoing projects, key relationships,
  current situation, plans, preferences, and any status that could answer a future question.
  These are easy to miss but critical for long-term memory. Never omit them.
  IDENTITY CHECKLIST — for each person mentioned, capture whenever revealed:
    origin/home country (write the EXACT name — never paraphrase to "her home country"
    or "their homeland"; if the conversation states "Sweden", write "Sweden"),
    relationship/marital status, marriage duration,
    number of children (state as "X has N children"), pets (names and species),
    childhood favorites (books, food),
    hobbies and where they practise them, occupation and career stage.
- VERBATIM LISTS: For every enumerable fact — locations visited, items collected, activities
  practised, hobbies, places, names in a series — copy the EXACT words from the conversation.
  Never paraphrase or generalise. If the conversation says "she camps at the beach and forest",
  write "beach and forest", not "coastal and wooded areas" or "outdoor locations". If it says
  "Jane Austen, Mark Twain", write those names, not "classic authors".
- RESOLVE ALL DATES to absolute form. The session date is given above. If someone says
  "yesterday", "last week", "a few months ago", "next month", compute the actual calendar
  date/period and state it explicitly. Never leave a date relative.
- DATED EVENTS: For every specific event, trip, visit, activity, or milestone in this session,
  produce a compact entry: "[Who] [did what] on [absolute date]". Include even brief mentions.
  Example: "Melanie camping trip mountains 19-25 June 2023 [D4:6]". Missing a dated event is
  a critical failure — err on the side of capturing too many, not too few.
- Note how facts connect (cause, sequence, contrast) — understanding, not just a list.

Each turn is labeled with an ID like [D5:3]. Cite the turn ID(s) after each fact you record,
e.g. "Jon lost his banking job on 19 January 2023 [D5:3]".

SESSION:
{turn_text}

Write your understanding as a numbered list of resolved facts. Format each fact as:
  "N. [Name] | [Topic]: [fact] [turn ID]"
[Name]  = the PRIMARY SUBJECT (the person the fact is centrally about, not just mentioned)
[Topic] = a SHORT (1-3 words, Title Case) category describing WHAT the person IS or DOES —
          the subject matter, not their emotional state or attitude about it.
          Use BROAD, STABLE real-world categories. Be consistent across all facts.

          Typical categories for personal conversations:
            Identity     — who they are: origin, background, transition, keepsakes, pets
            Career       — job, education, professional aspirations, workplace
            Family       — partner, children, parenting, home life, adoption plans
            Health       — physical/mental health, therapy, self-care routines
            Hobbies      — arts, sports, music, crafts, pottery, reading
            Relationships — friendships, social bonds, mentorship, close connections
            Community    — activism, volunteering, events, group memberships, advocacy
          For other domains choose equivalents (e.g. Architecture, Projects, Tools).

          CRITICAL — NEVER use these as topics (they are attitudes, not subjects):
            Feelings, Emotions, Goals, Plans, Motivation, Values, Support System,
            Sentimental Items, Emotional Well-Being, Thoughts, Reflections, Mindset

          Examples of correct topic assignment:
            "feels hopeful about adoption" → Family  (adoption IS the subject)
            "treasures a bowl from her 18th birthday" → Identity  (keepsake = who she is)
            "plans to study counseling" → Career  (career aspiration)
            "attended LGBTQ pride parade" → Community  (civic participation)
Examples:
  "1. Caroline | Identity: is a transgender woman from Sweden, started transitioning June 2020 [D3:1]"
  "2. Caroline | Community: attended an LGBTQ support group on 7 May 2023 [D1:3]"
  "3. Melanie | Family: went camping in the mountains 20-26 June 2023 [D4:6]"
  "4. Melanie | Hobbies: signed up for a pottery class on 2 July 2023 [D5:4]"
Be complete and critical — capture everything that matters, with dates made absolute:"""

        facts_raw = compiler_call(understand_prompt, system=COMPILER_SYSTEM_PROMPT)
        if not facts_raw:
            return {"pages_created": 0, "pages_updated": 0, "facts_extracted": 0}

        structured_facts = self._parse_fact_json(facts_raw)  # will be empty for prose; used by verify
        fact_count = len([l for l in facts_raw.strip().split("\n") if l.strip()])
        print(f"      Understood {fact_count} facts")

        # ── PASS 2: TWO-PASS INTEGRATE (route all → integrate per page) ──
        schema = self.wiki.get_schema()
        created, updated, pages = self._two_pass_integrate(facts_raw, schema, session_label)

        # ── POST-PASS: verify resolved dates survived into the wiki ──
        # Deterministic — no LLM. Appends missing resolved dates directly.
        if n_dates:
            self._verify_anchors(anchors)

        # ── CRITIC PASS: the autonomous stand-in for Karpathy's human reviewer ──
        # Reviews each page touched this session against the source turns and schema,
        # using reasoning (not regex) to fix what a careful human would catch:
        # unresolved dates, contradictions, page corruption, claims absent from source.
        self._critique(pages, turns, session_label)

        # ── SPLIT PASS: mechanical granularity control ──
        # Page granularity oscillates run-to-run (7 mega-pages vs 38 tiny ones).
        # A collapsed wiki is just a wiki of oversized pages, so splitting oversized
        # pages by sub-topic fixes BOTH failure modes: it breaks up attractor
        # mega-pages and keeps every page independently retrievable.
        self._split_oversized()

        # ── CONSOLIDATION PASS: merge look-alike same-entity+topic fragments ──
        # Split breaks up oversized pages; consolidation merges the fragmentation that
        # causes routing ambiguity and forced multi-page assembly. Guarded against
        # over-merge (same-entity+topic only, provenance-preserving, page-count floor).
        self._consolidate_fragments()

        # Inject wikilinks (capped, relevance-based) — see _inject_wikilinks
        self._inject_wikilinks()

        # Ensure every page has YAML frontmatter
        self._ensure_frontmatter(session_label)

        # Log — report the TRUE structural delta across the whole pipeline.
        pages_after = set(self.wiki.list_pages())
        net_new = len(pages_after - pages_before)
        net_removed = len(pages_before - pages_after)
        page_list = ", ".join(sorted(pages_after))
        self.wiki.append_log(
            f"compile | {session_label} | {len(turns)} turns | {fact_count} facts | "
            f"integrate(+{created} new, ~{updated} touched) | "
            f"enforce={self._n_enforced} split={self._n_split} consolidate={self._n_consolidated} | "
            f"net pages: +{net_new}/-{net_removed} → {len(pages_after)} total")

        return {"pages_created": net_new, "pages_updated": updated,
                "facts_extracted": fact_count, "total_pages": len(pages_after),
                "enforced": self._n_enforced, "split": self._n_split,
                "consolidated": self._n_consolidated, "net_removed": net_removed}

    def _canonical_name(self, emitted: str, existing_names: list):
        """Return an existing page name that `emitted` is a near-duplicate of, else None.
        Match rule: SAME entity (text before ' - ') AND strong topic overlap. Conservative
        — won't merge different topics ('Career' vs 'Adoption'), but catches variants
        ('Adoption Journey'/'Adoption Research', 'Career Aspirations'/'Education and Career
        Aspirations'). Deterministic; no model involved."""
        def split(n):
            parts = n.split(" - ", 1)
            if len(parts) != 2:
                return None, None
            return parts[0].strip().lower(), parts[1].strip().lower()

        STOP = {"and", "the", "of", "a", "an", "in", "on", "for", "to", "with", "&"}
        def topic_tokens(topic):
            return {w for w in re.findall(r'[a-z0-9]+', topic) if w not in STOP}

        e_entity, e_topic = split(emitted)
        if not e_entity:
            return None
        e_tok = topic_tokens(e_topic)
        if not e_tok:
            return None

        best, best_score = None, 0.0
        for cand in existing_names:
            c_entity, c_topic = split(cand)
            if c_entity != e_entity:
                continue
            c_tok = topic_tokens(c_topic)
            if not c_tok:
                continue
            inter = len(e_tok & c_tok)
            overlap = inter / min(len(e_tok), len(c_tok))
            if e_topic in c_topic or c_topic in e_topic:
                overlap = max(overlap, 0.9)
            if inter >= 1 and overlap >= 0.5 and overlap > best_score:
                best, best_score = cand, overlap
        return best

    def _parse_fact_json(self, raw: str) -> list[dict]:
        """Parse the Understand pass JSON output into a list of fact dicts.
        Handles model noise (prose wrapper, markdown fences) with graceful fallback."""
        text = raw.strip()
        # Strip markdown code fences if present
        text = re.sub(r'^```[a-zA-Z]*\s*\n?', '', text)
        text = re.sub(r'\n?```\s*$', '', text)
        text = text.strip()
        # Extract JSON array if surrounded by prose
        m = re.search(r'\[.*\]', text, re.DOTALL)
        if m:
            text = m.group(0)
        try:
            facts = json.loads(text)
            if isinstance(facts, list):
                # Keep only dicts with at least entity + verbatim
                return [f for f in facts
                        if isinstance(f, dict) and f.get("entity") and f.get("verbatim")]
        except (json.JSONDecodeError, ValueError):
            pass
        return []  # fallback: caller uses raw prose

    def _format_facts_for_integrate(self, structured: list[dict], prose_fallback: str) -> str:
        """Format facts for the Integrate prompt.
        Structured facts get each verbatim quote on its own labeled line so the model
        cannot silently ignore it. Falls back to the raw prose string if no structured facts."""
        if not structured:
            return prose_fallback
        lines = []
        for i, f in enumerate(structured, 1):
            entity  = f.get("entity", "?")
            topic   = f.get("topic", "?")
            verbatim = f.get("verbatim", "")
            turn_id  = f.get("turn_id", "")
            date     = f.get("date", "")
            line = f"[{i}] {entity} / {topic}"
            if date:
                line += f"  ({date})"
            line += f"\n    VERBATIM: \"{verbatim}\""
            if turn_id:
                line += f"  [{turn_id}]"
            lines.append(line)
        return "\n".join(lines)

    def _verify_verbatim(self, structured_facts: list[dict]):
        """Post-integration guardrail: ensure each fact's verbatim text is represented
        in the target wiki page. If key tokens are missing, append the verbatim quote
        directly — no model call, fully deterministic."""
        STOP = {"i", "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
                "for", "of", "is", "was", "am", "are", "it", "my", "me", "we",
                "he", "she", "they", "his", "her", "their", "its"}
        MIN_OVERLAP = 0.5   # fraction of content tokens that must appear in page
        MIN_TOKENS  = 3     # only check facts with at least this many content tokens

        for f in structured_facts:
            verbatim = f.get("verbatim", "").strip()
            entity   = f.get("entity", "").strip()
            topic    = f.get("topic", "").strip()
            turn_id  = f.get("turn_id", "")
            if not verbatim or not entity:
                continue

            # Find the best matching page for this entity+topic
            all_pages = self.wiki.list_pages()
            target = None
            for pname in all_pages:
                if " - " in pname:
                    pent = pname.split(" - ")[0].strip().lower()
                    if pent == entity.lower():
                        if not target:
                            target = pname
                        elif topic and topic.lower() in pname.lower():
                            target = pname
                            break
            if not target:
                continue

            content = self.wiki.read_page(target) or ""
            content_lower = content.lower()

            # Token-level overlap check on meaningful words only
            v_tokens = [t for t in re.findall(r'[a-z0-9]+', verbatim.lower())
                        if t not in STOP]
            if len(v_tokens) < MIN_TOKENS:
                continue
            matched = sum(1 for t in v_tokens if t in content_lower)
            if matched / len(v_tokens) >= MIN_OVERLAP:
                continue  # verbatim is sufficiently represented

            # Append the missing verbatim quote directly to the page
            cite = f" [{turn_id}]" if turn_id else ""
            addition = f'\n\n"{verbatim}"{cite}'
            self.wiki.write_page(target, content.rstrip() + addition)
            self.wiki.append_log(
                f"verbatim-verify | appended to '{target}': {verbatim[:60]!r}")

    # ── DETERMINISTIC PRE/POST PASSES ────────────────────────────────────────

    _MONTHS = {
        'january':1,'february':2,'march':3,'april':4,'may':5,'june':6,
        'july':7,'august':8,'september':9,'october':10,'november':11,'december':12,
        'jan':1,'feb':2,'mar':3,'apr':4,'jun':6,'jul':7,'aug':8,
        'sep':9,'oct':10,'nov':11,'dec':12,
    }
    _DOW = {'monday':0,'tuesday':1,'wednesday':2,'thursday':3,
            'friday':4,'saturday':5,'sunday':6}

    @staticmethod
    def _fmt_date(d: datetime) -> str:
        """Format date as '7 May 2023' (no leading zero, cross-platform)."""
        return d.strftime('%d %B %Y').lstrip('0')

    @staticmethod
    def _fmt_month(d: datetime) -> str:
        return d.strftime('%B %Y')

    def _parse_session_date(self, session_label: str) -> Optional[datetime]:
        """Parse session date from label like 'D3 (1:56 pm on 8 May, 2023)'."""
        m = re.search(r'(\d{1,2})\s+(\w+),?\s+(\d{4})', session_label)
        if m:
            month = self._MONTHS.get(m.group(2).lower())
            if month:
                try:
                    return datetime(int(m.group(3)), month, int(m.group(1)))
                except ValueError:
                    pass
        return None

    def _resolve_date_expr(self, expr: str, sd: datetime) -> str:
        """Resolve a relative date expression to an absolute string using session date sd."""
        e = expr.lower().strip()
        if e == 'yesterday':
            return self._fmt_date(sd - timedelta(days=1))
        if e == 'today':
            return self._fmt_date(sd)
        if e in ('last week', 'the past week'):
            start = sd - timedelta(days=sd.weekday() + 7)
            end   = start + timedelta(days=6)
            return f"{self._fmt_date(start).rsplit(' ', 1)[0]}–{self._fmt_date(end)}"
        if e == 'next week':
            start = sd + timedelta(days=7 - sd.weekday())
            end   = start + timedelta(days=6)
            return f"{self._fmt_date(start).rsplit(' ', 1)[0]}–{self._fmt_date(end)}"
        if e in ('last month', 'the past month'):
            first = (sd.replace(day=1) - timedelta(days=1)).replace(day=1)
            return self._fmt_month(first)
        if e == 'next month':
            first = (sd.replace(day=28) + timedelta(days=4)).replace(day=1)
            return self._fmt_month(first)
        if e == 'last year':
            return str(sd.year - 1)
        if e == 'next year':
            return str(sd.year + 1)
        # "N days/weeks/months ago"
        m = re.match(r'(\d+)\s+(day|week|month|year)s?\s+ago', e)
        if m:
            n, unit = int(m.group(1)), m.group(2)
            if unit == 'day':
                return self._fmt_date(sd - timedelta(days=n))
            if unit == 'week':
                return f"week of {self._fmt_date(sd - timedelta(weeks=n))}"
            if unit == 'month':
                month = sd.month - n
                year  = sd.year + (month - 1) // 12
                month = ((month - 1) % 12) + 1
                return self._fmt_month(datetime(year, month, 1))
            if unit == 'year':
                return str(sd.year - n)
        # "in N days/weeks/months" or "N days/weeks/months from now"
        m = re.match(r'(?:in\s+)?(\d+)\s+(day|week|month|year)s?(?:\s+from\s+now)?', e)
        if m:
            n, unit = int(m.group(1)), m.group(2)
            if unit == 'day':
                return self._fmt_date(sd + timedelta(days=n))
            if unit == 'week':
                return f"week of {self._fmt_date(sd + timedelta(weeks=n))}"
            if unit == 'month':
                month = sd.month + n
                year  = sd.year + (month - 1) // 12
                month = ((month - 1) % 12) + 1
                return self._fmt_month(datetime(year, month, 1))
        # "last/next Monday/Tuesday/..."
        m = re.match(r'(last|next)\s+(\w+)', e)
        if m and m.group(2) in self._DOW:
            target = self._DOW[m.group(2)]
            delta  = (sd.weekday() - target) % 7
            if m.group(1) == 'last':
                d = sd - timedelta(days=delta if delta else 7)
            else:
                d = sd + timedelta(days=(target - sd.weekday()) % 7 or 7)
            return self._fmt_date(d)
        # "a few days/weeks/months ago" — approximate
        m = re.match(r'a\s+few\s+(day|week|month)s?\s+ago', e)
        if m:
            approx = {'day': timedelta(days=4), 'week': timedelta(weeks=3),
                      'month': timedelta(days=90)}
            return f"~{self._fmt_month(sd - approx[m.group(1)])}"
        return ""

    # Relative date patterns to scan for in turn text
    _REL_DATE_RE = re.compile(
        r'\b(yesterday|today|last\s+week|the\s+past\s+week|next\s+week|'
        r'last\s+month|the\s+past\s+month|next\s+month|last\s+year|next\s+year|'
        r'a\s+few\s+(?:days?|weeks?|months?)\s+ago|'
        r'\d+\s+(?:day|week|month|year)s?\s+ago|'
        r'in\s+\d+\s+(?:day|week|month|year)s?|'
        r'\d+\s+(?:day|week|month|year)s?\s+from\s+now|'
        r'(?:last|next)\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b',
        re.IGNORECASE)

    # Common sentence-starter words that should never be the first item of a list
    _LIST_STOPWORDS = frozenset({
        'wow','hey','yes','no','oh','thanks','thank','well','ok','okay','so','but',
        'and','or','also','great','sure','right','nice','good','bad','yep','nope',
    })

    # Explicit list: requires 2+ comma-separated capitalised items AND no stopword start.
    # Multi-word proper nouns ("Dr. Seuss") count as 2-item lists; single words need 3+.
    _LIST_RE = re.compile(
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+'       # multi-word first item (Dr. Seuss style)
        r'(?:,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)+' # comma + more items
        r'(?:,?\s*and\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)?'  # optional "and last"
        r'|'
        r'[A-Z][a-z]+'                               # single-word first item
        r'(?:,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*){2,}'  # NEEDS 2+ more items (3 total)
        r'(?:,?\s*and\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)?)\b'
    )
    # Lowercase activity/hobby lists after explicit trigger verbs
    _ACTIVITY_LIST_RE = re.compile(
        r'(?:like[sd]?|enjoy[sed]*|love[sd]?|practis[eing]+|into|hobbi(?:es|ed?)|'
        r'interest(?:s|ed)?(?:\s+in)?)\s+'
        r'((?:[a-z][a-z]+(?:\s+[a-z][a-z]+)*,\s*){1,}'
        r'(?:and\s+)?[a-z][a-z]+(?:\s+[a-z][a-z]+)*)',
        re.IGNORECASE)

    def _pre_extract_anchors(self, turns: list[dict], session_label: str) -> dict:
        """Deterministically extract Type A facts before any LLM call.

        Returns:
          dates  — list of {raw, resolved, turn_id, context}
          lists  — list of {items, turn_id, context}
        """
        sd = self._parse_session_date(session_label)
        anchors = {"session_date": sd, "dates": [], "lists": []}
        if not sd:
            return anchors

        seen_dates: set[str] = set()
        seen_lists: set[str] = set()

        for turn in turns:
            turn_id = turn.get("dia_id", "")
            text    = turn.get("text", "")

            # ── Date expressions ──
            for m in self._REL_DATE_RE.finditer(text):
                raw      = m.group(0)
                raw_lower = raw.lower()
                if raw_lower in seen_dates:
                    continue
                resolved = self._resolve_date_expr(raw, sd)
                if resolved:
                    # Context: up to 60 chars around the match
                    start = max(0, m.start() - 30)
                    ctx   = text[start: m.end() + 30].strip()
                    anchors["dates"].append({
                        "raw": raw, "resolved": resolved,
                        "turn_id": turn_id, "context": ctx})
                    seen_dates.add(raw_lower)

            # List extraction deferred — regex cannot reliably distinguish noun-phrase
            # enumerations from comma-separated clauses without NLP. Date extraction
            # (above) covers the highest-value Type A facts. Lists to be added later.

        return anchors

    def _format_anchor_section(self, anchors: dict) -> str:
        """Format extracted anchors for injection into the Understand prompt."""
        lines = []
        sd = anchors.get("session_date")
        if sd:
            lines.append(
                'RELATIVE TEMPORAL WORDS — rewrite as timeless persistent facts:\n'
                '  "currently", "now", "at the moment", "these days", "at present" describe\n'
                '  a state at session time. Drop the temporal qualifier and state the persistent\n'
                '  fact directly. Example: "she is currently a nurse" → "she is a nurse".\n'
                '  If the state is genuinely time-bound (a plan, an event), use the resolved date instead.')
        if anchors.get("dates"):
            lines.append("RESOLVED DATES (use these exact resolved values — do NOT recompute):")
            for d in anchors["dates"]:
                lines.append(f'  "{d["raw"]}" → {d["resolved"]}  [{d["turn_id"]}]'
                             + (f'  (context: "…{d["context"]}…")' if d.get("context") else ""))
        if anchors.get("lists"):
            lines.append("VERBATIM LISTS (copy these exact items into the facts you record):")
            for lst in anchors["lists"]:
                lines.append(f'  [{lst["turn_id"]}] {lst["raw"]}')
        return "\n".join(lines)

    def _verify_anchors(self, anchors: dict):
        """Post-integration: check resolved dates and lists survived into wiki pages.
        Appends missing anchors directly — no LLM, fully deterministic."""
        all_pages = self.wiki.list_pages()
        if not all_pages:
            return

        # Build a combined text of all wiki content for fast lookup
        all_content = {}
        for name in all_pages:
            all_content[name] = (self.wiki.read_page(name) or "").lower()
        combined = " ".join(all_content.values())

        # Check resolved dates
        for d in anchors.get("dates", []):
            resolved = d["resolved"].lower().strip("~")
            # Key tokens: year + month word (or just year for year-only resolutions)
            tokens = [t for t in re.findall(r'[a-z0-9]+', resolved) if len(t) > 2]
            if not tokens:
                continue
            if all(t in combined for t in tokens):
                continue  # present — OK
            # Find best page (entity from turn context, or first page)
            target = all_pages[0]
            if d.get("context"):
                ctx_lower = d["context"].lower()
                for name in all_pages:
                    ent = name.split(" - ")[0].strip().lower() if " - " in name else ""
                    if ent and ent in ctx_lower:
                        target = name
                        break
            existing = self.wiki.read_page(target) or ""
            addition = f'\n\n[{d["turn_id"]}] {d["raw"]} → {d["resolved"]}'
            self.wiki.write_page(target, existing.rstrip() + addition)
            self.wiki.append_log(
                f"anchor-verify | date appended to '{target}': {d['raw']} → {d['resolved']}")

        # Check verbatim lists
        for lst in anchors.get("lists", []):
            items = lst["items"]
            # At least half the items must appear in the wiki
            present = sum(1 for item in items if item.lower() in combined)
            if present / len(items) >= 0.5:
                continue
            # Append the raw list to the most relevant page
            target = all_pages[0]
            for name in all_pages:
                ent = name.split(" - ")[0].strip().lower() if " - " in name else ""
                if ent and ent in lst.get("raw", "").lower():
                    target = name
                    break
            existing = self.wiki.read_page(target) or ""
            addition = f'\n\n[{lst["turn_id"]}] {lst["raw"]}'
            self.wiki.write_page(target, existing.rstrip() + addition)
            self.wiki.append_log(
                f"anchor-verify | list appended to '{target}': {lst['raw'][:60]!r}")

    def _critique(self, pages: dict, turns: list[dict], session_label: str):
        """Critic pass — reviews this session's pages against source truth.
        Plays the judgment role Karpathy's human plays: catch and fix errors at
        ingest time, with reasoning, before they propagate into the wiki."""
        if not pages:
            return

        turn_text = "\n".join(
            f"[{t.get('dia_id','')}] {t['speaker']}: {t['text']}" for t in turns)

        for name in list(pages.keys()):
            name = name.strip()
            current = self.wiki.read_page_content(name)
            if not current:
                continue

            critic_prompt = f"""You are a careful wiki editor reviewing a page against its source.

This session took place on: {session_label}

SOURCE CONVERSATION:
{turn_text}

PAGE "{name}" (current content):
{current}

Review the page against the source and fix these issues using your judgment:
1. UNRESOLVED DATES: any relative reference ("last year", "last month", "yesterday",
   "recently", "next month") must be converted to an absolute date/period computed from
   the session date above. E.g. on a May 2023 session, "last year" → "2022".
2. RELATIVE TEMPORAL WORDS: "currently", "now", "at the moment", "these days", "at present"
   must be rewritten as timeless persistent facts — drop the temporal qualifier.
   E.g. "she is currently a nurse" → "she is a nurse". If the state is time-bound, use its resolved date.
3. CONTRADICTIONS or claims NOT supported by the source — remove or correct them.
4. CORRUPTION: if the content contains stray YAML, code fences, or duplicated
   frontmatter, clean it into plain prose.
5. Keep all [turn ID] citations like [D5:3]. Keep the prose faithful and concise.

Output ONLY the corrected page content (prose, no frontmatter block, no preamble).
If the page is already correct, output it unchanged:"""

            corrected = compiler_call(critic_prompt, system=COMPILER_SYSTEM_PROMPT)
            if corrected and len(corrected.strip()) > 20:
                # Strip any frontmatter the critic may have re-added
                c = corrected.strip()
                if c.startswith("---"):
                    parts = c.split("---", 2)
                    c = parts[2].strip() if len(parts) >= 3 else c
                self.wiki.write_page(name, c)

    SPLIT_THRESHOLD_WORDS    = 400  # was 250 — too aggressive, caused permanent fragmentation
    CONSOLIDATE_CEILING_WORDS = 700  # decoupled from split threshold; allows merging up to this size
                                     # (was implicitly 250 = SPLIT_THRESHOLD, creating a deadlock
                                     # where split fragments could never re-merge)

    MIN_PAGES_FLOOR = 8  # never consolidate below this many pages (anti over-merge)

    def _consolidate_fragments(self):
        """Merge look-alike fragments of the SAME entity+topic into one coherent page.
        Fragmentation ('John - Community Service' + 'Community Engagement' + ...) causes
        both routing ambiguity (which fragment?) and forced multi-page assembly. Merging
        same-entity+same-topic families fixes both. Guards against the over-merge attractor:
          - only merge SAME entity + SAME topic-prefix (never cross-entity/topic)
          - preserve every [turn ID] (abort a merge that drops provenance)
          - floor on total page count (never collapse below MIN_PAGES_FLOOR)
          - ceiling: skip a merge whose result would exceed SPLIT_THRESHOLD_WORDS
            (otherwise split would just undo it — pointless oscillation)
        Runs AFTER split, BEFORE link/frontmatter/summary."""
        pages = self.wiki.get_all_pages()
        if len(pages) <= self.MIN_PAGES_FLOOR:
            return

        # Group by family key = "Entity - TopicPrefix" (entity + first topic word).
        # Normalize trailing punctuation/symbols so "LGBTQ+" and "LGBTQ" fall in the same family.
        def _norm_topic_word(w: str) -> str:
            return re.sub(r'[+\-&]+$', '', w).strip().lower()

        families = {}
        for name in pages:
            parts = name.split(" - ")
            if len(parts) < 2:
                continue
            entity = parts[0].strip()
            topic_first_raw = parts[1].split()[0].strip() if parts[1].split() else ""
            if not topic_first_raw:
                continue
            topic_first = _norm_topic_word(topic_first_raw)
            key = f"{entity} - {topic_first}"
            families.setdefault(key, []).append(name)

        for key, members in families.items():
            if len(members) < 2:
                continue  # not a fragment family
            if len(self.wiki.list_pages()) - (len(members) - 1) < self.MIN_PAGES_FLOOR:
                continue  # merging would drop us below the floor

            # Gather bodies (strip frontmatter + wikilinks for the merge input)
            bodies = {}
            for m in members:
                raw = self.wiki.read_page(m) or ""
                body = raw
                if raw.strip().startswith("---"):
                    p = raw.split("---", 2)
                    body = p[2].strip() if len(p) >= 3 else raw
                bodies[m] = re.sub(r'\s*\[\[[^\]]+\]\]\.?', '', body).strip()

            combined_words = sum(len(b.split()) for b in bodies.values())
            if combined_words > self.CONSOLIDATE_CEILING_WORDS:
                continue  # merged page would be too large; skip

            merge_input = "\n\n".join(f'PAGE "{m}":\n{bodies[m]}' for m in members)
            merge_prompt = f"""These wiki pages are fragments of the SAME topic and should be ONE page.
Merge them into a single coherent page named "{key}".

RULES:
- Combine all information into one well-organized page. Lose NOTHING.
- Preserve every [turn ID] citation like [D5:3] exactly.
- Remove redundancy, but keep every distinct fact, date, name, and number.
- Do not invent anything not in the fragments. Do not merge in unrelated facts.

FRAGMENTS:
{merge_input}

Output exactly one page:
PAGE: {key}
CONTENT:
(coherent prose with all citations)

Begin:"""
            result = compiler_call(merge_prompt, system=COMPILER_SYSTEM_PROMPT)
            merged = self._parse_pages(result)
            if len(merged) != 1:
                continue  # expected exactly one merged page

            mname, mcontent = next(iter(merged.items()))
            # Verify provenance preserved: all original turn IDs must survive
            orig_ids = set()
            for b in bodies.values():
                orig_ids.update(re.findall(r'\[(D\d+:\d+)\]', b))
            new_ids = set(re.findall(r'\[(D\d+:\d+)\]', mcontent))
            if orig_ids and len(orig_ids - new_ids) > 2:
                continue  # merge dropped provenance — keep fragments, skip

            # Commit: delete fragments, write the single merged page
            for m in members:
                self.wiki.delete_page(m)
            self.wiki.write_page(key, mcontent.strip())
            self.wiki.append_log(f"consolidate | {len(members)} fragments → '{key}'")
            self._n_consolidated = getattr(self,'_n_consolidated',0) + 1

    def _split_oversized(self):
        """Split any page over the word threshold into coherent sub-topic pages.
        This is the anti-collapse mechanism: a wiki that collapsed to a few
        mega-pages is a wiki of oversized pages, so splitting them restores
        granularity. The compiler decides the split (it has the judgment);
        Python detects the trigger, verifies content is preserved, and rewires."""
        pages = self.wiki.get_all_pages()
        existing_names = set(pages.keys())

        for name, raw in pages.items():
            # Work on body without frontmatter
            body = raw
            if raw.strip().startswith("---"):
                parts = raw.split("---", 2)
                body = parts[2].strip() if len(parts) >= 3 else raw
            # Strip wikilinks for the word count / split input
            body_clean = re.sub(r'\s*\[\[[^\]]+\]\]\.?', '', body).strip()

            if len(body_clean.split()) <= self.SPLIT_THRESHOLD_WORDS:
                continue

            # Show sibling pages (same entity) so the model routes split content into
            # existing pages rather than creating new near-duplicate names.
            entity_prefix = name.split(" - ")[0] if " - " in name else ""
            siblings = [n for n in existing_names if n != name and n.startswith(entity_prefix + " - ")]
            sibling_hint = (f"\nEXISTING PAGES FOR {entity_prefix} (prefer these names for split targets "
                            f"when the content fits — do NOT create a new page if an existing one covers "
                            f"the same sub-topic):\n{', '.join(siblings)}\n"
                            if siblings else "")

            split_prompt = f"""This wiki page has grown too large and covers multiple sub-topics.
Split it into 2-4 smaller pages, each a single coherent sub-topic.

RULES:
- Prefer using EXISTING page names (listed below) as split targets when the content fits.
  Only create a NEW name if the sub-topic has no existing page at all.
- Each new page: a distinct, specific name in "Entity - Topic" form, different from "{name}".
- Preserve ALL information and ALL [turn ID] citations like [D5:3] — distribute them
  to the page they belong to. Lose nothing.
- Do not create vague "part 2" pages; split by actual sub-topic.
- If the page is genuinely a single topic that cannot be meaningfully split, output
  exactly: KEEP
{sibling_hint}
PAGE "{name}":
{body_clean}

Output each new page as:
PAGE: New Page Name
CONTENT:
(prose with citations)

Begin:"""

            result = compiler_call(split_prompt, system=COMPILER_SYSTEM_PROMPT)
            if not result or result.strip().upper().startswith("KEEP"):
                continue

            new_pages = self._parse_pages(result)
            if len(new_pages) < 2:
                continue  # not a real split

            # Verify content preservation: every [turn ID] in the original must survive
            orig_ids = set(re.findall(r'\[(D\d+:\d+)\]', body_clean))
            new_ids = set()
            for c in new_pages.values():
                new_ids.update(re.findall(r'\[(D\d+:\d+)\]', c))
            # Require that we didn't lose more than a couple of citations
            if orig_ids and len(orig_ids - new_ids) > 2:
                continue  # split dropped provenance — keep original, skip

            # Commit: remove the original, write the sub-pages (avoid name collisions)
            self.wiki.delete_page(name)
            self._n_split = getattr(self,'_n_split',0) + 1
            for nname, ncontent in new_pages.items():
                nname = nname.strip()
                if not nname or not ncontent.strip():
                    continue
                if nname in existing_names and nname != name:
                    nname = f"{nname} (cont.)"
                self.wiki.write_page(nname, ncontent.strip())
                existing_names.add(nname)

    def _ensure_frontmatter(self, session_label: str):
        """Add YAML frontmatter to any page missing it. Python-enforced."""
        import datetime
        pages = self.wiki.get_all_pages()
        all_entities = set()
        for name in pages:
            if " - " in name:
                all_entities.add(name.split(" - ")[0].strip())

        # Date patterns to extract date ranges from content
        date_pattern = re.compile(
            r'\b(\d{1,2}\s+)?(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b'
            r'|\b\d{4}\b', re.IGNORECASE)

        type_keywords = {
            "identity": ["is a", "identifies", "transgender", "years old", "background"],
            "event": ["attended", "happened", "on ", "went to", "participated"],
            "plan": ["plans to", "will", "intends", "goal", "hopes to", "considering"],
            "relationship": ["friend", "married", "partner", "supports", "relationship"],
            "activity": ["enjoys", "hobby", "practices", "does", "likes to"],
        }

        for name, raw in pages.items():
            stripped = raw.strip()
            # Detect corruption: a page that has frontmatter but whose body still
            # contains embedded fences or a second frontmatter block.
            body_after_fm = stripped
            has_fm = stripped.startswith("---")
            if has_fm:
                parts = stripped.split("---", 2)
                body_after_fm = parts[2].strip() if len(parts) >= 3 else stripped
            corrupted = ("```" in body_after_fm or
                         body_after_fm.lstrip().startswith("---") or
                         "type:" in body_after_fm[:200] and "created:" in body_after_fm[:200])

            if has_fm and not corrupted:
                continue  # clean page with frontmatter — leave it

            # Either no frontmatter, or corrupted — rebuild from cleaned body.
            content = self._clean_content(body_after_fm if has_fm else stripped)

            # Derive entities: page-name entity + any other known entity mentioned
            entities = []
            if " - " in name:
                entities.append(name.split(" - ")[0].strip())
            else:
                entities.append(name)
            for ent in all_entities:
                if ent not in entities and re.search(r'\b' + re.escape(ent) + r'\b', content, re.IGNORECASE):
                    entities.append(ent)

            # Derive type from keywords
            page_type = "general"
            content_lower = content.lower()
            for t, keywords in type_keywords.items():
                if any(kw in content_lower for kw in keywords):
                    page_type = t
                    break

            # Date range intentionally omitted — created/last_updated capture timing;
            # content-extracted ranges were computed out of chronological order.
            date_range = ""

            # Harvest source turn IDs (Tier 3 provenance) from inline [Dx:y] citations
            new_ids = re.findall(r'\[(D\d+:\d+)\]', content)

            # STRUCTURAL PROVENANCE: sources are CUMULATIVE — union new citations with any
            # already recorded on the page. The binding can only grow, never shrink, so a
            # later edit that drops an inline citation cannot lose earlier provenance. Also
            # preserve the original `created` label rather than overwriting it each session.
            prior_ids, prior_created = [], None
            existing_raw = self.wiki.read_page(name) or ""
            if existing_raw.strip().startswith("---"):
                fm = existing_raw.split("---", 2)
                if len(fm) >= 3:
                    pm = re.search(r'sources:\s*\[([^\]]+)\]', fm[1])
                    if pm:
                        prior_ids = [x.strip() for x in pm.group(1).split(",") if x.strip()]
                    cm = re.search(r'created:\s*(.+)', fm[1])
                    if cm:
                        prior_created = cm.group(1).strip()
            source_ids = list(dict.fromkeys(prior_ids + new_ids))  # union, preserve order
            created_label = prior_created if prior_created else session_label

            frontmatter = (
                "---\n"
                f"entities: [{', '.join(entities)}]\n"
                f"type: {page_type}\n"
                f"created: {created_label}\n"
                f"last_updated: {session_label}\n"
            )
            if date_range:
                frontmatter += f"dates: {date_range}\n"
            if source_ids:
                frontmatter += f"sources: [{', '.join(source_ids)}]\n"
            frontmatter += "---\n"

            self.wiki.write_page(name, frontmatter + content)

    def _inject_wikilinks(self):
        """Insert [[wikilinks]] based on TOPIC relevance, capped per page.
        Linking on entity name (Caroline/Melanie) made everything link to everything,
        since those names appear on every page. Instead, link only when another page's
        distinctive TOPIC is actually mentioned — and cap the count so the graph stays
        navigable rather than fully-connected noise."""
        pages = self.wiki.get_all_pages()
        page_names = list(pages.keys())
        valid_names = set(page_names)
        MAX_LINKS = 4

        # Strip broken wikilinks first
        for page_name, content in pages.items():
            links = re.findall(r'\[\[([^\]]+)\]\]', content)
            cleaned = content
            for link in links:
                if link not in valid_names:
                    cleaned = cleaned.replace(f"[[{link}]]", link)
            if cleaned != content:
                self.wiki.write_page(page_name, cleaned)
        pages = self.wiki.get_all_pages()

        # Topic term for each page: the part after " - ", or the whole name
        def topic_of(name):
            return name.split(" - ", 1)[1].strip() if " - " in name else name.strip()

        for page_name, content in pages.items():
            # Strip any existing links to rebuild cleanly (avoid accumulation across sessions)
            body = re.sub(r'\s*\[\[[^\]]+\]\]\.?', '', content).rstrip()
            base = body

            scored = []
            for target in page_names:
                if target == page_name:
                    continue
                topic = topic_of(target)
                # Link only if the target's distinctive topic phrase appears in this page
                if len(topic) >= 4 and re.search(r'\b' + re.escape(topic) + r'\b', base, re.IGNORECASE):
                    # Score by topic length (more specific = stronger signal)
                    scored.append((len(topic), target))

            scored.sort(reverse=True)
            chosen = [t for _, t in scored[:MAX_LINKS]]

            if chosen:
                if not base.endswith("."):
                    base += "."
                base += " " + " ".join(f"[[{t}]]" for t in chosen)

            if base != content:
                self.wiki.write_page(page_name, base)

    def _parse_pages(self, raw: str) -> dict[str, str]:
        pages = {}
        parts = re.split(r'\nPAGE:\s*', '\n' + raw)
        for part in parts[1:]:
            lines = part.strip().split('\n')
            if not lines: continue
            name = lines[0].strip()
            content_lines = []
            found_content = False
            for line in lines[1:]:
                if line.strip().startswith('CONTENT:'):
                    found_content = True
                    after = line.split('CONTENT:', 1)[1].strip()
                    if after: content_lines.append(after)
                elif found_content:
                    content_lines.append(line)
            if name and content_lines:
                pages[name] = self._clean_content('\n'.join(content_lines).strip())
        return pages

    @staticmethod
    def _clean_content(content: str) -> str:
        """Strip code fences and stray/duplicated frontmatter the LLM sometimes
        embeds in page content (cause of the nested-YAML corruption)."""
        c = content.strip()
        # Remove leading ```yaml / ``` fence and its closing fence
        c = re.sub(r'^```[a-zA-Z]*\s*\n', '', c)
        c = re.sub(r'\n```\s*$', '', c)
        c = c.strip()
        # If content now starts with a frontmatter block, strip it (real frontmatter
        # is added separately by _ensure_frontmatter — content should be prose only)
        if c.startswith("---"):
            parts = c.split("---", 2)
            if len(parts) >= 3:
                c = parts[2].strip()
        # Remove any remaining stray fences
        c = c.replace("```yaml", "").replace("```", "").strip()
        return c

    def generate_catalog(self) -> dict:
        """Build _catalog.json: page-name → routing description for query engine.

        Uses the compiler to write a retrieval-optimised description for each page:
        'What facts and questions does this page contain?' The LLM description preserves
        exact nouns, dates, and lists verbatim, giving the router far better signal than
        Python-extracted sentences which paraphrase compound facts.
        One compiler call per page — only runs during compilation (cached afterwards).
        """
        pages = self.wiki.get_all_pages()
        catalog = {}

        for name, content in pages.items():
            body = content
            if content.strip().startswith("---"):
                parts = content.split("---", 2)
                body = parts[2].strip() if len(parts) >= 3 else content
            body_clean = re.sub(r'\[\[[^\]]+\]\]', '', body)
            body_clean = re.sub(r'\[D\d+:\d+\]', '', body_clean).strip()

            prompt = (
                f"Wiki page title: {name}\n\n"
                f"Page content:\n{body_clean[:1200]}\n\n"
                "Write a 2-3 sentence routing description for this page. "
                "The description will be used to decide whether to open this page to answer a question. "
                "Include: the main entity and topic, key facts (use EXACT names, dates, and lists "
                "verbatim from the content — never paraphrase), and 2-3 example questions this page "
                "can answer. Format: plain sentences, no bullet points, max 120 words."
            )
            desc = compiler_call(prompt)
            if not desc:
                # Fallback: first 400 chars of body
                desc = body_clean[:400]
            catalog[name] = desc[:600]

        self.wiki.write_catalog(catalog)
        self.wiki.append_log(f"catalog | {len(catalog)} pages indexed")
        return catalog

    def generate_summary(self) -> str:
        """Generate summary.md — structured entity index with inline dates and [→Page] pointers."""
        pages = self.wiki.get_all_pages()
        if not pages: return ""

        schema = self.wiki.get_schema()

        def _body(content: str) -> str:
            if content.strip().startswith("---"):
                parts = content.split("---", 2)
                return parts[2].strip() if len(parts) >= 3 else content
            return content

        pages_text = "\n\n".join(
            f"=== {name} ===\n{_body(content)}" for name, content in pages.items())

        # Word budget: 60 words per page — enough for dense structured entries.
        # Not capped too low: a sparse summary loses dates and names, breaking temporal queries.
        word_budget = min(1800, max(300, len(pages) * 60))

        prompt = f"""Write a STRUCTURED MEMORY INDEX from these wiki pages.

A query model reads this index first when answering questions. It must:
- Answer simple questions directly (exact dates, names, numbers inline)
- Route harder questions to the right page via [→Page Name] pointers

FORMAT — group by entity, one bullet per topic, facts inline:

ENTITY: [Name]
  - [topic]: [exact fact with date/name/number if known] [→Page Name]
  - [topic]: [fact] [→Page Name]; [related fact on same topic] [→Page Name]

RULES:
1. EXACT facts always: "7 May 2023" not "recently"; "Sweden" not "abroad"; "28" not "late 20s".
   A temporal question must be answerable from the date written here.
2. Every page must appear as a [→Page Name] pointer target at least once.
3. Cover ALL entities and ALL pages. Do not omit any page.
4. One bullet per topic — pack related facts together on one line rather than splitting.
5. MANDATORY FIRST BULLET for each person: a "status" line giving their full identity profile.
   It MUST include (when known): gender identity or orientation, exact home country or origin,
   relationship status and duration, children count, pets (exact names and species), occupation.
   Example: "Jane is a 34-year-old nurse from Canada, married 5 years, 2 children;
   she has a cat named Whiskers [D1:4][D3:8]."
   ONLY use facts that appear in the wiki pages — do NOT guess or infer. If origin country
   is not in any page, omit it rather than writing "United States" or any other guess.
   Do NOT omit facts that ARE in the pages, or replace them with vague phrases like "from abroad".
6. Under {word_budget} words total.

WIKI PAGES ({len(pages)} pages):
{pages_text}

Write the index. Start directly with the first ENTITY: line:"""

        summary = compiler_call(prompt, system=COMPILER_SYSTEM_PROMPT)

        # Ensure every page is referenced with a short description hint
        page_names = list(pages.keys())
        missing = [name for name in page_names if f"[→{name}]" not in summary]
        if missing:
            hints = []
            for name in missing:
                content = pages[name]
                # Strip frontmatter
                if content.strip().startswith("---"):
                    parts = content.split("---", 2)
                    content = parts[2].strip() if len(parts) >= 3 else content
                # Take first 10 words as hint
                words = content.split()[:10]
                hint = " ".join(words).rstrip(".,;:")
                hints.append(f"[→{name}] ({hint})")
            summary += "\n\nAlso in memory: " + ", ".join(hints) + "."

        self.wiki.write_summary(summary)
        self.wiki.append_log(f"summary | {len(pages)} pages | ~{estimate_tokens(summary)} tokens | {len(missing)} pages auto-added")
        return summary

    def lint(self) -> str:
        """Post-compilation health check."""
        pages = self.wiki.get_all_pages()
        if not pages:
            return "No pages to lint."

        # Python-level checks first
        issues = []
        all_names = set(pages.keys())

        for name, content in pages.items():
            # Check for broken wikilinks
            links = re.findall(r'\[\[([^\]]+)\]\]', content)
            for link in links:
                if link not in all_names:
                    issues.append(f"BROKEN LINK: [[{link}]] in '{name}' — page does not exist")

            # Check page size
            words = len(content.split())
            if words > 300:
                issues.append(f"OVERSIZED: '{name}' is {words} words (target: 50-200)")
            elif words < 20:
                issues.append(f"UNDERSIZED: '{name}' is only {words} words — consider merging")

            # Check for missing frontmatter
            if not content.strip().startswith("---"):
                issues.append(f"NO FRONTMATTER: '{name}' missing YAML frontmatter")

        # Check for orphan pages (no inbound links from other pages)
        inbound = {name: 0 for name in all_names}
        for name, content in pages.items():
            links = re.findall(r'\[\[([^\]]+)\]\]', content)
            for link in links:
                if link in inbound:
                    inbound[link] += 1
        for name, count in inbound.items():
            if count == 0:
                issues.append(f"ORPHAN: '{name}' has no inbound links from other pages")

        # Check summary references all pages
        summary = self.wiki.get_summary()
        for name in all_names:
            if f"[→{name}]" not in summary:
                issues.append(f"NOT IN SUMMARY: '{name}' has no [→{name}] pointer in summary")

        # LLM-based check for duplicates
        if len(pages) > 5:
            page_list = "\n".join(f"- {name}" for name in sorted(all_names))
            dup_check = compiler_call(
                f"These are wiki page names. Identify any that likely cover the same topic "
                f"and should be merged. Reply ONLY with pairs like 'MERGE: Page A + Page B' "
                f"or 'NONE' if no duplicates.\n\n{page_list}",
                system=COMPILER_SYSTEM_PROMPT,
                temperature=0.0)
            if dup_check and "NONE" not in dup_check.upper():
                for line in dup_check.strip().split("\n"):
                    if line.strip():
                        issues.append(f"DUPLICATE: {line.strip()}")

        result = "\n".join(issues) if issues else "No issues found."
        self.wiki.append_log(f"lint | {len(issues)} issues found")

        if issues:
            print(f"  [LINT] {len(issues)} issues:")
            for issue in issues:
                print(f"    {issue}")
        else:
            print(f"  [LINT] Clean — no issues found")

        return result


# ─────────────────────────────────────────────
# Query Engine — reads summary, retrieves pages on demand
# ─────────────────────────────────────────────

@dataclass
class QueryTrace:
    query: str
    summary_shown: str = ""
    pages_retrieved: list = field(default_factory=list)
    answer: str = ""
    tokens_est: int = 0
    tokens_t1: int = 0       # T1 summary attempt + judge
    tokens_t2_pick: int = 0  # all _pick_pages calls
    tokens_t2_ans: int = 0   # all T2 _attempt_answer + judge calls
    tokens_t3: int = 0       # T3 raw attempt + judge
    hops: int = 1
    reasoning: str = ""
    steps: list = field(default_factory=list)  # ReAct trail: tier, thought, answer, verdict
    t3_debug: dict = field(default_factory=dict)  # Tier-3 retrieval diagnostics


MAX_PAGE_ATTEMPTS = 3   # measured on conv 0: attempt 2 converts 48 questions, attempt 3
                        # converts 6, attempt 4 converts 0 — a 4th iteration is pure cost
MAX_TOTAL_PAGES   = 10  # hard cap on total pages loaded across all Tier 2 iterations
MAX_CITED_TURNS   = 14  # cap on raw turns fetched via page citations per answer attempt

# Inline turn citation as written by the compiler: [D3:12]
_CITATION_RE = re.compile(r'\[(D\d+:\d+)\]')

# Inference questions require full character profile — different strategy than factual retrieval.
_INFERENCE_WORDS = frozenset({"would","likely","might","could","probably","seem","consider","tend"})

# Relative dates from T3 raw turns can't be resolved to calendar dates — refuse them for "when".
_RELATIVE_DATE_RE = re.compile(
    r'\b(yesterday|today|last week|last month|last year|next week|next month|next weekend|'
    r'last weekend|this week|this morning|recently|just now|a few days? ago|earlier today|'
    r'last (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|'
    r'next (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b',
    re.IGNORECASE)


class QueryEngine:
    def __init__(self, wiki: WikiStorage):
        self.wiki = wiki

    def query(self, question: str, recent_turns: list[dict] = None) -> QueryTrace:
        # ── TIERED ReAct: each tier does Thought → Act(answer attempt) → Judge ──
        # The 3-tier hierarchy is kept (summary → pages → raw turns). What's new is an
        # explicit per-tier cycle: attempt an answer, then a SEPARATE call judges whether
        # that answer is clear & specific (not vague). A clear answer surfaces and stops;
        # a vague/absent one descends. Tier 2 may try up to MAX_PAGE_ATTEMPTS distinct
        # pages before dropping to raw turns. Bounded by the 3 tiers — no open-ended loop.

        summary = self.wiki.get_summary()
        catalog_dict = self.wiki.get_catalog()   # name → one-line description
        trace = QueryTrace(query=question, summary_shown=summary)

        # ===== TIER 1: SUMMARY =====
        _tok_before = trace.tokens_est
        ans = self._attempt_answer(
            question,
            context=f"MEMORY SUMMARY:\n{summary or '(empty)'}",
            source_kind="summary", trace=trace)
        verdict = self._judge_clear(question, ans, trace)
        trace.tokens_t1 += trace.tokens_est - _tok_before
        trace.steps.append({"tier": 1, "source": "summary", "answer": ans, "verdict": verdict,
                             "tokens": trace.tokens_est - _tok_before})

        # When Tier 1 fails (vague/absent), extract summary pointers to seed Tier 2 picks.
        # When Tier 1 succeeds (clear), keep the fast path — avoids token waste and
        # preserves adversarial handling (Tier 1 "not available" is correctly clear).
        summary_picks = []
        if verdict != "clear":
            summary_picks = self._pointers_for_question(question, summary)

        if verdict == "clear":
            trace.hops = 1
            trace.answer = ans
            return trace

        # ===== TIER 2: PAGES (survey catalog → pick → attempt, up to N page picks) =====
        # Inference questions ("would/likely/might") need the entity's complete character
        # profile — catalog-based routing lands on the wrong pages. Load everything at once.
        is_inference = bool(set(question.lower().split()) & _INFERENCE_WORDS)

        retrieved_pages = {}
        tried_pages = set()
        absent_streak = 0   # consecutive T2 attempts yielding nothing at all
        for attempt in range(MAX_PAGE_ATTEMPTS):
            seeded_by_summary = False
            partial_answer = ""

            if attempt == 0 and is_inference:
                # Burst-load all wiki pages for inference, but entity-sort so the cap
                # (MAX_TOTAL_PAGES) hits entity-relevant pages first.
                # Without sorting, alphabetical order excluded ALL Melanie pages for
                # every inference question (10 Caroline pages filled the cap first).
                all_page_names = [n for n in self.wiki.get_all_pages() if n not in tried_pages]
                relevant = [
                    n for n in all_page_names if " - " in n and any(
                        self._entity_in_question(p.strip(), question)
                        for p in re.split(r'\s+and\s+', n.split(" - ")[0],
                                          flags=re.IGNORECASE) if p.strip())]
                other = [n for n in all_page_names if n not in set(relevant)]
                picks = relevant + other
            elif attempt == 0 and summary_picks:
                picks = summary_picks
                summary_picks = []
                seeded_by_summary = True
            else:
                best_t2 = [s["answer"] for s in trace.steps
                           if s.get("tier") == 2 and s.get("answer")
                           and "not available" not in (s.get("answer") or "").lower()]
                if best_t2:
                    partial_answer = best_t2[-1][:200]
                _tok_before_pick = trace.tokens_est
                # Descriptions on the first real pick (nothing or only summary-seeded
                # pages tried so far); names-only after that — partial_answer steers.
                picks = self._pick_pages(question, catalog_dict, tried_pages, retrieved_pages, trace,
                                         partial_answer=partial_answer,
                                         with_descriptions=len(tried_pages) <= 2)
                trace.tokens_t2_pick += trace.tokens_est - _tok_before_pick
            picks = [p for p in picks if p and p.lower() != "none"]
            if not picks:
                break
            newly = {}
            cap_hit = False
            for name in picks:
                tried_pages.add(name)
                if name in retrieved_pages:
                    continue
                if len(retrieved_pages) >= MAX_TOTAL_PAGES:
                    cap_hit = True
                    break
                content = self.wiki.read_page(name)
                if not content:
                    matches = self.wiki.find_pages(name.split())
                    if matches:
                        mn, mc = next(iter(matches.items()))
                        content, name = mc, mn
                if content:
                    retrieved_pages[name] = content
                    newly[name] = content
                    for link in self.wiki.get_wikilinks(content)[:1]:  # was 3; fewer to preserve cap budget
                        if len(retrieved_pages) >= MAX_TOTAL_PAGES:
                            cap_hit = True
                            break
                        if link not in retrieved_pages:
                            lc = self.wiki.read_page(link)
                            if lc:
                                retrieved_pages[link] = lc
                                newly[link] = lc
            trace.pages_retrieved = list(retrieved_pages.keys())
            if not newly:
                continue  # picks yielded nothing new; try a different page

            # Send ALL accumulated pages for the answer attempt.
            # Factual questions: entity-scope strips the other entity's pages (prevents
            # substitution hallucinations). Inference questions: don't scope — cross-entity
            # context is required (Melanie's LGBTQ-ally status is defined via Caroline pages).
            answer_pages = retrieved_pages if is_inference else self._entity_scoped_pages(question, retrieved_pages)
            all_text = "\n\n".join(f"=== {n} ===\n{c}" for n, c in answer_pages.items())
            # Grounding: the pages located the evidence — fetch the raw turns they cite
            # and answer from the original wording (span extraction), with the page text
            # supplying resolved absolute dates. Inference questions skip grounding:
            # they need the curated character profile, not verbatim spans.
            cited = [] if is_inference else self._cited_turns(question, answer_pages)
            if cited:
                turns_text = "\n".join(
                    f"[{t['dia_id']}{' | ' + t['date'] if t.get('date') else ''}] "
                    f"{t['speaker']}: {t['text']}" for t in cited)
                context = (f"WIKI NOTES:\n{all_text}\n\n"
                           f"CONVERSATION EXCERPTS (original wording of the turns the notes cite):\n"
                           f"{turns_text}")
                source_kind = "wiki notes and conversation excerpts"
            else:
                context = f"PAGES:\n{all_text}"
                source_kind = "pages"
            _tok_before_ans = trace.tokens_est
            ans = self._attempt_answer(question, context=context,
                                       source_kind=source_kind, trace=trace,
                                       is_inference=is_inference,
                                       grounded=bool(cited))
            verdict = self._judge_clear(question, ans, trace)
            _step_tok = trace.tokens_est - _tok_before_ans
            trace.tokens_t2_ans += _step_tok
            trace.steps.append({
                "tier": 2,
                "source": f"pages:{picks}",
                "answer": ans,
                "verdict": verdict,
                "seeded": seeded_by_summary,   # True = pointer-seeded (Item 2 fired)
                "partial": partial_answer[:80] if partial_answer else "",  # chain context (Item 1)
                "cap_hit": cap_hit,            # True = MAX_TOTAL_PAGES limit reached
                "cited_turns": [t["dia_id"] for t in cited],  # grounding evidence used
                "tokens": _step_tok,
            })
            if verdict == "clear":
                trace.hops = 2
                trace.answer = ans
                return trace
            # Two consecutive attempts that found NOTHING (absent, not even a vague
            # partial) mean the wiki doesn't have it — more page picks won't change
            # that. Drop to Tier 3 now instead of burning further attempts. A 'vague'
            # verdict resets the streak: partial info means we're on the right trail.
            if verdict == "absent":
                absent_streak += 1
                if absent_streak >= 2:
                    break
            else:
                absent_streak = 0
            # else: vague/absent → loop tries a DIFFERENT page (re-try), up to the cap

        # ===== TIER 3: RAW TURNS =====
        source_turns = self._gather_raw(question, retrieved_pages, trace=trace)
        if source_turns:
            turns_text = "\n".join(
                f"[{t['date']}] {t['speaker']}: {t['text']}" if t.get('date')
                else f"{t['speaker']}: {t['text']}" for t in source_turns)
            _tok_before_t3 = trace.tokens_est
            ans = self._attempt_answer(question, context=f"CONVERSATION EXCERPTS:\n{turns_text}",
                                       source_kind="raw", trace=trace,
                                       is_inference=is_inference,
                                       dated=any(t.get('date') for t in source_turns))
            # Raw turns carry relative dates that can't be resolved ("yesterday", "last week").
            # For "when" questions, refuse relative-date answers — they're wrong by definition.
            q_start = question.strip().lower()
            if (q_start.startswith(("when ", "what date", "what day", "what time"))
                    and _RELATIVE_DATE_RE.search(ans)):
                ans = "This information is not available."
            verdict = self._judge_clear(question, ans, trace)
            trace.tokens_t3 += trace.tokens_est - _tok_before_t3
            trace.steps.append({"tier": 3, "source": "raw", "answer": ans, "verdict": verdict,
                                 "tokens": trace.tokens_est - _tok_before_t3})
            trace.pages_retrieved += [f"[raw:{t['dia_id']}]" for t in source_turns]
            trace.hops = 3
            if verdict == "clear":
                trace.answer = ans
            else:
                trace.answer = self._best_vague_or_na(trace, is_inference)
            return trace

        trace.hops = 3
        trace.answer = self._best_vague_or_na(trace, is_inference)
        return trace

    # ── ReAct helpers ──

    def _attempt_answer(self, question, context, source_kind, trace,
                        is_inference: bool = False, grounded: bool = False,
                        dated: bool = False):
        """Act: attempt a TERSE answer from the given context only.
        grounded=True means the context pairs WIKI NOTES with the raw CONVERSATION
        EXCERPTS they cite — the answer should be a span copied from the excerpts,
        except dates, which must come from the notes (absolute, already resolved).
        dated=True means excerpts carry the session date they were spoken on, so
        relative wording in them is resolvable."""
        _binary_starters = (
            "would ", "could ", "is ", "are ", "does ", "did ",
            "has ", "have ", "will ", "was ", "were ", "can ",
        )
        _is_binary_q = is_inference and question.lower().strip().startswith(_binary_starters)

        if is_inference:
            _common = (
                "INFERENCE REQUIRED: This question asks you to reason about what someone "
                "would likely do, believe, or prefer. You MUST draw a logical conclusion "
                "from the facts in the context — do NOT output 'not available' when the "
                "context contains relevant facts about the person's values, personality, "
                "goals, experiences, or preferences. Reason from the evidence:\n"
                "- Stated interests, habits, and values extend to closely related ones.\n"
                "- A negative experience makes a repeat unlikely; an ongoing commitment "
                "makes abandoning it unlikely.\n"
                "- Supporting a group is not the same as belonging to it: if someone has "
                "never identified as a member, 'are they a member?' is likely No.\n"
                "- If the question asks about a preference or goal the person has NEVER "
                "mentioned, while the context shows a DIFFERENT stated preference or goal "
                "in that same area of life, answer 'Likely no' — do NOT output 'not "
                "available'; the absence plus the differing stated preference IS the "
                "evidence.\n"
            )
            if _is_binary_q:
                _fmt = (
                    "ANSWER FORMAT: This is a yes/no question — output 'Likely yes' or "
                    "'Likely no' followed by a brief reason (one phrase). Never output bare "
                    "'Yes' or 'No' — the word 'Likely' is required.\n"
                    "A 'Likely yes/no + reason' backed by evidence is always correct; "
                    "refusing to infer when context exists is the wrong behavior.\n"
                )
            else:
                _fmt = (
                    "ANSWER FORMAT: This question asks for a specific value, name, list, or "
                    "description — output the concrete inferred answer directly. Do NOT output "
                    "'Likely yes' or 'Likely no'; that format is only for yes/no questions. "
                    "Example: 'What might her degree be in?' → 'Political science' not "
                    "'Likely yes — she studied politics.'\n"
                    "A specific inference backed by evidence is always correct; refusing to "
                    "infer when context exists is the wrong behavior.\n"
                )
            inference_rule = _common + _fmt
        else:
            inference_rule = (
                "If the question asks what someone would 'likely' do or what they might prefer, "
                "you MAY reason from the facts in the text to draw a logical conclusion — "
                "but never fabricate facts not supported by the provided text.\n"
            )
        presupposition = (
            "" if is_inference else
            "PREMISE CHECK: Before answering, verify what the question assumes against the "
            "provided text. Decide on EVIDENCE, not on question type:\n"
            "- If the text STATES the asked-for fact (including a stated meaning, reason, or "
            "feeling), answer with it — even for subjective questions.\n"
            "- If the text CONTRADICTS the question's premise (the event happened to someone "
            "else, the assumed action never occurred), output exactly: This information is "
            "not available.\n"
            "- If the text neither confirms the assumed event/situation nor states the "
            "asked-for fact, do NOT fill the gap with a plausible guess from general "
            "knowledge — output exactly: This information is not available.\n"
            "- A single brief mention is not evidence of an ongoing habit, fandom, or "
            "status.\n"
        )
        _is_temporal_q = any(
            question.lower().strip().startswith(w)
            for w in ("when ", "what date", "what day", "what month", "what year",
                      "which date", "which day", "which month", "which year",
                      "around which", "around what")
        )
        temporal_rule = (
            "TEMPORAL COMPLETENESS: For questions about when something happened, output "
            "the date AND the key associated detail (location, activity, or person) if "
            "the context provides it. Example: 'July 5th, at the museum' not just 'July 5th'.\n"
            if _is_temporal_q else ""
        )
        grounded_rule = (
            "GROUNDED ANSWERING: The context pairs WIKI NOTES (curated facts with absolute "
            "dates) with CONVERSATION EXCERPTS (the speakers' original words, cited by the "
            "notes).\n"
            "- Answer with a SHORT SPAN copied from a CONVERSATION EXCERPT whenever the "
            "excerpt contains the answer — use the speaker's exact words, do not rephrase.\n"
            "- EXCEPTION for dates: prefer absolute dates from the WIKI NOTES. If only an "
            "excerpt has the timing and it uses relative wording, resolve it against the "
            "session date the excerpt is tagged with.\n"
            "- If notes and excerpts disagree on a date, the WIKI NOTES win; on anything "
            "else, the EXCERPTS win.\n"
            if grounded else ""
        )
        dated_rule = (
            "DATED EXCERPTS: Each conversation excerpt is tagged with the date it was "
            "spoken on. Resolve relative time wording against that date — 'last week' "
            "spoken on 8 May 2023 means 'the week before 8 May 2023'; 'yesterday' means "
            "7 May 2023. Never output unresolved relative wording as an answer.\n"
            if (dated or grounded) else ""
        )
        sys = (
            f"You answer strictly from the provided {source_kind}, and nothing else.\n"
            "Output ONLY the answer — a date, name, place, or short phrase. Never restate "
            "the question, never explain, never write 'The question asks' or 'I found'.\n"
            "CONCISENESS: Answer in the minimum words that fully answer the question. "
            "For factual questions output a phrase or list, not a sentence. "
            "For 'how many' questions output a numeral ('2', not 'twice' or 'two').\n"
            + grounded_rule
            + dated_rule
            + temporal_rule
            + inference_rule
            + presupposition +
            "If the provided text contains no relevant facts at all, output exactly: "
            "This information is not available.\n"
            + (
            "ENTITY FOCUS: Focus on facts about the named entity. You may draw logical "
            "conclusions even if the exact conclusion is not explicitly stated — inference "
            "from the evidence is required and correct.\n"
            if is_inference else
            "ENTITY ALIGNMENT: If the question asks about a specific person or entity, "
            "only use facts that explicitly belong to that exact entity. Do NOT answer "
            "using facts about a different person even if the topic seems related — "
            "that also counts as 'not available'."
            ))
        usr = f"{context}\n\nQUESTION: {question}\n\nAnswer:"
        out = query_call([{"role": "system", "content": sys},
                          {"role": "user", "content": usr}], temperature=0.0)
        trace.tokens_est += estimate_tokens(sys) + estimate_tokens(usr) + estimate_tokens(out)
        return out.strip()

    def _judge_clear(self, question, answer, trace):
        """Judge (separate call): is the answer RESPONSIVE to the question — or not?
        Accepts on-topic, substantive answers even when approximate ('a few years ago',
        'last week', 'by dancing'), because many real answers are legitimately approximate.
        Only descends when the answer is empty or genuinely does not address the question."""
        if not answer or "not available" in answer.lower():
            return "absent"
        sys = (
            "You judge whether an ANSWER is RESPONSIVE to a QUESTION — that is, whether it "
            "actually addresses what was asked, using whatever specificity the source "
            "provided.\n"
            "Reply with exactly one word:\n"
            "GOOD — the answer addresses the question and is substantive. Accept it EVEN IF "
            "it is approximate or relative ('a few years ago', 'last week', 'by dancing', "
            "'contemporary'), as long as that genuinely responds to what was asked. An "
            "approximate-but-on-point answer is GOOD.\n"
            "POOR — the answer does NOT address the question, is empty, or just restates the "
            "question without answering it. Only use POOR when the answer fails to respond.\n"
            "Reply only GOOD or POOR.")
        usr = f"QUESTION: {question}\nANSWER: {answer}\n\nVerdict:"
        out = query_call([{"role": "system", "content": sys},
                          {"role": "user", "content": usr}], temperature=0.0)
        trace.tokens_est += estimate_tokens(sys) + estimate_tokens(usr) + estimate_tokens(out)
        return "clear" if "GOOD" in out.upper() else "vague"

    def _best_vague_or_na(self, trace: QueryTrace, is_inference: bool) -> str:
        """For inference questions, salvage the best non-absent vague answer from prior steps
        rather than silently dropping to 'not available'. The judge may rate an inference
        answer POOR (vague) because it's brief ('No') — but that's still the right answer."""
        if is_inference:
            for s in reversed(trace.steps):
                a = s.get("answer", "")
                if s.get("verdict") == "vague" and a and "not available" not in a.lower():
                    return a
        return "This information is not available."

    def _pointers_for_question(self, question: str, summary: str) -> list[str]:
        """Extract [→Page Name] pointers from the summary, ranked by topic relevance to question.
        Matches entity AND topic keywords — entity-only matching loads irrelevant pages when a
        wiki has many pages per entity (e.g., 15 Caroline pages, all match on entity alone)."""
        STOP = {"what","when","where","who","why","how","did","does","is","the","a","an",
                "of","to","in","on","for","and","or","was","were","have","has","had","do"}
        q_lower = question.lower()
        q_words = {w.strip("?.,!") for w in q_lower.split()
                   if w.strip("?.,!") not in STOP and len(w.strip("?.,!")) > 2}
        all_ptrs = re.findall(r'\[→([^\]]+)\]', summary)
        scored, seen = [], set()
        for ptr in all_ptrs:
            if " - " not in ptr or ptr in seen:
                continue
            entity = ptr.split(" - ")[0].strip().lower()
            topic  = ptr.split(" - ")[1].strip().lower()
            if entity not in q_lower:
                continue
            topic_words = set(re.findall(r'[a-z]+', topic))
            overlap = len(q_words & topic_words)
            scored.append((overlap, ptr))
            seen.add(ptr)
        scored.sort(reverse=True)
        # Prefer topic-matching pointers; fall back to top entity match if none overlap
        relevant = [p for s, p in scored if s > 0]
        if not relevant and scored:
            relevant = [scored[0][1]]
        return relevant[:2]  # cap at 2 to preserve budget for model-guided iterations

    @staticmethod
    def _entity_in_question(entity: str, question: str) -> bool:
        """True if the question names the entity, tolerating shortened forms and
        near-miss spellings: a capitalized question word of ≥3 chars counts when
        either string is a prefix of the other ('Mel' → 'Melanie') or they are one
        edit apart with the same first letter ('John' → 'Jon'). The capitalization
        requirement keeps common nouns ('car') from matching entity names.
        General replacement for a hardcoded nickname map."""
        e = entity.lower()
        if e in question.lower():
            return True

        def _edit1(a: str, b: str) -> bool:
            """Levenshtein distance ≤ 1."""
            if abs(len(a) - len(b)) > 1:
                return False
            if len(a) > len(b):
                a, b = b, a
            i = j = diff = 0
            while i < len(a) and j < len(b):
                if a[i] == b[j]:
                    i += 1; j += 1
                    continue
                diff += 1
                if diff > 1:
                    return False
                if len(a) == len(b):
                    i += 1
                j += 1   # skip the extra/substituted char in the longer string
            return diff + (len(b) - j) <= 1

        for w in re.findall(r'\b[A-Z][a-zA-Z]*\b', question):
            wl = w.lower()
            if len(wl) < 3:
                continue
            if e.startswith(wl) or wl.startswith(e):
                return True
            if wl[0] == e[0] and _edit1(wl, e):
                return True
        return False

    def _entity_scoped_pages(self, question: str, pages: dict) -> dict:
        """Filter pages to only those owned by the entity explicitly named in the question.
        Page names follow 'Entity - Topic' or 'A and B - Topic' conventions.
        Only filters when exactly one entity name appears in the question — avoids
        over-filtering on multi-entity or entity-free questions."""
        if not pages:
            return pages
        entities: dict[str, str] = {}  # lower → canonical
        for name in pages:
            if " - " in name:
                left = name.split(" - ")[0].strip()
                for part in re.split(r'\s+and\s+', left, flags=re.IGNORECASE):
                    part = part.strip()
                    if part:
                        entities[part.lower()] = part
        if len(entities) <= 1:
            return pages
        mentioned = [ek for ek in entities
                     if self._entity_in_question(ek, question)]
        if len(mentioned) != 1:
            return pages  # 0 or 2+ entities — can't safely filter
        target = mentioned[0]
        filtered = {}
        for name, content in pages.items():
            if " - " not in name:
                filtered[name] = content
                continue
            left = name.split(" - ")[0].strip()
            page_ents = [p.strip().lower()
                         for p in re.split(r'\s+and\s+', left, flags=re.IGNORECASE) if p.strip()]
            if target in page_ents:
                filtered[name] = content
        return filtered if filtered else pages  # safety: never return empty

    def _cited_turns(self, question: str, pages: dict) -> list[dict]:
        """Fetch the raw turns cited inline ([DX:Y]) by the given pages.
        The pages locate the evidence; the cited turns ARE the evidence — the answer
        is extracted from their original wording, not from the page's paraphrase.
        Deterministic: regex scrape + dict lookup, no LLM."""
        ids = []
        for content in pages.values():
            ids.extend(_CITATION_RE.findall(content))
        ids = list(dict.fromkeys(ids))
        if not ids:
            return []
        turns = self.wiki.get_turns(ids)
        if len(turns) > MAX_CITED_TURNS:
            q_words = {w.strip("?.,!'\"").lower() for w in question.split()}
            q_words = {w for w in q_words if len(w) > 2}
            turns.sort(key=lambda t: sum(1 for w in q_words if w in t["text"].lower()),
                       reverse=True)
            turns = turns[:MAX_CITED_TURNS]
        # Chronological order reads naturally and keeps multi-hop chains intact.
        turns.sort(key=lambda t: tuple(int(x) for x in re.findall(r'\d+', t["dia_id"])))
        return turns

    def _pick_pages(self, question, catalog_dict: dict, tried_pages, retrieved_pages, trace,
                    partial_answer: str = "", with_descriptions: bool = True):
        """Thought+Act: pick the best-matching untried page(s) using the annotated catalog.
        Each catalog entry is 'Page Name: one-line description' — the model matches on
        content, not just name, so first-pick accuracy is much higher than name-only routing.
        partial_answer guides multi-hop: 'I found X — what page has the completing fact?'
        with_descriptions=False sends names only — descriptions earn their cost on the
        FIRST pick; later picks are steered by partial_answer, so re-sending the full
        catalog each iteration is pure token cost."""
        tried = ", ".join(sorted(tried_pages)) if tried_pages else "(none yet)"
        partial_note = (f"\nPARTIAL ANSWER SO FAR: \"{partial_answer}\"\n"
                        "Pick the page most likely to have the REMAINING information needed."
                        if partial_answer else "")
        # Format catalog with descriptions; exclude already-tried pages.
        # Use "---" block separator so long multi-sentence descriptions are readable.
        if with_descriptions:
            catalog_lines = [f"PAGE: {name}\n  {desc}"
                             for name, desc in catalog_dict.items()
                             if name not in tried_pages]
        else:
            catalog_lines = [f"PAGE: {name}" for name in catalog_dict
                             if name not in tried_pages]
        catalog_text = "\n---\n".join(catalog_lines) if catalog_lines else "(none available)"
        sys = (
            "You are choosing which wiki page most likely contains the answer to a question.\n"
            "Each page is shown as 'Name: paragraph description of its content (facts, dates, topics)'.\n"
            "Read the descriptions carefully — key facts such as identity terms, dates, names, and events\n"
            "appear anywhere in the paragraph, not just the first line.\n"
            "Pick the page(s) whose description best matches what the question is asking.\n"
            "Do NOT pick pages already tried.\n"
            "Reply with ONLY the exact page name(s), comma-separated, as listed.\n"
            "If no untried page plausibly matches, reply exactly: NONE")
        usr = (f"QUESTION: {question}{partial_note}\n\n"
               f"AVAILABLE PAGES:\n{catalog_text}\n\n"
               f"ALREADY TRIED: {tried}\n\nPage(s) to open:")
        out = query_call([{"role": "system", "content": sys},
                          {"role": "user", "content": usr}], temperature=0.0)
        trace.tokens_est += estimate_tokens(sys) + estimate_tokens(usr) + estimate_tokens(out)
        if not trace.reasoning:
            trace.reasoning = f"page pick: {out.strip()}"
        if out.strip().upper().startswith("NONE"):
            return []
        return [p.strip().strip('"').strip("'") for p in out.split(",") if p.strip()]

    def _gather_raw(self, question, retrieved_pages, trace=None):
        """Tier 3 retrieval. Combines two sources then re-ranks by keyword score.
        (a) Bound set: turns cited in Tier 2 pages (provenance-scoped, high precision).
        (b) Global keyword search: supplements when Tier 2 loaded the wrong page and the
            bound set is therefore from the wrong context. Re-ranking after merge ensures
            the best-matching turn wins regardless of which source it came from.
        Records diagnostics into trace.t3_debug for validation."""
        MAX_SOURCE_TURNS = 12  # raised from 8: feed more bound candidates to the model
        stop = {"what","when","where","who","why","how","did","does","is","the","a","an",
                "of","to","in","on","for","and","or","was","were","have","has","had","do",
                "with","about","that","this","would","could","should","still","want","she",
                "her","him","his","they","their","from","over","ago","long","much","many",
                "will","been","not","any","more","than","even","just","also","its","your",
                "are","some","which","been","get","got","let","put","via","per","all","by"}
        def _stem(w: str) -> str:
            """Minimal suffix-stripping so 'camped'/'camping'/'camps' all match 'camp'."""
            w = re.sub(r"ing$|ed$|s$", "", w)
            return w if len(w) > 2 else w + "s"  # don't over-strip short words

        kw = []
        for _w in question.split():
            _w = re.sub(r"'s$|n't$|'re$|'ve$|'ll$", "", _w.strip("?.,!'\"").lower())
            if _w and _w not in stop and len(_w) > 2:
                kw.append(_stem(_w))
        kw = list(dict.fromkeys(kw))  # deduplicate after stemming

        def _score_turn(text: str) -> int:
            # Stemmed keywords are already substrings of their inflected forms
            # ('camp' in 'camping', 'mov' in 'moved') — plain substring match suffices.
            text_lower = text.lower()
            return sum(1 for k in kw if k in text_lower)

        turns = []
        path = "none"
        prov_ids = []
        if retrieved_pages:
            # Entity-scope the bound set: extract turn IDs only from pages that belong to
            # the entity the question asks about. Prevents Melanie's turn IDs from leaking
            # into a bound set for a question about Caroline (and vice versa).
            scoped = self._entity_scoped_pages(question, retrieved_pages)
            ids = []
            for content in scoped.values():
                m = re.search(r'sources:\s*\[([^\]]+)\]', content)
                if m:
                    ids.extend(x.strip() for x in m.group(1).split(",") if x.strip())
                ids.extend(_CITATION_RE.findall(content))  # inline [DX:Y] citations
            ids = list(dict.fromkeys(ids))
            prov_ids = ids
            if ids:
                fetched = self.wiki.get_turns(ids)
                if len(fetched) > MAX_SOURCE_TURNS:
                    fetched.sort(key=lambda t: _score_turn(t["text"]), reverse=True)
                    turns = fetched[:MAX_SOURCE_TURNS]
                else:
                    turns = fetched
                if turns:
                    path = "within-bound-set"
        # Determine the single entity the question is about (for speaker-level filtering below).
        mentioned_ents: dict = {}
        for name in retrieved_pages:
            if " - " in name:
                left = name.split(" - ")[0].strip()
                for part in re.split(r'\s+and\s+', left, flags=re.IGNORECASE):
                    part = part.strip()
                    if part:
                        mentioned_ents[part.lower()] = part
        named = [ek for ek in mentioned_ents
                 if self._entity_in_question(ek, question)]

        # Supplement with global keyword search, then re-rank the combined pool.
        global_turns = self.wiki.search_turns(kw, max_results=8)
        if global_turns:
            bound_ids = {t.get("dia_id") for t in turns}
            supplemental = [t for t in global_turns if t.get("dia_id") not in bound_ids]
            if supplemental:
                turns.extend(supplemental)
                path = (path + "+global") if path else "global-only"

        # Filter the FULL pool (bound-set + global) to speaker=target when the question
        # names exactly one entity. Prevents Melanie's turns ("Been reading and painting
        # during my pottery break") from answering questions about Caroline.
        # Uses speaker identity only — "caroline" appearing as an addressee in Melanie's
        # turns must not let those turns through.
        if len(named) == 1:
            target_ent = named[0]
            entity_turns = [t for t in turns if t.get("speaker", "").lower() == target_ent]
            if entity_turns:
                turns = entity_turns
        # Re-rank using stemmed scoring so the best-matching turn wins regardless of source.
        if kw:
            turns.sort(key=lambda t: _score_turn(t["text"]), reverse=True)
        turns = turns[:MAX_SOURCE_TURNS]
        if trace is not None:
            trace.t3_debug = {
                "path": path,
                "keywords": kw,
                "bound_set_size": len(prov_ids),
                "provenance_ids_available": prov_ids,
                "retrieved_ids": [t.get("dia_id") for t in turns],
                "excerpts": [f"[{t.get('dia_id')}] {t['speaker']}: {t['text']}" for t in turns],
            }
        return turns



    @staticmethod
    def _wants_to_look(response: str) -> bool:
        """The single, uniform descent signal — same at every tier."""
        return response.strip().upper().startswith("LOOK:")

    @staticmethod
    def _parse_look_targets(response: str) -> list[str]:
        body = response.split(":", 1)[1].strip()
        return [t.strip().strip('"').strip("'") for t in body.split(",") if t.strip()]


# ─────────────────────────────────────────────
# Memory Manager
# ─────────────────────────────────────────────

class HLMAMemory:
    def __init__(self, wiki_dir: str = "wiki"):
        self.wiki = WikiStorage(wiki_dir)
        self.compiler = Compiler(self.wiki)
        self.query_engine = QueryEngine(self.wiki)
        self.all_turns: list[dict] = []
        self.recent_turns: list[dict] = []

    def reset(self):
        self.wiki.reset()
        self.all_turns = []
        self.recent_turns = []

    def ingest_session(self, turns: list[dict], session_label: str = "",
                       session_date: str = ""):
        # Stamp every turn with its session date before anything stores it —
        # the date is provenance, and downstream (store_turns, dated excerpts)
        # reads it from the turn dict.
        if session_date:
            for t in turns:
                t.setdefault("date_time", session_date)
        self.all_turns.extend(turns)
        self.recent_turns = turns[-config.RECENT_WINDOW_TURNS:]
        print(f"    Compiling {session_label} ({len(turns)} turns)...")
        result = self.compiler.compile_session(turns, session_label)
        print(f"    → {result['facts_extracted']} facts | +{result['pages_created']}/-{result.get('net_removed',0)} pages "
              f"(enforce={result.get('enforced',0)} split={result.get('split',0)} consolidate={result.get('consolidated',0)}) "
              f"→ {result.get('total_pages','?')} total")
        self.compiler.generate_summary()
        self.compiler.generate_catalog()
        return result

    def generate_summary(self):
        """Generate summary after all sessions are compiled."""
        return self.compiler.generate_summary()

    def lint(self):
        """Run post-compilation health check."""
        return self.compiler.lint()

    def query(self, question: str) -> QueryTrace:
        return self.query_engine.query(question, self.recent_turns)
