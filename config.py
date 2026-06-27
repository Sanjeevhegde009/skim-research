"""HLMA MVP Configuration"""

# ── Compiler (frontier model for entity pages, summaries, scoring) ──
COMPILER_PROVIDER = "openai"   # "anthropic", "openai", "openai_compatible"
COMPILER_BASE_URL = ""
COMPILER_API_KEY_ENV = "OPENAI_API_KEY"
COMPILER_MODEL = "gpt-4o-mini"

# ── Query model ──
# Set QUERY_PROVIDER to "ollama" for local SLM, or same as compiler for API-based
QUERY_PROVIDER = "openai"
QUERY_BASE_URL = ""
QUERY_API_KEY_ENV = "OPENAI_API_KEY"
QUERY_MODEL = "gpt-4.1-mini"

# Ollama settings (only used when QUERY_PROVIDER = "ollama")
OLLAMA_URL = "http://localhost:11434/api/chat"

# ── Memory settings ──
RECENT_WINDOW_TURNS = 6

# ── LoCoMo ──
LOCOMO_PATH = "locomo10.json"
QA_CATEGORIES = {1: "single-hop", 2: "temporal", 3: "multi-hop", 4: "open-domain", 5: "adversarial"}

# ── LongMemEval (second benchmark; per-question haystacks that overflow context) ──
# Download longmemeval_s.json from https://github.com/xiaowu0162/longmemeval and place at repo root.
LONGMEMEVAL_PATH = "longmemeval_s.json"
