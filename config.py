"""pageindex-rag configuration"""

from pathlib import Path

# ── Directory layout (runtime artifacts live outside the source tree) ──
DATA_DIR        = Path("data")              # datasets (gitignored; see download_data.sh)
CACHE_DIR       = Path("cache")             # regenerated caches (gitignored)
RESULTS_DIR     = Path("results")           # eval outputs (gitignored)
INDEX_CACHE_DIR = CACHE_DIR / "pageindex"   # per-session table-of-contents index
EMB_CACHE_DIR   = CACHE_DIR / "embeddings"  # pi_rag turn embeddings

# ── Compiler (frontier model for entity pages, summaries, scoring) ──
COMPILER_PROVIDER = "openai"   # "anthropic", "openai", "openai_compatible"
COMPILER_BASE_URL = ""
COMPILER_API_KEY_ENV = "OPENAI_API_KEY"
COMPILER_MODEL = "gpt-4o-mini"

# ── Judge (answer scoring only) ──
# Defaults to the compiler, so existing runs are unchanged. Decoupled so the judge can stay on a
# frontier API while the compiler/indexer runs locally (a fully-local pipeline, judged fairly).
JUDGE_PROVIDER    = COMPILER_PROVIDER
JUDGE_BASE_URL    = COMPILER_BASE_URL
JUDGE_API_KEY_ENV = COMPILER_API_KEY_ENV
JUDGE_MODEL       = COMPILER_MODEL

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
LOCOMO_PATH = str(DATA_DIR / "locomo10.json")
QA_CATEGORIES = {1: "single-hop", 2: "temporal", 3: "multi-hop", 4: "open-domain", 5: "adversarial"}

# ── LongMemEval (second benchmark; per-question haystacks that overflow context) ──
# Fetched by download_data.sh from the official xiaowu0162/longmemeval-cleaned HF dataset.
LONGMEMEVAL_PATH = str(DATA_DIR / "longmemeval_s.json")
