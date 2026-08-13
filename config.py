import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent

# ── Models ──────────────────────────────────────────────────────────────────
HAIKU = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-4-6"

# ── API Keys ─────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
API_NINJAS_KEY = os.getenv("API_NINJAS_KEY", "")
API_NINJAS_PREMIUM = os.getenv("API_NINJAS_PREMIUM", "").lower() == "true"
ROIC_API_KEY = os.getenv("ROIC_API_KEY", "")
FMP_API_KEY = os.getenv("FMP_API_KEY", "")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
PRODUCT_HUNT_TOKEN = os.getenv("PRODUCT_HUNT_TOKEN", "")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
EDGAR_USER_AGENT = os.getenv("EDGAR_USER_AGENT", "DeepDive Research vandit@deductive.ai")

# ── Storage ──────────────────────────────────────────────────────────────────
DB_PATH = ROOT / os.getenv("DEEPDIVE_DB_PATH", "cache.db")

# ── Cache TTLs (seconds) ─────────────────────────────────────────────────────
TTL_PRICES       = 3_600        # 1h
TTL_FUNDAMENTALS = 86_400       # 24h
TTL_ESTIMATES    = 86_400       # 24h
TTL_NEWS         = 21_600       # 6h
TTL_FILINGS      = 2_592_000   # 30d
TTL_TRANSCRIPTS  = 2_592_000   # 30d
TTL_WEB_PAGES    = 604_800     # 7d
TTL_IMAGES       = 2_592_000   # 30d
TTL_LLM          = 2_592_000   # 30d (content-hash cache)

# ── Prompt versions — bump to invalidate LLM content-hash cache ──────────────
PROMPT_VERSIONS: dict[str, str] = {
    "business_explainer":      "v1",
    "call_extraction_a":       "v2",  # bumped: signals field added to CallSummary
    "call_sentiment_b":        "v1",
    "call_synthesis_c":        "v1",
    "headline_classification": "v2",
    "state_of_play":          "v1",
    "kpi_extraction":          "v1",
    "quality_flags":           "v1",
    "thesis":                  "v1",
    "red_team":                "v1",
    "memo":                    "v1",
    "page_extraction":         "v1",
    "company_synthesis":       "v1",
    "product_explainer":       "v1",
    "screen_explanation":      "v1",
    "delta_narrative":         "v1",
}

# ── Cost per million tokens (approximate 2025 pricing — verify at console) ───
COST_PER_MTOK: dict[str, dict[str, float]] = {
    HAIKU: {
        "input":       0.80,
        "output":      4.00,
        "cache_write": 1.00,
        "cache_read":  0.08,
    },
    SONNET: {
        "input":       3.00,
        "output":      15.00,
        "cache_write": 3.75,
        "cache_read":  0.30,
    },
    "default": {
        "input":       3.00,
        "output":      15.00,
        "cache_write": 3.75,
        "cache_read":  0.30,
    },
}

# ── Budget & limits ───────────────────────────────────────────────────────────
SESSION_BUDGET_USD = float(os.getenv("DEEPDIVE_SESSION_BUDGET_USD", "5.0"))
EDGAR_RATE_LIMIT   = float(os.getenv("DEEPDIVE_EDGAR_RATE_LIMIT", "2.0"))  # req/s
WEB_RATE_LIMIT     = 2.0   # req/s for web intel
MAX_WEB_PAGES      = 40
MAX_DOCS_PAGES     = 25
MAX_IMAGES         = 15

# ── Transcript provider endpoints ───────────────────────────────────────────
API_NINJAS_TRANSCRIPT_URL    = "https://api.api-ninjas.com/v1/earningstranscript"
ROIC_TRANSCRIPT_LIST_URL     = "https://roic.ai/v3.0.0/earnings-calls"
ROIC_TRANSCRIPT_DETAIL_URL   = "https://roic.ai/v3.0.0/earnings-calls/{ecall_id}"
FMP_TRANSCRIPT_URL           = "https://financialmodelingprep.com/api/v3/earning_call_transcript/{symbol}"
FINNHUB_TRANSCRIPT_LIST_URL  = "https://finnhub.io/api/v1/stock/transcripts/list"
FINNHUB_TRANSCRIPT_URL       = "https://finnhub.io/api/v1/stock/transcripts"

# ── EDGAR endpoints ──────────────────────────────────────────────────────────
EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_XBRL_FACTS_URL  = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
EDGAR_COMPANY_TICKERS = "https://www.sec.gov/files/company_tickers.json"
