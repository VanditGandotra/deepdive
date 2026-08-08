# DeepDive Research

Equity deep dives (ticker mode) and product intelligence (URL mode) in a single Streamlit dashboard.

## Setup

```bash
cd deepdive
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Fill in ANTHROPIC_API_KEY, API_NINJAS_KEY, and (optionally) NEWSAPI_KEY
```

## Run

```bash
streamlit run app.py
```

## Architecture

```
app.py                  # Streamlit entry, mode detection, tab routing
config.py               # All constants, model IDs, TTLs, prompt versions
llm.py                  # Gateway: tiering, caching, streaming, batch, tool-use structured outputs
data/
  cache.py              # SQLite: general cache, LLM content-hash cache, run snapshots, call logs
  resilience.py         # retry decorator, fallback chains
  market.py             # yfinance: prices, fundamentals, estimates, insiders, holders, SI, news
  edgar.py              # edgartools + direct XBRL API fallback
  transcripts.py        # API Ninjas earnings transcripts
  news.py               # yfinance + NewsAPI, deduped
  webintel.py           # URL mode: sitemap/RSS/path discovery (Phase 9)
  docs_crawler.py       # URL mode: docs/llms.txt/image harvest (Phase 10)
analysis/
  schemas.py            # All Pydantic v2 models
  ratios.py             # Phase 2
  reconcile.py          # Phase 2
  business.py           # Phase 3
  calls.py              # Phase 3
  sentiment.py          # Phase 3
  kpis.py               # Phase 5
  quality.py            # Phase 5
  expectations.py       # Phase 5
  positioning.py        # Phase 5
  news_impact.py        # Phase 4
  thesis.py             # Phase 6
  delta.py              # Phase 7
  memo.py               # Phase 6
  company.py            # Phase 9
  product.py            # Phase 10
ui/
  components.py         # Freshness badges, citation badges, delta card, cost footer, streaming container
  charts.py             # Plotly builders: candlestick, sparklines, beat/miss, revenue, sentiment
evals/
  run_evals.py          # Phase 11: CLI runner, Batch API
  checks.py             # Phase 11: schema, extraction, grounding, regression, cost checks
  golden/               # Hand-labelled fixtures (synthetic stubs → swap in real data)
```

## Cost Architecture

- **Model tiering**: `claude-haiku-4-5-20251001` for extraction/classification; `claude-sonnet-4-6` for synthesis only
- **Prompt caching**: 1h TTL on system prompts and reusable context blocks via `cache_control`
- **Content-hash cache**: every LLM output stored in SQLite keyed by SHA-256(model + prompt_version + inputs)
- **Batch API**: 50% discount for non-interactive runs (evals, backfill, warm-cache)
- **Cost footer**: tokens + estimated $ per session in the UI

## .env keys

| Key | Required | Purpose |
|-----|----------|---------|
| `ANTHROPIC_API_KEY` | ✅ | Claude API |
| `API_NINJAS_KEY` | ✅ | Earnings call transcripts |
| `NEWSAPI_KEY` | optional | Supplement yfinance news |
| `EDGAR_USER_AGENT` | optional | SEC identity (default: `DeepDive Research vandit@deductive.ai`) |

## Build Phases

| Phase | Feature | Status |
|-------|---------|--------|
| 1 | Foundation: llm.py, cache, resilience, market, EDGAR, transcripts, news | ✅ |
| 2 | Ratios + reconciliation + charts (no LLM) | ✅ |
| 3 | Business explainer + deep call analysis (A/B/C) | ✅ |
| 4 | News impact monitor | ✅ |
| 5 | Analyst Mode: reverse DCF, KPIs, quality, positioning | ✅ |
| 6 | Thesis, red team, memo + grounding audits | ✅ |
| 7 | Delta engine: snapshots, diff, narrative card | ✅ |
| 8 | Ticker-mode assembly: full tabs, streaming, freshness, footer | ✅ |
| 9 | Company Intel URL mode | ✅ |
| 10 | Product Deep Dive URL mode | ✅ |
| 11 | Full eval suite on Batch API | 🔜 |

## Post-MVP Backlog

- Notion export: push memos to a research database with ticker/date/thesis properties
- URL-mode visual diff from stored snapshots
- Playwright rendering for JS-heavy sites + live screenshots
- Web-search-augmented Company Intel toggle
- Sector peer comparison for ratios
