"""Pre-earnings brief: auto-generated when next earnings ≤14 days away."""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Dict, List, Optional

import llm
from analysis.schemas import Fundamentals
from config import PROMPT_VERSIONS, SONNET, TTL_NEWS

logger = logging.getLogger(__name__)

_WINDOW_DAYS = 14  # show brief when earnings within this many days

_SYSTEM = llm.cached_system("""
You are a buy-side analyst preparing a portfolio manager for an upcoming earnings call.
Write a concise pre-earnings brief (under 400 words) covering exactly these sections:
## What To Watch
## Consensus vs Street Whisper
## Key KPIs To Monitor
## Q&A Traps (questions analysts will likely press on)
## Risk Events (non-consensus scenarios)
Be specific: name metrics, cite prior quarter numbers where given, note if guidance was raised/lowered last quarter.
End with one sentence: the single most important thing to watch.
Disclaimer: This is research synthesis, not investment advice.
""")


def days_to_earnings(fund: Fundamentals) -> Optional[int]:
    if fund.next_earnings_date is None:
        return None
    today = date.today()
    delta = (fund.next_earnings_date - today).days
    return delta if delta >= 0 else None


def should_show_brief(fund: Fundamentals) -> bool:
    days = days_to_earnings(fund)
    return days is not None and days <= _WINDOW_DAYS


def build_pre_earnings_brief(
    ticker: str,
    fund: Fundamentals,
    beat_miss_records: List[Dict],
    estimates_raw: Dict,
    kpi_summaries: List[str],
    prior_guidance_note: str = "",
) -> str:
    """
    Stream a pre-earnings brief. Returns the full text (non-streaming for simplicity;
    callers can wrap in st.spinner).
    Cached for 6h via content-hash so re-runs on same day are free.
    """
    days = days_to_earnings(fund)
    earnings_date_str = fund.next_earnings_date.strftime("%B %d, %Y") if fund.next_earnings_date else "upcoming"

    # Beat/miss summary
    valid_bm = [r for r in beat_miss_records if r.get("eps_surprise_pct") is not None]
    beats = sum(1 for r in valid_bm if (r["eps_surprise_pct"] or 0) >= 0)
    avg_surp = (
        sum(r["eps_surprise_pct"] for r in valid_bm) / len(valid_bm) if valid_bm else None
    )
    bm_summary = (
        f"Beat EPS {beats}/{len(valid_bm)} quarters, avg surprise {avg_surp:+.1f}%"
        if valid_bm else "No beat/miss history available"
    )

    # Consensus estimates from yfinance eps_trend
    consensus_lines = []
    eps_trend = estimates_raw.get("eps_trend") or []
    if eps_trend:
        for row in eps_trend[:2]:
            period = row.get("period") or row.get("index") or ""
            current_est = row.get("current") or row.get("0q")
            if period and current_est:
                consensus_lines.append(f"EPS estimate ({period}): {current_est}")
    rev_est = estimates_raw.get("revenue_estimate") or []
    if rev_est:
        for row in rev_est[:1]:
            period = row.get("period") or row.get("index") or ""
            avg = row.get("avg")
            if period and avg:
                try:
                    consensus_lines.append(f"Revenue estimate ({period}): ${float(avg)/1e9:.2f}B")
                except Exception:
                    pass
    consensus_text = "\n".join(consensus_lines) if consensus_lines else "Consensus estimates unavailable"

    # Build context
    ctx_parts = [
        f"Company: {fund.name or ticker} ({ticker})",
        f"Earnings date: {earnings_date_str} ({days} days away)",
        f"Current price: ${fund.current_price:.2f}" if fund.current_price else "",
        f"Market cap: ${fund.market_cap/1e9:.1f}B" if fund.market_cap else "",
        f"\nBeat/miss history: {bm_summary}",
        f"\nConsensus estimates:\n{consensus_text}",
    ]
    if kpi_summaries:
        ctx_parts.append("\nKey KPIs from last call:\n" + "\n".join(f"- {k}" for k in kpi_summaries[:5]))
    if prior_guidance_note:
        ctx_parts.append(f"\nPrior guidance: {prior_guidance_note}")

    context = "\n".join(p for p in ctx_parts if p)

    cache_key_raw = f"pre_earnings:{ticker.upper()}:{fund.next_earnings_date}"
    import hashlib
    cache_key = "llmcache:" + hashlib.md5(cache_key_raw.encode()).hexdigest()

    from data.cache import get_cache_obj, set_cache_obj
    cached = get_cache_obj(cache_key)
    if cached:
        return cached

    messages = [
        {
            "role": "user",
            "content": [
                *llm.cached_content(context),
                llm.text_block(
                    f"Write the pre-earnings brief for {ticker} ahead of their {earnings_date_str} report."
                ),
            ],
        }
    ]
    try:
        text = llm.call(
            SONNET, messages,
            system=_SYSTEM,
            prompt_version="v1",
            max_tokens=1200,
        )
    except Exception as exc:
        logger.warning("pre_earnings brief failed for %s: %s", ticker, exc)
        return ""

    set_cache_obj(cache_key, text, TTL_NEWS)
    return text
