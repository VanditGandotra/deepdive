"""Peer comps: multi-ticker ratio table + Sonnet narrative synthesis."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import llm
from analysis.schemas import Fundamentals, PeerComps, PeerRow
from config import HAIKU, PROMPT_VERSIONS, SONNET, TTL_FUNDAMENTALS
from data.cache import get_cache_obj, set_cache_obj

logger = logging.getLogger(__name__)

_SYSTEM_PEERS = llm.cached_system("""
You are a sell-side equity analyst writing a sector peer comparison note.
Given a table of financial ratios for a target company and its peers,
write one paragraph (under 150 words) identifying:
- where the target looks cheap vs peers (name the specific metrics)
- where it looks expensive vs peers
- whether the premium or discount appears justified given growth/quality differences
Be specific: use actual numbers. Do not use vague qualifiers.
""")

# Known sector peers for common tickers (user can always override in UI)
_DEFAULT_PEERS: Dict[str, List[str]] = {
    "NVDA": ["AMD", "INTC", "QCOM", "AVGO", "TSM"],
    "AAPL": ["MSFT", "GOOGL", "META", "AMZN", "DELL"],
    "MSFT": ["AAPL", "GOOGL", "AMZN", "CRM", "NOW"],
    "GOOGL": ["META", "MSFT", "AMZN", "SNAP", "TTD"],
    "META": ["GOOGL", "SNAP", "PINS", "TWTR", "TTD"],
    "AMZN": ["MSFT", "GOOGL", "SHOP", "WMT", "TGT"],
    "TSLA": ["GM", "F", "TM", "RIVN", "NIO"],
    "JPM": ["BAC", "WFC", "C", "GS", "MS"],
    "V": ["MA", "AXP", "PYPL", "SQ", "FIS"],
    "MA": ["V", "AXP", "PYPL", "SQ", "FIS"],
}


def _fund_to_peer_row(fund: Fundamentals, is_target: bool = False) -> PeerRow:
    fcf_yield = None
    if fund.fcf_ttm and fund.market_cap and fund.market_cap > 0:
        fcf_yield = fund.fcf_ttm / fund.market_cap
    return PeerRow(
        ticker=fund.ticker,
        name=fund.name,
        market_cap=fund.market_cap,
        pe_ttm=fund.pe_ttm,
        pe_forward=fund.pe_forward,
        ev_ebitda=fund.ev_ebitda,
        price_to_sales=fund.price_to_sales,
        gross_margin=fund.gross_margin,
        net_margin=fund.net_margin,
        revenue_growth_yoy=fund.revenue_growth_yoy,
        fcf_yield=fcf_yield,
        is_target=is_target,
    )


def _fetch_peer_fund(ticker: str):
    from data.market import get_fundamentals
    try:
        return get_fundamentals(ticker)
    except Exception as exc:
        logger.warning("peer fetch failed for %s: %s", ticker, exc)
        return None


def get_peer_comps(
    ticker: str,
    peer_tickers: Optional[List[str]] = None,
) -> PeerComps:
    """
    Fetch fundamentals for target + peers in parallel, build PeerComps.
    peer_tickers: override list. If None, uses _DEFAULT_PEERS or infers from sector.
    """
    cache_key = f"peers:{ticker.upper()}:{','.join(sorted(peer_tickers or []))}"
    cached = get_cache_obj(cache_key)
    if cached:
        try:
            return PeerComps.model_validate(cached)
        except Exception:
            pass

    # Determine peer list
    if not peer_tickers:
        peer_tickers = _DEFAULT_PEERS.get(ticker.upper(), [])

    all_tickers = [ticker.upper()] + [p.upper() for p in peer_tickers if p.upper() != ticker.upper()]

    rows: List[PeerRow] = []
    failed_tickers: List[str] = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(_fetch_peer_fund, t): t for t in all_tickers}
        for fut in as_completed(futures):
            t = futures[fut]
            fund = fut.result()
            if fund:
                rows.append(_fund_to_peer_row(fund, is_target=(t == ticker.upper())))
            else:
                failed_tickers.append(t)

    # Sort: target first, then by market cap descending
    rows.sort(key=lambda r: (not r.is_target, -(r.market_cap or 0)))

    synthesis = _synthesize_peers(ticker, rows) if rows else ""
    result = PeerComps(target_ticker=ticker.upper(), peers=rows, synthesis=synthesis, failed_tickers=failed_tickers)
    set_cache_obj(cache_key, result.model_dump(mode="json"), TTL_FUNDAMENTALS)
    return result


def _synthesize_peers(ticker: str, rows: List[PeerRow]) -> str:
    """One-paragraph Sonnet synthesis of the peer table."""
    lines = ["Ticker | Mkt Cap ($B) | P/E TTM | Fwd P/E | EV/EBITDA | P/S | Gross Margin | Net Margin | Rev Growth | FCF Yield"]
    lines.append("---" * 10)
    for r in rows:
        def _pct(v):
            return f"{v*100:.1f}%" if v is not None else "—"
        def _x(v):
            return f"{v:.1f}x" if v is not None else "—"
        def _b(v):
            return f"${v/1e9:.1f}" if v is not None else "—"
        marker = " ← TARGET" if r.is_target else ""
        lines.append(
            f"{r.ticker}{marker} | {_b(r.market_cap)} | {_x(r.pe_ttm)} | {_x(r.pe_forward)} | "
            f"{_x(r.ev_ebitda)} | {_x(r.price_to_sales)} | {_pct(r.gross_margin)} | "
            f"{_pct(r.net_margin)} | {_pct(r.revenue_growth_yoy)} | {_pct(r.fcf_yield)}"
        )

    table_text = "\n".join(lines)
    messages = [
        {
            "role": "user",
            "content": [
                llm.text_block(
                    f"Peer comparison table for {ticker}:\n\n{table_text}\n\n"
                    "Write one paragraph: where does the target look cheap vs peers, "
                    "where expensive, and is the gap justified?"
                ),
            ],
        }
    ]
    try:
        return llm.call(
            SONNET, messages,
            system=_SYSTEM_PEERS,
            prompt_version="v1",
            max_tokens=300,
        )
    except Exception as exc:
        logger.warning("peer synthesis failed: %s", exc)
        return ""
