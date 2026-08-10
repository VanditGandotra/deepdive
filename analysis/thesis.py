"""Phase 6: Bull/base/bear theses + red team. Sonnet, streamed, citation-tagged."""
from __future__ import annotations

import logging
from typing import Dict, Iterator, List, Optional, Tuple

import llm
from analysis.schemas import (
    CallDelta, Fundamentals, QualityPanel, PositioningSummary,
    RedTeam, ReverseDCFResult, Thesis,
)
from config import PROMPT_VERSIONS, SONNET

logger = logging.getLogger(__name__)

_SYSTEM_THESIS = llm.cached_system("""
You are a hedge fund analyst writing investment theses for an internal research meeting.
Use ONLY the data provided in the source blocks — every number must trace to a [TAG].
Write each scenario (bull/base/bear) as if arguing it forcefully but honestly.
For confirm_signals: observable leading indicators that would CONFIRM the thesis.
For kill_signals: observable events that would INVALIDATE the thesis.
Format implied_price_12mo as a round number.
Probability weights across the three scenarios must sum to 1.0.
DISCLAIMER: This is research synthesis for institutional use, not investment advice.
""")

_SYSTEM_REDTEAM = llm.cached_system("""
You are a devil's advocate analyst tasked with steelmanning the weakest parts of the investment case.
Be specific. Name the mechanisms. Don't just say "competition could increase" — say which competitor,
what they are doing, why it could win.
fastest_falsifier = the single most observable, falsifiable signal that would confirm the bear case
within 6-12 months.
""")


def _build_thesis_context(
    ticker: str,
    fund: Fundamentals,
    dcf: Optional[ReverseDCFResult],
    call_delta: Optional[CallDelta],
    quality: Optional[QualityPanel],
    positioning: Optional[PositioningSummary],
    high_mat_news: List[str],
    kpi_summaries: List[str],
    estimates_summary: str = "",
) -> Tuple[str, Dict[str, str]]:
    chunks: Dict[str, str] = {}
    parts = []

    # Fundamentals
    fund_text = (
        f"P/E TTM: {fund.pe_ttm}  Fwd P/E: {fund.pe_forward}  EV/EBITDA: {fund.ev_ebitda}\n"
        f"Gross margin: {fund.gross_margin}  Net margin: {fund.net_margin}  ROE: {fund.roe}\n"
        f"Revenue TTM: {fund.revenue_ttm}  FCF TTM: {fund.fcf_ttm}\n"
        f"Revenue growth YoY: {fund.revenue_growth_yoy}  EPS growth: {fund.eps_growth_yoy}"
    )
    chunks["[RATIOS]"] = fund_text
    parts.append(f'<source id="[RATIOS]">\n{fund_text}\n</source>')

    if dcf:
        dcf_text = f"Reverse DCF: {dcf.headline}"
        chunks["[XDCF]"] = dcf_text
        parts.append(f'<source id="[XDCF]">\n{dcf_text}\n</source>')

    if call_delta:
        delta_text = (
            f"Call delta narrative: {call_delta.what_changed_narrative}\n"
            f"Guidance changes: {'; '.join(f'{g.metric}: {g.change_description}' for g in call_delta.guidance_changes)}\n"
            f"Dropped topics: {'; '.join(call_delta.dropped_topics)}\n"
            f"Sentiment trend: {call_delta.sentiment_trend}"
        )
        chunks["[CALLΔ]"] = delta_text
        parts.append(f'<source id="[CALLΔ]">\n{delta_text}\n</source>')

    if quality:
        qual_text = f"Quality panel ({quality.overall}): {quality.summary}"
        for f in quality.flags:
            if f.status != "green":
                qual_text += f"\n  {f.name}: {f.status.upper()} — {f.explanation}"
        chunks["[QUAL]"] = qual_text
        parts.append(f'<source id="[QUAL]">\n{qual_text}\n</source>')

    if positioning:
        pos_text = (
            f"Positioning: {positioning.synthesis}\n"
            f"Short interest: {positioning.short_interest_pct_float}\n"
            f"Insider sentiment: {positioning.insider_net_sentiment}"
        )
        chunks["[POS]"] = pos_text
        parts.append(f'<source id="[POS]">\n{pos_text}\n</source>')

    if high_mat_news:
        news_text = "\n".join(f"- {n}" for n in high_mat_news[:5])
        chunks["[NEWS-HI]"] = news_text
        parts.append(f'<source id="[NEWS-HI]">\n{news_text}\n</source>')

    if kpi_summaries:
        kpi_text = "\n".join(f"- {k}" for k in kpi_summaries[:5])
        chunks["[KPI]"] = kpi_text
        parts.append(f'<source id="[KPI]">\n{kpi_text}\n</source>')

    if estimates_summary:
        chunks["[EST]"] = estimates_summary
        parts.append(f'<source id="[EST]">\n{estimates_summary}\n</source>')

    return "\n\n".join(parts), chunks


def stream_theses(
    ticker: str,
    fund: Fundamentals,
    dcf: Optional[ReverseDCFResult] = None,
    call_delta: Optional[CallDelta] = None,
    quality: Optional[QualityPanel] = None,
    positioning: Optional[PositioningSummary] = None,
    high_mat_news: Optional[List[str]] = None,
    kpi_summaries: Optional[List[str]] = None,
    estimates_summary: str = "",
) -> Tuple[Iterator[str], Dict[str, str]]:
    """Stream bull/base/bear narrative. Returns (token_iterator, chunks_map)."""
    context, chunks = _build_thesis_context(
        ticker, fund, dcf, call_delta, quality, positioning,
        high_mat_news or [], kpi_summaries or [], estimates_summary,
    )
    messages = [
        {
            "role": "user",
            "content": [
                *llm.cached_content(context),
                llm.text_block(
                    f"Write bull, base, and bear investment theses for {ticker}. "
                    "Structure:\n## Bull Case\n## Base Case\n## Bear Case\n\n"
                    "For each scenario include: narrative, 3 key drivers, "
                    "implied 12-month price target, probability weight (must sum to 1.0 across scenarios), "
                    "2 confirm signals, 2 kill signals.\n"
                    "Tag every factual claim with its source ID. "
                    "End with a one-line disclaimer."
                ),
            ],
        }
    ]
    return llm.call(SONNET, messages, system=_SYSTEM_THESIS, mode="stream", max_tokens=4096), chunks


def get_structured_theses(
    ticker: str,
    fund: Fundamentals,
    dcf: Optional[ReverseDCFResult] = None,
    call_delta: Optional[CallDelta] = None,
    quality: Optional[QualityPanel] = None,
    positioning: Optional[PositioningSummary] = None,
    high_mat_news: Optional[List[str]] = None,
    kpi_summaries: Optional[List[str]] = None,
) -> List[Thesis]:
    context, _ = _build_thesis_context(
        ticker, fund, dcf, call_delta, quality, positioning,
        high_mat_news or [], kpi_summaries or [],
    )
    results: List[Thesis] = []
    for scenario in ["bull", "base", "bear"]:
        messages = [
            {
                "role": "user",
                "content": [
                    *llm.cached_content(context),
                    llm.text_block(
                        f"Return a structured {scenario} case thesis for {ticker}. "
                        f"Set scenario={scenario}."
                    ),
                ],
            }
        ]
        try:
            t = llm.call(
                SONNET, messages,
                system=_SYSTEM_THESIS,
                schema=Thesis,
                prompt_version=PROMPT_VERSIONS["thesis"],
                max_tokens=2000,
            )
            results.append(t)
        except Exception as exc:
            logger.warning("Thesis %s failed for %s: %s", scenario, ticker, exc)
    return results


def get_red_team(
    ticker: str,
    context: str,
) -> Optional[RedTeam]:
    messages = [
        {
            "role": "user",
            "content": [
                *llm.cached_content(context),
                llm.text_block(
                    f"Devil's advocate analysis for {ticker}: "
                    "What is the strongest counter-argument to the prevailing thesis? "
                    "Be specific, concrete, and name mechanisms."
                ),
            ],
        }
    ]
    try:
        return llm.call(
            SONNET, messages,
            system=_SYSTEM_REDTEAM,
            schema=RedTeam,
            prompt_version=PROMPT_VERSIONS["red_team"],
            max_tokens=1500,
        )
    except Exception as exc:
        logger.warning("Red team failed for %s: %s", ticker, exc)
        return None


def fetch_web_context(ticker: str, company_name: str) -> str:
    """
    Fetch live analyst views, recent news, and market sentiment via web search.
    Returns a formatted context block tagged [WEB] for use in thesis generation.
    Cached 6h (TTL_NEWS). Returns empty string on failure.
    """
    cache_key = f"web_context:{ticker.upper()}"
    prompt = (
        f"Research {company_name} ({ticker}) for an equity investment memo. Find:\n"
        "1. Most recent analyst upgrades/downgrades and price target changes (last 30 days)\n"
        "2. Key news or events affecting the investment thesis (last 2 weeks)\n"
        "3. Main bull arguments vs bear arguments being discussed by investors\n"
        "4. Any management guidance updates or strategic announcements\n\n"
        "Be specific: cite sources, dates, and exact figures. "
        "Summarize in 3-4 concise paragraphs. Focus on what is NEW vs what the market already knows."
    )
    raw = llm.web_search_synthesis(prompt, cache_key=cache_key)
    if not raw:
        return ""
    return f'<source id="[WEB]" note="Live web search — {company_name}">\n{raw}\n</source>'
