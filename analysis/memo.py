"""Phase 6: One-page memo (<600 words) with inline citation tags. Sonnet, streamed."""
from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Dict, Iterator, List, Optional, Tuple

import llm
from analysis.schemas import (
    CallDelta, Fundamentals, KpiSeries, Memo, PositioningSummary, QualityPanel,
    RedTeam, ReverseDCFResult, Thesis,
)
from config import PROMPT_VERSIONS, SONNET

logger = logging.getLogger(__name__)

_SYSTEM = llm.cached_system("""
You are a senior analyst writing a one-page investment memo for a portfolio manager.
Rules:
1. Under 600 words total (count carefully)
2. Citation tags MANDATORY on every factual claim: [RATIOS], [XDCF], [CALLΔ], [QUAL], [POS], [KPI], [EST]
3. No hedge-fund jargon — write for a smart generalist
4. Structure: Company & Setup / Thesis / Variant View / Key Drivers / KPIs to Watch / Risks / Positioning
5. End with: "This is research synthesis, not investment advice."
""")


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def stream_memo(
    ticker: str,
    company_name: str,
    thesis_narrative: str,
    fund: Fundamentals,
    dcf: Optional[ReverseDCFResult] = None,
    call_delta: Optional[CallDelta] = None,
    quality: Optional[QualityPanel] = None,
    positioning: Optional[PositioningSummary] = None,
    kpis: Optional[List[KpiSeries]] = None,
    red_team: Optional[RedTeam] = None,
    chunk_context: str = "",
) -> Tuple[Iterator[str], Dict[str, str]]:
    """Stream the memo to the UI. Returns (token_iter, footnote_chunks)."""
    footnotes: Dict[str, str] = {}

    # Build compact context for memo
    lines = [f"Company: {company_name} ({ticker})"]
    if fund.current_price and fund.market_cap:
        lines.append(f"Price: ${fund.current_price:.2f}  Market cap: ${fund.market_cap/1e9:.1f}B")
    if fund.pe_ttm:
        lines.append(f"P/E TTM: {fund.pe_ttm:.1f}  Fwd P/E: {fund.pe_forward or 'N/A'}")
    lines.append(f"Revenue growth YoY: {(fund.revenue_growth_yoy or 0)*100:.1f}%")
    lines.append(f"Net margin: {(fund.net_margin or 0)*100:.1f}%  FCF margin: "
                 f"{(fund.fcf_ttm/fund.revenue_ttm*100 if fund.fcf_ttm and fund.revenue_ttm else 0):.1f}%")

    footnotes["[RATIOS]"] = "\n".join(lines)
    context = f'<source id="[RATIOS]">\n' + "\n".join(lines) + "\n</source>"

    if dcf:
        footnotes["[XDCF]"] = dcf.headline
        context += f'\n\n<source id="[XDCF]">\n{dcf.headline}\n</source>'

    if call_delta:
        delta_text = call_delta.what_changed_narrative[:500]
        footnotes["[CALLΔ]"] = delta_text
        context += f'\n\n<source id="[CALLΔ]">\n{delta_text}\n</source>'

    if quality:
        qual_text = f"{quality.overall}: {quality.summary}"
        footnotes["[QUAL]"] = qual_text
        context += f'\n\n<source id="[QUAL]">\n{qual_text}\n</source>'

    if positioning:
        pos_text = positioning.synthesis
        footnotes["[POS]"] = pos_text
        context += f'\n\n<source id="[POS]">\n{pos_text}\n</source>'

    if kpis:
        kpi_text = "\n".join(
            f"{k.kpi_name}: {k.trend_note}" for k in kpis[:3]
        )
        footnotes["[KPI]"] = kpi_text
        context += f'\n\n<source id="[KPI]">\n{kpi_text}\n</source>'

    if red_team:
        rt_text = f"Counter-argument: {red_team.strongest_counterargument}"
        footnotes["[REDTEAM]"] = rt_text
        context += f'\n\n<source id="[REDTEAM]">\n{rt_text}\n</source>'

    context += f'\n\n<thesis>\n{thesis_narrative[:2000]}\n</thesis>'

    messages = [
        {
            "role": "user",
            "content": [
                *llm.cached_content(context),
                llm.text_block(
                    f"Write the one-page investment memo for {ticker} ({company_name}). "
                    "Target: under 600 words. Cite every fact. "
                    "Structure:\n"
                    "## Company & Setup\n## Thesis\n## Variant View vs Consensus\n"
                    "## Key Drivers\n## KPIs to Watch\n## Risks & Falsifiers\n## Positioning\n\n"
                    "End with the disclaimer."
                ),
            ],
        }
    ]

    token_iter = llm.call(SONNET, messages, system=_SYSTEM, mode="stream", max_tokens=2000)
    return token_iter, footnotes


def build_memo_object(
    ticker: str,
    company_name: str,
    memo_text: str,
    footnotes: Dict[str, str],
) -> Memo:
    """Construct a Memo model from the streamed text + footnotes."""
    wc = _word_count(memo_text)
    return Memo(
        ticker=ticker,
        company_name=company_name,
        generated_at=datetime.utcnow(),
        setup=memo_text,
        thesis_summary="",
        variant_vs_consensus="",
        key_drivers=[],
        kpis_to_watch=[],
        risks_and_falsifiers=[],
        positioning_note="",
        word_count=wc,
        footnotes=footnotes,
    )


def memo_to_markdown(memo: Memo) -> str:
    """Export memo as markdown with footnotes section."""
    lines = [memo.setup, "\n\n---\n\n**Sources**\n"]
    for tag, text in memo.footnotes.items():
        lines.append(f"- **{tag}**: {text[:200]}")
    lines.append(f"\n_Word count: {memo.word_count}_")
    return "\n".join(lines)
