"""State of play: 3-4 sentence synthesis of where a stock sits right now. Sonnet, streamed."""
from __future__ import annotations

import logging
from typing import Iterator, List, Optional

import llm
from analysis.schemas import (
    CallDelta, CallSentiment, Fundamentals, QualityPanel, ReverseDCFResult,
)
from config import PROMPT_VERSIONS, SONNET

logger = logging.getLogger(__name__)

_SYSTEM = llm.cached_system("""
You are a senior equity analyst writing a 3-4 sentence "state of play" for a research briefing.
Synthesize ALL of the provided data into a single cohesive paragraph. Specifically address:
1. Valuation vs. own history (is the stock cheap, fairly valued, or expensive vs. its 5-year median?)
2. Estimate revision direction (are estimates rising, falling, or flat?)
3. Call sentiment trend (is management tone improving, deteriorating, or mixed?)
4. Any quality flags or structural concerns worth flagging

Rules:
- Anchor every claim to the data provided — never invent numbers
- Use [TAG] citation style for any specific figure you reference
- Be specific and direct; no filler phrases like "it is worth noting"
- Write for a professional investor who already knows the company basics
- Maximum 4 sentences, no bullet points, no headers
""")

_PROMPT_VERSION = "state_of_play_v1"


def stream_state_of_play(
    ticker: str,
    fund: Fundamentals,
    dcf: Optional[ReverseDCFResult],
    sentiment_trend: Optional[List[float]],
    quality: Optional[QualityPanel],
) -> Iterator[str]:
    """Stream the 3-4 sentence state-of-play synthesis. Cached by content hash."""
    parts: List[str] = []

    # Valuation context
    parts.append(
        f"[RATIOS] {ticker}: P/E TTM={fund.pe_ttm}, Fwd P/E={fund.pe_forward}, "
        f"EV/EBITDA={fund.ev_ebitda}, P/S={fund.price_to_sales}, "
        f"Net margin={fund.net_margin and f'{fund.net_margin*100:.1f}%'}, "
        f"Rev growth YoY={fund.revenue_growth_yoy and f'{fund.revenue_growth_yoy*100:.1f}%'}"
    )

    # DCF context
    if dcf and dcf.headline:
        parts.append(f"[DCF] {dcf.headline}")

    # Sentiment
    if sentiment_trend and len(sentiment_trend) >= 2:
        direction = "improving" if sentiment_trend[-1] > sentiment_trend[0] else "deteriorating"
        parts.append(
            f"[SENT] Call sentiment over last {len(sentiment_trend)} quarters: "
            f"{[f'{s:.2f}' for s in sentiment_trend]} — {direction} trend."
        )

    # Quality flags
    if quality:
        red = [f.name for f in quality.flags if f.status == "red"]
        yellow = [f.name for f in quality.flags if f.status == "yellow"]
        parts.append(
            f"[QF] Quality flags: {len(red)} red ({', '.join(red) or 'none'}), "
            f"{len(yellow)} yellow ({', '.join(yellow) or 'none'}). "
            f"Overall: {quality.overall}."
        )

    context = "\n\n".join(parts)
    messages = [
        {
            "role": "user",
            "content": [
                llm.text_block(
                    f"Write the state-of-play paragraph for {ticker} ({fund.name or ticker}).\n\n"
                    f"{context}"
                )
            ],
        }
    ]

    yield from llm.call(
        SONNET, messages,
        system=_SYSTEM,
        prompt_version=_PROMPT_VERSION,
        mode="stream",
        max_tokens=1200,
    )
