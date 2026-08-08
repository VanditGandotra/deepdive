"""Phase 5: Short interest, insider transactions, holder changes — positioning summary."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import List, Optional

from analysis.schemas import (
    InstitutionalHolder, InsiderTransaction, PositioningSummary, ShortInterest,
)
from data.market import get_holders, get_insiders, get_short_interest

logger = logging.getLogger(__name__)


def _net_insider_sentiment(transactions: List[InsiderTransaction], months: int = 6) -> str:
    cutoff = datetime.utcnow() - timedelta(days=months * 30)
    recent = [
        t for t in transactions
        if t.date and datetime.combine(t.date, datetime.min.time()) >= cutoff
    ]
    if not recent:
        return "minimal"
    buys = sum(t.shares for t in recent if "buy" in (t.transaction_type or "").lower()
               or "purchase" in (t.transaction_type or "").lower())
    sells = sum(t.shares for t in recent if "sale" in (t.transaction_type or "").lower()
                or "sell" in (t.transaction_type or "").lower())
    if buys > sells * 2:
        return "net_buying"
    if sells > buys * 2:
        return "net_selling"
    return "mixed"


def get_positioning(ticker: str) -> PositioningSummary:
    si = get_short_interest(ticker)
    insiders = get_insiders(ticker)
    holders = get_holders(ticker)

    sentiment = _net_insider_sentiment(insiders)

    # Compose one-line synthesis
    parts = []
    if si.pct_float:
        pf = si.pct_float * 100
        if pf > 20:
            parts.append(f"heavily shorted ({pf:.1f}% float)")
        elif pf > 10:
            parts.append(f"moderately shorted ({pf:.1f}% float)")
        else:
            parts.append(f"low short interest ({pf:.1f}% float)")

    if sentiment == "net_buying":
        parts.append("insiders net buying over 6 months")
    elif sentiment == "net_selling":
        parts.append("insiders net selling over 6 months")
    elif sentiment == "mixed":
        parts.append("mixed insider activity")

    # Holder changes (positive = increasing position)
    buyers = [h for h in holders if h.change and h.change > 0]
    sellers = [h for h in holders if h.change and h.change < 0]
    if len(buyers) > len(sellers) * 2:
        parts.append("institutions accumulating")
    elif len(sellers) > len(buyers) * 2:
        parts.append("institutions reducing")

    synthesis = "; ".join(parts).capitalize() + "." if parts else "Limited positioning data available."

    return PositioningSummary(
        short_interest_pct_float=si.pct_float,
        days_to_cover=si.days_to_cover,
        insider_net_sentiment=sentiment,
        insider_transactions=insiders[:20],
        top_holder_changes=holders[:10],
        synthesis=synthesis,
    )
