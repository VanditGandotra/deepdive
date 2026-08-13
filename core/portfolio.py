from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class EnrichmentResult:
    portfolio: "Portfolio"
    failed: List[str]   # tickers whose price fetch raised; their market_value stays None


@dataclass
class Holding:
    ticker: str
    shares: float
    cost_basis: Optional[float] = None   # per share
    account: Optional[str] = None
    notes: Optional[str] = None
    is_cash: bool = False
    # Filled in when market data is available:
    current_price: Optional[float] = None
    market_value: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    unrealized_pnl_pct: Optional[float] = None
    weight: Optional[float] = None
    sector: Optional[str] = None


@dataclass
class Portfolio:
    name: str
    holdings: List[Holding] = field(default_factory=list)

    @property
    def total_value(self) -> float:
        return sum(h.market_value or 0 for h in self.holdings)

    def compute_weights(self) -> None:
        """Compute weight for each holding as market_value / total_value."""
        tv = self.total_value
        for h in self.holdings:
            h.weight = (h.market_value or 0) / tv if tv > 0 else 0.0

    def sector_concentration(self) -> Dict[str, float]:
        """Returns {sector: total_weight}."""
        out: Dict[str, float] = {}
        for h in self.holdings:
            s = h.sector or "Unknown"
            out[s] = out.get(s, 0.0) + (h.weight or 0.0)
        return out

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for h in self.holdings:
            rows.append({
                "Ticker": h.ticker,
                "Shares": h.shares,
                "Cost Basis": h.cost_basis,
                "Account": h.account or "",
                "Notes": h.notes or "",
                "Price": h.current_price,
                "Market Value": h.market_value,
                "Unrealized P&L": h.unrealized_pnl,
                "Unrealized P&L %": h.unrealized_pnl_pct,
                "Weight": h.weight,
                "Sector": h.sector or "",
            })
        return pd.DataFrame(rows)


def enrich_with_prices(portfolio: Portfolio) -> EnrichmentResult:
    """
    Fill in current_price, market_value, unrealized_pnl, weight, sector.
    Always returns EnrichmentResult; failed list is empty on full success.
    """
    from data.market import get_fundamentals

    failed: List[str] = []
    for h in portfolio.holdings:
        if h.is_cash:
            h.current_price = 1.0
            h.market_value = h.shares  # shares = dollar amount
            h.sector = "Cash"
            continue
        try:
            fund = get_fundamentals(h.ticker)
            h.current_price = fund.current_price
            h.sector = fund.sector
            h.market_value = (fund.current_price or 0) * h.shares
            if h.cost_basis and h.cost_basis > 0:
                cost_total = h.cost_basis * h.shares
                h.unrealized_pnl = h.market_value - cost_total
                h.unrealized_pnl_pct = h.unrealized_pnl / cost_total if cost_total > 0 else None
        except Exception:
            failed.append(h.ticker)

    portfolio.compute_weights()
    return EnrichmentResult(portfolio=portfolio, failed=failed)
