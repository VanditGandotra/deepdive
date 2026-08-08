"""Phase 5: Reverse DCF — pure math, no LLM. Finds (CAGR, FCF margin) combos that justify price."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from analysis.schemas import ReverseDCFPoint, ReverseDCFResult
from data.market import get_fundamentals

logger = logging.getLogger(__name__)


def _dcf_implied_price(
    revenue_0: float,
    revenue_cagr: float,
    fcf_margin_terminal: float,
    horizon_years: int,
    discount_rate: float,
    terminal_growth: float,
    net_debt: float,
    shares: float,
) -> Optional[float]:
    """
    Compute implied share price given assumptions.
    FCF margin grows linearly from current to terminal over the horizon.
    """
    if shares <= 0 or discount_rate <= terminal_growth:
        return None

    try:
        pv = 0.0
        for t in range(1, horizon_years + 1):
            rev_t = revenue_0 * (1 + revenue_cagr) ** t
            # FCF margin converges linearly to terminal over horizon
            margin_t = fcf_margin_terminal  # simplified: assume terminal margin throughout
            fcf_t = rev_t * margin_t
            pv += fcf_t / (1 + discount_rate) ** t

        # Terminal value (Gordon Growth Model)
        rev_T = revenue_0 * (1 + revenue_cagr) ** horizon_years
        fcf_T = rev_T * fcf_margin_terminal
        tv = fcf_T * (1 + terminal_growth) / (discount_rate - terminal_growth)
        tv_pv = tv / (1 + discount_rate) ** horizon_years

        ev = pv + tv_pv
        equity_value = ev - net_debt
        return max(0.0, equity_value / shares)
    except Exception:
        return None


def reverse_dcf(
    ticker: str,
    discount_rate: float = 0.10,
    terminal_growth: float = 0.025,
    horizon_years: int = 10,
) -> Optional[ReverseDCFResult]:
    """
    Compute the isocurve of (revenue_cagr, fcf_margin) pairs that imply today's stock price.
    Also returns ±2% discount-rate sensitivity table.
    """
    fund = get_fundamentals(ticker)
    price = fund.current_price
    shares = fund.shares_outstanding
    revenue_0 = fund.revenue_ttm

    if not all([price, shares, revenue_0]) or revenue_0 <= 0 or shares <= 0:
        logger.warning("Insufficient data for reverse DCF on %s", ticker)
        return None

    net_debt = (fund.total_debt or 0) - (fund.cash or 0)
    target_ev = price * shares + net_debt

    # Isocurve: scan (CAGR, margin) grid and find combos that ≈ imply current price
    cagr_range = np.arange(0.00, 0.41, 0.02)      # 0% to 40%
    margin_range = np.arange(0.01, 0.41, 0.02)    # 1% to 40%

    isocurve: List[ReverseDCFPoint] = []
    target_price = price

    for cagr in cagr_range:
        # For each CAGR, find the FCF margin that implies current price
        for margin in margin_range:
            implied = _dcf_implied_price(
                revenue_0, float(cagr), float(margin),
                horizon_years, discount_rate, terminal_growth,
                net_debt, shares,
            )
            if implied is not None:
                match = abs(implied - target_price) / target_price < 0.05
                isocurve.append(ReverseDCFPoint(
                    revenue_cagr=round(float(cagr), 3),
                    fcf_margin=round(float(margin), 3),
                    implied_price=round(implied, 2),
                    matches_current=match,
                ))

    # Find the single best-match combination for the headline
    best = min(isocurve, key=lambda p: abs(p.implied_price - target_price), default=None)
    if best:
        headline = (
            f"At ${target_price:.2f}, the market prices in ~{best.revenue_cagr*100:.0f}% "
            f"revenue CAGR at ~{best.fcf_margin*100:.0f}% FCF margins over {horizon_years} years."
        )
    else:
        headline = f"Could not solve reverse DCF for {ticker} at current price."

    # Discount rate sensitivity (keep margin fixed at best-match, vary discount rate by ±2%)
    sensitivity: List[Dict[str, float]] = []
    for dr_delta in [-0.02, -0.01, 0.00, 0.01, 0.02]:
        dr = discount_rate + dr_delta
        implied_list = []
        for cagr in [0.05, 0.10, 0.15, 0.20, 0.25]:
            for margin in [0.05, 0.10, 0.15, 0.20]:
                p = _dcf_implied_price(revenue_0, cagr, margin, horizon_years, dr, terminal_growth, net_debt, shares)
                if p:
                    implied_list.append(p)
        sensitivity.append({
            "discount_rate": round(dr, 3),
            "avg_implied_price": round(float(np.mean(implied_list)), 2) if implied_list else 0.0,
        })

    return ReverseDCFResult(
        current_price=price,
        shares_outstanding=shares,
        net_debt=net_debt,
        discount_rate=discount_rate,
        terminal_growth=terminal_growth,
        horizon_years=horizon_years,
        headline=headline,
        isocurve_points=isocurve[:200],  # cap for display
        sensitivity_table=sensitivity,
    )
