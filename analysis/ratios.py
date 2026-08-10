"""Phase 2: Ratio engine — 4 groups with 5yr sparkline history. Pure math, no LLM."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from analysis.schemas import Fundamentals, RatioGroup, RatioHistory
from data.market import get_fundamentals, get_prices

logger = logging.getLogger(__name__)


# ── DataFrame helpers ─────────────────────────────────────────────────────────

def _get_row(df: Optional[pd.DataFrame], *names: str) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    for name in names:
        for idx in df.index:
            if isinstance(idx, str) and (idx == name or name.lower() in idx.lower()):
                series = df.loc[idx]
                return series if not series.isnull().all() else None
    return None


def _to_float(val: Any) -> Optional[float]:
    try:
        f = float(val)
        return None if (f != f or abs(f) > 1e18) else f
    except (TypeError, ValueError):
        return None


def _series_to_history(series: Optional[pd.Series], n: int = 5) -> List[Optional[float]]:
    if series is None or series.empty:
        return []
    sorted_s = series.sort_index(ascending=True)
    return [_to_float(v) for v in sorted_s.values[-n:]]


def _pct_history(series: Optional[pd.Series], n: int = 5) -> List[Optional[float]]:
    if series is None or series.empty:
        return []
    sorted_s = series.sort_index(ascending=True).dropna()
    if len(sorted_s) < 2:
        return []
    pct = sorted_s.pct_change().values[1:]
    return [_to_float(v) for v in pct[-(n - 1):]]


def _ratio_history(num: Optional[pd.Series], denom: Optional[pd.Series], n: int = 5) -> List[Optional[float]]:
    if num is None or denom is None:
        return []
    combined = pd.concat([num.rename("n"), denom.rename("d")], axis=1).dropna().sort_index(ascending=True)
    vals = []
    for _, row in combined.tail(n).iterrows():
        d = _to_float(row["d"])
        num_v = _to_float(row["n"])
        if d and abs(d) > 0 and num_v is not None:
            vals.append(num_v / d)
        else:
            vals.append(None)
    return vals


def _sparkline_stats(hist: List[Optional[float]]) -> tuple:
    vals = [v for v in hist if v is not None]
    if not vals:
        return None, None, None
    return float(np.min(vals)), float(np.median(vals)), float(np.max(vals))


# ── Annual data fetch ─────────────────────────────────────────────────────────

def _fetch_annual_data(ticker: str) -> Dict[str, Optional[pd.DataFrame]]:
    import yfinance as yf
    yf_t = yf.Ticker(ticker)
    result: Dict[str, Optional[pd.DataFrame]] = {}
    for attr, key in [
        ("financials",    "income"),
        ("balance_sheet", "balance"),
        ("cashflow",      "cashflow"),
        ("quarterly_financials",    "q_income"),
        ("quarterly_balance_sheet", "q_balance"),
        ("quarterly_cashflow",      "q_cashflow"),
    ]:
        try:
            df = getattr(yf_t, attr, None)
            result[key] = df if (df is not None and not df.empty) else None
        except Exception as exc:
            logger.debug("yfinance %s error for %s: %s", attr, ticker, exc)
            result[key] = None
    return result


def _year_end_prices(ticker: str) -> Dict[int, float]:
    try:
        pd_data = get_prices(ticker, "5y")
        year_prices: Dict[int, tuple] = {}
        for bar in pd_data.bars:
            yr = bar.date.year
            if yr not in year_prices or bar.date > year_prices[yr][0]:
                year_prices[yr] = (bar.date, bar.close)
        return {yr: price for yr, (_, price) in year_prices.items()}
    except Exception:
        return {}


# ── Ratio group builders ──────────────────────────────────────────────────────

def _valuation_group(fund: Fundamentals, data: Dict, year_prices: Dict) -> RatioGroup:
    income = data.get("income")
    rev = _get_row(income, "Total Revenue", "Revenue")
    net_inc = _get_row(income, "Net Income", "Net Income Common Stockholders")

    # Historical P/E: year-end price / (annual net income / current shares)
    pe_hist: List[Optional[float]] = []
    if net_inc is not None and fund.shares_outstanding:
        for dt, ni in sorted(net_inc.items(), key=lambda x: x[0])[-5:]:
            yr = dt.year if hasattr(dt, "year") else None
            ni_f = _to_float(ni)
            if yr and yr in year_prices and ni_f and ni_f > 0:
                eps = ni_f / fund.shares_outstanding
                pe_hist.append(year_prices[yr] / eps)
            else:
                pe_hist.append(None)

    # Historical P/S: year-end market cap / annual revenue
    ps_hist: List[Optional[float]] = []
    if rev is not None and fund.shares_outstanding:
        for dt, r_val in sorted(rev.items(), key=lambda x: x[0])[-5:]:
            yr = dt.year if hasattr(dt, "year") else None
            r_f = _to_float(r_val)
            if yr and yr in year_prices and r_f and r_f > 0:
                mkt = year_prices[yr] * fund.shares_outstanding
                ps_hist.append(mkt / r_f)
            else:
                ps_hist.append(None)

    return RatioGroup(group="Valuation", ratios=[
        RatioHistory(
            name="P/E (TTM)", current=fund.pe_ttm,
            min_5y=_sparkline_stats(pe_hist)[0],
            median_5y=_sparkline_stats(pe_hist)[1],
            max_5y=_sparkline_stats(pe_hist)[2],
            history=pe_hist,
            description="Price-to-trailing-earnings. Lower vs history = cheaper; >40 often growth-priced.",
        ),
        RatioHistory(
            name="P/E (Fwd)", current=fund.pe_forward,
            description="Market's expectation for next-12-month earnings.",
        ),
        RatioHistory(
            name="PEG", current=fund.peg,
            description="P/E ÷ earnings growth rate. <1 historically attractive.",
        ),
        RatioHistory(
            name="EV/EBITDA", current=fund.ev_ebitda,
            description="Enterprise-value to EBITDA. Capital-structure-neutral. <10 typically value territory.",
        ),
        RatioHistory(
            name="P/S", current=fund.price_to_sales,
            min_5y=_sparkline_stats(ps_hist)[0],
            median_5y=_sparkline_stats(ps_hist)[1],
            max_5y=_sparkline_stats(ps_hist)[2],
            history=ps_hist,
            description="Price-to-sales. Useful pre-profitability; >10 prices in significant growth.",
        ),
        RatioHistory(
            name="P/FCF", current=fund.price_to_fcf,
            description="Price-to-free-cash-flow. Often the most honest valuation anchor.",
        ),
    ])


def _profitability_group(fund: Fundamentals, data: Dict) -> RatioGroup:
    income = data.get("income")
    balance = data.get("balance")
    rev = _get_row(income, "Total Revenue", "Revenue")
    gross = _get_row(income, "Gross Profit")
    op_inc = _get_row(income, "Operating Income", "EBIT", "Total Operating Income As Reported")
    net_inc = _get_row(income, "Net Income", "Net Income Common Stockholders")
    equity = _get_row(balance, "Stockholders Equity", "Total Stockholders Equity", "Common Stock Equity")

    gm_hist = _ratio_history(gross, rev)
    om_hist = _ratio_history(op_inc, rev)
    nm_hist = _ratio_history(net_inc, rev)
    roe_hist = _ratio_history(net_inc, equity)

    return RatioGroup(group="Profitability", ratios=[
        RatioHistory(
            name="Gross Margin", current=fund.gross_margin,
            min_5y=_sparkline_stats(gm_hist)[0], median_5y=_sparkline_stats(gm_hist)[1],
            max_5y=_sparkline_stats(gm_hist)[2], history=gm_hist,
            description="Revenue minus COGS. Structural competitive moat signal.",
        ),
        RatioHistory(
            name="Operating Margin", current=fund.operating_margin,
            min_5y=_sparkline_stats(om_hist)[0], median_5y=_sparkline_stats(om_hist)[1],
            max_5y=_sparkline_stats(om_hist)[2], history=om_hist,
            description="EBIT / Revenue. Efficiency of core operations before financing.",
        ),
        RatioHistory(
            name="Net Margin", current=fund.net_margin,
            min_5y=_sparkline_stats(nm_hist)[0], median_5y=_sparkline_stats(nm_hist)[1],
            max_5y=_sparkline_stats(nm_hist)[2], history=nm_hist,
            description="Bottom-line profitability after all costs and taxes.",
        ),
        RatioHistory(
            name="ROE", current=fund.roe,
            min_5y=_sparkline_stats(roe_hist)[0], median_5y=_sparkline_stats(roe_hist)[1],
            max_5y=_sparkline_stats(roe_hist)[2], history=roe_hist,
            description="Return on equity. Reflects compounding power of retained earnings.",
        ),
        RatioHistory(
            name="ROIC", current=fund.roic,
            description="Return on invested capital. Best measure of capital allocation quality.",
        ),
    ])


def _health_group(fund: Fundamentals, data: Dict) -> RatioGroup:
    balance = data.get("balance")
    income = data.get("income")
    curr_assets = _get_row(balance, "Current Assets", "Total Current Assets")
    curr_liab = _get_row(balance, "Current Liabilities", "Total Current Liabilities")
    total_debt = _get_row(balance, "Total Debt", "Long Term Debt")
    equity = _get_row(balance, "Stockholders Equity", "Total Stockholders Equity", "Common Stock Equity")
    cash = _get_row(balance, "Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments")
    ebitda_s = _get_row(income, "EBITDA")

    cr_hist = _ratio_history(curr_assets, curr_liab)
    de_hist = _ratio_history(total_debt, equity)

    nd_ebitda: List[Optional[float]] = []
    if total_debt is not None and cash is not None and ebitda_s is not None:
        combined = pd.concat(
            [total_debt.rename("d"), cash.rename("c"), ebitda_s.rename("e")], axis=1
        ).dropna().sort_index(ascending=True)
        for _, row in combined.tail(5).iterrows():
            e = _to_float(row["e"]); d = _to_float(row["d"]); c = _to_float(row["c"])
            if e and e > 0 and d is not None and c is not None:
                nd_ebitda.append((d - c) / e)
            else:
                nd_ebitda.append(None)

    return RatioGroup(group="Balance Sheet Health", ratios=[
        RatioHistory(
            name="Current Ratio", current=fund.current_ratio,
            min_5y=_sparkline_stats(cr_hist)[0], median_5y=_sparkline_stats(cr_hist)[1],
            max_5y=_sparkline_stats(cr_hist)[2], history=cr_hist,
            description="Current assets / current liabilities. >1 = can cover near-term obligations.",
        ),
        RatioHistory(
            name="Debt / Equity", current=fund.debt_to_equity,
            min_5y=_sparkline_stats(de_hist)[0], median_5y=_sparkline_stats(de_hist)[1],
            max_5y=_sparkline_stats(de_hist)[2], history=de_hist,
            description="Leverage ratio. Context-dependent by sector and capital intensity.",
        ),
        RatioHistory(
            name="Net Debt / EBITDA", current=fund.net_debt_ebitda,
            min_5y=_sparkline_stats(nd_ebitda)[0], median_5y=_sparkline_stats(nd_ebitda)[1],
            max_5y=_sparkline_stats(nd_ebitda)[2], history=nd_ebitda,
            description="Debt payback in operating-profit years. >4× triggers credit concern.",
        ),
        RatioHistory(
            name="Interest Coverage", current=fund.interest_coverage,
            description="EBIT / interest expense. <2× is financially stressed territory.",
        ),
    ])


def _growth_group(fund: Fundamentals, data: Dict) -> RatioGroup:
    income = data.get("income")
    cashflow = data.get("cashflow")
    rev = _get_row(income, "Total Revenue", "Revenue")
    net_inc = _get_row(income, "Net Income", "Net Income Common Stockholders")
    ocf = _get_row(cashflow, "Operating Cash Flow", "Total Cash From Operating Activities")
    capex = _get_row(cashflow, "Capital Expenditure", "Capital Expenditures",
                     "Purchase Of Property Plant And Equipment")

    rev_growth = _pct_history(rev)
    eps_growth = _pct_history(net_inc)

    fcf_margin: List[Optional[float]] = []
    if ocf is not None and capex is not None and rev is not None:
        combined = pd.concat(
            [ocf.rename("o"), capex.rename("c"), rev.rename("r")], axis=1
        ).dropna().sort_index(ascending=True)
        for _, row in combined.tail(5).iterrows():
            o = _to_float(row["o"]); c = _to_float(row["c"]); r = _to_float(row["r"])
            if o is not None and c is not None and r and r > 0:
                fcf = o + c if c < 0 else o - c
                fcf_margin.append(fcf / r)
            else:
                fcf_margin.append(None)

    current_fcf_margin = None
    if fund.fcf_ttm and fund.revenue_ttm and fund.revenue_ttm > 0:
        current_fcf_margin = fund.fcf_ttm / fund.revenue_ttm

    return RatioGroup(group="Growth & Cash Generation", ratios=[
        RatioHistory(
            name="Revenue Growth YoY", current=fund.revenue_growth_yoy,
            min_5y=_sparkline_stats(rev_growth)[0],
            median_5y=_sparkline_stats(rev_growth)[1],
            max_5y=_sparkline_stats(rev_growth)[2],
            history=rev_growth,
            description="Year-over-year revenue growth. Trend direction matters as much as level.",
        ),
        RatioHistory(
            name="EPS Growth YoY", current=fund.eps_growth_yoy,
            min_5y=_sparkline_stats(eps_growth)[0],
            median_5y=_sparkline_stats(eps_growth)[1],
            max_5y=_sparkline_stats(eps_growth)[2],
            history=eps_growth,
            description="Net income per share growth. Divergence from revenue reveals margin change.",
        ),
        RatioHistory(
            name="FCF Margin", current=current_fcf_margin,
            min_5y=_sparkline_stats(fcf_margin)[0],
            median_5y=_sparkline_stats(fcf_margin)[1],
            max_5y=_sparkline_stats(fcf_margin)[2],
            history=fcf_margin,
            description="Free cash flow as % of revenue. Expanding = improving earnings quality.",
        ),
    ])


# ── Public API ────────────────────────────────────────────────────────────────

def get_ratio_groups(ticker: str) -> List[RatioGroup]:
    fund = get_fundamentals(ticker)
    data = _fetch_annual_data(ticker)
    year_prices = _year_end_prices(ticker)
    return [
        _valuation_group(fund, data, year_prices),
        _profitability_group(fund, data),
        _health_group(fund, data),
        _growth_group(fund, data),
    ]
