"""Phase 5: Rules-based quality-of-earnings flags. Pure math, no LLM."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from analysis.schemas import QualityFlag, QualityPanel

logger = logging.getLogger(__name__)


def _safe(val: Any) -> Optional[float]:
    try:
        f = float(val)
        return None if (f != f or abs(f) > 1e18) else f
    except (TypeError, ValueError):
        return None


def _get_q_series(df: Optional[pd.DataFrame], *names: str) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    for name in names:
        for idx in df.index:
            if isinstance(idx, str) and name.lower() in idx.lower():
                s = df.loc[idx].dropna().sort_index(ascending=True)
                return s if not s.empty else None
    return None


def _pct_change(series: Optional[pd.Series]) -> Optional[pd.Series]:
    if series is None or len(series) < 2:
        return None
    return series.pct_change().dropna()


# ── Individual flag checkers ──────────────────────────────────────────────────

def _flag_receivables_growing_faster_than_revenue(
    q_income: Optional[pd.DataFrame],
    q_balance: Optional[pd.DataFrame],
) -> QualityFlag:
    rev = _get_q_series(q_income, "Total Revenue", "Revenue")
    rec = _get_q_series(q_balance, "Net Receivables", "Accounts Receivable", "Receivables")

    if rev is None or rec is None or len(rev) < 3:
        return QualityFlag(
            name="Receivables vs Revenue Growth",
            status="green",
            trigger_condition="AR growth > revenue growth for 2 consecutive quarters",
            threshold="2 consecutive quarters",
            explanation="Insufficient quarterly data to evaluate.",
        )

    rev_chg = _pct_change(rev)
    rec_chg = _pct_change(rec)
    if rev_chg is None or rec_chg is None:
        status = "green"
        observed = "Insufficient data"
    else:
        combined = pd.concat([rev_chg.rename("rev"), rec_chg.rename("rec")], axis=1).dropna()
        flag_qs = (combined["rec"] > combined["rev"]).tail(4)
        consecutive = sum(1 for i in range(len(flag_qs) - 1) if flag_qs.iloc[i] and flag_qs.iloc[i + 1])
        if consecutive >= 1:
            status = "red"
            observed = f"AR outpaced revenue in {flag_qs.sum()}/{len(flag_qs)} recent quarters"
        elif flag_qs.sum() > 0:
            status = "yellow"
            observed = f"AR outpaced revenue in {flag_qs.sum()}/{len(flag_qs)} recent quarters"
        else:
            status = "green"
            observed = "AR growth in line with or below revenue growth"

    return QualityFlag(
        name="Receivables vs Revenue Growth",
        status=status,
        trigger_condition="AR growth > revenue growth for 2 consecutive quarters",
        observed_value=observed if observed else None,
        threshold="2 consecutive quarters trigger yellow/red",
        explanation=(
            "Receivables growing faster than revenue may indicate channel stuffing, "
            "aggressive revenue recognition, or collection issues." if status != "green"
            else "No anomaly detected."
        ),
    )


def _flag_inventory(
    q_income: Optional[pd.DataFrame],
    q_balance: Optional[pd.DataFrame],
) -> QualityFlag:
    rev = _get_q_series(q_income, "Total Revenue", "Revenue")
    inv = _get_q_series(q_balance, "Inventory")

    if inv is None or rev is None:
        return QualityFlag(
            name="Inventory vs Revenue Growth",
            status="green",
            trigger_condition="Inventory growth > revenue growth for 2 consecutive quarters",
            threshold="2 consecutive quarters",
            explanation="No inventory data or not applicable for this business.",
        )

    rev_chg = _pct_change(rev)
    inv_chg = _pct_change(inv)
    if rev_chg is None or inv_chg is None:
        status, observed = "green", "Insufficient data"
    else:
        combined = pd.concat([rev_chg.rename("rev"), inv_chg.rename("inv")], axis=1).dropna()
        flag_qs = (combined["inv"] > combined["rev"]).tail(4)
        if flag_qs.sum() >= 2:
            status = "yellow"
            observed = f"Inventory outpaced revenue in {flag_qs.sum()} of last {len(flag_qs)} quarters"
        else:
            status = "green"
            observed = "Inventory growth in line with revenue"

    return QualityFlag(
        name="Inventory vs Revenue Growth",
        status=status,
        trigger_condition="Inventory growth > revenue growth for 2 consecutive quarters",
        observed_value=observed,
        threshold="2 consecutive quarters",
        explanation=(
            "Inventory building faster than revenue could signal demand softness or over-ordering."
            if status != "green" else "No anomaly detected."
        ),
    )


def _flag_dso(
    q_income: Optional[pd.DataFrame],
    q_balance: Optional[pd.DataFrame],
) -> QualityFlag:
    rev = _get_q_series(q_income, "Total Revenue", "Revenue")
    rec = _get_q_series(q_balance, "Net Receivables", "Accounts Receivable", "Receivables")

    if rev is None or rec is None or len(rev) < 5:
        return QualityFlag(
            name="DSO Trend",
            status="green",
            trigger_condition="DSO rising >15% YoY",
            threshold=">15% YoY DSO increase",
            explanation="Insufficient data to compute DSO trend.",
        )

    dso = (rec / rev) * 91.25  # approximate quarterly DSO (days)
    dso_yoy = dso.pct_change(periods=4).dropna()
    if dso_yoy.empty:
        return QualityFlag(
            name="DSO Trend",
            status="green",
            trigger_condition="DSO rising >15% YoY",
            threshold=">15% YoY DSO increase",
            explanation="Not enough history for YoY comparison.",
        )

    latest_yoy = _safe(dso_yoy.iloc[-1])
    if latest_yoy and latest_yoy > 0.15:
        status = "red"
        observed = f"DSO up {latest_yoy*100:.1f}% YoY"
    elif latest_yoy and latest_yoy > 0.05:
        status = "yellow"
        observed = f"DSO up {latest_yoy*100:.1f}% YoY"
    else:
        status = "green"
        observed = f"DSO YoY change: {(latest_yoy or 0)*100:.1f}%"

    return QualityFlag(
        name="DSO Trend",
        status=status,
        trigger_condition="DSO rising >15% YoY",
        observed_value=observed,
        threshold=">15% YoY increase",
        explanation=(
            "Rising DSO indicates customers are taking longer to pay, which may signal "
            "collection risk or aggressive revenue recognition." if status != "green"
            else "DSO stable or improving."
        ),
    )


def _flag_ocf_vs_netincome(
    q_income: Optional[pd.DataFrame],
    q_cashflow: Optional[pd.DataFrame],
) -> QualityFlag:
    net_inc = _get_q_series(q_income, "Net Income", "Net Income Common Stockholders")
    ocf = _get_q_series(q_cashflow, "Operating Cash Flow", "Total Cash From Operating Activities")

    if net_inc is None or ocf is None:
        return QualityFlag(
            name="OCF vs Net Income",
            status="green",
            trigger_condition="Trailing 4Q OCF < trailing 4Q Net Income",
            threshold="OCF / NI < 1.0 on TTM basis",
            explanation="Insufficient data.",
        )

    combined = pd.concat([net_inc.rename("ni"), ocf.rename("ocf")], axis=1).dropna()
    if len(combined) < 4:
        return QualityFlag(
            name="OCF vs Net Income",
            status="green",
            trigger_condition="Trailing 4Q OCF < trailing 4Q Net Income",
            threshold="OCF / NI < 1.0 on TTM basis",
            explanation="Less than 4 quarters available.",
        )

    ttm_ni = float(combined["ni"].tail(4).sum())
    ttm_ocf = float(combined["ocf"].tail(4).sum())
    if ttm_ni == 0:
        return QualityFlag(
            name="OCF vs Net Income",
            status="green",
            trigger_condition="Trailing 4Q OCF < trailing 4Q Net Income",
            threshold="OCF / NI < 1.0 on TTM basis",
            explanation="Net income is zero or negative — ratio not meaningful.",
            not_meaningful=True if False else False,
        )

    ratio = ttm_ocf / ttm_ni
    if ratio < 0.5:
        status = "red"
    elif ratio < 1.0:
        status = "yellow"
    else:
        status = "green"

    return QualityFlag(
        name="OCF vs Net Income",
        status=status,
        trigger_condition="Trailing 4Q OCF < trailing 4Q Net Income",
        observed_value=f"OCF/NI ratio (TTM): {ratio:.2f}x  |  OCF=${ttm_ocf/1e9:.1f}B  NI=${ttm_ni/1e9:.1f}B",
        threshold="Ratio < 1.0 = yellow; < 0.5 = red",
        explanation=(
            "OCF persistently below net income suggests accruals are flattering GAAP earnings. "
            "Common causes: rising working capital, capitalised costs, or non-cash income." if status != "green"
            else "OCF exceeds net income — strong earnings quality."
        ),
    )


def _flag_sbc(
    q_income: Optional[pd.DataFrame],
    q_cashflow: Optional[pd.DataFrame],
) -> QualityFlag:
    rev = _get_q_series(q_income, "Total Revenue", "Revenue")
    sbc = _get_q_series(q_cashflow, "Stock Based Compensation", "Share Based Compensation")

    if rev is None or sbc is None:
        return QualityFlag(
            name="Stock-Based Compensation",
            status="green",
            trigger_condition="SBC % revenue rising AND >10%",
            threshold=">10% revenue AND rising trend",
            explanation="Insufficient data to evaluate SBC.",
        )

    combined = pd.concat([rev.rename("rev"), sbc.rename("sbc")], axis=1).dropna()
    if len(combined) < 4:
        return QualityFlag(
            name="Stock-Based Compensation",
            status="green",
            trigger_condition="SBC % revenue rising AND >10%",
            threshold=">10% revenue AND rising trend",
            explanation="Less than 4 quarters available.",
        )

    sbc_pct = combined["sbc"] / combined["rev"]
    recent_pct = float(sbc_pct.tail(4).mean())
    trend = float(sbc_pct.tail(4).values[-1]) - float(sbc_pct.tail(4).values[0])
    rising = trend > 0.01

    if recent_pct > 0.10 and rising:
        status = "red"
    elif recent_pct > 0.10:
        status = "yellow"
    elif rising and recent_pct > 0.05:
        status = "yellow"
    else:
        status = "green"

    return QualityFlag(
        name="Stock-Based Compensation",
        status=status,
        trigger_condition="SBC % revenue rising AND >10%",
        observed_value=f"Avg SBC/revenue last 4Q: {recent_pct*100:.1f}%  |  Trend: {'rising' if rising else 'stable/falling'}",
        threshold=">10% revenue AND rising trend",
        explanation=(
            "High and rising SBC dilutes shareholders and often overstates non-GAAP earnings quality."
            if status != "green" else "SBC within normal range."
        ),
    )


# ── Orchestrator ──────────────────────────────────────────────────────────────

def get_quality_panel(ticker: str) -> QualityPanel:
    yf_t = yf.Ticker(ticker)
    q_income, q_balance, q_cashflow = None, None, None
    try:
        q_income = yf_t.quarterly_financials
        q_balance = yf_t.quarterly_balance_sheet
        q_cashflow = yf_t.quarterly_cashflow
    except Exception as exc:
        logger.warning("Quarterly data fetch failed for %s: %s", ticker, exc)

    flags = [
        _flag_receivables_growing_faster_than_revenue(q_income, q_balance),
        _flag_inventory(q_income, q_balance),
        _flag_dso(q_income, q_balance),
        _flag_ocf_vs_netincome(q_income, q_cashflow),
        _flag_sbc(q_income, q_cashflow),
    ]

    reds = [f for f in flags if f.status == "red"]
    yellows = [f for f in flags if f.status == "yellow"]
    if reds:
        overall = "concerning"
        summary = f"{len(reds)} red flag(s) detected: {', '.join(f.name for f in reds)}."
    elif yellows:
        overall = "mixed"
        summary = f"{len(yellows)} yellow flag(s): {', '.join(f.name for f in yellows)}."
    else:
        overall = "clean"
        summary = "No quality-of-earnings flags triggered on available data."

    return QualityPanel(flags=flags, overall=overall, summary=summary)
