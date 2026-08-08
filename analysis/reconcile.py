"""Phase 2: Cross-source reconciliation — yfinance TTM vs EDGAR XBRL TTM. No LLM."""
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

from analysis.schemas import Fundamentals, MetricReconciliation
from data.edgar import build_xbrl_composite, compute_edgar_ttm, extract_xbrl_instant, get_xbrl_facts
from data.market import get_fundamentals

logger = logging.getLogger(__name__)

# How many days since EDGAR TTM end before we flag staleness
_STALE_DAYS = 180
# Relative diff above which we flag a pipeline error on aligned, non-composite periods
_SANITY_THRESHOLD = 0.20


def _pct_diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b == 0:
        return None
    return abs(a - b) / abs(b)


def _days_since(date_str: str) -> Optional[int]:
    try:
        return (date.today() - date.fromisoformat(date_str)).days
    except (ValueError, TypeError):
        return None


def _stale_note(end_date: str) -> str:
    days = _days_since(end_date)
    if days is not None and days > _STALE_DAYS:
        return f" (EDGAR data ends {end_date}, {days}d ago — may lag yfinance)"
    return ""


def _diff_note(
    metric: str,
    diff_pct: Optional[float],
    edgar_end: str = "",
    classification_mismatch: bool = False,
) -> str:
    if diff_pct is None:
        return "Only one source available."
    stale = _stale_note(edgar_end) if edgar_end else ""
    if diff_pct <= 0.02:
        return f"Values agree within 2%.{stale}"
    if diff_pct > _SANITY_THRESHOLD and not stale and not classification_mismatch:
        return (
            f"⚠ {diff_pct:.1%} diff on aligned periods — likely pipeline error, "
            f"check field mapping for '{metric}'."
        )
    notes = {
        "revenue":     "TTM mismatch: possible revenue recognition or restatement difference.",
        "net_income":  "TTM mismatch: possible non-recurring items or discontinued ops.",
        "shares":      "Possible mismatch: basic vs diluted, or reporting lag.",
        "total_debt":  "Possible mismatch: operating vs finance leases, or current debt classification.",
        "cash":        "Possible mismatch: marketable securities classification varies by company.",
    }
    base = notes.get(metric, "Possible classification or restatement mismatch.")
    return base + stale


# ── EDGAR TTM for flow metrics ────────────────────────────────────────────────

def _edgar_flow_ttm(
    facts: Dict,
    concepts: List[str],
    namespace: str = "us-gaap",
) -> Tuple[Optional[float], str]:
    for concept in concepts:
        result = compute_edgar_ttm(facts, concept, namespace)
        if result is not None:
            return result
    return (None, "")


# ── Main reconciliation ───────────────────────────────────────────────────────

def get_reconciliation(ticker: str) -> List[MetricReconciliation]:
    """
    Compare yfinance vs EDGAR for 5 key metrics.

    Flow metrics (revenue, net_income): EDGAR TTM from 4 quarterly records.
    Balance-sheet metrics (shares, total_debt, cash): EDGAR composites matching
    yfinance's field definitions exactly.

    yfinance totalDebt = CommercialPaper + LongTermDebtCurrent + LongTermDebtNoncurrent
    yfinance totalCash = CashAndCashEquivalentsAtCarryingValue + MarketableSecuritiesCurrent

    All composite components must share the same balance-sheet date.
    """
    fund: Fundamentals = get_fundamentals(ticker)
    facts = get_xbrl_facts(ticker)

    if facts is None:
        return [
            MetricReconciliation(
                metric=m, yfinance_value=v, edgar_value=None,
                diff_pct=None, note="EDGAR data unavailable.", canonical="yfinance",
            )
            for m, v in [
                ("revenue",    fund.revenue_ttm),
                ("net_income", fund.net_income_ttm),
                ("shares",     fund.shares_outstanding),
                ("total_debt", fund.total_debt),
                ("cash",       fund.cash),
            ]
        ]

    recons: List[MetricReconciliation] = []

    # ── Flow metrics — TTM from 4 quarterly records ──────────────────────────
    for metric, concepts, yf_val in [
        ("revenue", [
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "Revenues",
            "SalesRevenueNet",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
        ], fund.revenue_ttm),
        ("net_income", [
            "NetIncomeLoss",
            "ProfitLoss",
            "NetIncome",
        ], fund.net_income_ttm),
    ]:
        edgar_val, edgar_end = _edgar_flow_ttm(facts, concepts)
        diff = _pct_diff(yf_val, edgar_val)
        canonical = "edgar" if edgar_val is not None else ("yfinance" if yf_val is not None else "na")
        recons.append(MetricReconciliation(
            metric=metric,
            yfinance_value=yf_val,
            edgar_value=edgar_val,
            diff_pct=diff,
            note=_diff_note(metric, diff, edgar_end),
            canonical=canonical,
        ))

    # ── Shares (point-in-time) ───────────────────────────────────────────────
    # Prefer dei:EntityCommonStockSharesOutstanding (point-in-time outstanding)
    # over us-gaap:CommonStockSharesOutstanding (which can be weighted-average diluted)
    shares_dei, shares_dei_end = _edgar_instant(facts, ["EntityCommonStockSharesOutstanding"], "dei")
    shares_gaap, shares_gaap_end = _edgar_instant(facts, ["CommonStockSharesOutstanding"], "us-gaap")
    edgar_shares = shares_dei if shares_dei is not None else shares_gaap
    edgar_shares_end = shares_dei_end if shares_dei is not None else shares_gaap_end
    shares_diff = _pct_diff(fund.shares_outstanding, edgar_shares)
    shares_note = _diff_note("shares", shares_diff, edgar_shares_end)
    if shares_dei is None and shares_gaap is not None:
        shares_note = "(using weighted-avg diluted; DEI point-in-time concept unavailable) " + shares_note
    recons.append(MetricReconciliation(
        metric="shares",
        yfinance_value=fund.shares_outstanding,
        edgar_value=edgar_shares,
        diff_pct=shares_diff,
        note=shares_note,
        canonical="edgar" if edgar_shares is not None else ("yfinance" if fund.shares_outstanding else "na"),
    ))

    # ── Total Debt (composite) ────────────────────────────────────────────────
    # yfinance totalDebt sums commercial paper + current LTD + noncurrent LTD.
    # We build the same composite from EDGAR components.
    # If LongTermDebt (consolidated) is available AND noncurrent is not filed separately,
    # use LongTermDebt as the noncurrent slot's fallback (it may already include current).
    debt_composite = build_xbrl_composite(facts, [
        {
            "label": "CommercialPaper",
            "aliases": ["CommercialPaper", "ShortTermBorrowings"],
            "required": False,  # not all companies issue commercial paper
        },
        {
            "label": "LongTermDebtCurrent",
            "aliases": ["LongTermDebtCurrent", "DebtCurrent"],
            "required": True,
        },
        {
            "label": "LongTermDebtNoncurrent",
            "aliases": [
                "LongTermDebtNoncurrent",
                "LongTermDebt",               # fallback: some companies file consolidated LTD
            ],
            "required": True,
        },
    ])

    if debt_composite["incomplete"]:
        # If component decomposition failed, try the consolidated LongTermDebt tag directly
        # (it often includes both current + noncurrent for simpler balance sheets)
        fallback_debt, fallback_debt_end = _edgar_instant(facts, ["LongTermDebt", "DebtAndCapitalLeaseObligations"])
        edgar_debt = fallback_debt
        debt_end = fallback_debt_end
        debt_components = {"LongTermDebt (consolidated fallback)": fallback_debt} if fallback_debt else {}
        composite_note = (
            f"composition incomplete: missing {', '.join(debt_composite['missing'])}; "
            f"used consolidated LongTermDebt as fallback"
        ) if fallback_debt else f"composition incomplete: missing {', '.join(debt_composite['missing'])}"
        classification_mismatch = True
    else:
        # Check for date consistency across components
        if debt_composite["date_mismatch"]:
            edgar_debt = None
            debt_end = ""
            debt_components = debt_composite["components"]
            composite_note = "component balance-sheet dates do not match — not comparable"
            classification_mismatch = False
        else:
            edgar_debt = debt_composite["value"]
            debt_end = debt_composite["end"]
            debt_components = debt_composite["components"]
            composite_note = ""
            classification_mismatch = False

    debt_diff = _pct_diff(fund.total_debt, edgar_debt)
    recons.append(MetricReconciliation(
        metric="total_debt",
        yfinance_value=fund.total_debt,
        edgar_value=edgar_debt,
        diff_pct=debt_diff,
        note=_diff_note("total_debt", debt_diff, debt_end, classification_mismatch),
        canonical="edgar" if edgar_debt is not None else ("yfinance" if fund.total_debt else "na"),
        components=debt_components or None,
        composite_note=composite_note,
    ))

    # ── Cash + Short-Term Investments (composite) ─────────────────────────────
    # yfinance totalCash = CashAndCashEquivalentsAtCarryingValue + short-term investments.
    # AAPL calls short-term investments "MarketableSecuritiesCurrent" in XBRL;
    # other companies use ShortTermInvestments or AvailableForSaleSecuritiesCurrent.
    cash_composite = build_xbrl_composite(facts, [
        {
            "label": "CashAndEquivalents",
            "aliases": ["CashAndCashEquivalentsAtCarryingValue", "CashAndCashEquivalents"],
            "required": True,
        },
        {
            "label": "ShortTermInvestments",
            "aliases": [
                "MarketableSecuritiesCurrent",
                "ShortTermInvestments",
                "AvailableForSaleSecuritiesCurrent",
                "OtherShortTermInvestments",
            ],
            "required": False,  # not all companies hold ST investments; omit gracefully
        },
    ])

    if cash_composite["incomplete"]:
        # CashAndCashEquivalents is required — if missing EDGAR data is truly unavailable
        edgar_cash = None
        cash_end = ""
        cash_components: Dict = {}
        cash_composite_note = f"composition incomplete: missing {', '.join(cash_composite['missing'])}"
        cash_classification_mismatch = False
    elif cash_composite["date_mismatch"]:
        edgar_cash = None
        cash_end = ""
        cash_components = cash_composite["components"]
        cash_composite_note = "component balance-sheet dates do not match — not comparable"
        cash_classification_mismatch = False
    else:
        edgar_cash = cash_composite["value"]
        cash_end = cash_composite["end"]
        cash_components = cash_composite["components"]
        cash_composite_note = ""
        # If short-term investments not found, note the partial match
        if "ShortTermInvestments" not in cash_composite["components"]:
            cash_classification_mismatch = True
            cash_composite_note = "short-term investments tag not found for this company; EDGAR value = cash & equivalents only"
        else:
            cash_classification_mismatch = False

    cash_diff = _pct_diff(fund.cash, edgar_cash)
    recons.append(MetricReconciliation(
        metric="cash",
        yfinance_value=fund.cash,
        edgar_value=edgar_cash,
        diff_pct=cash_diff,
        note=_diff_note("cash", cash_diff, cash_end, cash_classification_mismatch),
        canonical="edgar" if edgar_cash is not None else ("yfinance" if fund.cash else "na"),
        components=cash_components or None,
        composite_note=cash_composite_note,
    ))

    return recons


def _edgar_instant(
    facts: Dict,
    concepts: List[str],
    namespace: str = "us-gaap",
) -> Tuple[Optional[float], str]:
    """Try concepts in order; return (value, end_date) for first match."""
    for concept in concepts:
        result = extract_xbrl_instant(facts, concept, namespace)
        if result is not None:
            return result
    return (None, "")
