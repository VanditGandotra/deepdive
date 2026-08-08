"""Tests for net_margin derivation in get_fundamentals: must use annual income statement,
not yfinance info.profitMargins (which reflects TTM and can be polluted by one-time items)."""
from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data.market import get_fundamentals


def _make_income_stmt(revenues: list[float], net_incomes: list[float]) -> pd.DataFrame:
    """Build a fake yfinance financials DataFrame (columns = dates, rows = line items)."""
    dates = pd.to_datetime(["2025-12-31", "2024-12-31", "2023-12-31", "2022-12-31"][: len(revenues)])
    df = pd.DataFrame(
        {
            "Total Revenue": revenues,
            "Net Income": net_incomes,
            "Gross Profit": [r * 0.6 for r in revenues],
            "Operating Income": [r * 0.3 for r in revenues],
            "Interest Expense": [-500_000_000.0] * len(revenues),
        },
        index=dates,
    ).T  # yfinance returns rows=line items, cols=dates
    return df


def _make_cashflow(ocf: float = 100e9, capex: float = -20e9) -> pd.DataFrame:
    dates = pd.to_datetime(["2025-12-31"])
    df = pd.DataFrame(
        {"Operating Cash Flow": [ocf], "Capital Expenditure": [capex]},
        index=dates,
    ).T
    return df


def _mock_yf_ticker(
    info_overrides: dict | None = None,
    revenue: float = 400e9,
    net_income: float = 130e9,
    # intentionally bad profitMargins to prove we don't use it
    profit_margins: float = 0.99,
) -> MagicMock:
    mock = MagicMock()
    base_info = {
        "longName": "Test Corp",
        "sector": "Technology",
        "industry": "Software",
        "marketCap": 2e12,
        "enterpriseValue": 2e12,
        "trailingPE": 25.0,
        "forwardPE": 20.0,
        "pegRatio": 1.5,
        "enterpriseToEbitda": 15.0,
        "priceToSalesTrailing12Months": 5.0,
        "grossMargins": 0.60,
        "operatingMargins": 0.32,
        # purposely wrong to verify we override from income statement
        "profitMargins": profit_margins,
        "returnOnEquity": 0.30,
        "returnOnAssets": 0.15,
        "currentRatio": 2.0,
        "debtToEquity": 0.1,
        "ebitdaMargins": 0.38,
        "revenueGrowth": 0.15,
        "earningsGrowth": 0.30,
        "totalRevenue": revenue,
        "ebitda": revenue * 0.38,
        "netIncomeToCommon": net_income,
        "totalDebt": 20e9,
        "totalCash": 100e9,
        "sharesOutstanding": 12e9,
        "currentPrice": 175.0,
        "beta": 1.1,
    }
    if info_overrides:
        base_info.update(info_overrides)
    mock.info = base_info
    mock.financials = _make_income_stmt([revenue], [net_income])
    mock.cashflow = _make_cashflow()
    return mock


class TestNetMarginFromAnnualStatement:

    def _run(self, revenue: float, net_income: float, profit_margins_polluted: float):
        mock_ticker = _mock_yf_ticker(
            revenue=revenue,
            net_income=net_income,
            profit_margins=profit_margins_polluted,
        )
        with (
            patch("data.market.yf.Ticker", return_value=mock_ticker),
            patch("data.market.get_cache_obj", return_value=None),
            patch("data.market.set_cache_obj"),
            patch("data.market.record_freshness"),
        ):
            return get_fundamentals("TEST")

    def test_normal_margin_derived_from_annual_statement(self) -> None:
        """When income statement is available, net_margin uses it, not profitMargins."""
        fund = self._run(revenue=400e9, net_income=128e9, profit_margins_polluted=0.99)
        # 128/400 = 0.32 — should NOT be 0.99
        assert fund.net_margin == pytest.approx(128e9 / 400e9, rel=1e-4)

    def test_one_time_item_does_not_pollute_margin(self) -> None:
        """Simulates GOOG Q2 2026 scenario: huge one-time gain in TTM (profitMargins=0.548)
        while annual statement correctly shows 32.8%."""
        fund = self._run(
            revenue=402e9,
            net_income=132e9,
            profit_margins_polluted=0.548,  # the polluted TTM value
        )
        expected = 132e9 / 402e9  # ~0.328
        assert fund.net_margin == pytest.approx(expected, rel=1e-3)
        assert fund.net_margin < 0.40, "Net margin must not reflect TTM one-time items"

    def test_fallback_to_info_when_no_income_statement(self) -> None:
        """If income statement unavailable, fall back to info.profitMargins."""
        mock_ticker = _mock_yf_ticker(profit_margins=0.28)
        mock_ticker.financials = None
        with (
            patch("data.market.yf.Ticker", return_value=mock_ticker),
            patch("data.market.get_cache_obj", return_value=None),
            patch("data.market.set_cache_obj"),
            patch("data.market.record_freshness"),
        ):
            fund = get_fundamentals("TEST")
        assert fund.net_margin == pytest.approx(0.28, rel=1e-4)

    def test_interest_coverage_from_ebit_over_interest(self) -> None:
        """interest_coverage must be EBIT / |interest expense|, not ebitdaMargins."""
        # income stmt: Operating Income = 0.32 * 400B = 128B, Interest Expense = -500M
        fund = self._run(revenue=400e9, net_income=130e9, profit_margins_polluted=0.99)
        # EBIT ~128B / 0.5B = ~256x
        if fund.interest_coverage is not None:
            assert fund.interest_coverage > 10, (
                f"interest_coverage={fund.interest_coverage:.2f} — looks like ebitdaMargins "
                "proxy was used instead of EBIT/interest"
            )
