"""EDGAR-backed fundamentals provider — keyless, uses SEC XBRL + submissions API."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import httpx

import config
from analysis.schemas import Fundamentals
from data.edgar import (
    _get_cik_direct,
    _rate_limit,
    compute_edgar_ttm,
    extract_xbrl_instant,
    get_xbrl_facts,
)


def _sic_to_sector(sic_str: str) -> Optional[str]:
    """Map SEC SIC code to a broad GICS-like sector name."""
    try:
        sic = int(sic_str)
    except (ValueError, TypeError):
        return None
    # Ordered from most-specific to least-specific
    if 7370 <= sic <= 7379:
        return "Technology"           # Computer programming / software
    if 3670 <= sic <= 3679:
        return "Technology"           # Semiconductors
    if 3600 <= sic <= 3699:
        return "Technology"           # Electronic components
    if 8000 <= sic <= 8099:
        return "Healthcare"           # Health services
    if 3800 <= sic <= 3899:
        return "Healthcare"           # Medical instruments
    if 4800 <= sic <= 4899:
        return "Communication Services"
    if 7800 <= sic <= 7999:
        return "Communication Services"   # Amusement / entertainment
    if 6500 <= sic <= 6699:
        return "Real Estate"
    if 6000 <= sic <= 6999:
        return "Financials"           # Banks, insurance, investment
    if 3700 <= sic <= 3799:
        return "Consumer Discretionary"   # Autos
    if 2000 <= sic <= 2199:
        return "Consumer Staples"         # Food & tobacco
    if 5000 <= sic <= 5999:
        return "Consumer Discretionary"   # Wholesale & retail
    if 7000 <= sic <= 7799:
        return "Consumer Discretionary"   # Hotels, services
    if 2800 <= sic <= 2999:
        return "Materials"                # Chemicals
    if 1000 <= sic <= 1499:
        return "Materials"                # Mining
    if 100 <= sic <= 999:
        return "Materials"                # Agriculture
    if 3000 <= sic <= 3399:
        return "Materials"                # Primary metals
    if 1500 <= sic <= 1799:
        return "Industrials"              # Construction
    if 3400 <= sic <= 3599:
        return "Industrials"              # Machinery
    if 4000 <= sic <= 4799:
        return "Industrials"              # Transportation
    if 8700 <= sic <= 8999:
        return "Industrials"              # Engineering / management
    return None


def _ttm(facts: dict, *concepts: str) -> Optional[float]:
    """Try each XBRL concept in order; return first successful TTM value."""
    for concept in concepts:
        result = compute_edgar_ttm(facts, concept)
        if result is not None:
            return result[0]
    return None


def _instant(facts: dict, *concepts: str) -> Optional[float]:
    """Try each XBRL concept in order; return first successful balance-sheet value."""
    for concept in concepts:
        result = extract_xbrl_instant(facts, concept)
        if result is not None:
            return result[0]
    return None


def _stooq_price(ticker: str) -> Optional[float]:
    """Best-effort current price from Stooq; returns None on any failure."""
    try:
        from data.providers.stooq import StooqProvider
        return StooqProvider().get_price(ticker)
    except Exception:
        return None


class EdgarFundamentalsProvider:
    name = "edgar"

    def available(self) -> bool:
        return True

    def get_price(self, ticker: str) -> float:
        raise NotImplementedError("EDGAR does not provide real-time prices")

    def get_fundamentals(self, ticker: str) -> Fundamentals:
        cik = _get_cik_direct(ticker)
        if not cik:
            raise ValueError(f"EDGAR: CIK not found for {ticker}")

        # Company profile from SEC submissions endpoint
        _rate_limit()
        resp = httpx.get(
            config.EDGAR_SUBMISSIONS_URL.format(cik=cik),
            headers={"User-Agent": config.EDGAR_USER_AGENT},
            timeout=15,
        )
        if resp.status_code != 200:
            raise ValueError(
                f"EDGAR submissions HTTP {resp.status_code} for {ticker}"
            )
        sub = resp.json()

        name: Optional[str] = sub.get("name") or None
        sic = str(sub.get("sic", ""))
        sic_desc: Optional[str] = sub.get("sicDescription") or None
        sector = _sic_to_sector(sic)
        industry = sic_desc

        # Financial data from XBRL facts (large JSON, 30-day cached)
        facts = get_xbrl_facts(ticker)

        revenue_ttm: Optional[float] = None
        net_income_ttm: Optional[float] = None
        gross_profit_ttm: Optional[float] = None
        operating_income_ttm: Optional[float] = None
        total_debt: Optional[float] = None
        cash: Optional[float] = None
        shares_out: Optional[float] = None
        equity: Optional[float] = None

        if facts:
            revenue_ttm = _ttm(
                facts,
                "Revenues",
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax",
                "SalesRevenueGoodsNet",
            )
            net_income_ttm = _ttm(
                facts,
                "NetIncomeLoss",
                "ProfitLoss",
                "NetIncomeLossAvailableToCommonStockholdersBasic",
            )
            gross_profit_ttm = _ttm(facts, "GrossProfit")
            operating_income_ttm = _ttm(facts, "OperatingIncomeLoss")

            total_debt = _instant(
                facts,
                "DebtAndCapitalLeaseObligations",
                "LongTermDebt",
                "LongTermDebtAndCapitalLeaseObligations",
                "LongTermDebtNoncurrent",
            )
            cash = _instant(
                facts,
                "CashAndCashEquivalentsAtCarryingValue",
                "CashCashEquivalentsAndShortTermInvestments",
                "CashAndCashEquivalentsAndShortTermInvestments",
            )
            shares_out = _instant(
                facts,
                "CommonStockSharesOutstanding",
                "EntityCommonStockSharesOutstanding",
            )
            equity = _instant(
                facts,
                "StockholdersEquity",
                "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
            )

        # Derive margin ratios
        gross_margin: Optional[float] = None
        operating_margin: Optional[float] = None
        net_margin: Optional[float] = None
        roe: Optional[float] = None
        debt_to_equity: Optional[float] = None

        if revenue_ttm and revenue_ttm > 0:
            if gross_profit_ttm is not None:
                gross_margin = gross_profit_ttm / revenue_ttm
            if operating_income_ttm is not None:
                operating_margin = operating_income_ttm / revenue_ttm
            if net_income_ttm is not None:
                net_margin = net_income_ttm / revenue_ttm

        if net_income_ttm and equity and equity > 0:
            roe = net_income_ttm / equity
        if total_debt is not None and equity and equity > 0:
            debt_to_equity = total_debt / equity

        # Current price from Stooq (keyless, reliable)
        current_price = _stooq_price(ticker)

        # Valuation ratios that need price
        market_cap: Optional[float] = None
        pe_ttm: Optional[float] = None
        price_to_sales: Optional[float] = None

        if current_price and shares_out and shares_out > 0:
            market_cap = current_price * shares_out
        if market_cap and net_income_ttm and net_income_ttm > 0:
            pe_ttm = market_cap / net_income_ttm
        if market_cap and revenue_ttm and revenue_ttm > 0:
            price_to_sales = market_cap / revenue_ttm

        # Require at least a company name to call this a success
        if not name:
            raise ValueError(f"EDGAR: no company name resolved for {ticker}")

        return Fundamentals(
            ticker=ticker,
            name=name,
            sector=sector,
            industry=industry,
            market_cap=market_cap,
            current_price=current_price,
            pe_ttm=pe_ttm,
            price_to_sales=price_to_sales,
            gross_margin=gross_margin,
            operating_margin=operating_margin,
            net_margin=net_margin,
            roe=roe,
            debt_to_equity=debt_to_equity,
            revenue_ttm=revenue_ttm,
            net_income_ttm=net_income_ttm,
            total_debt=total_debt,
            cash=cash,
            shares_outstanding=shares_out,
            fetched_at=datetime.utcnow(),
        )
