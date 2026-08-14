"""FMP (Financial Modeling Prep) market data provider."""
from __future__ import annotations

from datetime import datetime

import httpx

import config
from analysis.schemas import Fundamentals


def _check_fmp_response(resp: httpx.Response, ticker: str, endpoint: str) -> None:
    """Raise ValueError with a secret-free message on non-200 responses."""
    if resp.status_code != 200:
        raise ValueError(
            f"FMP HTTP {resp.status_code} for {ticker} ({endpoint}) — "
            "check key validity or endpoint availability"
        )


class FmpMarketProvider:
    name = "fmp"

    def available(self) -> bool:
        return bool(config.FMP_API_KEY)

    def get_price(self, ticker: str) -> float:
        url = config.FMP_QUOTE_URL.format(symbol=ticker)
        resp = httpx.get(url, params={"apikey": config.FMP_API_KEY}, timeout=15)
        _check_fmp_response(resp, ticker, "quote")
        data = resp.json()
        if isinstance(data, dict) and "Error Message" in data:
            raise ValueError(f"FMP quote error for {ticker}: {data['Error Message']}")
        if not data:
            raise ValueError(f"FMP quote: no data for {ticker}")
        return float(data[0]["price"])

    def get_fundamentals(self, ticker: str) -> Fundamentals:
        # Fetch profile
        profile_url = config.FMP_PROFILE_URL.format(symbol=ticker)
        profile_resp = httpx.get(
            profile_url, params={"apikey": config.FMP_API_KEY}, timeout=15
        )
        _check_fmp_response(profile_resp, ticker, "profile")
        profile_data = profile_resp.json()
        if isinstance(profile_data, dict) and "Error Message" in profile_data:
            raise ValueError(f"FMP profile error for {ticker}: {profile_data['Error Message']}")
        if not isinstance(profile_data, list) or not profile_data:
            raise ValueError(
                f"FMP profile: unexpected response for {ticker}: {str(profile_data)[:120]}"
            )
        profile = profile_data[0]

        # Fetch key metrics TTM
        metrics_url = config.FMP_KEY_METRICS_TTM_URL.format(symbol=ticker)
        metrics_resp = httpx.get(
            metrics_url, params={"apikey": config.FMP_API_KEY}, timeout=15
        )
        _check_fmp_response(metrics_resp, ticker, "key-metrics-ttm")
        metrics_data = metrics_resp.json()
        if isinstance(metrics_data, dict) and "Error Message" in metrics_data:
            raise ValueError(f"FMP key-metrics error for {ticker}: {metrics_data['Error Message']}")
        metrics = metrics_data[0] if isinstance(metrics_data, list) and metrics_data else {}

        return Fundamentals(
            ticker=ticker,
            name=profile.get("companyName"),
            sector=profile.get("sector"),
            industry=profile.get("industry"),
            market_cap=profile.get("mktCap"),
            current_price=profile.get("price"),
            beta=profile.get("beta"),
            pe_ttm=metrics.get("peRatioTTM"),
            ev_ebitda=metrics.get("evToEbitdaTTM"),
            net_margin=metrics.get("netProfitMarginTTM"),
            roe=metrics.get("roeTTM"),
            roic=metrics.get("roicTTM"),
            gross_margin=metrics.get("grossProfitMarginTTM"),
            operating_margin=metrics.get("operatingProfitMarginTTM"),
            current_ratio=metrics.get("currentRatioTTM"),
            debt_to_equity=metrics.get("debtToEquityTTM"),
            revenue_growth_yoy=metrics.get("revenueGrowthTTM"),
            price_to_sales=metrics.get("priceToSalesRatioTTM"),
            price_to_fcf=metrics.get("pfcfRatioTTM"),
            peg=metrics.get("pegRatioTTM"),
            fetched_at=datetime.utcnow(),
        )
