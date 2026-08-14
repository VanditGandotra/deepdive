"""Stooq keyless CSV price provider."""
from __future__ import annotations

import csv
import io
from datetime import date, timedelta

import httpx

import config
from analysis.schemas import Fundamentals, PriceBar, PriceData


class StooqProvider:
    name = "stooq"

    def available(self) -> bool:
        return True

    def _fetch_csv(self, ticker: str) -> str:
        stooq_sym = ticker.lower()
        url = config.STOOQ_PRICE_URL.format(ticker=stooq_sym)
        resp = httpx.get(url, timeout=30)
        resp.raise_for_status()
        text = resp.content.decode("utf-8")
        stripped = text.strip()
        if not stripped or stripped.lower().startswith("no data"):
            raise ValueError(f"Stooq: no data for {ticker}")
        return text

    def get_price(self, ticker: str) -> float:
        text = self._fetch_csv(ticker)
        try:
            reader = csv.DictReader(io.StringIO(text))
            rows = list(reader)
        except Exception as exc:
            raise ValueError(f"Stooq: CSV parse failed for {ticker}: {exc}") from exc
        if not rows:
            raise ValueError(f"Stooq: no rows for {ticker}")
        last_row = rows[-1]
        close_val = last_row.get("Close", "")
        if not close_val or close_val.strip() == "":
            raise ValueError(f"Stooq: missing Close field for {ticker}")
        return float(close_val)

    def get_prices(self, ticker: str, period: str = "5y") -> PriceData:
        """Return full daily OHLCV history from Stooq as a PriceData object."""
        text = self._fetch_csv(ticker)
        try:
            reader = csv.DictReader(io.StringIO(text))
            rows = list(reader)
        except Exception as exc:
            raise ValueError(f"Stooq: CSV parse failed for {ticker}: {exc}") from exc
        if not rows:
            raise ValueError(f"Stooq: no rows for {ticker}")

        years = int(period.rstrip("y")) if period.endswith("y") else 5
        cutoff = date.today() - timedelta(days=years * 365)

        bars: list[PriceBar] = []
        for row in rows:
            try:
                dt = date.fromisoformat(row["Date"])
                if dt < cutoff:
                    continue
                close_val = row.get("Close", "")
                if not close_val or close_val.strip() == "":
                    continue
                bars.append(PriceBar(
                    date=dt,
                    open=float(row.get("Open") or row["Close"]),
                    high=float(row.get("High") or row["Close"]),
                    low=float(row.get("Low") or row["Close"]),
                    close=float(row["Close"]),
                    volume=int(float(row.get("Volume") or 0)),
                ))
            except (KeyError, ValueError):
                continue

        if not bars:
            raise ValueError(f"Stooq: no usable price bars for {ticker}")
        return PriceData(ticker=ticker, currency="USD", bars=bars)

    def get_fundamentals(self, ticker: str) -> Fundamentals:
        raise NotImplementedError("Stooq only provides price data")
