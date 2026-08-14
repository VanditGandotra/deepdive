"""Stooq keyless CSV price provider."""
from __future__ import annotations

import csv
import io

import httpx

import config
from analysis.schemas import Fundamentals


class StooqProvider:
    name = "stooq"

    def available(self) -> bool:
        return True

    def get_price(self, ticker: str) -> float:
        url = config.STOOQ_PRICE_URL.format(ticker=ticker)
        resp = httpx.get(url, timeout=15)
        resp.raise_for_status()
        try:
            text = resp.content.decode("utf-8")
            reader = csv.DictReader(io.StringIO(text))
            rows = list(reader)
        except Exception as exc:
            raise ValueError(f"Stooq: CSV parse failed for {ticker}: {exc}") from exc
        if not rows:
            raise ValueError(f"Stooq: no rows for {ticker}")
        last_row = rows[-1]
        return float(last_row["Close"])

    def get_fundamentals(self, ticker: str) -> Fundamentals:
        raise NotImplementedError("Stooq only provides price data")
