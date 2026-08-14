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
        stooq_sym = ticker.lower()
        url = config.STOOQ_PRICE_URL.format(ticker=stooq_sym)
        resp = httpx.get(url, timeout=15)
        resp.raise_for_status()
        try:
            text = resp.content.decode("utf-8")
            stripped = text.strip()
            if not stripped or stripped.lower().startswith("no data"):
                raise ValueError(f"Stooq: no data for {ticker}")
            reader = csv.DictReader(io.StringIO(text))
            rows = list(reader)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Stooq: CSV parse failed for {ticker}: {exc}") from exc
        if not rows:
            raise ValueError(f"Stooq: no rows for {ticker}")
        last_row = rows[-1]
        close_val = last_row.get("Close", "")
        if not close_val or close_val.strip() == "":
            raise ValueError(f"Stooq: missing Close field for {ticker}")
        return float(close_val)

    def get_fundamentals(self, ticker: str) -> Fundamentals:
        raise NotImplementedError("Stooq only provides price data")
