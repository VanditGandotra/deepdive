"""Market data provider protocol and shared types."""
from __future__ import annotations

from typing import Protocol

from analysis.schemas import Fundamentals


class MarketProvider(Protocol):
    name: str

    def available(self) -> bool: ...
    def get_price(self, ticker: str) -> float: ...
    def get_fundamentals(self, ticker: str) -> Fundamentals: ...
