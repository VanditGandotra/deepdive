from __future__ import annotations

from datetime import date, datetime
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class Source(BaseModel):
    label: str
    url: Optional[str] = None
    retrieved_at: datetime = Field(default_factory=datetime.utcnow)


class Driver(BaseModel):
    name: str
    value: float = 0.0
    unit: str = ""
    direction: Literal["positive", "negative", "neutral"] = "neutral"


class Scenario(BaseModel):
    scenario: Literal["bull", "base", "bear"]
    price_target: Optional[float] = None
    probability: float = Field(ge=0.0, le=1.0)
    horizon_years: int = 1
    drivers: List[Driver] = []
    narrative: str = ""
    implied_return: Optional[float] = None


class TickerAnalysis(BaseModel):
    ticker: str
    company_name: Optional[str] = None
    as_of: date
    analysis_version: str = "v1"
    current_price: Optional[float] = None
    market_cap: Optional[float] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    # Flat fundamentals for compare table
    pe_ttm: Optional[float] = None
    pe_forward: Optional[float] = None
    ev_ebitda: Optional[float] = None
    price_to_sales: Optional[float] = None
    gross_margin: Optional[float] = None
    net_margin: Optional[float] = None
    revenue_growth_yoy: Optional[float] = None
    fcf_yield: Optional[float] = None
    debt_to_equity: Optional[float] = None
    roe: Optional[float] = None
    # Sub-models stored as Any to avoid circular imports
    fundamentals: Optional[Any] = None
    dcf: Optional[Any] = None
    call_delta: Optional[Any] = None
    quality: Optional[Any] = None
    positioning: Optional[Any] = None
    headlines: List[Any] = []
    kpis: List[Any] = []
    # Scenarios
    scenarios: List[Scenario] = []
    expected_return_1y: Optional[float] = None
    expected_return_3y: Optional[float] = None
    expected_return_5y: Optional[float] = None
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    data_gaps: List[str] = []
    sources: List[Source] = []
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    @model_validator(mode="after")
    def _normalize_and_derive(self) -> "TickerAnalysis":
        # Normalize probabilities
        if self.scenarios:
            total = sum(s.probability for s in self.scenarios)
            if total > 0 and abs(total - 1.0) > 0.001:
                for s in self.scenarios:
                    s.probability = round(s.probability / total, 6)
        # Derive implied_return per scenario
        if self.current_price and self.current_price > 0:
            for s in self.scenarios:
                if s.price_target is not None:
                    s.implied_return = (s.price_target / self.current_price) - 1.0
        # Derive expected returns
        self._compute_expected_returns()
        return self

    def _compute_expected_returns(self) -> None:
        if not self.scenarios or not self.current_price:
            return
        for horizon in [1, 3, 5]:
            weighted = 0.0
            for s in self.scenarios:
                if s.implied_return is not None:
                    # annualize: (1 + total_return)^(1/horizon) - 1
                    annual = (1 + s.implied_return) ** (1.0 / horizon) - 1.0
                    weighted += s.probability * annual
            if horizon == 1:
                self.expected_return_1y = round(weighted, 6)
            elif horizon == 3:
                self.expected_return_3y = round(weighted, 6)
            else:
                self.expected_return_5y = round(weighted, 6)
