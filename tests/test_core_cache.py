"""Tests for core/cache.py — cache round-trip, no API calls."""
from __future__ import annotations

import time
from datetime import date, datetime
from unittest.mock import patch

import pytest

from core.schemas import Scenario, Source, TickerAnalysis
from core.cache import get_cached_analysis, save_analysis, ANALYSIS_VERSION


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_analysis(ticker: str = "TSLA", as_of: date = date(2026, 1, 1), **kwargs) -> TickerAnalysis:
    return TickerAnalysis(ticker=ticker, as_of=as_of, **kwargs)


# ── round-trip ────────────────────────────────────────────────────────────────

class TestCacheRoundTrip:
    def test_save_and_retrieve(self) -> None:
        ta = _make_analysis(
            ticker="AAPL",
            as_of=date(2026, 6, 1),
            company_name="Apple Inc.",
            current_price=195.5,
            market_cap=3_000_000_000_000.0,
            sector="Technology",
            confidence=0.8,
            data_gaps=["no transcripts"],
        )
        save_analysis(ta)
        retrieved = get_cached_analysis("AAPL", date(2026, 6, 1), ANALYSIS_VERSION)
        assert retrieved is not None
        assert retrieved.ticker == "AAPL"
        assert retrieved.company_name == "Apple Inc."
        assert retrieved.current_price == pytest.approx(195.5)
        assert retrieved.market_cap == pytest.approx(3_000_000_000_000.0)
        assert retrieved.sector == "Technology"
        assert retrieved.confidence == pytest.approx(0.8)
        assert "no transcripts" in retrieved.data_gaps

    def test_ticker_is_upper_cased_as_db_key(self) -> None:
        """save_analysis uppercases the ticker in the DB primary key so get_cached_analysis
        finds it with the uppercased key.  The model field retains the original casing."""
        ta = _make_analysis(ticker="msft", as_of=date(2026, 6, 2))
        save_analysis(ta)
        # DB key is uppercased — lookup must succeed
        retrieved = get_cached_analysis("MSFT", date(2026, 6, 2), ANALYSIS_VERSION)
        assert retrieved is not None
        # The deserialized model field retains whatever was in the JSON (lowercase "msft")
        assert retrieved.ticker.upper() == "MSFT"

    def test_cache_miss_returns_none(self) -> None:
        result = get_cached_analysis("NONEXISTENT_ZZZZZ", date(2099, 12, 31), ANALYSIS_VERSION)
        assert result is None

    def test_different_dates_are_independent(self) -> None:
        ta1 = _make_analysis(ticker="GOOG", as_of=date(2026, 1, 1), current_price=100.0)
        ta2 = _make_analysis(ticker="GOOG", as_of=date(2026, 1, 2), current_price=200.0)
        save_analysis(ta1)
        save_analysis(ta2)
        r1 = get_cached_analysis("GOOG", date(2026, 1, 1), ANALYSIS_VERSION)
        r2 = get_cached_analysis("GOOG", date(2026, 1, 2), ANALYSIS_VERSION)
        assert r1 is not None and r1.current_price == pytest.approx(100.0)
        assert r2 is not None and r2.current_price == pytest.approx(200.0)

    def test_different_versions_are_independent(self) -> None:
        ta_v1 = _make_analysis(ticker="META", as_of=date(2026, 3, 1), analysis_version="v1", current_price=500.0)
        ta_v2 = _make_analysis(ticker="META", as_of=date(2026, 3, 1), analysis_version="v2", current_price=600.0)
        save_analysis(ta_v1)
        save_analysis(ta_v2)
        r1 = get_cached_analysis("META", date(2026, 3, 1), "v1")
        r2 = get_cached_analysis("META", date(2026, 3, 1), "v2")
        assert r1 is not None and r1.current_price == pytest.approx(500.0)
        assert r2 is not None and r2.current_price == pytest.approx(600.0)

    def test_overwrite_updates_value(self) -> None:
        ta_old = _make_analysis(ticker="AMZN", as_of=date(2026, 4, 1), current_price=180.0)
        ta_new = _make_analysis(ticker="AMZN", as_of=date(2026, 4, 1), current_price=220.0)
        save_analysis(ta_old)
        save_analysis(ta_new)
        retrieved = get_cached_analysis("AMZN", date(2026, 4, 1), ANALYSIS_VERSION)
        assert retrieved is not None
        assert retrieved.current_price == pytest.approx(220.0)


# ── expiry ────────────────────────────────────────────────────────────────────

class TestCacheExpiry:
    def test_expired_entry_returns_none(self) -> None:
        """Save with TTL=0 so the entry is immediately expired."""
        ta = _make_analysis(ticker="NVDA", as_of=date(2026, 5, 1), current_price=888.0)
        # Patch TTL_PRICES to 0 so the entry expires immediately
        with patch("core.cache.TTL_PRICES", 0):
            save_analysis(ta)
        # A tiny sleep ensures expires_at < time.time()
        time.sleep(0.01)
        result = get_cached_analysis("NVDA", date(2026, 5, 1), ANALYSIS_VERSION)
        assert result is None


# ── nested model round-trip ───────────────────────────────────────────────────

class TestNestedModelSerialization:
    def test_scenarios_survive_round_trip(self) -> None:
        scenarios = [
            Scenario(scenario="bull", probability=0.3, price_target=200.0, narrative="Strong growth"),
            Scenario(scenario="base", probability=0.5, price_target=150.0),
            Scenario(scenario="bear", probability=0.2, price_target=100.0),
        ]
        ta = _make_analysis(
            ticker="SPY",
            as_of=date(2026, 7, 1),
            current_price=130.0,
            scenarios=scenarios,
        )
        save_analysis(ta)
        retrieved = get_cached_analysis("SPY", date(2026, 7, 1), ANALYSIS_VERSION)
        assert retrieved is not None
        assert len(retrieved.scenarios) == 3
        bull = next(s for s in retrieved.scenarios if s.scenario == "bull")
        assert bull.narrative == "Strong growth"
        assert bull.implied_return is not None  # derived field survives

    def test_sources_survive_round_trip(self) -> None:
        sources = [Source(label="yfinance", url="https://example.com")]
        ta = _make_analysis(ticker="QQQ", as_of=date(2026, 8, 1), sources=sources)
        save_analysis(ta)
        retrieved = get_cached_analysis("QQQ", date(2026, 8, 1), ANALYSIS_VERSION)
        assert retrieved is not None
        assert retrieved.sources[0].label == "yfinance"
        assert retrieved.sources[0].url == "https://example.com"

    def test_data_gaps_list_survives(self) -> None:
        gaps = ["no transcripts", "dcf failed", "quality panel error"]
        ta = _make_analysis(ticker="IWM", as_of=date(2026, 9, 1), data_gaps=gaps)
        save_analysis(ta)
        retrieved = get_cached_analysis("IWM", date(2026, 9, 1), ANALYSIS_VERSION)
        assert retrieved is not None
        assert retrieved.data_gaps == gaps
