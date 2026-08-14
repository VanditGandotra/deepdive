# tests/test_provider_chain.py
"""Provider chain integration tests: 429 failover, stale cache, all-down hard fail, circuit breaker, call-count."""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch
import pytest

from analysis.schemas import Fundamentals
from data.resilience import SourceUnavailable


def _make_fundamentals(ticker="PLTR"):
    from datetime import datetime
    return Fundamentals(
        ticker=ticker, name="Test Corp", sector="Technology", industry="Software",
        market_cap=90e9, current_price=45.0, fetched_at=datetime.utcnow(),
    )


class TestYfinance429FallsToFmp:

    def test_yfinance_429_triggers_fmp_fallback(self):
        """When yfinance raises SourceUnavailable, FMP is called and its result returned."""
        import data.market as mkt
        fmp_result = _make_fundamentals()

        with patch("data.market.get_cache_obj", return_value=None), \
             patch("data.market.get_stale_cache_obj", return_value=None), \
             patch("data.market.set_cache_obj"), \
             patch("data.market.record_freshness"), \
             patch("data.market._fetch_fundamentals_yfinance",
                   side_effect=SourceUnavailable("Yahoo 429")), \
             patch("data.market._CB_YFINANCE") as mock_yf_cb, \
             patch("data.market._CB_FMP") as mock_fmp_cb, \
             patch("data.market._fetch_fundamentals_fmp", return_value=fmp_result) as mock_fmp, \
             patch("data.market._cfg") as mock_cfg:
            mock_yf_cb.is_open = False
            mock_fmp_cb.is_open = False
            mock_cfg.FMP_API_KEY = "testkey"
            result = mkt.get_fundamentals("PLTR")

        assert result.ticker == "PLTR"
        mock_fmp.assert_called_once_with("PLTR")


class TestStaleCacheServedOnAllProviderFailure:

    def test_stale_data_returned_when_all_providers_fail(self):
        """When all providers fail but stale cache exists, stale data is returned (no exception)."""
        import data.market as mkt
        stale_fund = _make_fundamentals()
        stale_serialized = stale_fund.model_dump(mode="json")
        past_expiry = time.time() - 100

        with patch("data.market.get_cache_obj", return_value=None), \
             patch("data.market.get_stale_cache_obj", return_value=(stale_serialized, past_expiry)), \
             patch("data.market.set_cache_obj"), \
             patch("data.market.record_freshness"), \
             patch("data.market._fetch_fundamentals_yfinance",
                   side_effect=SourceUnavailable("429")), \
             patch("data.market._fetch_fundamentals_fmp",
                   side_effect=ValueError("FMP down")), \
             patch("data.market._CB_YFINANCE") as mock_yf_cb, \
             patch("data.market._CB_FMP") as mock_fmp_cb:
            mock_yf_cb.is_open = False
            mock_fmp_cb.is_open = False
            # Patch config.FMP_API_KEY to ensure FMP branch is tried
            with patch("data.market._cfg") as mock_cfg:
                mock_cfg.FMP_API_KEY = "testkey"
                result = mkt.get_fundamentals("PLTR")

        assert result is not None
        assert result.ticker == "PLTR"


class TestAllProvidersDownColdCacheRaises:

    def test_raises_source_unavailable_when_no_cache_and_all_providers_fail(self):
        import data.market as mkt

        with patch("data.market.get_cache_obj", return_value=None), \
             patch("data.market.get_stale_cache_obj", return_value=None), \
             patch("data.market._fetch_fundamentals_yfinance",
                   side_effect=SourceUnavailable("429")), \
             patch("data.market._fetch_fundamentals_fmp",
                   side_effect=ValueError("FMP down")), \
             patch("data.market._CB_YFINANCE") as mock_yf_cb, \
             patch("data.market._CB_FMP") as mock_fmp_cb:
            mock_yf_cb.is_open = False
            mock_fmp_cb.is_open = False
            with patch("data.market._cfg") as mock_cfg:
                mock_cfg.FMP_API_KEY = "testkey"
                with pytest.raises(SourceUnavailable):
                    mkt.get_fundamentals("PLTR")


class TestCircuitBreakerSkipsProvider:

    def test_open_yfinance_cb_skips_to_fmp(self):
        """When yfinance CB is open, yfinance fetch is never called."""
        import data.market as mkt
        fmp_result = _make_fundamentals()

        with patch("data.market.get_cache_obj", return_value=None), \
             patch("data.market.get_stale_cache_obj", return_value=None), \
             patch("data.market.set_cache_obj"), \
             patch("data.market.record_freshness"), \
             patch("data.market._CB_YFINANCE") as mock_yf_cb, \
             patch("data.market._CB_FMP") as mock_fmp_cb, \
             patch("data.market._fetch_fundamentals_yfinance") as mock_yf_fetch, \
             patch("data.market._fetch_fundamentals_fmp", return_value=fmp_result):
            mock_yf_cb.is_open = True
            mock_fmp_cb.is_open = False
            with patch("data.market._cfg") as mock_cfg:
                mock_cfg.FMP_API_KEY = "testkey"
                result = mkt.get_fundamentals("PLTR")

        mock_yf_fetch.assert_not_called()
        assert result is not None


class TestCallCountOnWarmCache:

    def test_warm_sqlite_cache_makes_zero_provider_calls(self):
        """SQLite cache hit prevents any provider call."""
        import data.market as mkt
        cached_fund = _make_fundamentals()
        serialized = cached_fund.model_dump(mode="json")

        with patch("data.market.get_cache_obj", return_value=serialized), \
             patch("data.market._fetch_fundamentals_yfinance") as mock_yf, \
             patch("data.market._fetch_fundamentals_fmp") as mock_fmp:
            result = mkt.get_fundamentals("PLTR")

        mock_yf.assert_not_called()
        mock_fmp.assert_not_called()
        assert result.ticker == "PLTR"


class TestPrices429FallsToStooq:

    def test_yfinance_prices_429_falls_to_stooq(self):
        """When yfinance prices fail, Stooq provides the fallback close price."""
        import data.market as mkt

        with patch("data.market.get_cache_obj", return_value=None), \
             patch("data.market.get_stale_cache_obj", return_value=None), \
             patch("data.market.set_cache_obj"), \
             patch("data.market.record_freshness"), \
             patch("data.market._CB_YFINANCE") as mock_yf_cb, \
             patch("data.market._CB_STOOQ") as mock_stooq_cb, \
             patch("data.market._fetch_prices_yfinance",
                   side_effect=SourceUnavailable("429")), \
             patch("data.market._fetch_price_stooq", return_value=45.50) as mock_stooq:
            mock_yf_cb.is_open = False
            mock_stooq_cb.is_open = False
            result = mkt.get_prices("PLTR")

        mock_stooq.assert_called_once_with("PLTR")
        assert result is not None
        assert result.bars[-1].close == pytest.approx(45.50)


class TestCircuitBreakerOpensAfterThresholdFailures:

    def test_cb_opens_and_stays_open(self):
        """After 3 consecutive failures, CB opens and stays open during cooldown."""
        from data.resilience import CircuitBreaker
        cb = CircuitBreaker("test_chain_open", failure_threshold=3, cooldown_secs=60)
        cb.record_failure()
        cb.record_failure()
        assert not cb.is_open
        cb.record_failure()
        assert cb.is_open
        assert cb.state() == "open"

    def test_open_cb_makes_zero_calls_to_yfinance(self):
        """With yfinance CB open, the yfinance fetch function is never invoked."""
        import data.market as mkt
        fmp_result = _make_fundamentals()

        with patch("data.market.get_cache_obj", return_value=None), \
             patch("data.market.get_stale_cache_obj", return_value=None), \
             patch("data.market.set_cache_obj"), \
             patch("data.market.record_freshness"), \
             patch("data.market._CB_YFINANCE") as mock_yf_cb, \
             patch("data.market._CB_FMP") as mock_fmp_cb, \
             patch("data.market._fetch_fundamentals_yfinance") as mock_yf_fetch, \
             patch("data.market._fetch_fundamentals_fmp", return_value=fmp_result):
            mock_yf_cb.is_open = True   # CB is open
            mock_fmp_cb.is_open = False
            with patch("data.market._cfg") as mock_cfg:
                mock_cfg.FMP_API_KEY = "testkey"
                mkt.get_fundamentals("AAPL")

        mock_yf_fetch.assert_not_called()


class TestStaleBadgeMetadata:

    def test_provider_health_set_to_stale_when_serving_expired_cache(self):
        """When stale cache is served, _PROVIDER_HEALTH[ticker]['status'] == 'stale'."""
        import data.market as mkt
        stale_fund = _make_fundamentals("MSFT")
        stale_data = stale_fund.model_dump(mode="json")
        past_expiry = time.time() - 500

        with patch("data.market.get_cache_obj", return_value=None), \
             patch("data.market.get_stale_cache_obj", return_value=(stale_data, past_expiry)), \
             patch("data.market.set_cache_obj"), \
             patch("data.market.record_freshness"), \
             patch("data.market._fetch_fundamentals_yfinance",
                   side_effect=SourceUnavailable("429")), \
             patch("data.market._fetch_fundamentals_fmp",
                   side_effect=ValueError("FMP down")), \
             patch("data.market._CB_YFINANCE") as mock_yf_cb, \
             patch("data.market._CB_FMP") as mock_fmp_cb:
            mock_yf_cb.is_open = False
            mock_fmp_cb.is_open = False
            with patch("data.market._cfg") as mock_cfg:
                mock_cfg.FMP_API_KEY = "testkey"
                result = mkt.get_fundamentals("MSFT")

        assert result is not None
        health = mkt.get_provider_health()
        assert health.get("MSFT", {}).get("status") == "stale"


class TestCallCountVerification:

    def test_two_provider_calls_on_cold_cache_all_fail(self):
        """On cold cache with all providers down: exactly 1 yfinance call + 1 FMP call made."""
        import data.market as mkt

        with patch("data.market.get_cache_obj", return_value=None), \
             patch("data.market.get_stale_cache_obj", return_value=None), \
             patch("data.market._fetch_fundamentals_yfinance",
                   side_effect=SourceUnavailable("yf down")) as mock_yf, \
             patch("data.market._fetch_fundamentals_fmp",
                   side_effect=ValueError("fmp down")) as mock_fmp, \
             patch("data.market._CB_YFINANCE") as mock_yf_cb, \
             patch("data.market._CB_FMP") as mock_fmp_cb:
            mock_yf_cb.is_open = False
            mock_fmp_cb.is_open = False
            with patch("data.market._cfg") as mock_cfg:
                mock_cfg.FMP_API_KEY = "testkey"
                with pytest.raises(SourceUnavailable):
                    mkt.get_fundamentals("PLTR")

        assert mock_yf.call_count == 1, f"Expected 1 yfinance call, got {mock_yf.call_count}"
        assert mock_fmp.call_count == 1, f"Expected 1 FMP call, got {mock_fmp.call_count}"

    def test_zero_provider_calls_on_warm_cache(self):
        """Warm SQLite cache: zero provider calls regardless of circuit breaker state."""
        import data.market as mkt
        cached_fund = _make_fundamentals("NVDA")
        serialized = cached_fund.model_dump(mode="json")

        with patch("data.market.get_cache_obj", return_value=serialized), \
             patch("data.market._fetch_fundamentals_yfinance") as mock_yf, \
             patch("data.market._fetch_fundamentals_fmp") as mock_fmp:
            result = mkt.get_fundamentals("NVDA")

        assert result.ticker == "NVDA"
        assert mock_yf.call_count == 0
        assert mock_fmp.call_count == 0
