# tests/test_provider_chain.py
"""Provider chain integration tests: failover, stale cache, all-down hard fail, circuit breaker, call-count.

Chain order (as of current implementation):
  get_fundamentals: EDGAR → FMP → yfinance → Stooq (price-only)
  get_prices:       Stooq full-history → yfinance
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch
import pytest

from analysis.schemas import Fundamentals, PriceBar, PriceData
from data.resilience import SourceUnavailable
from datetime import date


def _make_fundamentals(ticker="PLTR"):
    from datetime import datetime
    return Fundamentals(
        ticker=ticker, name="Test Corp", sector="Technology", industry="Software",
        market_cap=90e9, current_price=45.0, fetched_at=datetime.utcnow(),
    )


def _make_price_data(ticker="PLTR", close=45.0):
    return PriceData(
        ticker=ticker, currency="USD",
        bars=[PriceBar(date=date.today(), open=close, high=close, low=close, close=close, volume=0)],
    )


# ── Helper: standard patch set for fundamentals calls (all providers open/failing) ──────

def _edgar_skip():
    """Return patches that set EDGAR CB as open (skipped without error)."""
    return patch("data.market._CB_EDGAR")


# ── Fundamentals chain tests ──────────────────────────────────────────────────

class TestEdgarPrimaryFallsToFmp:
    """EDGAR is tried first; when it fails, FMP is the next provider."""

    def test_edgar_fail_triggers_fmp_fallback(self):
        import data.market as mkt
        fmp_result = _make_fundamentals()

        with patch("data.market.get_cache_obj", return_value=None), \
             patch("data.market.get_stale_cache_obj", return_value=None), \
             patch("data.market.set_cache_obj"), \
             patch("data.market.record_freshness"), \
             patch("data.market._CB_EDGAR") as mock_edgar_cb, \
             patch("data.market._CB_FMP") as mock_fmp_cb, \
             patch("data.market._CB_YFINANCE") as mock_yf_cb, \
             patch("data.market._fetch_fundamentals_edgar",
                   side_effect=ValueError("EDGAR: CIK not found")), \
             patch("data.market._fetch_fundamentals_fmp", return_value=fmp_result) as mock_fmp, \
             patch("data.market._cfg") as mock_cfg:
            mock_edgar_cb.is_open = False
            mock_fmp_cb.is_open = False
            mock_yf_cb.is_open = False
            mock_cfg.FMP_API_KEY = "testkey"
            result = mkt.get_fundamentals("PLTR")

        assert result.ticker == "PLTR"
        mock_fmp.assert_called_once_with("PLTR")

    def test_edgar_success_never_calls_fmp_or_yfinance(self):
        import data.market as mkt
        edgar_result = _make_fundamentals()

        with patch("data.market.get_cache_obj", return_value=None), \
             patch("data.market.get_stale_cache_obj", return_value=None), \
             patch("data.market.set_cache_obj"), \
             patch("data.market.record_freshness"), \
             patch("data.market._CB_EDGAR") as mock_edgar_cb, \
             patch("data.market._fetch_fundamentals_edgar", return_value=edgar_result), \
             patch("data.market._fetch_fundamentals_fmp") as mock_fmp, \
             patch("data.market._fetch_fundamentals_yfinance") as mock_yf:
            mock_edgar_cb.is_open = False
            result = mkt.get_fundamentals("PLTR")

        assert result.ticker == "PLTR"
        mock_fmp.assert_not_called()
        mock_yf.assert_not_called()


class TestStaleCacheServedOnAllProviderFailure:

    def test_stale_data_returned_when_all_providers_fail(self):
        import data.market as mkt
        stale_fund = _make_fundamentals()
        stale_serialized = stale_fund.model_dump(mode="json")
        past_expiry = time.time() - 100

        with patch("data.market.get_cache_obj", return_value=None), \
             patch("data.market.get_stale_cache_obj", return_value=(stale_serialized, past_expiry)), \
             patch("data.market.set_cache_obj"), \
             patch("data.market.record_freshness"), \
             patch("data.market._CB_EDGAR") as mock_edgar_cb, \
             patch("data.market._CB_FMP") as mock_fmp_cb, \
             patch("data.market._CB_YFINANCE") as mock_yf_cb, \
             patch("data.market._fetch_fundamentals_edgar",
                   side_effect=ValueError("EDGAR down")), \
             patch("data.market._fetch_fundamentals_yfinance",
                   side_effect=SourceUnavailable("429")), \
             patch("data.market._fetch_fundamentals_fmp",
                   side_effect=ValueError("FMP down")), \
             patch("data.market._cfg") as mock_cfg:
            mock_edgar_cb.is_open = False
            mock_fmp_cb.is_open = False
            mock_yf_cb.is_open = False
            mock_cfg.FMP_API_KEY = "testkey"
            result = mkt.get_fundamentals("PLTR")

        assert result is not None
        assert result.ticker == "PLTR"


class TestAllProvidersDownColdCacheRaises:

    def test_raises_source_unavailable_when_no_cache_and_all_providers_fail(self):
        import data.market as mkt

        with patch("data.market.get_cache_obj", return_value=None), \
             patch("data.market.get_stale_cache_obj", return_value=None), \
             patch("data.market._CB_EDGAR") as mock_edgar_cb, \
             patch("data.market._CB_FMP") as mock_fmp_cb, \
             patch("data.market._CB_YFINANCE") as mock_yf_cb, \
             patch("data.market._CB_STOOQ") as mock_stooq_cb, \
             patch("data.market._fetch_fundamentals_edgar",
                   side_effect=ValueError("EDGAR down")), \
             patch("data.market._fetch_fundamentals_yfinance",
                   side_effect=SourceUnavailable("429")), \
             patch("data.market._fetch_fundamentals_fmp",
                   side_effect=ValueError("FMP down")), \
             patch("data.market._fetch_fundamentals_from_stooq",
                   side_effect=ValueError("Stooq down")):
            mock_edgar_cb.is_open = False
            mock_fmp_cb.is_open = False
            mock_yf_cb.is_open = False
            mock_stooq_cb.is_open = False
            with patch("data.market._cfg") as mock_cfg:
                mock_cfg.FMP_API_KEY = "testkey"
                with pytest.raises(SourceUnavailable):
                    mkt.get_fundamentals("PLTR")


class TestCircuitBreakerSkipsProvider:

    def test_open_edgar_cb_falls_through_to_fmp(self):
        """When EDGAR CB is open, FMP is tried without calling edgar fetch."""
        import data.market as mkt
        fmp_result = _make_fundamentals()

        with patch("data.market.get_cache_obj", return_value=None), \
             patch("data.market.get_stale_cache_obj", return_value=None), \
             patch("data.market.set_cache_obj"), \
             patch("data.market.record_freshness"), \
             patch("data.market._CB_EDGAR") as mock_edgar_cb, \
             patch("data.market._CB_FMP") as mock_fmp_cb, \
             patch("data.market._fetch_fundamentals_edgar") as mock_edgar_fetch, \
             patch("data.market._fetch_fundamentals_fmp", return_value=fmp_result):
            mock_edgar_cb.is_open = True
            mock_edgar_cb.state = lambda: "open"
            mock_fmp_cb.is_open = False
            with patch("data.market._cfg") as mock_cfg:
                mock_cfg.FMP_API_KEY = "testkey"
                result = mkt.get_fundamentals("PLTR")

        mock_edgar_fetch.assert_not_called()
        assert result is not None

    def test_open_yfinance_cb_does_not_call_yfinance_fetch(self):
        """When yfinance CB is open, yfinance fetch is never called."""
        import data.market as mkt
        edgar_result = _make_fundamentals()

        with patch("data.market.get_cache_obj", return_value=None), \
             patch("data.market.get_stale_cache_obj", return_value=None), \
             patch("data.market.set_cache_obj"), \
             patch("data.market.record_freshness"), \
             patch("data.market._CB_EDGAR") as mock_edgar_cb, \
             patch("data.market._CB_YFINANCE") as mock_yf_cb, \
             patch("data.market._fetch_fundamentals_edgar", return_value=edgar_result), \
             patch("data.market._fetch_fundamentals_yfinance") as mock_yf_fetch:
            mock_edgar_cb.is_open = False
            mock_yf_cb.is_open = True
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
             patch("data.market._fetch_fundamentals_edgar") as mock_edgar, \
             patch("data.market._fetch_fundamentals_yfinance") as mock_yf, \
             patch("data.market._fetch_fundamentals_fmp") as mock_fmp:
            result = mkt.get_fundamentals("PLTR")

        mock_edgar.assert_not_called()
        mock_yf.assert_not_called()
        mock_fmp.assert_not_called()
        assert result.ticker == "PLTR"


class TestPricesChain:

    def test_stooq_provides_full_price_history_as_primary(self):
        """Stooq is tried first for prices and returns full OHLCV history."""
        import data.market as mkt
        stooq_result = _make_price_data("PLTR", 45.50)

        with patch("data.market.get_cache_obj", return_value=None), \
             patch("data.market.get_stale_cache_obj", return_value=None), \
             patch("data.market.set_cache_obj"), \
             patch("data.market.record_freshness"), \
             patch("data.market._CB_STOOQ") as mock_stooq_cb, \
             patch("data.market._CB_YFINANCE") as mock_yf_cb, \
             patch("data.market._fetch_prices_stooq_full", return_value=stooq_result) as mock_stooq, \
             patch("data.market._fetch_prices_yfinance") as mock_yf:
            mock_stooq_cb.is_open = False
            mock_yf_cb.is_open = False
            result = mkt.get_prices("PLTR")

        mock_stooq.assert_called_once()
        mock_yf.assert_not_called()
        assert result.bars[-1].close == pytest.approx(45.50)

    def test_stooq_prices_fail_falls_to_yfinance(self):
        """When Stooq prices fail, yfinance provides the fallback."""
        import data.market as mkt
        yf_result = _make_price_data("PLTR", 46.0)

        with patch("data.market.get_cache_obj", return_value=None), \
             patch("data.market.get_stale_cache_obj", return_value=None), \
             patch("data.market.set_cache_obj"), \
             patch("data.market.record_freshness"), \
             patch("data.market._CB_STOOQ") as mock_stooq_cb, \
             patch("data.market._CB_YFINANCE") as mock_yf_cb, \
             patch("data.market._fetch_prices_stooq_full",
                   side_effect=ValueError("Stooq 404")), \
             patch("data.market._fetch_prices_yfinance", return_value=yf_result) as mock_yf:
            mock_stooq_cb.is_open = False
            mock_yf_cb.is_open = False
            result = mkt.get_prices("PLTR")

        mock_yf.assert_called_once()
        assert result.bars[-1].close == pytest.approx(46.0)


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

    def test_open_cb_makes_zero_calls_to_that_provider(self):
        """With yfinance CB open, the yfinance fetch function is never invoked."""
        import data.market as mkt
        fmp_result = _make_fundamentals()

        with patch("data.market.get_cache_obj", return_value=None), \
             patch("data.market.get_stale_cache_obj", return_value=None), \
             patch("data.market.set_cache_obj"), \
             patch("data.market.record_freshness"), \
             patch("data.market._CB_EDGAR") as mock_edgar_cb, \
             patch("data.market._CB_FMP") as mock_fmp_cb, \
             patch("data.market._CB_YFINANCE") as mock_yf_cb, \
             patch("data.market._fetch_fundamentals_edgar",
                   side_effect=ValueError("EDGAR down")), \
             patch("data.market._fetch_fundamentals_yfinance") as mock_yf_fetch, \
             patch("data.market._fetch_fundamentals_fmp", return_value=fmp_result):
            mock_edgar_cb.is_open = False
            mock_fmp_cb.is_open = False
            mock_yf_cb.is_open = True   # yfinance CB open
            with patch("data.market._cfg") as mock_cfg:
                mock_cfg.FMP_API_KEY = "testkey"
                mkt.get_fundamentals("AAPL")

        mock_yf_fetch.assert_not_called()


class TestStaleBadgeMetadata:

    def test_provider_health_set_to_stale_when_serving_expired_cache(self):
        import data.market as mkt
        stale_fund = _make_fundamentals("MSFT")
        stale_data = stale_fund.model_dump(mode="json")
        past_expiry = time.time() - 500

        with patch("data.market.get_cache_obj", return_value=None), \
             patch("data.market.get_stale_cache_obj", return_value=(stale_data, past_expiry)), \
             patch("data.market.set_cache_obj"), \
             patch("data.market.record_freshness"), \
             patch("data.market._CB_EDGAR") as mock_edgar_cb, \
             patch("data.market._CB_FMP") as mock_fmp_cb, \
             patch("data.market._CB_YFINANCE") as mock_yf_cb, \
             patch("data.market._CB_STOOQ") as mock_stooq_cb, \
             patch("data.market._fetch_fundamentals_edgar",
                   side_effect=ValueError("EDGAR down")), \
             patch("data.market._fetch_fundamentals_yfinance",
                   side_effect=SourceUnavailable("429")), \
             patch("data.market._fetch_fundamentals_fmp",
                   side_effect=ValueError("FMP down")), \
             patch("data.market._fetch_fundamentals_from_stooq",
                   side_effect=ValueError("Stooq down")):
            mock_edgar_cb.is_open = False
            mock_fmp_cb.is_open = False
            mock_yf_cb.is_open = False
            mock_stooq_cb.is_open = False
            with patch("data.market._cfg") as mock_cfg:
                mock_cfg.FMP_API_KEY = "testkey"
                result = mkt.get_fundamentals("MSFT")

        assert result is not None
        health = mkt.get_provider_health()
        assert health.get("MSFT", {}).get("status") == "stale"


class TestCallCountVerification:

    def test_all_four_providers_called_on_cold_cache_all_fail(self):
        """On cold cache with all providers down: exactly 1 call each to edgar, fmp, yf, stooq."""
        import data.market as mkt

        with patch("data.market.get_cache_obj", return_value=None), \
             patch("data.market.get_stale_cache_obj", return_value=None), \
             patch("data.market._CB_EDGAR") as mock_edgar_cb, \
             patch("data.market._CB_FMP") as mock_fmp_cb, \
             patch("data.market._CB_YFINANCE") as mock_yf_cb, \
             patch("data.market._CB_STOOQ") as mock_stooq_cb, \
             patch("data.market._fetch_fundamentals_edgar",
                   side_effect=ValueError("edgar down")) as mock_edgar, \
             patch("data.market._fetch_fundamentals_yfinance",
                   side_effect=SourceUnavailable("yf down")) as mock_yf, \
             patch("data.market._fetch_fundamentals_fmp",
                   side_effect=ValueError("fmp down")) as mock_fmp, \
             patch("data.market._fetch_fundamentals_from_stooq",
                   side_effect=ValueError("stooq down")) as mock_stooq:
            mock_edgar_cb.is_open = False
            mock_fmp_cb.is_open = False
            mock_yf_cb.is_open = False
            mock_stooq_cb.is_open = False
            with patch("data.market._cfg") as mock_cfg:
                mock_cfg.FMP_API_KEY = "testkey"
                with pytest.raises(SourceUnavailable):
                    mkt.get_fundamentals("PLTR")

        assert mock_edgar.call_count == 1
        assert mock_fmp.call_count == 1
        assert mock_yf.call_count == 1
        assert mock_stooq.call_count == 1

    def test_zero_provider_calls_on_warm_cache(self):
        """Warm SQLite cache: zero provider calls regardless of circuit breaker state."""
        import data.market as mkt
        cached_fund = _make_fundamentals("NVDA")
        serialized = cached_fund.model_dump(mode="json")

        with patch("data.market.get_cache_obj", return_value=serialized), \
             patch("data.market._fetch_fundamentals_edgar") as mock_edgar, \
             patch("data.market._fetch_fundamentals_yfinance") as mock_yf, \
             patch("data.market._fetch_fundamentals_fmp") as mock_fmp:
            result = mkt.get_fundamentals("NVDA")

        assert result.ticker == "NVDA"
        assert mock_edgar.call_count == 0
        assert mock_yf.call_count == 0
        assert mock_fmp.call_count == 0


class TestChainAdvancesToStooqOnAllOtherFailure:
    """Regression: chain must reach Stooq when edgar+fmp+yfinance all fail."""

    def test_stooq_called_and_succeeds_when_others_fail(self):
        import data.market as mkt
        from datetime import datetime
        stooq_result = Fundamentals(ticker="NVDA", current_price=130.0, fetched_at=datetime.utcnow())

        with patch("data.market.get_cache_obj", return_value=None), \
             patch("data.market.get_stale_cache_obj", return_value=None), \
             patch("data.market.set_cache_obj"), \
             patch("data.market.record_freshness"), \
             patch("data.market._CB_EDGAR") as mock_edgar_cb, \
             patch("data.market._CB_FMP") as mock_fmp_cb, \
             patch("data.market._CB_YFINANCE") as mock_yf_cb, \
             patch("data.market._CB_STOOQ") as mock_stooq_cb, \
             patch("data.market._fetch_fundamentals_edgar",
                   side_effect=ValueError("EDGAR: CIK not found")) as mock_edgar, \
             patch("data.market._fetch_fundamentals_yfinance",
                   side_effect=SourceUnavailable("429")) as mock_yf, \
             patch("data.market._fetch_fundamentals_fmp",
                   side_effect=ValueError("FMP 403")) as mock_fmp, \
             patch("data.market._fetch_fundamentals_from_stooq",
                   return_value=stooq_result) as mock_stooq:
            mock_edgar_cb.is_open = False
            mock_fmp_cb.is_open = False
            mock_yf_cb.is_open = False
            mock_stooq_cb.is_open = False
            with patch("data.market._cfg") as mock_cfg:
                mock_cfg.FMP_API_KEY = "testkey"
                result = mkt.get_fundamentals("NVDA")

        assert result is not None
        assert result.current_price == pytest.approx(130.0)
        mock_edgar.assert_called_once()
        mock_fmp.assert_called_once()
        mock_yf.assert_called_once()
        mock_stooq.assert_called_once()


class TestProviderOutcomesAllRecorded:
    """Every provider attempt must appear in per-provider outcomes."""

    def test_all_four_outcomes_recorded_on_all_fail(self):
        import data.market as mkt

        with patch("data.market.get_cache_obj", return_value=None), \
             patch("data.market.get_stale_cache_obj", return_value=None), \
             patch("data.market._CB_EDGAR") as mock_edgar_cb, \
             patch("data.market._CB_FMP") as mock_fmp_cb, \
             patch("data.market._CB_YFINANCE") as mock_yf_cb, \
             patch("data.market._CB_STOOQ") as mock_stooq_cb, \
             patch("data.market._fetch_fundamentals_edgar",
                   side_effect=ValueError("EDGAR down")), \
             patch("data.market._fetch_fundamentals_yfinance",
                   side_effect=SourceUnavailable("429")), \
             patch("data.market._fetch_fundamentals_fmp",
                   side_effect=ValueError("FMP err")), \
             patch("data.market._fetch_fundamentals_from_stooq",
                   side_effect=ValueError("Stooq err")):
            mock_edgar_cb.is_open = False
            mock_fmp_cb.is_open = False
            mock_yf_cb.is_open = False
            mock_stooq_cb.is_open = False
            with patch("data.market._cfg") as mock_cfg:
                mock_cfg.FMP_API_KEY = "testkey"
                with pytest.raises(SourceUnavailable):
                    mkt.get_fundamentals("TSLA")

        outcomes = mkt.get_provider_outcomes("TSLA")
        providers_seen = {o["provider"] for o in outcomes}
        assert "edgar" in providers_seen
        assert "fmp" in providers_seen
        assert "yfinance" in providers_seen
        assert "stooq" in providers_seen

    def test_open_cb_shows_skipped_in_outcomes(self):
        import data.market as mkt
        from datetime import datetime
        edgar_result = Fundamentals(ticker="AMD", current_price=150.0, fetched_at=datetime.utcnow())

        with patch("data.market.get_cache_obj", return_value=None), \
             patch("data.market.get_stale_cache_obj", return_value=None), \
             patch("data.market.set_cache_obj"), \
             patch("data.market.record_freshness"), \
             patch("data.market._CB_EDGAR") as mock_edgar_cb, \
             patch("data.market._CB_YFINANCE") as mock_yf_cb, \
             patch("data.market._fetch_fundamentals_edgar", return_value=edgar_result), \
             patch("data.market._fetch_fundamentals_yfinance") as mock_yf_fetch:
            mock_edgar_cb.is_open = False
            mock_yf_cb.is_open = True    # yfinance CB is tripped
            mock_yf_cb.state = lambda: "open"
            mkt.get_fundamentals("AMD")

        mock_yf_fetch.assert_not_called()
        outcomes = mkt.get_provider_outcomes("AMD")
        edgar_outcome = next((o for o in outcomes if o["provider"] == "edgar"), None)
        assert edgar_outcome is not None
        assert edgar_outcome["status"] == "ok"


class TestErrorSummaryNeverNone:
    """Error message must not be 'None' when providers were skipped."""

    def test_error_message_contains_skip_reason_when_all_cbs_open(self):
        import data.market as mkt

        with patch("data.market.get_cache_obj", return_value=None), \
             patch("data.market.get_stale_cache_obj", return_value=None), \
             patch("data.market._CB_EDGAR") as mock_edgar_cb, \
             patch("data.market._CB_YFINANCE") as mock_yf_cb, \
             patch("data.market._CB_FMP") as mock_fmp_cb, \
             patch("data.market._CB_STOOQ") as mock_stooq_cb:
            mock_edgar_cb.is_open = True
            mock_edgar_cb.state = lambda: "open"
            mock_yf_cb.is_open = True
            mock_yf_cb.state = lambda: "open"
            mock_fmp_cb.is_open = True
            mock_fmp_cb.state = lambda: "open"
            mock_stooq_cb.is_open = True
            mock_stooq_cb.state = lambda: "open"
            with patch("data.market._cfg") as mock_cfg:
                mock_cfg.FMP_API_KEY = "testkey"
                with pytest.raises(SourceUnavailable) as exc_info:
                    mkt.get_fundamentals("GOOG")

        msg = str(exc_info.value)
        assert "None" not in msg or "skipped" in msg, \
            f"Error message should not contain bare 'None': {msg}"
        assert "circuit breakers open" in msg or "skipped" in msg


class TestFmpHandlesErrorDictResponse:
    """Bug 4 regression: FMP returning error dict must raise ValueError, not KeyError."""

    def test_fmp_error_dict_raises_value_error(self):
        import config as cfg
        original_key = cfg.FMP_API_KEY
        cfg.FMP_API_KEY = "bad_key"
        try:
            from data.providers.fmp import FmpMarketProvider
            fake_resp = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
            fake_resp.status_code = 200
            fake_resp.raise_for_status = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
            fake_resp.json.return_value = {"Error Message": "Invalid API KEY. Please retry or visit..."}

            with patch("httpx.get", return_value=fake_resp):
                with pytest.raises(ValueError, match="FMP profile error"):
                    FmpMarketProvider().get_fundamentals("AAPL")
        finally:
            cfg.FMP_API_KEY = original_key


class TestRedactSecurity:
    """P0: API keys must never appear in outcome detail strings."""

    def test_redact_strips_url_query_string(self):
        from data.resilience import redact
        url = "https://financialmodelingprep.com/api/v3/profile/AAPL?apikey=SUPERSECRETKEY123"
        result = redact(url)
        assert "SUPERSECRETKEY123" not in result
        assert "financialmodelingprep.com" in result
        assert "?<redacted>" in result

    def test_redact_url_embedded_in_exception(self):
        from data.resilience import redact
        msg = (
            "Client error '403 Forbidden' for url "
            "'https://financialmodelingprep.com/api/v3/profile/NVDA?apikey=ABC123XYZ'"
        )
        result = redact(msg)
        assert "ABC123XYZ" not in result
        assert "financialmodelingprep.com" in result

    def test_redact_leaves_no_url_query_strings(self):
        from data.resilience import redact
        msg = "GET https://example.com/endpoint?token=secret&other=value failed"
        result = redact(msg)
        assert "secret" not in result
        assert "?<redacted>" in result

    def test_fmp_http_error_message_is_secret_free(self):
        """FmpMarketProvider must not include the API key in any raised exception."""
        import config as cfg
        from data.providers.fmp import FmpMarketProvider
        fake_resp = MagicMock()
        fake_resp.status_code = 403
        fake_resp.json.return_value = {}
        with patch("httpx.get", return_value=fake_resp):
            with pytest.raises(ValueError) as exc_info:
                FmpMarketProvider().get_fundamentals("NVDA")
        msg = str(exc_info.value)
        if cfg.FMP_API_KEY:
            assert cfg.FMP_API_KEY not in msg
        assert "apikey=" not in msg


class TestEdgarProvider:
    """Unit tests for the EDGAR fundamentals provider."""

    def test_edgar_raises_when_cik_not_found(self):
        from data.providers.edgar import EdgarFundamentalsProvider
        with patch("data.providers.edgar._get_cik_direct", return_value=None):
            with pytest.raises(ValueError, match="CIK not found"):
                EdgarFundamentalsProvider().get_fundamentals("FAKEXYZ")

    def test_edgar_raises_on_submissions_http_error(self):
        from data.providers.edgar import EdgarFundamentalsProvider
        fake_resp = MagicMock()
        fake_resp.status_code = 503
        with patch("data.providers.edgar._get_cik_direct", return_value="0001234567"), \
             patch("data.providers.edgar._rate_limit"), \
             patch("httpx.get", return_value=fake_resp):
            with pytest.raises(ValueError, match="HTTP 503"):
                EdgarFundamentalsProvider().get_fundamentals("AAPL")

    def test_edgar_raises_when_name_is_empty(self):
        from data.providers.edgar import EdgarFundamentalsProvider
        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = {"name": None, "sic": "3674", "sicDescription": "Semiconductors"}
        with patch("data.providers.edgar._get_cik_direct", return_value="0001234567"), \
             patch("data.providers.edgar._rate_limit"), \
             patch("data.providers.edgar.get_xbrl_facts", return_value=None), \
             patch("data.providers.edgar._stooq_price", return_value=None), \
             patch("httpx.get", return_value=fake_resp):
            with pytest.raises(ValueError, match="no company name"):
                EdgarFundamentalsProvider().get_fundamentals("FAKE")

    def test_edgar_builds_fundamentals_from_profile(self):
        from data.providers.edgar import EdgarFundamentalsProvider
        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = {
            "name": "Palantir Technologies Inc.",
            "sic": "7372",
            "sicDescription": "Prepackaged Software",
        }
        with patch("data.providers.edgar._get_cik_direct", return_value="0001321732"), \
             patch("data.providers.edgar._rate_limit"), \
             patch("data.providers.edgar.get_xbrl_facts", return_value=None), \
             patch("data.providers.edgar._stooq_price", return_value=42.0), \
             patch("httpx.get", return_value=fake_resp):
            result = EdgarFundamentalsProvider().get_fundamentals("PLTR")

        assert result.ticker == "PLTR"
        assert result.name == "Palantir Technologies Inc."
        assert result.current_price == pytest.approx(42.0)

    def test_sic_sector_mapping(self):
        from data.providers.edgar import _sic_to_sector
        assert _sic_to_sector("7372") == "Technology"   # Prepackaged software
        assert _sic_to_sector("3674") == "Technology"   # Semiconductors
        assert _sic_to_sector("6022") == "Financials"   # Banks
        assert _sic_to_sector("8011") == "Healthcare"   # Health services
        assert _sic_to_sector("4813") == "Communication Services"
        assert _sic_to_sector("invalid") is None


class TestStooqProvider:
    """Unit tests for stooq price provider."""

    def test_get_price_lowercases_ticker(self):
        """Stooq requires lowercase ticker symbols."""
        from data.providers.stooq import StooqProvider
        csv_body = "Date,Open,High,Low,Close,Volume\n2026-01-10,140.0,142.0,139.0,141.0,1000000\n"
        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.raise_for_status = MagicMock()
        fake_resp.content = csv_body.encode()
        with patch("httpx.get", return_value=fake_resp) as mock_get:
            StooqProvider().get_price("NVDA")
        url_called = mock_get.call_args[0][0]
        assert "nvda" in url_called
        assert "NVDA" not in url_called

    def test_get_price_raises_on_no_data_response(self):
        from data.providers.stooq import StooqProvider
        fake_resp = MagicMock()
        fake_resp.raise_for_status = MagicMock()
        fake_resp.content = b"No data"
        with patch("httpx.get", return_value=fake_resp):
            with pytest.raises(ValueError, match="no data"):
                StooqProvider().get_price("FAKE")

    def test_get_prices_returns_price_data_object(self):
        from data.providers.stooq import StooqProvider
        csv_body = (
            "Date,Open,High,Low,Close,Volume\n"
            "2026-01-09,139.0,141.0,138.0,140.0,900000\n"
            "2026-01-10,140.0,142.0,139.0,141.0,1000000\n"
        )
        fake_resp = MagicMock()
        fake_resp.raise_for_status = MagicMock()
        fake_resp.content = csv_body.encode()
        with patch("httpx.get", return_value=fake_resp):
            result = StooqProvider().get_prices("NVDA", period="5y")
        assert len(result.bars) == 2
        assert result.bars[-1].close == pytest.approx(141.0)
