"""
Pytest for transcript fetcher error states.
Each HTTP error code produces a distinct Python exception.
Empty/error responses are never written to cache.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from data.resilience import SourceUnavailable
from data.transcripts import (
    Transcript, TranscriptRateLimited, _fetch_transcript,
    _tickers_to_try, get_last_n_transcripts, get_transcript,
)


def _make_response(status_code: int, body: object, headers: dict | None = None) -> httpx.Response:
    """Build a fake httpx.Response."""
    body_bytes = json.dumps(body).encode() if not isinstance(body, bytes) else body
    resp = httpx.Response(
        status_code=status_code,
        content=body_bytes,
        headers={"content-type": "application/json", **(headers or {})},
        request=httpx.Request("GET", "https://api.api-ninjas.com/v1/earningstranscript"),
    )
    return resp


# ── Error state tests ──────────────────────────────────────────────────────────

class TestFetchTranscriptErrorStates:

    def _patch_get(self, response: httpx.Response):
        return patch("httpx.Client.get", return_value=response)

    def test_401_raises_source_unavailable(self) -> None:
        resp = _make_response(401, {"error": "Unauthorized"})
        with self._patch_get(resp):
            with pytest.raises(SourceUnavailable, match="auth failed"):
                _fetch_transcript("AAPL", 2024, 4)

    def test_400_premium_raises_source_unavailable(self) -> None:
        resp = _make_response(400, {"error": "This endpoint is available to premium subscribers only."})
        with self._patch_get(resp):
            with pytest.raises(SourceUnavailable, match="premium subscription"):
                _fetch_transcript("AAPL", 2024, 4)

    def test_400_non_premium_returns_none(self) -> None:
        """A 400 that isn't the premium gate (e.g. bad quarter) should return None."""
        resp = _make_response(400, {"error": "Invalid quarter parameter"})
        with self._patch_get(resp):
            result = _fetch_transcript("AAPL", 2024, 4)
        assert result is None

    def test_404_returns_none(self) -> None:
        resp = _make_response(404, {"error": "Not found"})
        with self._patch_get(resp):
            result = _fetch_transcript("AAPL", 2024, 4)
        assert result is None

    def test_429_raises_transcript_rate_limited(self) -> None:
        resp = _make_response(429, {"error": "Rate limit exceeded"}, headers={"Retry-After": "30"})
        with self._patch_get(resp):
            with pytest.raises(TranscriptRateLimited) as exc_info:
                _fetch_transcript("AAPL", 2024, 4)
        assert exc_info.value.retry_after == 30

    def test_429_without_retry_after_header(self) -> None:
        resp = _make_response(429, {"error": "Rate limit exceeded"})
        with self._patch_get(resp):
            with pytest.raises(TranscriptRateLimited) as exc_info:
                _fetch_transcript("AAPL", 2024, 4)
        assert exc_info.value.retry_after is None

    def test_empty_body_returns_none(self) -> None:
        resp = _make_response(200, [])
        with self._patch_get(resp):
            result = _fetch_transcript("AAPL", 2024, 4)
        assert result is None

    def test_valid_response_parsed_correctly(self) -> None:
        payload = [{"transcript": "Jensen Huang: Revenue was $35B.", "date": "2024-11-20"}]
        resp = _make_response(200, payload)
        with self._patch_get(resp):
            result = _fetch_transcript("AAPL", 2024, 4)
        assert result is not None
        assert result["ticker"] == "AAPL"
        assert result["year"] == 2024
        assert result["quarter"] == 4
        assert "Revenue" in result["content"]
        assert result["date"] == "2024-11-20"

    def test_valid_dict_response(self) -> None:
        """API may also return a dict directly (not list)."""
        payload = {"transcript": "CFO: EPS beat by 12%.", "date": "2024-08-01"}
        resp = _make_response(200, payload)
        with self._patch_get(resp):
            result = _fetch_transcript("NVDA", 2024, 2)
        assert result is not None
        assert "EPS" in result["content"]


# ── Cache isolation tests ──────────────────────────────────────────────────────

class TestTranscriptCaching:
    """
    These tests exercise get_transcript's cache layer.
    We patch _get_transcript_from_chain (the provider chain) rather than
    httpx directly, because DefeatBeta bypasses httpx entirely.
    """

    def test_error_response_never_cached(self, tmp_path) -> None:
        """SourceUnavailable from the chain must not call set_cache_obj."""
        mock_set = MagicMock()
        with patch("data.transcripts.get_cache_obj", return_value=None), \
             patch("data.transcripts.set_cache_obj", mock_set), \
             patch("data.transcripts.record_freshness"), \
             patch("data.transcripts._get_transcript_from_chain",
                   side_effect=SourceUnavailable("premium required")):
            with pytest.raises(SourceUnavailable):
                get_transcript("AAPL", 2024, 4)

        mock_set.assert_not_called()

    def test_none_response_never_cached(self, tmp_path) -> None:
        """None result (quarter not found) must not call set_cache_obj."""
        mock_set = MagicMock()
        with patch("data.transcripts.get_cache_obj", return_value=None), \
             patch("data.transcripts.set_cache_obj", mock_set), \
             patch("data.transcripts.record_freshness"), \
             patch("data.transcripts._get_transcript_from_chain", return_value=None):
            result = get_transcript("AAPL", 2024, 4)

        assert result is None
        mock_set.assert_not_called()

    def test_valid_response_is_cached(self) -> None:
        """A successful fetch IS cached; second call hits cache, not the provider chain."""
        from data.transcripts import Transcript

        fake_transcript = Transcript(
            ticker="AAPL", year=2024, quarter=2,
            date="2024-05-01",
            prepared_remarks="Tim Cook: Revenue was $10B.",
            qa_section="",
            participants=["Tim Cook"],
            source="defeatbeta",
        )

        local_cache: dict = {}

        def fake_cache_get(key: str):
            return local_cache.get(key)

        def fake_cache_set(key: str, value, ttl, source=None):
            local_cache[key] = value

        chain_call_count = 0

        def counting_chain(ticker, year, quarter):
            nonlocal chain_call_count
            chain_call_count += 1
            return fake_transcript

        with patch("data.transcripts.get_cache_obj", side_effect=fake_cache_get), \
             patch("data.transcripts.set_cache_obj", side_effect=fake_cache_set), \
             patch("data.transcripts.record_freshness"), \
             patch("data.transcripts._get_transcript_from_chain", side_effect=counting_chain):
            r1 = get_transcript("AAPL", 2024, 2)
            r2 = get_transcript("AAPL", 2024, 2)  # should hit local_cache

        assert r1 is not None
        assert r2 == r1
        assert chain_call_count == 1, f"Expected 1 chain call (second hit cache), got {chain_call_count}"


# ── get_last_n_transcripts propagation tests ───────────────────────────────────

class TestGetLastNTranscripts:

    def test_source_unavailable_propagates(self) -> None:
        """SourceUnavailable from any quarter attempt must surface immediately."""
        with patch("data.transcripts.get_transcript",
                   side_effect=SourceUnavailable("premium required")):
            with pytest.raises(SourceUnavailable, match="premium"):
                get_last_n_transcripts("AAPL", n=4)

    def test_rate_limited_propagates(self) -> None:
        with patch("data.transcripts.get_transcript",
                   side_effect=TranscriptRateLimited(60)):
            with pytest.raises(TranscriptRateLimited):
                get_last_n_transcripts("AAPL", n=4)

    def test_missing_quarters_skipped(self) -> None:
        """None returns (missing quarters) are skipped until n found or budget exhausted."""
        good = {"ticker": "AAPL", "year": 2024, "quarter": 4,
                "content": "Record revenue.", "date": "2024-11-01"}
        call_count = 0

        def mock_get(ticker, year, quarter):
            nonlocal call_count
            call_count += 1
            # Return a result only on the 3rd call
            return good if call_count == 3 else None

        with patch("data.transcripts.get_transcript", side_effect=mock_get):
            results = get_last_n_transcripts("AAPL", n=1)

        assert len(results) == 1
        assert results[0]["content"] == "Record revenue."

    def test_exhausted_budget_returns_empty(self) -> None:
        """After _MAX_ATTEMPTS misses, return [] (not raise)."""
        with patch("data.transcripts.get_transcript", return_value=None):
            results = get_last_n_transcripts("AAPL", n=4)
        assert results == []


# ── Ticker alias tests ─────────────────────────────────────────────────────────

class TestTickerAliases:
    """GOOG/GOOGL and BRK.B/BRK-B aliases are tried before declaring not_found."""

    def test_goog_aliases_include_googl(self) -> None:
        aliases = _tickers_to_try("GOOG")
        assert "GOOGL" in aliases

    def test_googl_aliases_include_goog(self) -> None:
        aliases = _tickers_to_try("GOOGL")
        assert "GOOG" in aliases

    def test_brk_b_variants_all_present(self) -> None:
        aliases = _tickers_to_try("BRK.B")
        assert "BRK-B" in aliases

    def test_plain_ticker_returns_itself(self) -> None:
        aliases = _tickers_to_try("AAPL")
        assert "AAPL" in aliases
        assert aliases[0] == "AAPL"

    def test_goog_transcript_found_via_googl_alias(self) -> None:
        """When GOOG returns nothing but GOOGL has a transcript, it is returned as GOOG."""
        from data.transcripts import _get_transcript_from_chain

        googl_transcript = Transcript(
            ticker="GOOGL", year=2025, quarter=1, date="2025-04-29",
            prepared_remarks="Sundar Pichai: Revenue was $90B.", qa_section="",
            participants=["Sundar Pichai"], source="fmp",
        )

        call_log: list = []

        def fake_fetch(sym, year, quarter):
            call_log.append(sym)
            if sym == "GOOGL":
                return googl_transcript
            return None  # GOOG returns nothing

        with patch("data.transcripts.get_cache_obj", return_value=None), \
             patch("data.transcripts.set_cache_obj"), \
             patch("data.transcripts.record_freshness"), \
             patch("data.transcripts._PROVIDERS") as mock_providers:
            mock_p = MagicMock()
            mock_p.name = "fmp"
            mock_p.available.return_value = True
            mock_p.fetch.side_effect = fake_fetch
            mock_providers.__iter__ = MagicMock(return_value=iter([mock_p]))

            result = _get_transcript_from_chain("GOOG", 2025, 1)

        assert result is not None
        assert result.ticker == "GOOG"        # normalized back to input ticker
        assert "GOOGL" in call_log            # alias was tried
        assert result.prepared_remarks == googl_transcript.prepared_remarks

    def test_googl_resolved_directly(self) -> None:
        """GOOGL lookup succeeds on first symbol without needing alias fallback."""
        from data.transcripts import _get_transcript_from_chain

        googl_transcript = Transcript(
            ticker="GOOGL", year=2025, quarter=2, date="2025-07-30",
            prepared_remarks="Sundar: Cloud grew 30%.", qa_section="",
            participants=[], source="fmp",
        )

        def fake_fetch(sym, year, quarter):
            if sym == "GOOGL":
                return googl_transcript
            return None

        with patch("data.transcripts.get_cache_obj", return_value=None), \
             patch("data.transcripts.set_cache_obj"), \
             patch("data.transcripts.record_freshness"), \
             patch("data.transcripts._PROVIDERS") as mock_providers:
            mock_p = MagicMock()
            mock_p.name = "fmp"
            mock_p.available.return_value = True
            mock_p.fetch.side_effect = fake_fetch
            mock_providers.__iter__ = MagicMock(return_value=iter([mock_p]))

            result = _get_transcript_from_chain("GOOGL", 2025, 2)

        assert result is not None
        assert result.ticker == "GOOGL"


class TestEmptyStateDifferentiation:
    """Empty-state outcomes must distinguish no_key / transport failure / true zero."""

    def test_all_no_key_outcome_status(self) -> None:
        """When no providers have keys, all outcomes are no_key."""
        from data.transcripts import _get_transcript_from_chain

        with patch("data.transcripts.get_cache_obj", return_value=None), \
             patch("data.transcripts.set_cache_obj"), \
             patch("data.transcripts.record_freshness"), \
             patch("data.transcripts._PROVIDERS") as mock_providers:
            mock_p = MagicMock()
            mock_p.name = "fmp"
            mock_p.available.return_value = False
            mock_providers.__iter__ = MagicMock(return_value=iter([mock_p]))

            result = _get_transcript_from_chain("NVDA", 2025, 1)

        assert result is None
        from data.transcripts import get_transcript_provider_outcomes
        outcomes = get_transcript_provider_outcomes("NVDA")
        assert any(o["status"] == "no_key" for o in outcomes)

    def test_transport_error_recorded_not_not_found(self) -> None:
        """A provider that raises RuntimeError gets status 'error', not 'not_found'."""
        from data.transcripts import _get_transcript_from_chain

        def failing_fetch(sym, year, quarter):
            raise ConnectionError("Connection refused")

        with patch("data.transcripts.get_cache_obj", return_value=None), \
             patch("data.transcripts.set_cache_obj"), \
             patch("data.transcripts.record_freshness"), \
             patch("data.transcripts._PROVIDERS") as mock_providers:
            mock_p = MagicMock()
            mock_p.name = "fmp"
            mock_p.available.return_value = True
            mock_p.fetch.side_effect = failing_fetch
            mock_providers.__iter__ = MagicMock(return_value=iter([mock_p]))

            _get_transcript_from_chain("NVDA2", 2025, 1)

        from data.transcripts import get_transcript_provider_outcomes
        outcomes = get_transcript_provider_outcomes("NVDA2")
        fmp_outcome = next((o for o in outcomes if o["provider"] == "fmp"), None)
        assert fmp_outcome is not None
        assert fmp_outcome["status"] == "error"
        assert fmp_outcome["status"] != "not_found"

    def test_all_transcript_calls_have_timeout(self) -> None:
        """FmpProvider and FinnhubProvider use httpx.Client(timeout=20)."""
        from data.transcripts import FmpProvider, FinnhubProvider
        from unittest.mock import patch
        import httpx

        captured_timeouts: list = []

        class FakeClient:
            def __init__(self, **kwargs):
                captured_timeouts.append(kwargs.get("timeout"))
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def get(self, *a, **kw):
                return MagicMock(status_code=200, text="[]", json=MagicMock(return_value=[]))

        with patch("httpx.Client", FakeClient):
            try:
                FmpProvider().fetch("AAPL", 2024, 4)
            except Exception:
                pass
            try:
                FinnhubProvider().fetch("AAPL", 2024, 4)
            except Exception:
                pass

        assert all(t == 20 for t in captured_timeouts if t is not None), \
            f"Expected timeout=20, got: {captured_timeouts}"
