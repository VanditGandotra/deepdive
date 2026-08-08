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
from data.transcripts import TranscriptRateLimited, _fetch_transcript, get_last_n_transcripts, get_transcript


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
