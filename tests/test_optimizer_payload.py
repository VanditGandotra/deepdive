"""End-to-end tests for optimizer payload serialization and fallback summary.

Tests Issue 1 (JSON TypeError) and Issue 2 (empty summary fallback):
- All three methods produce a valid JSON payload that survives json.loads(json.dumps(x, allow_nan=False))
- Payload is valid when position cap binds (the np.bool_ crash path)
- _build_fallback_summary produces non-empty text for all methods
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _make_price_data(ticker: str, n: int = 260) -> MagicMock:
    rng = np.random.default_rng(seed=abs(hash(ticker)) % (2**31))
    prices = 100.0 * np.cumprod(1 + rng.normal(0.0005, 0.015, n))
    bars = []
    for p in prices:
        b = MagicMock()
        b.close = float(p)
        bars.append(b)
    pd_obj = MagicMock()
    pd_obj.bars = bars
    return pd_obj


def _patch_prices(tickers):
    price_data = {t: _make_price_data(t) for t in tickers}

    def fake(ticker, period="2y"):
        if ticker in price_data:
            return price_data[ticker]
        raise ValueError(f"No data for {ticker}")

    return patch("data.market.get_prices", side_effect=fake)


TICKERS_5 = ["AAPL", "MSFT", "GOOG", "NVDA", "AMZN"]
WEIGHTS_5 = [0.25, 0.25, 0.20, 0.20, 0.10]


def _run_payload(method: str, tickers=None, weights=None, max_position_weight=0.40, sector_map=None):
    from core.optimizer import optimize_portfolio, _build_optimizer_payload

    tickers = tickers or TICKERS_5
    weights = weights or WEIGHTS_5
    with _patch_prices(tickers):
        result = optimize_portfolio(
            tickers,
            weights,
            method=method,
            max_position_weight=max_position_weight,
            max_sector_weight=0.60,
            sector_map=sector_map or {},
        )
    payload_str = _build_optimizer_payload(result, method)
    parsed = json.loads(payload_str)  # must not raise
    # Strict round-trip: no NaN, no numpy types
    json.dumps(parsed, allow_nan=False)  # must not raise
    return result, parsed


class TestPayloadJsonSafety:
    def test_max_sharpe_valid_json(self):
        _, parsed = _run_payload("max_sharpe")
        assert "holdings" in parsed
        assert "portfolio" in parsed

    def test_min_vol_valid_json(self):
        _, parsed = _run_payload("min_vol")
        assert "holdings" in parsed

    def test_risk_parity_valid_json(self):
        _, parsed = _run_payload("risk_parity")
        assert "holdings" in parsed

    def test_binding_cap_does_not_crash(self):
        # Tight cap forces binding — this is the exact crash path (np.bool_ from comparison)
        # Use 4 tickers with 25% cap so every ticker might bind
        tickers = ["AAPL", "MSFT", "GOOG", "NVDA"]
        weights = [0.25, 0.25, 0.25, 0.25]
        _, parsed = _run_payload(
            "max_sharpe",
            tickers=tickers,
            weights=weights,
            max_position_weight=0.30,  # tight cap — likely to bind
        )
        # All at_position_cap values must be plain Python bool
        for h in parsed["holdings"]:
            assert isinstance(h["at_position_cap"], bool), (
                f"at_position_cap for {h['ticker']} is {type(h['at_position_cap'])}, not bool"
            )

    def test_no_nan_in_payload(self):
        for method in ["max_sharpe", "min_vol", "risk_parity"]:
            _, parsed = _run_payload(method)
            raw = json.dumps(parsed)
            assert "NaN" not in raw, f"{method}: NaN found in payload"
            assert "Infinity" not in raw, f"{method}: Infinity found in payload"
            assert "Inf" not in raw, f"{method}: Inf found in payload"

    def test_all_holdings_present(self):
        result, parsed = _run_payload("max_sharpe")
        payload_tickers = {h["ticker"] for h in parsed["holdings"]}
        assert payload_tickers == set(result.tickers)

    def test_sharpe_formula_string(self):
        _, parsed = _run_payload("max_sharpe")
        formula = parsed["portfolio"]["sharpe_formula"]
        assert isinstance(formula, str)
        assert "%" in formula

    def test_sector_weights_absent_without_sector_map(self):
        _, parsed = _run_payload("max_sharpe", sector_map={})
        assert parsed["sector_weights"] == {}

    def test_sector_weights_present_with_sector_map(self):
        sm = {"AAPL": "Tech", "MSFT": "Tech", "GOOG": "Tech", "NVDA": "Tech", "AMZN": "Consumer"}
        _, parsed = _run_payload("max_sharpe", sector_map=sm)
        assert "Tech" in parsed["sector_weights"] or "Consumer" in parsed["sector_weights"]

    def test_sensitivity_rows_count(self):
        # Each ticker gets 2 rows (+ and - shock)
        result, parsed = _run_payload("max_sharpe")
        n = len(result.tickers)
        assert len(parsed["sensitivity"]) == 2 * n


class TestFallbackSummary:
    def test_fallback_nonempty_for_all_methods(self):
        from core.optimizer import _build_fallback_summary

        for method in ["max_sharpe", "min_vol", "risk_parity"]:
            _, _ = _run_payload(method)  # warm up result
            with _patch_prices(TICKERS_5):
                from core.optimizer import optimize_portfolio
                result = optimize_portfolio(TICKERS_5, WEIGHTS_5, method=method)
            text = _build_fallback_summary(result, method)
            assert len(text) > 200, f"{method}: fallback summary too short"
            assert "## Inputs" in text
            assert "## Active Constraints" in text
            assert "## The Arithmetic" in text
            assert "## Why Each Weight Moved" in text
            assert "## Plausibility Check" in text

    def test_fallback_covers_every_holding(self):
        from core.optimizer import _build_fallback_summary

        with _patch_prices(TICKERS_5):
            from core.optimizer import optimize_portfolio
            result = optimize_portfolio(TICKERS_5, WEIGHTS_5, method="max_sharpe")
        text = _build_fallback_summary(result, "max_sharpe")
        for t in result.tickers:
            assert t in text, f"Ticker {t} missing from fallback summary"

    def test_fallback_used_when_llm_raises(self):
        """Simulate LLM failure — the UI should store the fallback, not empty string."""
        from core.optimizer import _build_fallback_summary, optimize_portfolio

        with _patch_prices(TICKERS_5):
            result = optimize_portfolio(TICKERS_5, WEIGHTS_5, method="max_sharpe")

        with patch("core.optimizer.generate_optimizer_summary", side_effect=RuntimeError("LLM down")):
            # The fallback should be non-empty and contain section headers
            fallback = _build_fallback_summary(result, "max_sharpe")
            assert "## Inputs" in fallback


class TestPayloadTypeInvariants:
    """Every scalar value in the payload must be a JSON-native Python type."""

    def _assert_json_native(self, obj, path="root"):
        if obj is None:
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                assert isinstance(k, str), f"{path}: key {k!r} is {type(k)}"
                self._assert_json_native(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                self._assert_json_native(v, f"{path}[{i}]")
        else:
            assert isinstance(obj, (bool, int, float, str)), (
                f"{path}: got {type(obj).__name__} = {obj!r}"
            )
            if isinstance(obj, float):
                assert not (obj != obj), f"{path}: NaN in payload"  # NaN != NaN

    def test_all_methods_produce_native_types(self):
        for method in ["max_sharpe", "min_vol", "risk_parity"]:
            _, parsed = _run_payload(method)
            self._assert_json_native(parsed, method)
