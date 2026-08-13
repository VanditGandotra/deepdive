"""Tests for core/optimizer.py and core/montecarlo.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


def _make_price_data(ticker: str, n: int = 253) -> MagicMock:
    """Build a fake PriceData object with deterministic prices."""
    rng = np.random.default_rng(seed=abs(hash(ticker)) % (2**31))
    prices = 100.0 * np.cumprod(1 + rng.normal(0.0005, 0.015, n))

    bar_mock = []
    for p in prices:
        b = MagicMock()
        b.close = float(p)
        bar_mock.append(b)

    pd_obj = MagicMock()
    pd_obj.bars = bar_mock
    return pd_obj


def _patch_get_prices(tickers: list[str], n: int = 253):
    """Return a context manager that patches data.market.get_prices."""
    price_data = {t: _make_price_data(t, n) for t in tickers}

    def fake_get_prices(ticker, period="2y"):
        if ticker in price_data:
            return price_data[ticker]
        raise ValueError(f"No data for {ticker}")

    return patch("data.market.get_prices", side_effect=fake_get_prices)


TICKERS = ["AAPL", "MSFT", "GOOGL"]
WEIGHTS = [0.4, 0.35, 0.25]


class TestOptimizer:
    def test_optimize_max_sharpe_returns_valid_weights(self):
        from core.optimizer import optimize_portfolio
        with _patch_get_prices(TICKERS):
            result = optimize_portfolio(TICKERS, WEIGHTS, method="max_sharpe")
        assert abs(sum(result.proposed_weights) - 1.0) < 1e-6
        assert all(w >= -1e-8 for w in result.proposed_weights)
        assert result.tickers == TICKERS

    def test_optimize_min_vol_returns_valid_weights(self):
        from core.optimizer import optimize_portfolio
        with _patch_get_prices(TICKERS):
            result = optimize_portfolio(TICKERS, WEIGHTS, method="min_vol")
        assert abs(sum(result.proposed_weights) - 1.0) < 1e-6
        assert all(w >= -1e-8 for w in result.proposed_weights)

    def test_optimize_risk_parity_returns_valid_weights(self):
        from core.optimizer import optimize_portfolio
        with _patch_get_prices(TICKERS):
            result = optimize_portfolio(TICKERS, WEIGHTS, method="risk_parity")
        assert abs(sum(result.proposed_weights) - 1.0) < 1e-6
        assert all(w >= -1e-8 for w in result.proposed_weights)

    def test_optimize_single_ticker_raises(self):
        from core.optimizer import optimize_portfolio
        with pytest.raises(ValueError, match="2 tickers"):
            optimize_portfolio(["AAPL"], [1.0])

    def test_sensitivity_rows_match_tickers(self):
        from core.optimizer import optimize_portfolio, SensitivityRow
        with _patch_get_prices(TICKERS):
            result = optimize_portfolio(TICKERS, WEIGHTS, method="max_sharpe")
        # sensitivity is now a list of SensitivityRow (±shock per ticker)
        assert isinstance(result.sensitivity, list)
        row_tickers = {r.ticker for r in result.sensitivity}
        assert row_tickers == set(result.tickers)
        # each ticker should have two rows: +delta and -delta
        from collections import Counter
        counts = Counter(r.ticker for r in result.sensitivity)
        for t in result.tickers:
            assert counts[t] == 2, f"Expected 2 rows (±shock) for {t}, got {counts[t]}"
        for row in result.sensitivity:
            assert isinstance(row, SensitivityRow)
            assert isinstance(row.weight_delta, float)
        # legacy dict still populated for UI compatibility
        assert set(result.sensitivity_legacy.keys()) == set(result.tickers)


class TestMonteCarlo:
    def test_montecarlo_percentile_ordering(self):
        from core.montecarlo import run_montecarlo
        with _patch_get_prices(TICKERS):
            mc = run_montecarlo(WEIGHTS, TICKERS, n_paths=1000, horizon_days=252)
        p = mc.percentiles
        assert p[10] < p[25] < p[50] < p[75] < p[90]

    def test_montecarlo_paths_sample_length(self):
        from core.montecarlo import run_montecarlo
        horizon = 126
        with _patch_get_prices(TICKERS):
            mc = run_montecarlo(WEIGHTS, TICKERS, n_paths=200, horizon_days=horizon)
        assert len(mc.paths_sample) <= 50
        for path in mc.paths_sample:
            assert len(path) == horizon + 1

    def test_montecarlo_reproducible(self):
        from core.montecarlo import run_montecarlo
        with _patch_get_prices(TICKERS):
            mc1 = run_montecarlo(WEIGHTS, TICKERS, n_paths=500, horizon_days=63)
        with _patch_get_prices(TICKERS):
            mc2 = run_montecarlo(WEIGHTS, TICKERS, n_paths=500, horizon_days=63)
        assert abs(mc1.percentiles[50] - mc2.percentiles[50]) < 1e-10
