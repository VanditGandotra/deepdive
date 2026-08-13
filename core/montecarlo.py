from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloResult:
    horizon_days: int
    percentiles: dict[int, float] = field(default_factory=dict)
    paths_sample: list[list[float]] = field(default_factory=list)


def _fetch_log_returns(tickers: list[str], lookback_days: int) -> pd.DataFrame:
    from data.market import get_prices

    series = {}
    for t in tickers:
        try:
            pd_obj = get_prices(t, period="2y")
            bars = pd_obj.bars[-(lookback_days + 1):]
            if len(bars) < 2:
                raise ValueError("Insufficient data")
            closes = np.array([b.close for b in bars], dtype=float)
            log_rets = np.diff(np.log(closes))
            series[t] = log_rets
        except Exception as exc:
            logger.warning("Skipping %s in Monte Carlo: %s", t, exc)

    if not series:
        return pd.DataFrame()

    min_len = min(len(v) for v in series.values())
    return pd.DataFrame({t: v[-min_len:] for t, v in series.items()})


def run_montecarlo(
    weights: list[float],
    tickers: list[str],
    n_paths: int = 10_000,
    horizon_days: int = 252,
    lookback_days: int = 252,
    initial_value: float = 1.0,
) -> MonteCarloResult:
    rng = np.random.default_rng(42)

    returns_df = _fetch_log_returns(tickers, lookback_days)
    valid_tickers = [t for t in tickers if t in returns_df.columns]

    valid_idx = [tickers.index(t) for t in valid_tickers]
    w = np.array([weights[i] for i in valid_idx], dtype=float)
    if w.sum() > 1e-9:
        w = w / w.sum()
    else:
        w = np.ones(len(valid_tickers)) / len(valid_tickers)

    returns_arr = returns_df[valid_tickers].values
    mu_daily = returns_arr.mean(axis=0)
    lw = LedoitWolf()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lw.fit(returns_arr)
    cov_daily = lw.covariance_

    shocks = rng.multivariate_normal(mu_daily, cov_daily, size=(n_paths, horizon_days))
    port_log_rets = shocks @ w

    cum_paths = np.exp(np.cumsum(port_log_rets, axis=1))
    terminal = cum_paths[:, -1] * initial_value

    pct_levels = [10, 25, 50, 75, 90]
    percentiles = {p: float(np.percentile(terminal, p)) for p in pct_levels}

    n_sample = min(50, n_paths)
    sample_idx = rng.choice(n_paths, size=n_sample, replace=False)
    sample_paths = cum_paths[sample_idx] * initial_value

    paths_sample = []
    for path in sample_paths:
        full_path = [initial_value] + path.tolist()
        paths_sample.append(full_path)

    return MonteCarloResult(
        horizon_days=horizon_days,
        percentiles=percentiles,
        paths_sample=paths_sample,
    )
