from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)


@dataclass
class OptimizeResult:
    tickers: list[str]
    current_weights: list[float]
    proposed_weights: list[float]
    expected_return: float
    expected_vol: float
    sharpe: float
    sensitivity: dict[str, tuple[float, float]] = field(default_factory=dict)


def _prices_to_returns(price_data, lookback_days: int) -> Optional[pd.Series]:
    bars = price_data.bars[-lookback_days - 1:]
    if len(bars) < 2:
        return None
    closes = pd.Series([b.close for b in bars])
    return closes.pct_change().dropna()


def _build_returns_matrix(
    tickers: list[str], lookback_days: int
) -> tuple[pd.DataFrame, list[str]]:
    from data.market import get_prices

    series = {}
    dropped = []
    for t in tickers:
        try:
            pd_obj = get_prices(t, period="2y")
            ret = _prices_to_returns(pd_obj, lookback_days)
            if ret is None or len(ret) < 20:
                raise ValueError("Insufficient data")
            series[t] = ret.values
        except Exception as exc:
            logger.warning("Dropping %s from optimizer: %s", t, exc)
            dropped.append(t)

    if dropped:
        logger.warning("Tickers dropped from optimizer: %s", dropped)

    lengths = [len(v) for v in series.values()]
    if not lengths:
        return pd.DataFrame(), []

    min_len = min(lengths)
    df = pd.DataFrame({t: v[-min_len:] for t, v in series.items()})
    return df, dropped


def _ledoit_wolf_cov(returns: np.ndarray) -> np.ndarray:
    lw = LedoitWolf()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lw.fit(returns)
    return lw.covariance_ * 252


def _sharpe(w: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float) -> float:
    port_ret = mu @ w
    port_vol = np.sqrt(w @ cov @ w)
    if port_vol < 1e-12:
        return 0.0
    return (port_ret - rf) / port_vol


def _max_sharpe(mu: np.ndarray, cov: np.ndarray, rf: float) -> np.ndarray:
    n = len(mu)
    w0 = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    def neg_sharpe(w):
        return -_sharpe(w, mu, cov, rf)

    res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=constraints,
                   options={"maxiter": 1000, "ftol": 1e-9})
    if not res.success:
        logger.warning("max_sharpe did not converge: %s. Falling back to equal weights.", res.message)
        return w0
    return res.x


def _min_vol(cov: np.ndarray) -> np.ndarray:
    n = cov.shape[0]
    w0 = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    def port_vol(w):
        return np.sqrt(w @ cov @ w)

    res = minimize(port_vol, w0, method="SLSQP", bounds=bounds, constraints=constraints,
                   options={"maxiter": 1000, "ftol": 1e-9})
    if not res.success:
        logger.warning("min_vol did not converge: %s. Falling back to equal weights.", res.message)
        return w0
    return res.x


def _risk_parity(cov: np.ndarray) -> np.ndarray:
    vols = np.sqrt(np.diag(cov))
    inv_vols = 1.0 / np.where(vols > 1e-12, vols, 1e-12)
    w = inv_vols / inv_vols.sum()
    return w


def _compute_sensitivity(
    tickers: list[str],
    proposed_weights: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    rf: float,
    delta: float,
) -> dict[str, tuple[float, float]]:
    base_sharpe = _sharpe(proposed_weights, mu, cov, rf)
    result = {}
    n = len(tickers)
    for i, t in enumerate(tickers):
        bump = min(proposed_weights[i] + delta, 1.0) - proposed_weights[i]
        if bump < 1e-9:
            bump = 0.0
        w_bumped = proposed_weights.copy()
        w_bumped[i] += bump
        rest_sum = sum(w_bumped[j] for j in range(n) if j != i)
        if rest_sum > 1e-9:
            scale = (1.0 - w_bumped[i]) / rest_sum
            for j in range(n):
                if j != i:
                    w_bumped[j] *= scale
        else:
            for j in range(n):
                if j != i:
                    w_bumped[j] = (1.0 - w_bumped[i]) / (n - 1)
        w_bumped = np.clip(w_bumped, 0.0, 1.0)
        w_bumped /= w_bumped.sum()
        new_sharpe = _sharpe(w_bumped, mu, cov, rf)
        result[t] = (bump, new_sharpe - base_sharpe)
    return result


def optimize_portfolio(
    tickers: list[str],
    current_weights: list[float],
    method: str = "max_sharpe",
    risk_free_rate: float = 0.05,
    lookback_days: int = 252,
    sensitivity_delta: float = 0.02,
) -> OptimizeResult:
    if len(tickers) < 2:
        raise ValueError("At least 2 tickers required for optimization.")

    returns_df, _dropped = _build_returns_matrix(tickers, lookback_days)

    valid_tickers = [t for t in tickers if t in returns_df.columns]
    if len(valid_tickers) < 2:
        raise ValueError("Fewer than 2 tickers have sufficient price history after fetching data.")

    valid_idx = [tickers.index(t) for t in valid_tickers]
    valid_current = np.array([current_weights[i] for i in valid_idx], dtype=float)
    if valid_current.sum() > 1e-9:
        valid_current = valid_current / valid_current.sum()
    else:
        valid_current = np.ones(len(valid_tickers)) / len(valid_tickers)

    returns_arr = returns_df[valid_tickers].values
    mu = returns_arr.mean(axis=0) * 252
    cov = _ledoit_wolf_cov(returns_arr)

    if method == "max_sharpe":
        proposed = _max_sharpe(mu, cov, risk_free_rate)
    elif method == "min_vol":
        proposed = _min_vol(cov)
    elif method == "risk_parity":
        proposed = _risk_parity(cov)
    else:
        raise ValueError(f"Unknown method: {method}")

    proposed = np.clip(proposed, 0.0, 1.0)
    proposed /= proposed.sum()

    exp_ret = float(mu @ proposed)
    exp_vol = float(np.sqrt(proposed @ cov @ proposed))
    sharpe = _sharpe(proposed, mu, cov, risk_free_rate)

    sensitivity = _compute_sensitivity(
        valid_tickers, proposed, mu, cov, risk_free_rate, sensitivity_delta
    )

    return OptimizeResult(
        tickers=valid_tickers,
        current_weights=valid_current.tolist(),
        proposed_weights=proposed.tolist(),
        expected_return=exp_ret,
        expected_vol=exp_vol,
        sharpe=float(sharpe),
        sensitivity=sensitivity,
    )
