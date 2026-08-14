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
class SensitivityRow:
    ticker: str
    return_shock: float        # applied shock to expected return (e.g. +0.02)
    weight_before: float       # proposed weight before shock
    weight_after: float        # re-optimized weight after shock
    weight_delta: float        # weight_after - weight_before


@dataclass
class OptimizeResult:
    tickers: list[str]
    current_weights: list[float]
    proposed_weights: list[float]
    expected_return: float
    expected_vol: float
    sharpe: float
    # Inputs exposed for prose summary
    mu: list[float]             # annualized expected return per ticker
    vols: list[float]           # annualized vol per ticker
    corr_matrix: list[list[float]]  # pairwise correlation matrix
    risk_free_rate: float
    lookback_days: int
    risk_contributions: list[float]  # each ticker's fractional risk contribution
    # Constraint params and which ones bound
    max_position_weight: float = 1.0
    max_sector_weight: float = 1.0
    turnover_penalty: float = 0.0
    binding_constraints: list[str] = field(default_factory=list)
    sector_map: dict[str, str] = field(default_factory=dict)
    # Sensitivity: expected-return shocks → weight changes
    sensitivity: list[SensitivityRow] = field(default_factory=list)
    # Legacy dict kept for the existing UI table; populated from sensitivity rows
    sensitivity_legacy: dict[str, tuple[float, float]] = field(default_factory=dict)


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


def _build_constraints_and_bounds(
    n: int,
    valid_tickers: list[str],
    max_position_weight: float,
    max_sector_weight: float,
    sector_map: dict[str, str],
    min_position: float = 0.0,
) -> tuple[list[tuple[float, float]], list[dict]]:
    """Build SLSQP bounds + constraint list.

    Ensures the position cap is at least 1/n so the sum-to-one constraint is
    always feasible (e.g. 20% cap with 4 stocks would be infeasible at 4×20%=80%).
    """
    min_feasible = 1.0 / n
    effective_max = max(max_position_weight, min_feasible)
    if effective_max > max_position_weight + 1e-6:
        logger.info(
            "Position cap %.0f%% relaxed to %.0f%% to keep sum-to-one feasible with %d assets.",
            max_position_weight * 100, effective_max * 100, n,
        )

    bounds = [(min_position, effective_max)] * n
    constraints: list[dict] = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    if sector_map:
        from collections import defaultdict
        sector_to_idx: dict[str, list[int]] = defaultdict(list)
        for i, t in enumerate(valid_tickers):
            sector_to_idx[sector_map.get(t, "Unknown")].append(i)
        for indices in sector_to_idx.values():
            if len(indices) >= 2:
                constraints.append({
                    "type": "ineq",
                    "fun": lambda w, idx=indices: max_sector_weight - sum(w[i] for i in idx),
                })

    return bounds, constraints


def _max_sharpe(
    mu: np.ndarray,
    cov: np.ndarray,
    rf: float,
    bounds: list[tuple[float, float]],
    constraints: list[dict],
    current_w: Optional[np.ndarray] = None,
    turnover_penalty: float = 0.0,
) -> np.ndarray:
    n = len(mu)
    w0 = np.ones(n) / n

    def neg_sharpe(w):
        obj = -_sharpe(w, mu, cov, rf)
        if turnover_penalty > 0.0 and current_w is not None:
            obj += turnover_penalty * float(np.sum(np.abs(w - current_w)))
        return obj

    res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=constraints,
                   options={"maxiter": 1000, "ftol": 1e-9})
    if not res.success:
        logger.warning("max_sharpe did not converge: %s. Falling back to equal weights.", res.message)
        return w0
    return res.x


def _min_vol(
    cov: np.ndarray,
    bounds: list[tuple[float, float]],
    constraints: list[dict],
    current_w: Optional[np.ndarray] = None,
    turnover_penalty: float = 0.0,
) -> np.ndarray:
    n = cov.shape[0]
    w0 = np.ones(n) / n

    def port_vol(w):
        obj = np.sqrt(w @ cov @ w)
        if turnover_penalty > 0.0 and current_w is not None:
            obj += turnover_penalty * float(np.sum(np.abs(w - current_w)))
        return obj

    res = minimize(port_vol, w0, method="SLSQP", bounds=bounds, constraints=constraints,
                   options={"maxiter": 1000, "ftol": 1e-9})
    if not res.success:
        logger.warning("min_vol did not converge: %s. Falling back to equal weights.", res.message)
        return w0
    return res.x


def _risk_parity(
    cov: np.ndarray,
    bounds: list[tuple[float, float]],
    constraints: list[dict],
    current_w: Optional[np.ndarray] = None,
    turnover_penalty: float = 0.0,
) -> np.ndarray:
    """True risk parity: equalize each asset's marginal risk contribution.

    Solves min Σ(RC_i - target)² subject to Σw=1, w≥0, where
    RC_i = w_i * (Σw)_i is asset i's risk contribution.
    """
    n = cov.shape[0]
    w0 = np.ones(n) / n

    def _risk_contributions(w: np.ndarray) -> np.ndarray:
        pv = np.sqrt(w @ cov @ w)
        if pv < 1e-12:
            return np.ones(n) / n
        mrc = (cov @ w) / pv
        return w * mrc

    def objective(w: np.ndarray) -> float:
        rc = _risk_contributions(w)
        target = rc.sum() / n
        obj = float(np.sum((rc - target) ** 2))
        if turnover_penalty > 0.0 and current_w is not None:
            obj += turnover_penalty * float(np.sum(np.abs(w - current_w)))
        return obj

    # Risk parity needs a strictly positive lower bound for the risk-contribution gradient to be defined
    rp_bounds = [(max(b[0], 1e-4), b[1]) for b in bounds]

    res = minimize(objective, w0, method="SLSQP", bounds=rp_bounds, constraints=constraints,
                   options={"maxiter": 2000, "ftol": 1e-12})
    if not res.success:
        logger.warning("risk_parity did not converge: %s. Falling back to inverse-vol.", res.message)
        vols = np.sqrt(np.diag(cov))
        inv_vols = 1.0 / np.where(vols > 1e-12, vols, 1e-12)
        return inv_vols / inv_vols.sum()

    w = np.clip(res.x, 0.0, 1.0)
    return w / w.sum()


def _compute_risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Return each asset's fractional risk contribution (sums to 1)."""
    port_vol = np.sqrt(w @ cov @ w)
    if port_vol < 1e-12:
        return np.ones(len(w)) / len(w)
    mrc = (cov @ w) / port_vol
    rc = w * mrc
    total = rc.sum()
    return rc / total if total > 1e-12 else np.ones(len(w)) / len(w)


def _compute_sensitivity(
    tickers: list[str],
    proposed_weights: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    rf: float,
    method: str,
    delta: float = 0.02,
    bounds: Optional[list[tuple[float, float]]] = None,
    constraints: Optional[list[dict]] = None,
    current_w: Optional[np.ndarray] = None,
    turnover_penalty: float = 0.0,
) -> tuple[list[SensitivityRow], dict[str, tuple[float, float]]]:
    """Shock each holding's expected return ±delta, report weight changes.

    Re-optimizes with the same constraints as the original run so sensitivity
    rows reflect constrained responses, not unconstrained corner solutions.
    """
    n = len(tickers)
    if bounds is None:
        bounds = [(0.0, 1.0)] * n
    if constraints is None:
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    rows: list[SensitivityRow] = []
    legacy: dict[str, tuple[float, float]] = {}

    def _reoptimize(mu_shocked: np.ndarray) -> np.ndarray:
        if method == "max_sharpe":
            w = _max_sharpe(mu_shocked, cov, rf, bounds, constraints, current_w, turnover_penalty)
        elif method == "min_vol":
            w = _min_vol(cov, bounds, constraints, current_w, turnover_penalty)
        elif method == "risk_parity":
            w = _risk_parity(cov, bounds, constraints, current_w, turnover_penalty)
        else:
            return proposed_weights.copy()
        w = np.clip(w, 0.0, 1.0)
        return w / w.sum()

    for i, ticker in enumerate(tickers):
        for shock in (delta, -delta):
            mu_shocked = mu.copy()
            mu_shocked[i] += shock
            w_new = _reoptimize(mu_shocked)
            w_delta = float(w_new[i] - proposed_weights[i])
            rows.append(SensitivityRow(
                ticker=ticker,
                return_shock=shock,
                weight_before=float(proposed_weights[i]),
                weight_after=float(w_new[i]),
                weight_delta=w_delta,
            ))

    # Legacy shape: {ticker: (delta, sharpe_impact)} — repurpose as (return_shock, weight_delta)
    for r in rows:
        if r.return_shock > 0:   # only one direction for legacy display
            legacy[r.ticker] = (r.return_shock, r.weight_delta)

    return rows, legacy


def optimize_portfolio(
    tickers: list[str],
    current_weights: list[float],
    method: str = "max_sharpe",
    risk_free_rate: float = 0.05,
    lookback_days: int = 252,
    sensitivity_delta: float = 0.02,
    max_position_weight: float = 0.40,
    max_sector_weight: float = 0.60,
    turnover_penalty: float = 0.0,
    sector_map: Optional[dict[str, str]] = None,
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

    sm = sector_map or {}
    returns_arr = returns_df[valid_tickers].values
    mu = returns_arr.mean(axis=0) * 252
    cov = _ledoit_wolf_cov(returns_arr)

    bounds, constraints = _build_constraints_and_bounds(
        len(valid_tickers), valid_tickers,
        max_position_weight, max_sector_weight, sm,
    )

    if method == "max_sharpe":
        proposed = _max_sharpe(mu, cov, risk_free_rate, bounds, constraints, valid_current, turnover_penalty)
    elif method == "min_vol":
        proposed = _min_vol(cov, bounds, constraints, valid_current, turnover_penalty)
    elif method == "risk_parity":
        proposed = _risk_parity(cov, bounds, constraints, valid_current, turnover_penalty)
    else:
        raise ValueError(f"Unknown method: {method}")

    proposed = np.clip(proposed, 0.0, 1.0)
    proposed /= proposed.sum()

    # Detect binding constraints: positions at their upper bound
    effective_max = max(max_position_weight, 1.0 / len(valid_tickers))
    binding: list[str] = []
    for i, t in enumerate(valid_tickers):
        if proposed[i] >= effective_max - 1e-3:
            binding.append(f"{t} at position cap ({effective_max * 100:.0f}%)")

    if sm:
        from collections import defaultdict
        sector_to_idx: dict[str, list[int]] = defaultdict(list)
        for i, t in enumerate(valid_tickers):
            sector_to_idx[sm.get(t, "Unknown")].append(i)
        for sector, indices in sector_to_idx.items():
            if len(indices) >= 2:
                sw = sum(proposed[i] for i in indices)
                if sw >= max_sector_weight - 1e-3:
                    binding.append(f"{sector} sector at cap ({max_sector_weight * 100:.0f}%)")

    exp_ret = float(mu @ proposed)
    exp_vol = float(np.sqrt(proposed @ cov @ proposed))
    sharpe = _sharpe(proposed, mu, cov, risk_free_rate)

    sensitivity_rows, sensitivity_legacy = _compute_sensitivity(
        valid_tickers, proposed, mu, cov, risk_free_rate, method, sensitivity_delta,
        bounds=bounds, constraints=constraints,
        current_w=valid_current, turnover_penalty=turnover_penalty,
    )

    # Correlation matrix from cov
    vols_arr = np.sqrt(np.diag(cov))
    corr = cov / np.outer(
        np.where(vols_arr > 1e-12, vols_arr, 1e-12),
        np.where(vols_arr > 1e-12, vols_arr, 1e-12),
    )
    rc = _compute_risk_contributions(proposed, cov)

    return OptimizeResult(
        tickers=valid_tickers,
        current_weights=valid_current.tolist(),
        proposed_weights=proposed.tolist(),
        expected_return=exp_ret,
        expected_vol=exp_vol,
        sharpe=float(sharpe),
        mu=mu.tolist(),
        vols=vols_arr.tolist(),
        corr_matrix=corr.tolist(),
        risk_free_rate=risk_free_rate,
        lookback_days=lookback_days,
        risk_contributions=rc.tolist(),
        max_position_weight=effective_max,
        max_sector_weight=max_sector_weight,
        turnover_penalty=turnover_penalty,
        binding_constraints=binding,
        sector_map=sm,
        sensitivity=sensitivity_rows,
        sensitivity_legacy=sensitivity_legacy,
    )


def _build_optimizer_payload(result: OptimizeResult, method: str) -> str:
    """Serialize optimizer result into a compact payload for LLM prose summary."""
    import json

    n = len(result.tickers)
    corr = result.corr_matrix
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((result.tickers[i], result.tickers[j], corr[i][j]))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    # Sector groupings for payload
    sector_to_tickers: dict[str, list[str]] = {}
    if result.sector_map:
        for t in result.tickers:
            s = result.sector_map.get(t, "Unknown")
            sector_to_tickers.setdefault(s, []).append(t)

    proposed_arr = np.array(result.proposed_weights)
    payload = {
        "method": method,
        "risk_free_rate_pct": round(result.risk_free_rate * 100, 2),
        "lookback_days": result.lookback_days,
        "covariance_shrinkage": "Ledoit-Wolf",
        "n_assets": n,
        "constraints": {
            "max_position_pct": round(result.max_position_weight * 100, 1),
            "max_sector_pct": round(result.max_sector_weight * 100, 1),
            "turnover_penalty": round(result.turnover_penalty, 3),
            "binding": result.binding_constraints,
        },
        "holdings": [
            {
                "ticker": t,
                "sector": result.sector_map.get(t, "Unknown"),
                "expected_return_pct": round(result.mu[i] * 100, 2),
                "vol_pct": round(result.vols[i] * 100, 2),
                "current_weight_pct": round(result.current_weights[i] * 100, 2),
                "proposed_weight_pct": round(result.proposed_weights[i] * 100, 2),
                "weight_delta_pp": round(
                    (result.proposed_weights[i] - result.current_weights[i]) * 100, 2
                ),
                "risk_contribution_pct": round(result.risk_contributions[i] * 100, 2),
                "weight_x_return": round(result.proposed_weights[i] * result.mu[i] * 100, 2),
                "at_position_cap": bool(proposed_arr[i] >= result.max_position_weight - 1e-3),
            }
            for i, t in enumerate(result.tickers)
        ],
        "portfolio": {
            "expected_return_pct": round(result.expected_return * 100, 2),
            "vol_pct": round(result.expected_vol * 100, 2),
            "sharpe": round(result.sharpe, 4),
            "sharpe_formula": (
                f"({round(result.expected_return*100,2)}% - {round(result.risk_free_rate*100,2)}%) "
                f"/ {round(result.expected_vol*100,2)}% = {round(result.sharpe, 4)}"
            ),
        },
        "sector_weights": {
            s: round(sum(result.proposed_weights[result.tickers.index(t)] for t in ts) * 100, 1)
            for s, ts in sector_to_tickers.items()
        } if sector_to_tickers else {},
        "correlation_pairs": [
            {"pair": f"{a}/{b}", "corr": round(c, 3)} for a, b, c in pairs
        ],
        "sensitivity": [
            {
                "ticker": r.ticker,
                "return_shock_pp": round(r.return_shock * 100, 2),
                "weight_before_pct": round(r.weight_before * 100, 2),
                "weight_after_pct": round(r.weight_after * 100, 2),
                "weight_delta_pp": round(r.weight_delta * 100, 2),
            }
            for r in result.sensitivity
        ],
    }
    return json.dumps(payload, indent=2)


def generate_optimizer_summary(result: OptimizeResult, method: str) -> str:
    """Generate prose explanation of optimizer output via LLM.

    All numbers in the summary are drawn from result; the LLM writes prose
    around values it is given and must not introduce figures not in the payload.
    """
    import llm
    from config import PROMPT_VERSIONS, SONNET

    payload = _build_optimizer_payload(result, method)

    method_labels = {
        "max_sharpe": "Maximum Sharpe",
        "min_vol": "Minimum Volatility",
        "risk_parity": "Risk Parity",
    }
    method_label = method_labels.get(method, method)

    long_run_baseline = 9.0  # approximate 8–10% long-run equity nominal return

    system = llm.cached_system(f"""
You are a senior portfolio manager explaining an optimizer's output to a sophisticated investor.
Every number you write MUST come from the payload below. Do NOT introduce estimates, forecasts,
or figures that are not in the JSON. If a number seems high relative to long-run equity baselines
(~{long_run_baseline}% nominal), say so plainly and trace it to its source.

Structure your response as:
## Inputs
## Active Constraints
## The Arithmetic
## Why Each Weight Moved
## Dominant Assumption
## Plausibility Check
## What Would Change the Recommendation

Guidelines:
- In "Inputs": state the risk-free rate, lookback window, and shrinkage method explicitly.
- In "Active Constraints": name the position cap and sector cap; name any binding constraints
  (weights at their limit); if no constraint bound, say so.
- In "The Arithmetic": show weight × expected_return per holding summing to portfolio E[r],
  then Sharpe using the exact formula in sharpe_formula.
- In "Why Each Weight Moved": cover every holding. If a weight is zero or at its cap,
  say whether that is a genuine optimizer preference or a constraint forcing it.
- In "Plausibility Check": compare portfolio expected return to the long-run baseline.
  If the portfolio E[r] is more than double the baseline, name the holding driving it.
- Keep each section to 2–4 sentences. Be direct; name numbers.
""")

    messages = [
        {
            "role": "user",
            "content": [
                *llm.cached_content(f"<optimizer_result method='{method_label}'>\n{payload}\n</optimizer_result>"),
                llm.text_block(
                    f"Write the optimizer explanation for the {method_label} result. "
                    f"The long-run equity nominal return baseline is approximately {long_run_baseline}%. "
                    "Compare the portfolio expected return to that baseline in the Plausibility Check section. "
                    "Name the dominant assumption explicitly. "
                    "For 'Why Each Weight Moved', cover every holding."
                ),
            ],
        }
    ]
    return llm.call(
        SONNET,
        messages,
        system=system,
        mode="stream",
        prompt_version=PROMPT_VERSIONS["optimizer_summary"],
        max_tokens=2000,
    )
