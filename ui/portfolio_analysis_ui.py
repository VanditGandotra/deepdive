"""Portfolio analysis page: optimizer + Monte Carlo, triggered by ?view=portfolio_analysis."""
from __future__ import annotations

import contextlib
import hashlib
import re

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from core.schemas import Scenario
from core.text_render import render_md, sanitize_markdown

_VALID_TICKER_RE = re.compile(r"^[A-Z]{1,5}$")


def _holdings_hash(
    tickers: list[str],
    weights: list[float],
    max_pos: float = 0.40,
    max_sec: float = 0.60,
    turnover: float = 0.0,
) -> str:
    """Stable hash of holdings + constraint params — changing any param busts the cache."""
    content = "|".join(f"{t}:{w:.6f}" for t, w in sorted(zip(tickers, weights)))
    content += f"|p{max_pos:.3f}|s{max_sec:.3f}|t{turnover:.3f}"
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def _back_button(key: str = "portfolio_analysis__back") -> None:
    if st.button("<- Back to portfolio", key=key):
        for k in ["view", "portfolio", "tickers"]:
            st.query_params.pop(k, None)
        st.rerun()


import logging as _logging

_log = _logging.getLogger(__name__)


def _one_way_turnover(current: list[float], proposed: list[float]) -> float:
    """Fraction of the portfolio that trades hands (one-way)."""
    return sum(abs(p - c) for p, c in zip(proposed, current)) / 2.0


def _render_constraints_panel(
    result,
    method: str,
    holdings_hash: str,
    max_pos: float,
    max_sec: float,
    turnover_penalty: float,
) -> None:
    """Rewritten constraints panel: interpretation over raw numbers."""
    from core.optimizer import optimize_portfolio

    n = len(result.tickers)

    # ── Binding constraints ────────────────────────────────────────────────────
    if result.binding_constraints:
        bc_names = ", ".join(
            bc.split(" at ")[0] for bc in result.binding_constraints
        )
        st.warning(
            f"**Corner solution — {bc_names} are pinned at their cap.** "
            f"The optimizer wanted *more* of these names and was stopped by the "
            f"{result.max_position_weight*100:.0f}% position cap. "
            "The reported weights are constraint-driven, not optimizer conviction. "
            "Relax the cap to see the unconstrained optima."
        )
    else:
        st.success(
            "No constraints binding — all weights are interior solutions "
            "(the optimizer's genuine preference given the input data)."
        )

    # ── Sector utilization ─────────────────────────────────────────────────────
    if result.sector_map:
        from collections import defaultdict
        sec_to_w: dict[str, float] = defaultdict(float)
        for t, w in zip(result.tickers, result.proposed_weights):
            sec_to_w[result.sector_map.get(t, "Unknown")] += w
        max_sec_name = max(sec_to_w, key=sec_to_w.__getitem__)
        max_sec_w = sec_to_w[max_sec_name]
        sec_util_pct = max_sec_w / result.max_sector_weight * 100
        if sec_util_pct >= 90:
            st.warning(
                f"**{max_sec_name} sector: {max_sec_w*100:.1f}% / {result.max_sector_weight*100:.0f}% cap "
                f"({sec_util_pct:.0f}% utilized — near binding).**"
            )
        else:
            st.info(
                f"Max sector exposure: {max_sec_name} at {max_sec_w*100:.1f}% "
                f"(cap is {result.max_sector_weight*100:.0f}% — {sec_util_pct:.0f}% utilized, inactive)."
            )

    # ── Turnover consequence ───────────────────────────────────────────────────
    ot = _one_way_turnover(result.current_weights, result.proposed_weights)
    if result.turnover_penalty == 0.0:
        if ot > 0.05:
            st.warning(
                f"**Turnover penalty = 0 — optimizer ignored trading costs.** "
                f"Proposed portfolio requires {ot*100:.1f}% one-way turnover "
                f"({sum(max(0.0, p-c) for p, c in zip(result.proposed_weights, result.current_weights))*100:.1f}% in buys). "
                "At 0.10–0.30% round-trip cost, this erases meaningful alpha."
            )
        else:
            st.caption(
                f"One-way turnover: {ot*100:.1f}% — low enough that trading costs are immaterial."
            )
    else:
        st.caption(
            f"Turnover penalty: {result.turnover_penalty:.2f} · "
            f"One-way turnover: {ot*100:.1f}%"
        )

    # ── Shadow cost of the position cap ───────────────────────────────────────
    shadow_key = f"_opt_shadow_{method}_{holdings_hash}"
    if result.binding_constraints:
        if shadow_key not in st.session_state:
            if st.button(
                "Calculate shadow cost of position cap →",
                key=f"portfolio_analysis__shadow__{method}",
                help="Re-solves with the position cap raised by 10pp to show what the constraint costs.",
            ):
                relaxed_cap = min(max_pos + 0.10, 1.0)
                with st.spinner(f"Re-solving with {relaxed_cap*100:.0f}% cap…"):
                    try:
                        shadow_result = optimize_portfolio(
                            tickers=result.tickers,
                            current_weights=result.current_weights,
                            method=method,
                            max_position_weight=relaxed_cap,
                            max_sector_weight=max_sec,
                            turnover_penalty=turnover_penalty,
                            sector_map=result.sector_map,
                        )
                        st.session_state[shadow_key] = {
                            "relaxed_cap": relaxed_cap,
                            "delta_sharpe": shadow_result.sharpe - result.sharpe,
                            "delta_vol": shadow_result.expected_vol - result.expected_vol,
                            "delta_ret": shadow_result.expected_return - result.expected_return,
                        }
                    except Exception as exc:
                        st.session_state[shadow_key] = {"error": str(exc)}
                st.rerun()
        else:
            sc = st.session_state[shadow_key]
            if "error" in sc:
                st.caption(f"Shadow cost computation failed: {sc['error']}")
            else:
                ds = sc["delta_sharpe"]
                dv = sc["delta_vol"] * 100
                dr = sc["delta_ret"] * 100
                cap_pct = sc["relaxed_cap"] * 100
                sign_s = "+" if ds >= 0 else ""
                sign_v = "+" if dv >= 0 else ""
                st.info(
                    f"**Shadow cost:** raising the cap from {max_pos*100:.0f}% → {cap_pct:.0f}% "
                    f"would change Sharpe {sign_s}{ds:.3f} "
                    f"and Vol {sign_v}{dv:.1f}pp. "
                    + (f"The cap is costing you {-ds:.3f} Sharpe units." if ds < 0 else
                       "The cap is not the binding constraint on Sharpe.")
                )

    # ── Inputs disclosure ─────────────────────────────────────────────────────
    st.caption(
        f"Expected returns: historical mean daily return × 252 (noisy — dominates Max Sharpe weighting). "
        f"Lookback: {result.lookback_days}d · Covariance: Ledoit-Wolf shrinkage · "
        f"rf = {result.risk_free_rate*100:.1f}% · "
        f"n = {n} assets (small matrix — treat weights as direction, not exact allocation)."
    )


def _render_insights_panel(
    result,
    portfolio_value: float,
    priced_holdings,
) -> None:
    """Trade list, metrics comparison, risk contributions, correlation."""
    import math

    n = len(result.tickers)

    # ── Trade list ────────────────────────────────────────────────────────────
    st.subheader("Trade List")
    st.caption("Buys and sells needed to move from current to proposed weights.")
    hold_by_ticker = {h.ticker: h for h in (priced_holdings or [])}
    buys, sells = [], []
    total_ot = 0.0
    for i, t in enumerate(result.tickers):
        cur_w = result.current_weights[i]
        prop_w = result.proposed_weights[i]
        delta_w = prop_w - cur_w
        delta_pct = delta_w * 100
        dollar_delta = delta_w * portfolio_value
        h = hold_by_ticker.get(t)
        if h and h.current_price and h.current_price > 0:
            shares_delta = abs(dollar_delta) / h.current_price
            shares_str = f"{shares_delta:,.0f} sh"
        else:
            shares_str = "—"
        row = {
            "Ticker": t,
            "Current": f"{cur_w*100:.1f}%",
            "Proposed": f"{prop_w*100:.1f}%",
            "Δ Weight": f"{delta_pct:+.1f}pp",
            "Δ $ (est.)": f"${dollar_delta:+,.0f}" if portfolio_value > 0 else "—",
            "Shares (est.)": shares_str if delta_w != 0 else "0",
        }
        if delta_w > 0.001:
            buys.append(row)
        elif delta_w < -0.001:
            sells.append(row)
        total_ot += abs(delta_w)
    total_ot /= 2.0
    all_trades = sells + buys
    if all_trades:
        st.dataframe(pd.DataFrame(all_trades), hide_index=True, use_container_width=True)
    else:
        st.info("No trades — proposed weights match current weights.")
    st.caption(
        f"Total one-way turnover: {total_ot*100:.1f}% of portfolio. "
        + (f"≈ ${total_ot * portfolio_value:,.0f}" if portfolio_value > 0 else "")
    )

    # ── Before / after metrics ─────────────────────────────────────────────────
    st.subheader("Before / After Metrics")
    import numpy as np

    cur_w = np.array(result.current_weights)
    prop_w = np.array(result.proposed_weights)
    mu = np.array(result.mu)
    vols = np.array(result.vols)

    # Rebuild cov from corr and vols (stored in result)
    corr = np.array(result.corr_matrix)
    cov = corr * np.outer(vols, vols)

    def _port_metrics(w: np.ndarray):
        ret = float(mu @ w)
        vol = float(np.sqrt(w @ cov @ w))
        rf = result.risk_free_rate
        sharpe = (ret - rf) / vol if vol > 1e-12 else 0.0
        # HHI-based effective N
        hhi = float(np.sum(w ** 2))
        eff_n = 1.0 / hhi if hhi > 1e-12 else float(n)
        return ret, vol, sharpe, eff_n

    cur_ret, cur_vol, cur_sharpe, cur_eff_n = _port_metrics(cur_w)
    prop_ret, prop_vol, prop_sharpe, prop_eff_n = _port_metrics(prop_w)

    metrics_rows = [
        {"Metric": "Expected Return", "Current": f"{cur_ret*100:.1f}%", "Proposed": f"{prop_ret*100:.1f}%",
         "Δ": f"{(prop_ret-cur_ret)*100:+.1f}pp"},
        {"Metric": "Volatility", "Current": f"{cur_vol*100:.1f}%", "Proposed": f"{prop_vol*100:.1f}%",
         "Δ": f"{(prop_vol-cur_vol)*100:+.1f}pp"},
        {"Metric": "Sharpe", "Current": f"{cur_sharpe:.3f}", "Proposed": f"{prop_sharpe:.3f}",
         "Δ": f"{prop_sharpe-cur_sharpe:+.3f}"},
        {"Metric": "Effective holdings (1/HHI)", "Current": f"{cur_eff_n:.1f}", "Proposed": f"{prop_eff_n:.1f}",
         "Δ": f"{prop_eff_n-cur_eff_n:+.1f}"},
    ]
    st.dataframe(pd.DataFrame(metrics_rows), hide_index=True, use_container_width=True)
    if abs(prop_sharpe - cur_sharpe) < 0.05:
        st.caption(
            "Δ Sharpe < 0.05 — within estimation error for a portfolio this size. "
            "Don't over-interpret small improvements."
        )

    # ── Risk contributions ────────────────────────────────────────────────────
    st.subheader("Risk Contribution — Capital Weight vs Risk Weight")
    st.caption(
        "Risk weight = each holding's share of total portfolio variance. "
        "A name with 10% capital weight can drive 40% of risk if it is volatile and correlated."
    )
    rc_rows = []
    for i, t in enumerate(result.tickers):
        rc_rows.append({
            "Ticker": t,
            "Capital (current)": f"{result.current_weights[i]*100:.1f}%",
            "Capital (proposed)": f"{result.proposed_weights[i]*100:.1f}%",
            "Risk share (proposed)": f"{result.risk_contributions[i]*100:.1f}%",
            "Risk/Capital ratio": f"{result.risk_contributions[i]/result.proposed_weights[i]:.2f}x"
            if result.proposed_weights[i] > 0.001 else "—",
        })
    st.dataframe(pd.DataFrame(rc_rows), hide_index=True, use_container_width=True)

    # ── Correlation ───────────────────────────────────────────────────────────
    if n >= 2:
        st.subheader("Highest Pairwise Correlations")
        pair_rows = []
        for i in range(n):
            for j in range(i + 1, n):
                c = corr[i][j]
                pair_rows.append({
                    "Pair": f"{result.tickers[i]} / {result.tickers[j]}",
                    "Correlation": f"{c:.2f}",
                    "Note": "Redundant exposure — they move together" if abs(c) > 0.75 else "",
                })
        pair_rows.sort(key=lambda r: float(r["Correlation"]), reverse=True)
        st.dataframe(pd.DataFrame(pair_rows[:6]), hide_index=True, use_container_width=True)


def _render_sensitivity_interpretation(result) -> None:
    """Explain what the sensitivity table is really telling you."""
    st.caption(
        "**Reading the table:** Names sitting *at a cap or at zero* show near-zero Δweight "
        "under a ±2pp return shock — not because the optimizer is indifferent, but because "
        "the constraint prevents them from moving. Those weights are constraint-driven. "
        "Names far from any bound show the optimizer's actual responsiveness to return estimates."
    )
    # Identify immobile names (at cap or zero)
    immobile = []
    free = []
    for row in result.sensitivity:
        at_cap = any(row.ticker in bc for bc in result.binding_constraints)
        at_zero = result.proposed_weights[result.tickers.index(row.ticker)] < 0.01
        if at_cap or at_zero:
            if row.ticker not in immobile:
                immobile.append(row.ticker)
        else:
            if row.ticker not in free:
                free.append(row.ticker)
    if immobile:
        st.caption(
            f"**{', '.join(immobile)}**: Δweight ≈ 0pp — constrained (at cap or zero). "
            "The optimizer can't move these names under a small shock."
        )
    if free:
        st.caption(
            f"**{', '.join(free)}**: responds to return shocks — these weights reflect optimizer conviction."
        )


def _render_optimizer_tab(
    method: str,
    tickers: list[str],
    current_weights: list[float],
    holdings_hash: str = "",
    sector_map: dict[str, str] | None = None,
    max_pos: float = 0.40,
    max_sec: float = 0.60,
    turnover: float = 0.0,
    portfolio_value: float = 0.0,
    priced_holdings=None,
) -> None:
    from core.optimizer import optimize_portfolio

    result_key = f"_opt_{method}_{holdings_hash}"
    if result_key not in st.session_state:
        try:
            st.session_state[result_key] = optimize_portfolio(
                tickers=tickers,
                current_weights=current_weights,
                method=method,
                max_position_weight=max_pos,
                max_sector_weight=max_sec,
                turnover_penalty=turnover,
                sector_map=sector_map or {},
            )
        except Exception as exc:
            st.error(f"Optimizer error: {exc}")
            return

    result = st.session_state[result_key]

    col_chart, col_metrics = st.columns([3, 1])

    with col_chart:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Current",
            x=result.tickers,
            y=[w * 100 for w in result.current_weights],
            marker_color="rgba(99,110,250,0.5)",
        ))
        fig.add_trace(go.Bar(
            name="Proposed",
            x=result.tickers,
            y=[w * 100 for w in result.proposed_weights],
            marker_color="rgba(0,204,150,0.85)",
        ))
        fig.update_layout(
            barmode="group",
            yaxis_title="Portfolio Weight (%)",
            height=320,
            legend=dict(orientation="h", y=1.1),
            margin=dict(l=4, r=4, t=32, b=4),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_metrics:
        st.metric("Expected Return", f"{result.expected_return * 100:.1f}%")
        st.metric("Volatility", f"{result.expected_vol * 100:.1f}%")
        st.metric("Sharpe", f"{result.sharpe:.2f}")
        st.caption(f"rf = {result.risk_free_rate * 100:.1f}%")

    # Constraints panel — interpretive, not just raw numbers (Issue 3)
    with st.expander("Constraints & interpretation", expanded=True):
        _render_constraints_panel(result, method, holdings_hash, max_pos, max_sec, turnover)

    # Insights: trade list, metrics, risk, correlations (Issue 4)
    with st.expander("Portfolio insights — trade list, metrics, risk", expanded=False):
        _render_insights_panel(result, portfolio_value, priced_holdings)

    st.subheader("Sensitivity — how weights move when expected returns are shocked ±2%")
    _render_sensitivity_interpretation(result)
    # Use structured sensitivity rows when available (new API), fall back to legacy dict
    sens_rows = []
    if result.sensitivity:
        proposed_map = dict(zip(result.tickers, result.proposed_weights))
        current_map = dict(zip(result.tickers, result.current_weights))
        by_ticker: dict[str, dict] = {}
        for row in result.sensitivity:
            entry = by_ticker.setdefault(row.ticker, {
                "Ticker": row.ticker,
                "Current Weight": f"{current_map.get(row.ticker, 0) * 100:.1f}%",
                "Proposed Weight": f"{proposed_map.get(row.ticker, 0) * 100:.1f}%",
            })
            if row.return_shock > 0:
                entry["+2pp return shock → weight Δ"] = f"{row.weight_delta * 100:+.1f}pp"
            else:
                entry["-2pp return shock → weight Δ"] = f"{row.weight_delta * 100:+.1f}pp"
        sens_rows = list(by_ticker.values())
    elif result.sensitivity_legacy:
        proposed_map = dict(zip(result.tickers, result.proposed_weights))
        current_map = dict(zip(result.tickers, result.current_weights))
        for t, (shock, w_delta) in result.sensitivity_legacy.items():
            sens_rows.append({
                "Ticker": t,
                "Current Weight": f"{current_map.get(t, 0) * 100:.1f}%",
                "Proposed Weight": f"{proposed_map.get(t, 0) * 100:.1f}%",
                "+2pp return shock → weight Δ": f"{w_delta * 100:+.1f}pp",
            })
    if sens_rows:
        st.dataframe(pd.DataFrame(sens_rows), hide_index=True, use_container_width=True)

    # Prose summary — collapsible, default open on first render, does not re-run optimizer
    summary_key = f"_opt_summary_{method}_{holdings_hash or '_'.join(result.tickers)}"
    with st.expander("Written explanation", expanded=(summary_key not in st.session_state)):
        if summary_key not in st.session_state:
            from core.optimizer import generate_optimizer_summary, _build_fallback_summary
            from ui.components import streaming_container
            container = st.empty()
            try:
                with st.spinner("Generating explanation (Sonnet)…"):
                    summary_iter = generate_optimizer_summary(result, method)
                    st.session_state[summary_key] = streaming_container(summary_iter, container)
            except Exception as exc:
                _log.exception("Optimizer summary LLM call failed for method=%s", method)
                fallback = _build_fallback_summary(result, method)
                st.session_state[summary_key] = fallback
                container.markdown(fallback)
                st.caption(
                    f"⚠ LLM explanation unavailable ({type(exc).__name__}). "
                    "Showing template-based summary. Click 'Regenerate' to retry."
                )
        else:
            st.markdown(st.session_state[summary_key])
        if st.button("Regenerate explanation", key=f"portfolio_analysis__regen_summary__{method}"):
            del st.session_state[summary_key]
            st.rerun()


def _render_montecarlo(weights: list[float], tickers: list[str], holdings_hash: str = "") -> None:
    from core.montecarlo import run_montecarlo

    mc_key = f"_mc_{holdings_hash}"
    if mc_key not in st.session_state:
        try:
            st.session_state[mc_key] = run_montecarlo(weights=weights, tickers=tickers)
        except Exception as exc:
            st.error(f"Monte Carlo error: {exc}")
            return

    mc = st.session_state[mc_key]

    days = list(range(mc.horizon_days + 1))

    fig = go.Figure()

    for path in mc.paths_sample:
        fig.add_trace(go.Scatter(
            x=days,
            y=path,
            mode="lines",
            line=dict(color="rgba(150,150,150,0.15)", width=1),
            showlegend=False,
            hoverinfo="skip",
        ))

    percentile_colors = {
        10: "rgb(220,50,50)",
        25: "rgb(255,140,0)",
        50: "rgb(60,180,80)",
        75: "rgb(30,144,255)",
        90: "rgb(148,0,211)",
    }

    for p_level in [10, 25, 50, 75, 90]:
        val = mc.percentiles[p_level]
        color = percentile_colors[p_level]
        fig.add_trace(go.Scatter(
            x=[0, mc.horizon_days],
            y=[val, val],
            mode="lines",
            name=f"P{p_level}",
            line=dict(color=color, width=2, dash="dot"),
        ))

    fig.update_layout(
        title="Monte Carlo — 1-Year Portfolio Paths (10,000 simulations)",
        xaxis_title="Trading Day",
        yaxis_title="Portfolio Value",
        height=450,
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=4, r=4, t=48, b=4),
    )
    st.plotly_chart(fig, use_container_width=True)

    p = mc.percentiles
    st.caption(
        f"Terminal values at 1 year — "
        f"P10: {p[10]:.2f}x  ·  "
        f"P50: {p[50]:.2f}x  ·  "
        f"P90: {p[90]:.2f}x"
    )


def render_portfolio_analysis_page() -> None:
    from core.portfolio import enrich_with_prices, Portfolio, Holding
    from data.portfolio_store import get_holdings, get_portfolio_id

    st.title("Portfolio Optimizer & Monte Carlo")
    _back_button(key="portfolio_analysis__back_top")

    portfolio_name = st.query_params.get("portfolio", "")
    tickers_param = st.query_params.get("tickers", "")
    tickers = [t.strip() for t in tickers_param.split(",") if t.strip()]

    if not portfolio_name or not tickers:
        st.warning("Missing portfolio name or tickers. Go back and re-run analysis.")
        return

    portfolio_id = get_portfolio_id(portfolio_name)
    if portfolio_id is None:
        st.error(f"Portfolio '{portfolio_name}' not found in database.")
        return

    with st.spinner("Loading portfolio and fetching prices..."):
        raw = get_holdings(portfolio_id)
        holdings = [Holding(
            ticker=h["ticker"],
            shares=h["shares"],
            cost_basis=h["cost_basis"],
            account=h["account"],
            notes=h["notes"],
            is_cash=bool(h["is_cash"]),
        ) for h in raw]
        portfolio = Portfolio(name=portfolio_name, holdings=holdings)
        from core.portfolio import EnrichmentResult
        enrichment: EnrichmentResult = enrich_with_prices(portfolio)
        portfolio = enrichment.portfolio
        failed = enrichment.failed

    # Surface pricing failures and gate optimization
    equity_holdings_all = [h for h in portfolio.holdings if not h.is_cash and h.ticker in tickers]
    failed_eq = [t for t in failed if t in tickers]
    priced_eq = [h for h in equity_holdings_all if h.ticker not in failed]

    if failed_eq:
        from data.cache import delete_cache, get_cache_obj

        # Distinguish tickers that have never successfully been cached from transient failures
        never_cached = {t for t in failed_eq if get_cache_obj(f"market:{t}:fundamentals") is None}

        st.error(
            f"**Price fetch failed for {len(failed_eq)} ticker(s).** "
            "Optimizer weights from incomplete data would be misleading."
        )
        for t in failed_eq:
            h_match = next((h for h in equity_holdings_all if h.ticker == t), None)
            pct_str = f"{(h_match.weight or 0) * 100:.1f}% of portfolio" if h_match else "unknown weight"
            if not _VALID_TICKER_RE.match(t):
                reason = "⚠ likely invalid — ticker symbols are 1–5 uppercase letters"
            elif t in never_cached:
                reason = "never successfully fetched — may be a new or unrecognized ticker"
            else:
                reason = "previously fetched; transient failure — retry likely to succeed"
            st.markdown(f"- **{t}** ({pct_str}): {reason}")

        col_retry, _ = st.columns([2, 3])
        with col_retry:
            if st.button("↺ Retry failed tickers", key="portfolio_analysis__retry_failed"):
                for t in failed_eq:
                    delete_cache(f"market:{t}:fundamentals")
                st.rerun()

        if not st.checkbox(
            f"Proceed without {', '.join(failed_eq)} — optimize over remaining {len(priced_eq)} positions",
            key="portfolio_analysis__proceed_with_missing",
        ):
            st.stop()

    eq_tickers = [h.ticker for h in priced_eq]
    tv = sum(h.market_value or 0 for h in priced_eq)
    current_weights = [(h.market_value or 0) / tv if tv > 0 else 1.0 / len(priced_eq)
                       for h in priced_eq]
    sector_map: dict[str, str] = {h.ticker: (h.sector or "Unknown") for h in priced_eq}

    if len(eq_tickers) < 2:
        st.warning("Need at least 2 priced equity positions to run optimizer.")
        return

    # Constraint controls — changing a slider busts the session-state cache via the hash
    with st.expander("Optimizer constraints", expanded=False):
        n_eq = len(eq_tickers)
        min_feasible_pct = max(5, int(100 / n_eq))  # smallest cap that keeps sum-to-1 feasible
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            max_pos = st.slider(
                "Max position weight",
                min_value=min_feasible_pct,
                max_value=100,
                value=40,
                step=5,
                format="%d%%",
                key="opt_max_pos",
                help=f"Per-stock upper bound. Min feasible with {n_eq} stocks is {min_feasible_pct}%.",
            ) / 100.0
        with cc2:
            max_sec = st.slider(
                "Max sector weight",
                min_value=min_feasible_pct,
                max_value=100,
                value=60,
                step=5,
                format="%d%%",
                key="opt_max_sec",
                help="Upper bound on total weight allocated to any single GICS sector.",
            ) / 100.0
        with cc3:
            turnover = st.slider(
                "Turnover penalty",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.05,
                format="%.2f",
                key="opt_turnover",
                help="L1 penalty on |proposed − current| added to the objective. 0 = unconstrained turnover.",
            )

    holdings_hash = _holdings_hash(eq_tickers, current_weights, max_pos, max_sec, turnover)

    st.subheader("Optimizer")
    opt_tabs = st.tabs(["Max Sharpe", "Min Vol", "Risk Parity"])

    _methods = ["max_sharpe", "min_vol", "risk_parity"]
    _needs_opt = any(f"_opt_{m}_{holdings_hash}" not in st.session_state for m in _methods)
    _opt_ctx = st.spinner("Running optimizers...") if _needs_opt else contextlib.nullcontext()

    with _opt_ctx:
        with opt_tabs[0]:
            _render_optimizer_tab(
                "max_sharpe", eq_tickers, current_weights, holdings_hash,
                sector_map=sector_map, max_pos=max_pos, max_sec=max_sec, turnover=turnover,
                portfolio_value=tv, priced_holdings=priced_eq,
            )
        with opt_tabs[1]:
            _render_optimizer_tab(
                "min_vol", eq_tickers, current_weights, holdings_hash,
                sector_map=sector_map, max_pos=max_pos, max_sec=max_sec, turnover=turnover,
                portfolio_value=tv, priced_holdings=priced_eq,
            )
        with opt_tabs[2]:
            _render_optimizer_tab(
                "risk_parity", eq_tickers, current_weights, holdings_hash,
                sector_map=sector_map, max_pos=max_pos, max_sec=max_sec, turnover=turnover,
                portfolio_value=tv, priced_holdings=priced_eq,
            )

    st.divider()
    st.subheader("Monte Carlo Simulation")

    _needs_mc = f"_mc_{holdings_hash}" not in st.session_state
    _mc_ctx = st.spinner("Running Monte Carlo (10,000 paths)...") if _needs_mc else contextlib.nullcontext()

    with _mc_ctx:
        _render_montecarlo(current_weights, eq_tickers, holdings_hash)

    st.divider()
    st.subheader("Per-Stock Scenario Cards")

    # Collect optimizer results for portfolio aggregation (use max_sharpe tab)
    opt_result = st.session_state.get(f"_opt_max_sharpe_{holdings_hash}")
    proposed_map = (
        dict(zip(opt_result.tickers, opt_result.proposed_weights)) if opt_result else {}
    )
    current_map_global = dict(zip(eq_tickers, current_weights))

    _render_scenario_cards_section(eq_tickers, current_map_global, proposed_map)

    st.divider()
    _back_button(key="portfolio_analysis__back_bottom")
    st.caption("Research tooling only — not investment advice.")


def _scenario_card(scenario: Scenario, current_price: float | None) -> None:
    CASE_ICON = {"bull": "🟢", "base": "🔵", "bear": "🔴"}
    icon = CASE_ICON.get(scenario.scenario, "⚪")
    st.markdown(f"**{icon} {scenario.scenario.title()}** · P = {scenario.probability * 100:.0f}%")
    if scenario.price_target is not None and current_price:
        st.metric(
            "Price Target",
            f"${scenario.price_target:.0f}",
            delta=f"{(scenario.price_target/current_price - 1)*100:+.1f}%",
        )
    elif scenario.price_target is not None:
        st.metric("Price Target", f"${scenario.price_target:.0f}")
    if scenario.implied_return is not None:
        sign = "+" if scenario.implied_return >= 0 else ""
        st.metric("Implied Return", f"{sign}{scenario.implied_return * 100:.1f}%")
    if scenario.narrative:
        render_md(scenario.narrative)
    if scenario.drivers:
        for d in scenario.drivers:
            render_md(f"- **{d.name}**")


def _render_scenario_cards_section(
    tickers: list[str],
    current_weights: dict[str, float],
    proposed_weights: dict[str, float],
) -> None:
    """Auto-load scenarios per holding and show portfolio-level aggregation."""
    from core.analyze import analyze_ticker

    # Load each ticker — auto-load with spinner (no intermediate button)
    for ticker in tickers:
        cache_key = f"_scenario_{ticker}"
        with st.expander(f"{ticker}", expanded=True):
            if cache_key not in st.session_state:
                with st.spinner(f"Loading scenarios for {ticker}…"):
                    try:
                        st.session_state[cache_key] = analyze_ticker(ticker)
                    except Exception as exc:
                        st.session_state[cache_key] = exc

            result = st.session_state.get(cache_key)
            if result is None:
                st.caption("Not yet loaded.")
                continue
            if isinstance(result, Exception):
                st.error(f"Analysis failed: {result}")
                col_retry, _ = st.columns([1, 4])
                with col_retry:
                    if st.button("Retry", key=f"portfolio_analysis__scenario_retry__{ticker}"):
                        del st.session_state[cache_key]
                        st.rerun()
                continue

            analysis = result
            label = analysis.company_name or ticker
            price_str = f"${analysis.current_price:.2f}" if analysis.current_price else "—"
            st.caption(f"{label} · Current: {price_str}")

            if not analysis.scenarios:
                st.info("No scenario data available for this ticker.")
            else:
                ordered = sorted(
                    analysis.scenarios,
                    key=lambda s: ["bull", "base", "bear"].index(s.scenario)
                    if s.scenario in ["bull", "base", "bear"] else 99,
                )
                cols = st.columns(len(ordered))
                for col, scenario in zip(cols, ordered):
                    with col:
                        _scenario_card(scenario, analysis.current_price)

            portfolio_name = st.query_params.get("portfolio", "")
            drill_url = f"?view=ticker&symbol={ticker}&from=portfolio"
            if portfolio_name:
                drill_url += f"&portfolio={portfolio_name}"
            st.link_button(
                f"Drill into {ticker} →",
                url=drill_url,
                help="Opens in this tab. Cmd-click (Mac) or Ctrl-click (Win) to open in a new tab.",
            )

    # Portfolio-level aggregation
    _render_scenario_aggregation(tickers, current_weights, proposed_weights)


def _render_scenario_aggregation(
    tickers: list[str],
    current_weights: dict[str, float],
    proposed_weights: dict[str, float],
) -> None:
    """Weighted bull/base/bear portfolio return under current and proposed weights."""
    # Collect per-ticker implied returns for each scenario label
    scenario_returns: dict[str, dict[str, float]] = {}  # ticker → {label: implied_return}
    for ticker in tickers:
        analysis = st.session_state.get(f"_scenario_{ticker}")
        if not analysis or isinstance(analysis, Exception) or not analysis.scenarios:
            continue
        for s in analysis.scenarios:
            if s.implied_return is not None:
                scenario_returns.setdefault(ticker, {})[s.scenario] = s.implied_return

    if not scenario_returns:
        return

    # Only aggregate over tickers that have all three cases
    complete_tickers = [t for t in tickers if set(scenario_returns.get(t, {}).keys()) >= {"bull", "base", "bear"}]
    if not complete_tickers:
        return

    st.divider()
    st.subheader("Portfolio Scenario Aggregation")
    st.caption(
        "Weighted return = Σ (position weight × holding's implied return) under each scenario. "
        "Only includes holdings with all three scenarios loaded."
    )

    agg_rows = []
    for label in ["bull", "base", "bear"]:
        cur_ret = sum(
            current_weights.get(t, 0) * scenario_returns[t][label]
            for t in complete_tickers
            if label in scenario_returns.get(t, {})
        )
        prop_ret = sum(
            proposed_weights.get(t, 0) * scenario_returns[t][label]
            for t in complete_tickers
            if label in scenario_returns.get(t, {})
        )
        agg_rows.append({
            "Scenario": label.title(),
            "Weighted Return (current weights)": f"{cur_ret*100:+.1f}%",
            "Weighted Return (proposed weights)": f"{prop_ret*100:+.1f}%",
            "Δ": f"{(prop_ret-cur_ret)*100:+.1f}pp",
        })
    st.dataframe(pd.DataFrame(agg_rows), hide_index=True, use_container_width=True)
    st.caption(
        f"Aggregated over: {', '.join(complete_tickers)}. "
        "Missing tickers excluded — load them above to include in aggregation."
    )
