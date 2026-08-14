"""Portfolio analysis page: optimizer + Monte Carlo, triggered by ?view=portfolio_analysis."""
from __future__ import annotations

import contextlib
import hashlib
import re

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from core.schemas import Scenario

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


def _render_optimizer_tab(
    method: str,
    tickers: list[str],
    current_weights: list[float],
    holdings_hash: str = "",
    sector_map: dict[str, str] | None = None,
    max_pos: float = 0.40,
    max_sec: float = 0.60,
    turnover: float = 0.0,
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

    # Constraint status — always visible so the reader knows what bounds were active
    with st.expander("Constraints & inputs", expanded=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("Position cap", f"{result.max_position_weight * 100:.0f}%")
        c2.metric("Sector cap", f"{result.max_sector_weight * 100:.0f}%")
        c3.metric("Turnover penalty", f"{result.turnover_penalty:.2f}")
        if result.binding_constraints:
            st.warning("**Binding:** " + " · ".join(result.binding_constraints))
        else:
            st.success("No constraints binding — weights are unconstrained optima.")
        st.caption(
            f"Lookback: {result.lookback_days}d · Covariance: Ledoit-Wolf shrinkage · "
            f"n={len(result.tickers)} assets (small matrix — treat weights as estimates, not precise allocations)"
        )

    st.subheader("Sensitivity — how weights move when expected returns are shocked ±2%")
    st.caption(
        "At a portfolio optimum, first-order weight sensitivity is ≈ 0 by construction, "
        "so bumping weights tells you nothing. This table shocks each holding's expected "
        "return estimate ±2pp and shows how the optimizer re-allocates."
    )
    # Use structured sensitivity rows when available (new API), fall back to legacy dict
    sens_rows = []
    if result.sensitivity:
        from core.optimizer import SensitivityRow
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
            from core.optimizer import generate_optimizer_summary
            from ui.components import streaming_container
            container = st.empty()
            with st.spinner("Generating explanation (Sonnet)…"):
                summary_iter = generate_optimizer_summary(result, method)
                st.session_state[summary_key] = streaming_container(summary_iter, container)
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
            )
        with opt_tabs[1]:
            _render_optimizer_tab(
                "min_vol", eq_tickers, current_weights, holdings_hash,
                sector_map=sector_map, max_pos=max_pos, max_sec=max_sec, turnover=turnover,
            )
        with opt_tabs[2]:
            _render_optimizer_tab(
                "risk_parity", eq_tickers, current_weights, holdings_hash,
                sector_map=sector_map, max_pos=max_pos, max_sec=max_sec, turnover=turnover,
            )

    st.divider()
    st.subheader("Monte Carlo Simulation")

    _needs_mc = f"_mc_{holdings_hash}" not in st.session_state
    _mc_ctx = st.spinner("Running Monte Carlo (10,000 paths)...") if _needs_mc else contextlib.nullcontext()

    with _mc_ctx:
        _render_montecarlo(current_weights, eq_tickers, holdings_hash)

    st.divider()
    st.subheader("Per-Stock Scenario Cards")
    _render_scenario_cards_section(eq_tickers)

    st.divider()
    _back_button(key="portfolio_analysis__back_bottom")
    st.caption("Research tooling only — not investment advice.")


def _scenario_card(scenario: Scenario, current_price: float | None) -> None:
    name_lower = scenario.scenario.lower()
    if "bull" in name_lower:
        color = "🟢"
    elif "bear" in name_lower:
        color = "🔴"
    else:
        color = "🔵"
    st.markdown(f"**{color} {scenario.scenario.title()}**")
    st.caption(f"P = {scenario.probability * 100:.0f}%")
    if scenario.price_target is not None:
        st.metric("Price Target", f"${scenario.price_target:.0f}")
    if scenario.implied_return is not None:
        sign = "+" if scenario.implied_return >= 0 else ""
        st.metric("Implied Return", f"{sign}{scenario.implied_return * 100:.1f}%")
    if scenario.narrative:
        st.markdown(scenario.narrative[:300])
    if scenario.drivers:
        for d in scenario.drivers[:3]:
            st.markdown(f"- {d.name}")


def _render_scenario_cards_section(tickers: list[str]) -> None:
    from core.analyze import analyze_ticker

    for ticker in tickers:
        with st.expander(f"{ticker} — click to expand scenarios"):
            cache_key = f"_scenario_{ticker}"
            if cache_key not in st.session_state:
                if st.button(f"Load scenarios for {ticker}", key=f"portfolio_analysis__scenario_load__{ticker}"):
                    with st.spinner(f"Analyzing {ticker}…"):
                        try:
                            st.session_state[cache_key] = analyze_ticker(ticker)
                        except Exception as exc:
                            st.session_state[cache_key] = exc
                    st.rerun()
            else:
                result = st.session_state[cache_key]
                if isinstance(result, Exception):
                    st.error(f"Analysis failed: {result}")
                    if st.button(f"Retry {ticker}", key=f"portfolio_analysis__scenario_retry__{ticker}"):
                        del st.session_state[cache_key]
                        st.rerun()
                else:
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
