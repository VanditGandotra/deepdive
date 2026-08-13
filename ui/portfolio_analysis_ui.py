"""Portfolio analysis page: optimizer + Monte Carlo, triggered by ?view=portfolio_analysis."""
from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from core.schemas import Scenario


def _back_button(key: str = "portfolio_analysis__back") -> None:
    if st.button("<- Back to portfolio", key=key):
        for k in ["view", "portfolio", "tickers"]:
            st.query_params.pop(k, None)
        st.rerun()


def _render_optimizer_tab(method: str, tickers: list[str], current_weights: list[float]) -> None:
    from core.optimizer import optimize_portfolio

    try:
        result = optimize_portfolio(
            tickers=tickers,
            current_weights=current_weights,
            method=method,
        )
    except Exception as exc:
        st.error(f"Optimizer error: {exc}")
        return

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
            yaxis_title="Weight (%)",
            height=320,
            legend=dict(orientation="h", y=1.1),
            margin=dict(l=4, r=4, t=32, b=4),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_metrics:
        st.metric("Expected Return", f"{result.expected_return * 100:.1f}%")
        st.metric("Volatility", f"{result.expected_vol * 100:.1f}%")
        st.metric("Sharpe", f"{result.sharpe:.2f}")

    st.subheader("Sensitivity (±2% weight bump)")
    sens_rows = []
    proposed_map = dict(zip(result.tickers, result.proposed_weights))
    current_map = dict(zip(result.tickers, result.current_weights))
    for t, (w_delta, sharpe_delta) in result.sensitivity.items():
        sens_rows.append({
            "Ticker": t,
            "Current Weight": f"{current_map.get(t, 0) * 100:.1f}%",
            "Proposed Weight": f"{proposed_map.get(t, 0) * 100:.1f}%",
            "Sharpe Impact (+2%)": f"{sharpe_delta:+.3f}",
        })
    if sens_rows:
        st.dataframe(pd.DataFrame(sens_rows), hide_index=True, use_container_width=True)


def _render_montecarlo(weights: list[float], tickers: list[str]) -> None:
    from core.montecarlo import run_montecarlo

    try:
        mc = run_montecarlo(weights=weights, tickers=tickers)
    except Exception as exc:
        st.error(f"Monte Carlo error: {exc}")
        return

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
        failed_weight = sum(h.weight or 0 for h in equity_holdings_all if h.ticker in failed_eq)
        st.error(
            f"**Price fetch failed for: {', '.join(failed_eq)}** "
            f"(combined portfolio weight: {failed_weight * 100:.1f}%). "
            "Optimizer weights from incomplete data would be misleading."
        )
        if not st.checkbox(
            f"Proceed anyway — exclude {', '.join(failed_eq)} and optimize over remaining positions",
            key="portfolio_analysis__proceed_with_missing",
        ):
            st.stop()

    eq_tickers = [h.ticker for h in priced_eq]
    tv = sum(h.market_value or 0 for h in priced_eq)
    current_weights = [(h.market_value or 0) / tv if tv > 0 else 1.0 / len(priced_eq)
                       for h in priced_eq]

    if len(eq_tickers) < 2:
        st.warning("Need at least 2 priced equity positions to run optimizer.")
        return

    st.subheader("Optimizer")
    opt_tabs = st.tabs(["Max Sharpe", "Min Vol", "Risk Parity"])

    with st.spinner("Running optimizers..."):
        with opt_tabs[0]:
            _render_optimizer_tab("max_sharpe", eq_tickers, current_weights)
        with opt_tabs[1]:
            _render_optimizer_tab("min_vol", eq_tickers, current_weights)
        with opt_tabs[2]:
            _render_optimizer_tab("risk_parity", eq_tickers, current_weights)

    st.divider()
    st.subheader("Monte Carlo Simulation")

    with st.spinner("Running Monte Carlo (10,000 paths)..."):
        _render_montecarlo(current_weights, eq_tickers)

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
