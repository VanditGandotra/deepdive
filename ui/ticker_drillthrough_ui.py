from __future__ import annotations

import streamlit as st

from ui.portfolio_analysis_ui import _scenario_card


def render_ticker_drillthrough_page() -> None:
    def _back() -> None:
        for k in ["view", "symbol", "portfolio", "tickers"]:
            st.query_params.pop(k, None)
        st.rerun()

    symbol = st.query_params.get("symbol", "").upper().strip()
    if not symbol:
        st.warning("No symbol specified.")
        if st.button("← Back"):
            _back()
        return

    from core.analyze import analyze_ticker

    st.title(f"Deep Dive: {symbol}")

    if st.button("← Back"):
        _back()

    cache_key = f"_drillthrough_{symbol}"
    if cache_key not in st.session_state:
        with st.spinner(f"Analyzing {symbol}…"):
            try:
                st.session_state[cache_key] = analyze_ticker(symbol)
            except Exception as exc:
                st.session_state[cache_key] = exc

    result = st.session_state[cache_key]

    if isinstance(result, Exception):
        st.error(f"Analysis failed for {symbol}: {result}")
        if st.button("← Back", key="back_err"):
            _back()
        return

    analysis = result

    company_label = analysis.company_name or symbol
    price_str = f"${analysis.current_price:,.2f}" if analysis.current_price else "—"
    st.markdown(f"## {company_label} ({symbol}) · {price_str}")
    if analysis.sector:
        st.caption(f"{analysis.sector}" + (f" · {analysis.industry}" if analysis.industry else ""))

    st.divider()

    if not analysis.scenarios:
        st.info("No scenario data available.")
    else:
        ordered = sorted(
            analysis.scenarios,
            key=lambda s: ["bull", "base", "bear"].index(s.scenario)
            if s.scenario in ["bull", "base", "bear"] else 99,
        )
        for scenario in ordered:
            with st.container(border=True):
                _scenario_card(scenario, analysis.current_price)

    st.divider()

    m1, m2, m3 = st.columns(3)
    with m1:
        if analysis.expected_return_1y is not None:
            sign = "+" if analysis.expected_return_1y >= 0 else ""
            st.metric("Expected Return (1Y)", f"{sign}{analysis.expected_return_1y * 100:.1f}%")
        else:
            st.metric("Expected Return (1Y)", "—")
    with m2:
        if analysis.expected_return_3y is not None:
            sign = "+" if analysis.expected_return_3y >= 0 else ""
            st.metric("Expected Return (3Y ann.)", f"{sign}{analysis.expected_return_3y * 100:.1f}%")
        else:
            st.metric("Expected Return (3Y ann.)", "—")
    with m3:
        if analysis.expected_return_5y is not None:
            sign = "+" if analysis.expected_return_5y >= 0 else ""
            st.metric("Expected Return (5Y ann.)", f"{sign}{analysis.expected_return_5y * 100:.1f}%")
        else:
            st.metric("Expected Return (5Y ann.)", "—")

    if analysis.data_gaps:
        with st.expander("Data gaps"):
            for gap in analysis.data_gaps:
                st.caption(f"- {gap}")

    st.divider()
    if st.button("← Back", key="back_bottom"):
        _back()

    st.caption("Research tooling only — not investment advice.")
