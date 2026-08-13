"""Batch multi-ticker analysis UI."""
from __future__ import annotations
import re
import threading
from typing import Dict, List, Optional

import streamlit as st

from core.batch import BatchRunner, BatchStatus, TickerState
from core.schemas import TickerAnalysis

# ── Session state keys ────────────────────────────────────────────────────────
_RUNNER_KEY = "_batch_runner"
_STARTED_KEY = "_batch_started"

_VALID_TICKER_RE = re.compile(r"^[A-Z]{1,5}$")


def parse_ticker_input(raw: str) -> List[str]:
    """Split raw input on commas/spaces/newlines/tabs, strip $, uppercase, filter, deduplicate."""
    tokens = re.split(r"[,\s\t\n]+", raw)
    seen: dict[str, bool] = {}
    result: List[str] = []
    for token in tokens:
        t = token.strip().lstrip("$").upper()
        if not t:
            continue
        if _VALID_TICKER_RE.match(t) and t not in seen:
            seen[t] = True
            result.append(t)
    return result


@st.fragment(run_every=2)
def _status_strip():
    runner: Optional[BatchRunner] = st.session_state.get(_RUNNER_KEY)
    if not runner:
        return
    states = runner.states
    done, total = runner.progress
    st.progress(done / max(total, 1), text=f"{done}/{total} complete")
    cols = st.columns(min(len(states), 8))
    status_colors = {
        BatchStatus.QUEUED: "gray",
        BatchStatus.FETCHING: "blue",
        BatchStatus.ANALYZING: "orange",
        BatchStatus.DONE: "green",
        BatchStatus.FAILED: "red",
        BatchStatus.CANCELLED: "gray",
    }
    for i, (ticker, state) in enumerate(sorted(states.items())):
        color = status_colors.get(state.status, "gray")
        elapsed = f" {state.elapsed:.0f}s" if state.elapsed else ""
        with cols[i % len(cols)]:
            st.markdown(f":{color}[**{ticker}**]  \n`{state.status.value}{elapsed}`")


def _stop_button():
    runner = st.session_state.get(_RUNNER_KEY)
    if runner and not runner.is_done:
        if st.button("Stop", type="secondary"):
            runner.cancel()
            st.rerun()


def _generate_synthesis(results: Dict[str, TickerAnalysis]) -> str:
    """Single Sonnet call summarizing relationships across tickers. No re-derivation of numbers."""
    import llm
    from config import SONNET

    context_parts = []
    for ticker, r in results.items():
        er1y = f"{r.expected_return_1y*100:.1f}%" if r.expected_return_1y is not None else "N/A"
        scenarios_summary = "; ".join(
            f"{s.scenario} p={s.probability:.0%} target={'$'+str(round(s.price_target,2)) if s.price_target else 'N/A'}"
            for s in r.scenarios
        ) if r.scenarios else "no scenarios"
        context_parts.append(
            f"{ticker} ({r.company_name or ticker}) | sector={r.sector} | "
            f"P/E={r.pe_ttm} | exp_return_1y={er1y} | "
            f"scenarios: {scenarios_summary}"
        )
    context = "\n".join(context_parts)

    messages = [{"role": "user", "content": [
        llm.text_block(
            f"You are analyzing a basket of {len(results)} stocks for a portfolio manager.\n\n"
            f"Stock data:\n{context}\n\n"
            "In 3-4 paragraphs: (1) Where do the theses overlap or conflict? "
            "(2) What shared factor exposure does this group have? "
            "(3) Which stock offers the best risk-adjusted expected return and why? "
            "Do not re-derive any numbers — use only the data provided."
        )
    ]}]
    try:
        return llm.call(SONNET, messages, max_tokens=800, prompt_version="batch_synthesis_v1")
    except Exception as exc:
        return f"Synthesis unavailable: {exc}"


def _render_compare_tab(runner: BatchRunner, done_tickers: List[str]):
    import pandas as pd

    if not done_tickers:
        st.info("No completed analyses yet.")
        return

    states = runner.states
    results = {t: states[t].result for t in done_tickers if states[t].result}

    # Normalized metrics table
    rows = []
    for ticker, r in results.items():
        rows.append({
            "Ticker": ticker,
            "Name": (r.company_name or "")[:20],
            "Price": f"${r.current_price:,.2f}" if r.current_price else "—",
            "Mkt Cap": f"${r.market_cap/1e9:.1f}B" if r.market_cap else "—",
            "P/E": f"{r.pe_ttm:.1f}x" if r.pe_ttm else "—",
            "Fwd P/E": f"{r.pe_forward:.1f}x" if r.pe_forward else "—",
            "EV/EBITDA": f"{r.ev_ebitda:.1f}x" if r.ev_ebitda else "—",
            "Net Margin": f"{r.net_margin*100:.1f}%" if r.net_margin else "—",
            "Rev Growth": f"{r.revenue_growth_yoy*100:.1f}%" if r.revenue_growth_yoy else "—",
            "FCF Yield": f"{r.fcf_yield*100:.1f}%" if r.fcf_yield else "—",
            "Exp Ret 1y": f"{r.expected_return_1y*100:.1f}%" if r.expected_return_1y is not None else "—",
            "Confidence": f"{r.confidence:.0%}",
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, use_container_width=True)

    # Export button
    csv = df.to_csv(index=False)
    st.download_button("Export CSV", csv, file_name="batch_compare.csv", mime="text/csv")

    # Expected return vs downside scatter
    import plotly.graph_objects as go

    st.subheader("Expected Return vs Downside (Bear Case)")
    scatter_data = []
    for ticker, r in results.items():
        er = r.expected_return_1y
        bear = next((s for s in r.scenarios if s.scenario == "bear"), None)
        downside = bear.implied_return if bear and bear.implied_return is not None else None
        if er is not None and downside is not None:
            scatter_data.append({"ticker": ticker, "er": er * 100, "downside": downside * 100})

    if scatter_data:
        fig = go.Figure()
        for d in scatter_data:
            fig.add_trace(go.Scatter(
                x=[d["downside"]], y=[d["er"]],
                mode="markers+text",
                text=[d["ticker"]], textposition="top center",
                marker=dict(size=14),
                name=d["ticker"],
            ))
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(
            xaxis_title="Bear Case Downside (%)",
            yaxis_title="Expected Return 1y (%)",
            height=400, showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Correlation matrix (daily returns, configurable lookback)
    st.subheader("Return Correlation Matrix")
    if len(done_tickers) >= 2:
        lookback = st.selectbox("Lookback", ["1y", "3y", "5y"], index=1, key="corr_lookback")
        try:
            from data.market import get_prices
            import numpy as np
            price_series = {}
            for ticker in done_tickers:
                pd_data = get_prices(ticker, period=lookback)
                if pd_data and pd_data.bars:
                    closes = pd.Series(
                        [b.close for b in pd_data.bars],
                        index=[b.date for b in pd_data.bars],
                    )
                    price_series[ticker] = closes
            if len(price_series) >= 2:
                prices_df = pd.DataFrame(price_series).dropna()
                returns_df = prices_df.pct_change().dropna()
                corr = returns_df.corr()
                import plotly.express as px
                fig_corr = px.imshow(
                    corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, title="Return Correlation"
                )
                fig_corr.update_layout(height=350)
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.caption("Need price data for at least 2 tickers.")
        except Exception as exc:
            st.caption(f"Correlation unavailable: {exc}")
    else:
        st.caption("Add 2+ tickers for correlation matrix.")

    # Cross-ticker LLM synthesis (runs AFTER all tickers done, cached in session_state)
    synthesis_key = f"_batch_synthesis_{'_'.join(sorted(done_tickers))}"
    if runner.is_done and done_tickers:
        if synthesis_key not in st.session_state:
            if st.button("Generate Cross-Ticker Synthesis", key="synthesis_btn"):
                with st.spinner("Synthesizing (Sonnet)…"):
                    st.session_state[synthesis_key] = _generate_synthesis(results)
        if synthesis_key in st.session_state:
            st.markdown(st.session_state[synthesis_key])


def _render_ticker_tab(ticker: str, result: TickerAnalysis):
    st.markdown(f"### {result.company_name or ticker} ({ticker})")
    col1, col2, col3 = st.columns(3)
    col1.metric("Price", f"${result.current_price:,.2f}" if result.current_price else "—")
    col2.metric("Sector", result.sector or "—")
    col3.metric("Confidence", f"{result.confidence:.0%}")

    if result.scenarios:
        st.markdown("**Scenarios**")
        for s in result.scenarios:
            pt = f"${s.price_target:.2f}" if s.price_target else "N/A"
            ir = f"{s.implied_return*100:.1f}%" if s.implied_return is not None else "N/A"
            st.markdown(f"- **{s.scenario.title()}** (p={s.probability:.0%}): target={pt} implied={ir}")

    if result.data_gaps:
        with st.expander("Data gaps"):
            for g in result.data_gaps:
                st.caption(f"• {g}")

    # Deep-link to full single-ticker analysis
    if st.button(f"Full analysis → {ticker}", key=f"drilldown_{ticker}"):
        # Navigate to single-ticker mode by clearing tickers param
        st.query_params.clear()
        st.query_params["view"] = "ticker"
        st.query_params["symbol"] = ticker
        st.rerun()


def _render_results(runner: BatchRunner):
    states = runner.states
    done_tickers = [t for t, s in sorted(states.items()) if s.status == BatchStatus.DONE and s.result]
    failed_tickers = [t for t, s in sorted(states.items()) if s.status == BatchStatus.FAILED]

    if not done_tickers and not failed_tickers:
        st.info("Analysis in progress…")
        return

    tab_labels = ["Compare"] + done_tickers + [f"X {t}" for t in failed_tickers]
    tabs = st.tabs(tab_labels)

    # Tab 0: Compare
    with tabs[0]:
        _render_compare_tab(runner, done_tickers)

    # Per-ticker tabs
    for i, ticker in enumerate(done_tickers):
        with tabs[i + 1]:
            _render_ticker_tab(ticker, states[ticker].result)

    # Failed tickers
    for i, ticker in enumerate(failed_tickers):
        with tabs[len(done_tickers) + 1 + i]:
            st.error(f"Analysis failed for **{ticker}**")
            st.code(states[ticker].error or "Unknown error")
            if st.button(f"Retry {ticker}", key=f"retry_{ticker}"):
                # Re-launch a single-ticker batch for this one
                single_runner = BatchRunner([ticker], config=runner.config)
                st.session_state[f"retry_runner_{ticker}"] = single_runner
                threading.Thread(target=single_runner.run, daemon=True).start()
                st.rerun()


def render_batch_page():
    st.title("Batch Analysis")

    # Parse tickers from query params
    raw_tickers_param = st.query_params.get("tickers", "")

    # Input section (always visible, even when results are showing)
    with st.expander("Configure batch", expanded=not st.session_state.get(_STARTED_KEY)):
        raw_input = st.text_area(
            "Tickers",
            value=raw_tickers_param.replace(",", "\n"),
            placeholder="MSFT\nNVDA\nAMZN",
            height=120,
            key="batch_ticker_input",
        )

        col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
        concurrency = col_cfg1.number_input("Concurrency", 1, 10, 3, key="batch_concurrency")
        discount_rate = col_cfg2.number_input("Discount rate", 0.05, 0.20, 0.10, 0.01, format="%.2f", key="batch_dr")
        terminal_growth = col_cfg3.number_input("Terminal growth", 0.01, 0.05, 0.025, 0.005, format="%.3f", key="batch_tg")

        valid_tickers = parse_ticker_input(raw_input)
        invalid_tokens = [t for t in raw_input.split() if t and t not in valid_tickers and len(t) <= 10]

        if valid_tickers:
            st.success(f"Valid tickers: {', '.join(valid_tickers)}")
        if invalid_tokens:
            st.warning(f"Ignored (invalid): {', '.join(set(invalid_tokens))}")

        runner = st.session_state.get(_RUNNER_KEY)
        already_running = runner is not None and not runner.is_done

        if st.button(
            "Analyze",
            type="primary",
            disabled=not valid_tickers or already_running,
            key="batch_launch_btn",
        ):
            config = {
                "discount_rate": discount_rate,
                "terminal_growth": terminal_growth,
                "horizon_years": 10,
            }
            new_runner = BatchRunner(valid_tickers, config=config, concurrency=int(concurrency))
            st.session_state[_RUNNER_KEY] = new_runner
            st.session_state[_STARTED_KEY] = True
            # Clear any previous synthesis cache
            for k in list(st.session_state.keys()):
                if k.startswith("_batch_synthesis_"):
                    del st.session_state[k]
            # Update query params
            st.query_params["tickers"] = ",".join(valid_tickers)
            threading.Thread(target=new_runner.run, daemon=True).start()
            st.rerun()

    # Stop button
    _stop_button()

    # Status strip
    _status_strip()

    # Results
    runner = st.session_state.get(_RUNNER_KEY)
    if runner:
        _render_results(runner)

    # Back to single-ticker link
    st.divider()
    if st.button("Single ticker mode"):
        st.query_params.clear()
        st.rerun()
