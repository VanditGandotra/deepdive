"""DeepDive — Streamlit entry point. Auto-detects ticker vs URL mode."""
from __future__ import annotations

import re
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterator, List, Optional

import streamlit as st

st.set_page_config(
    page_title="DeepDive Research",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Session init ──────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())
    import llm
    llm.set_session_id(st.session_state["session_id"])

_URL_PATTERN = re.compile(r"(https?://|www\.)|[a-zA-Z0-9-]+\.[a-zA-Z]{2,}")
_TICKER_PATTERN = re.compile(r"^[A-Z]{1,5}$")


def detect_mode(text: str) -> tuple[str, str]:
    stripped = text.strip()
    if _URL_PATTERN.search(stripped):
        url = stripped.lower()
        if not url.startswith("http"):
            url = "https://" + url
        return "url", url
    return "ticker", stripped.upper()


def validate_ticker(t: str) -> bool:
    return bool(_TICKER_PATTERN.match(t))



# ═══════════════════════════════════════════════════════════════════════════════
# TICKER MODE TABS
# ═══════════════════════════════════════════════════════════════════════════════

def tab_overview(ticker: str, settings: Dict) -> None:
    from analysis.expectations import reverse_dcf
    from analysis.news_impact import classify_headlines, high_materiality
    from analysis.quality import get_quality_panel
    from analysis.sentiment import aggregate_sentiment_scores
    from analysis.calls import analyse_all_calls
    from analysis.state_of_play import stream_state_of_play
    from data.cache import get_last_run_snapshot
    from data.market import get_fundamentals, get_short_interest
    from ui.charts import sentiment_mini_sparkline
    from ui.components import (
        analyst_consensus_bar, analyst_target_bar, delta_card, error_card,
        fmt_money, fmt_pct, fmt_ratio, metric_card, streaming_container, week_52_bar,
    )

    # ── 1. Delta card ─────────────────────────────────────────────────────────
    last_snap_raw = get_last_run_snapshot(ticker)
    if "delta_narrative" in st.session_state.get(f"computed_{ticker}", {}):
        delta_card(st.session_state[f"computed_{ticker}"]["delta_narrative"])
    else:
        delta_card(last_snap_raw)

    # ── 2. Load core data ─────────────────────────────────────────────────────
    try:
        fund = get_fundamentals(ticker)
        si = get_short_interest(ticker)
    except Exception as exc:
        error_card("Market data unavailable", str(exc))
        return

    # ── 3. Header strip ───────────────────────────────────────────────────────
    price = fund.current_price
    prev = fund.previous_close
    chg_pct = fund.day_change_pct
    chg_str = fmt_pct(chg_pct) if chg_pct is not None else "—"
    chg_cls = "dd-change-pos" if (chg_pct or 0) >= 0 else "dd-change-neg"
    price_str = f"${price:,.2f}" if price else "—"
    earnings_str = fund.next_earnings_date.strftime("%b %d, %Y") if fund.next_earnings_date else "—"
    mktcap_str = fmt_money(fund.market_cap)

    st.html(f"""
<div class="dd-header-strip">
  <div>
    <div class="dd-price">{price_str}</div>
    <span class="{chg_cls}">{chg_str} today</span>
  </div>
  <div>
    <div class="dd-header-label">Market cap</div>
    <div class="dd-header-val">{mktcap_str}</div>
  </div>
  <div>
    <div class="dd-header-label">Next earnings</div>
    <div class="dd-header-val">{earnings_str}</div>
  </div>
  <div>
    <div class="dd-header-label">Sector</div>
    <div class="dd-header-val">{fund.sector or '—'}</div>
  </div>
</div>
""")

    # 52-week range bar
    if price and fund.week_52_low and fund.week_52_high:
        week_52_bar(price, fund.week_52_low, fund.week_52_high)

    # ── 3b. Pre-earnings brief banner ────────────────────────────────────────
    try:
        from analysis.pre_earnings import build_pre_earnings_brief, days_to_earnings, should_show_brief
        if should_show_brief(fund):
            days_left = days_to_earnings(fund)
            date_str = fund.next_earnings_date.strftime("%b %d") if fund.next_earnings_date else ""
            with st.expander(
                f"Earnings in {days_left} day{'s' if days_left != 1 else ''} ({date_str}) — click for pre-earnings brief",
                expanded=(days_left is not None and days_left <= 3),
            ):
                brief_key = f"pre_earnings_{ticker}"
                if brief_key not in st.session_state:
                    from data.market import get_beat_miss_history, get_estimates
                    from analysis.kpis import extract_kpis
                    from analysis.calls import analyse_all_calls
                    with st.spinner("Generating pre-earnings brief…"):
                        bm = get_beat_miss_history(ticker)
                        ests = get_estimates(ticker)
                        kpi_sums: List[str] = []
                        try:
                            cd = analyse_all_calls(ticker, n=2)
                            kpis = extract_kpis(ticker, cd)
                            kpi_sums = [f"{k.kpi_name}: {k.trend_note}" for k in kpis]
                        except Exception:
                            pass
                        brief = build_pre_earnings_brief(ticker, fund, bm, ests, kpi_sums)
                    st.session_state[brief_key] = brief
                st.markdown(st.session_state.get(brief_key, "Brief unavailable."))
    except Exception:
        pass

    st.divider()

    # ── 4. State of play (streamed, cached by content hash) ───────────────────
    st.markdown("**State of play**")
    sop_key = f"sop_{ticker}"
    if sop_key not in st.session_state:
        # Gather light context — skip analyses that are slow; they cache anyway
        dcf_for_sop = None
        quality_for_sop = None
        sentiment_trend_for_sop: List[float] = []
        try:
            dcf_for_sop = reverse_dcf(ticker)
        except Exception:
            pass
        try:
            quality_for_sop = get_quality_panel(ticker)
        except Exception:
            pass
        try:
            call_data = analyse_all_calls(ticker, n=4)
            from analysis.sentiment import aggregate_sentiment_scores
            sentiment_trend_for_sop = aggregate_sentiment_scores(call_data.get("sentiments", []))
            sentiment_trend_for_sop = [s for s in sentiment_trend_for_sop if s is not None]
        except Exception:
            pass
        sop_container = st.empty()
        try:
            sop_iter = stream_state_of_play(ticker, fund, dcf_for_sop, sentiment_trend_for_sop, quality_for_sop)
            sop_text = streaming_container(sop_iter, sop_container)
            st.session_state[sop_key] = sop_text
        except Exception as exc:
            st.caption(f"State-of-play unavailable: {exc}")
    else:
        st.markdown(st.session_state[sop_key])

    st.divider()

    # ── 5. Metrics grid (3×3) ─────────────────────────────────────────────────
    # Load quality flags for grid (cached)
    quality = None
    try:
        quality = get_quality_panel(ticker)
    except Exception:
        pass

    # Load ratio history for P/E median
    pe_median = None
    ev_ebitda_median = None
    try:
        from analysis.ratios import get_ratio_groups, _fetch_annual_data, _year_end_prices, _valuation_group
        _data = _fetch_annual_data(ticker)
        _yp = _year_end_prices(ticker)
        _vg = _valuation_group(fund, _data, _yp)
        for r in _vg.ratios:
            if r.name == "P/E (TTM)":
                pe_median = r.median_5y
            elif r.name == "EV/EBITDA":
                ev_ebitda_median = r.median_5y
    except Exception:
        pass

    col1, col2, col3 = st.columns(3)

    with col1:
        pe_ctx = f"5yr median: {fmt_ratio(pe_median)}" if pe_median else ""
        metric_card("P/E (TTM)", fmt_ratio(fund.pe_ttm), pe_ctx)
        st.write("")
        ev_ctx = f"5yr median: {fmt_ratio(ev_ebitda_median)}" if ev_ebitda_median else ""
        metric_card("EV/EBITDA", fmt_ratio(fund.ev_ebitda), ev_ctx)
        st.write("")
        nm_val = fmt_pct(fund.net_margin)
        nm_ctx = f"operating: {fmt_pct(fund.operating_margin)}" if fund.operating_margin else ""
        metric_card("Net Margin", nm_val, nm_ctx)

    with col2:
        rev_cagr_ctx = f"YoY: {fmt_pct(fund.revenue_growth_yoy)}" if fund.revenue_growth_yoy else ""
        metric_card("Rev Growth YoY", fmt_pct(fund.revenue_growth_yoy), rev_cagr_ctx)
        st.write("")
        fcf_yield = None
        if fund.fcf_ttm and fund.market_cap and fund.market_cap > 0:
            fcf_yield = fund.fcf_ttm / fund.market_cap
        metric_card("FCF Yield", fmt_pct(fcf_yield), "FCF TTM / market cap")
        st.write("")
        si_pct = si.pct_float if si else None
        si_ctx = f"{si.days_to_cover:.1f}d to cover" if si and si.days_to_cover else ""
        metric_card("Short Interest", fmt_pct(si_pct), si_ctx)

    with col3:
        # Analyst consensus bar
        st.markdown('<div class="dd-metric-card"><div class="dd-metric-label">Analyst Consensus</div>', unsafe_allow_html=True)
        buys = fund.analyst_buy_count or 0
        holds = fund.analyst_hold_count or 0
        sells = fund.analyst_sell_count or 0
        if buys + holds + sells > 0:
            analyst_consensus_bar(buys, holds, sells)
        else:
            st.caption("No consensus data")
        st.html("</div>")

        st.write("")

        # Analyst target range
        if (fund.analyst_target_low and fund.analyst_target_high
                and fund.analyst_target_mean and price):
            st.markdown('<div class="dd-metric-card"><div class="dd-metric-label">Price Target Range</div>', unsafe_allow_html=True)
            analyst_target_bar(price, fund.analyst_target_low, fund.analyst_target_high, fund.analyst_target_mean)
            st.html("</div>")
        else:
            metric_card("Price Target", "N/A", "No analyst targets available")

        st.write("")

        # Quality flags summary
        if quality:
            red_n = sum(1 for f in quality.flags if f.status == "red")
            yellow_n = sum(1 for f in quality.flags if f.status == "yellow")
            green_n = sum(1 for f in quality.flags if f.status == "green")
            qf_val = f"{green_n} green · {yellow_n} yellow · {red_n} red"
            qf_ctx = quality.overall.title()
            metric_card("Quality Flags", qf_val, qf_ctx)
        else:
            metric_card("Quality Flags", "—", "See Analyst Mode tab")

    st.divider()

    # ── 6. Sentiment sparkline ────────────────────────────────────────────────
    try:
        call_data = analyse_all_calls(ticker, n=4)
        transcripts = call_data.get("transcripts", [])
        sentiments = call_data.get("sentiments", [])
        if transcripts and sentiments:
            quarter_labels = [f"Q{t.get('quarter')} {t.get('year')}" for t in reversed(transcripts)]
            prep_scores = [s.prepared_remarks_score if s else 0 for s in sentiments]
            qa_scores = [s.qa_score if s else 0 for s in sentiments]
            st.markdown("**Management tone — last 4 quarters**")
            fig = sentiment_mini_sparkline(quarter_labels, prep_scores, qa_scores)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Sentiment trend — no transcript data available.")
    except Exception:
        st.caption("Sentiment trend unavailable.")

    st.divider()

    # ── 7. Earnings beat/miss tracker ─────────────────────────────────────────
    st.markdown("**Earnings beat/miss — last 8 quarters**")
    try:
        from data.market import get_beat_miss_history
        from ui.charts import eps_surprise_bars
        bm = get_beat_miss_history(ticker)
        if bm:
            valid = [r for r in bm if r.get("eps_surprise_pct") is not None]
            beats = sum(1 for r in valid if (r["eps_surprise_pct"] or 0) >= 0)
            misses = len(valid) - beats
            avg_surprise = (
                sum(r["eps_surprise_pct"] for r in valid) / len(valid) if valid else 0
            )
            beat_color = "green" if beats >= misses else "red"
            st.caption(
                f":{beat_color}[Beat EPS {beats} of {len(valid)} quarters] · "
                f"avg surprise {avg_surprise:+.1f}%"
            )
            fig = eps_surprise_bars(bm)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No earnings history data available.")
    except Exception:
        st.caption("Earnings beat/miss tracker unavailable.")

    st.divider()

    # ── 9. High-materiality news ──────────────────────────────────────────────
    st.markdown("**High-materiality news**")
    try:
        impacts = classify_headlines(ticker, fund.name or ticker)
        hi = high_materiality(impacts)
        if hi:
            for item in hi[:5]:
                dir_marker = "+" if item.direction == "positive" else "-" if item.direction == "negative" else "·"
                url = item.url or "#"
                st.markdown(
                    f"`{dir_marker} {item.category}` [{item.title}]({url})",
                )
                st.caption(item.one_line_why)
        else:
            st.caption(f"No high-materiality news in the last 30 days ({len(impacts)} items classified).")
    except Exception as exc:
        error_card("News unavailable", str(exc))


def tab_financials(ticker: str, settings: Dict) -> None:
    from analysis.ratios import get_ratio_groups
    from analysis.reconcile import get_reconciliation
    from ui.charts import price_candlestick, ratio_sparkline, revenue_bars
    from ui.components import error_card, freshness_badge
    import plotly.graph_objects as go

    col_price, col_recon = st.columns([3, 2])

    with col_price:
        st.subheader("Price History")
        try:
            from data.market import get_prices
            period = st.radio("Period", ["1y", "3y", "5y"], horizontal=True, key="price_period")
            prices = get_prices(ticker, period)
            fig = price_candlestick(prices, f"{ticker} — {period}")
            st.plotly_chart(fig, use_container_width=True)
            freshness_badge(f"market:{ticker}:prices:{period}", "Prices")
        except Exception as exc:
            error_card("Price chart unavailable", str(exc))

    with col_recon:
        st.subheader("Source Reconciliation")
        st.caption("yfinance vs EDGAR XBRL. EDGAR is canonical where both exist.")
        try:
            recons = get_reconciliation(ticker)
            for r in recons:
                diff_str = f"{r.diff_pct*100:.1f}% diff" if r.diff_pct is not None else "—"
                flag = "!" if (r.diff_pct or 0) > 0.02 else "ok"
                with st.expander(f"[{flag}] {r.metric}: {diff_str}"):
                    col1, col2 = st.columns(2)
                    col1.metric("yfinance", f"{r.yfinance_value:,.0f}" if r.yfinance_value else "N/A")
                    col2.metric("EDGAR", f"{r.edgar_value:,.0f}" if r.edgar_value else "N/A")
                    st.caption(r.note)
                    if r.composite_note:
                        st.caption(r.composite_note)
                    if r.components:
                        st.markdown("**EDGAR composition:**")
                        for tag, val in r.components.items():
                            st.text(f"  {tag}: {val:,.0f}")
        except Exception as exc:
            error_card("Reconciliation unavailable", str(exc))

    st.divider()
    st.subheader("Ratio Groups")
    try:
        with st.spinner("Computing ratio history (fetching 5yr data)…"):
            groups = get_ratio_groups(ticker)
        for group in groups:
            with st.expander(f"**{group.group}**", expanded=(group.group == "Valuation")):
                cols = st.columns(min(3, len(group.ratios)))
                for i, ratio in enumerate(group.ratios):
                    with cols[i % len(cols)]:
                        cur = f"{ratio.current:.2f}" if ratio.current is not None else "N/A"
                        st.metric(ratio.name, cur)
                        if ratio.median_5y is not None:
                            st.caption(f"5yr: min {ratio.min_5y:.1f} / med {ratio.median_5y:.1f} / max {ratio.max_5y:.1f}")
                        if ratio.history:
                            fig = ratio_sparkline(ratio)
                            st.plotly_chart(fig, use_container_width=True, key=f"ratio_{ratio.name}_{ticker}")
                        if ratio.not_meaningful:
                            st.caption(f"N/A — {ratio.not_meaningful_reason}")
                        else:
                            st.caption(ratio.description)
    except Exception as exc:
        error_card("Ratio engine error", str(exc))


def tab_earnings_calls(ticker: str, settings: Dict) -> None:
    from analysis.calls import analyse_all_calls, stream_call_synthesis
    from analysis.sentiment import aggregate_sentiment_scores, prepared_qa_gap, sentiment_label
    from ui.charts import sentiment_trend_chart, beat_miss_chart
    from ui.components import error_card, freshness_badge, streaming_container, unavailable_tab
    from data.resilience import SourceUnavailable

    from data.transcripts import TranscriptRateLimited

    try:
        with st.spinner("Loading transcripts + running Pass A/B…"):
            call_data = analyse_all_calls(ticker, n=4)
    except SourceUnavailable as exc:
        unavailable_tab("Earnings Calls", str(exc))
        return
    except TranscriptRateLimited as exc:
        error_card("Transcript API rate limited", str(exc))
        return
    except Exception as exc:
        error_card("Earnings call pipeline error", str(exc))
        return

    transcripts = call_data["transcripts"]
    summaries = call_data["summaries"]
    sentiments = call_data["sentiments"]

    if not transcripts:
        st.info(f"No transcripts available for **{ticker}** in the last 8 quarters from any configured provider.")
        return

    # Sentiment trend chart
    scores = aggregate_sentiment_scores(sentiments)
    quarter_labels = [f"Q{t.get('quarter')} {t.get('year')}" for t in reversed(transcripts)]
    if scores:
        fig = sentiment_trend_chart(quarter_labels, [s or 0 for s in scores])
        st.plotly_chart(fig, use_container_width=True)

    # Per-call summaries
    col_tabs = st.tabs([f"Q{t.get('quarter')} {t.get('year')}" for t in reversed(transcripts)])
    for i, (col, summary, sentiment) in enumerate(zip(col_tabs, summaries, sentiments)):
        with col:
            if summary:
                st.markdown("**Key Themes**")
                for theme in summary.key_themes:
                    st.markdown(f"- {theme}")
                if summary.guidance_items:
                    st.markdown("**Guidance**")
                    for g in summary.guidance_items:
                        dir_tag = {"raised": "+", "lowered": "-", "maintained": "="}.get(g.direction, "")
                        st.markdown(f"- {dir_tag} **{g.metric}**: {g.value or '—'}")
                if summary.top_analyst_concerns_from_qa:
                    st.markdown("**Top Analyst Concerns (Q&A)**")
                    for c in summary.top_analyst_concerns_from_qa:
                        st.markdown(f"- {c}")
            if sentiment:
                st.divider()
                c1, c2, c3 = st.columns(3)
                c1.metric("Overall", sentiment_label(sentiment.overall_score))
                c2.metric("Prepared", f"{sentiment.prepared_remarks_score:+.2f}")
                c3.metric("Q&A", f"{sentiment.qa_score:+.2f}")
                gap = prepared_qa_gap(sentiment)
                if abs(gap) > 0.2:
                    st.warning(
                        f"Prepared vs Q&A gap: {gap:+.2f} — "
                        f"{'management more upbeat in scripted remarks' if gap > 0 else 'management more candid in Q&A'}"
                    )
                if sentiment.evasiveness_flags:
                    with st.expander("Evasiveness flags"):
                        for flag in sentiment.evasiveness_flags:
                            st.markdown(f"**{flag.analyst_question_topic}**: {flag.why_answer_seemed_indirect}")
                if sentiment.hedging_index.example_phrases:
                    with st.expander(f"Hedging ({sentiment.hedging_index.level})"):
                        for phrase in sentiment.hedging_index.example_phrases:
                            st.markdown(f'*"{phrase}"*')

    st.divider()
    st.markdown("**What Changed — 4-Quarter Synthesis**")
    if len(transcripts) >= 2:
        container = st.empty()
        with st.spinner("Synthesising cross-quarter evolution (Sonnet)…"):
            synthesis_iter = stream_call_synthesis(summaries, sentiments, ticker)
            streaming_container(synthesis_iter, container)
    else:
        st.caption("Need at least 2 quarters for synthesis.")

    freshness_badge(f"transcript:{ticker}:{transcripts[0].get('year')}:Q{transcripts[0].get('quarter')}", "Transcripts")

    st.divider()
    st.markdown("**Ask a question across all loaded transcripts**")
    st.caption("Answers are cited to specific quarters. Retrieval is keyword-based — be specific.")

    qa_input = st.text_input(
        "Your question",
        placeholder='e.g. "What did management say about gross margin expansion?"',
        key=f"qa_input_{ticker}",
    )
    if st.button("Ask", type="secondary", key=f"qa_ask_{ticker}") and qa_input.strip():
        from analysis.transcript_qa import answer_question
        with st.spinner("Searching transcripts…"):
            answer = answer_question(qa_input, transcripts)
        st.markdown(answer)
    elif f"qa_last_answer_{ticker}" in st.session_state:
        st.markdown(st.session_state[f"qa_last_answer_{ticker}"])


def tab_news(ticker: str, settings: Dict) -> None:
    from analysis.news_impact import classify_headlines
    from data.market import get_fundamentals
    from ui.components import error_card, freshness_badge

    try:
        fund = get_fundamentals(ticker)
        with st.spinner("Classifying headlines…"):
            impacts = classify_headlines(ticker, fund.name or ticker)
    except Exception as exc:
        error_card("News tab error", str(exc))
        return

    if not impacts:
        st.info("No news found for the last 30 days.")
        return

    # Category filter
    all_cats = sorted(set(h.category for h in impacts))
    selected = st.multiselect("Filter by category", all_cats, default=all_cats)
    filtered = [h for h in impacts if h.category in selected]

    for item in filtered:
        with st.container(border=True):
            url = item.url or "#"
            st.markdown(f"**[{item.title}]({url})**")
            st.caption(
                f"{item.materiality.upper()} · {item.direction} · "
                f"{item.category} · {item.source or 'unknown'} · "
                f"{item.published_at.strftime('%b %d') if item.published_at else '—'}"
            )
            st.caption(item.one_line_why)

    freshness_badge(f"news:{ticker}:30", "News")


def tab_analyst_mode(ticker: str, settings: Dict) -> None:
    from analysis.expectations import reverse_dcf
    from analysis.kpis import extract_kpis
    from analysis.quality import get_quality_panel
    from analysis.positioning import get_positioning
    from analysis.calls import analyse_all_calls
    from ui.components import error_card, freshness_badge
    import plotly.graph_objects as go
    import plotly.express as px

    # ── Inline DCF controls ───────────────────────────────────────────────────
    dcf_c1, dcf_c2, dcf_c3 = st.columns(3)
    with dcf_c1:
        dr_pct = st.slider(
            "Discount rate", 7.0, 15.0, 10.0, 0.5,
            format="%.1f%%", key="dcf_discount_rate",
            help="Used in reverse DCF",
        )
    with dcf_c2:
        tg_pct = st.slider(
            "Terminal growth", 1.0, 4.0, 2.5, 0.5,
            format="%.1f%%", key="dcf_terminal_growth",
        )
    with dcf_c3:
        horizon = st.selectbox("DCF horizon (years)", [5, 7, 10], index=2, key="dcf_horizon")

    discount_rate = dr_pct / 100.0
    terminal_growth = tg_pct / 100.0

    sub_tabs = st.tabs(["Reverse DCF", "Beat / Miss", "KPIs", "Quality Flags", "Positioning"])

    # ── Reverse DCF ───────────────────────────────────────────────────────────
    with sub_tabs[0]:
        st.markdown("**Reverse DCF — What does the market require?**")
        st.caption(f"Discount rate: {discount_rate*100:.1f}%  ·  "
                   f"Terminal growth: {terminal_growth*100:.1f}%  ·  "
                   f"Horizon: {horizon} years")
        try:
            from ui.charts import dcf_contour_heatmap
            from data.market import get_fundamentals as _get_fund
            import pandas as pd

            with st.spinner("Computing DCF grid…"):
                dcf = reverse_dcf(
                    ticker,
                    discount_rate=discount_rate,
                    terminal_growth=terminal_growth,
                    horizon_years=horizon,
                )
            if dcf:
                st.info(dcf.headline)
                fund_dcf = _get_fund(ticker)
                if fund_dcf.revenue_ttm and fund_dcf.revenue_ttm > 0:
                    fig = dcf_contour_heatmap(dcf, fund_dcf.revenue_ttm)
                    st.plotly_chart(fig, use_container_width=True)
                if dcf.sensitivity_table:
                    df = pd.DataFrame(dcf.sensitivity_table)
                    df["discount_rate"] = df["discount_rate"].map(lambda x: f"{x*100:.1f}%")
                    df["avg_implied_price"] = df["avg_implied_price"].map(lambda x: f"${x:,.0f}")
                    df.columns = ["Discount rate", "Avg implied price"]
                    st.caption("Discount rate sensitivity (avg across revenue CAGR and margin combos)")
                    st.dataframe(df, hide_index=True, use_container_width=False)
            else:
                st.warning("Insufficient data for reverse DCF.")
        except Exception as exc:
            error_card("Reverse DCF error", str(exc))

    # ── Beat/Miss ─────────────────────────────────────────────────────────────
    with sub_tabs[1]:
        st.subheader("EPS Beat/Miss History")
        try:
            from data.market import get_estimates
            import pandas as pd
            estimates = get_estimates(ticker)
            eh = estimates.get("earnings_history")
            if eh:
                df = pd.DataFrame(eh)
                # yfinance earnings_history columns vary — find EPS columns
                est_col = next((c for c in df.columns if "estimate" in str(c).lower() and "eps" in str(c).lower()), None)
                act_col = next((c for c in df.columns if "actual" in str(c).lower() or "surprise" in str(c).lower()), None)
                period_col = next((c for c in df.columns if "quarter" in str(c).lower() or "period" in str(c).lower() or "date" in str(c).lower()), None)

                if all([est_col, act_col, period_col]):
                    df_clean = df[[period_col, est_col, act_col]].dropna().tail(8)
                    from ui.charts import beat_miss_chart
                    fig = beat_miss_chart(
                        [str(p) for p in df_clean[period_col].values],
                        df_clean[est_col].tolist(),
                        df_clean[act_col].tolist(),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.dataframe(df.tail(8), use_container_width=True)
            else:
                st.info("Earnings history not available from yfinance.")
        except Exception as exc:
            error_card("Beat/miss chart error", str(exc))

    # ── KPIs ──────────────────────────────────────────────────────────────────
    with sub_tabs[2]:
        st.subheader("Company-Specific KPIs")
        try:
            call_data = analyse_all_calls(ticker, n=4)
            if not call_data["transcripts"]:
                st.info("KPI extraction requires earnings transcripts (none available for this ticker).")
            else:
                with st.spinner("Extracting KPIs from transcripts (Haiku)…"):
                    kpi_list = extract_kpis(ticker, call_data)
                if kpi_list:
                    for kpi in kpi_list:
                        with st.expander(f"{'[absent] ' if kpi.disappeared else ''}{kpi.kpi_name}"):
                            st.caption(f"**Definition:** {kpi.definition_as_company_uses_it}")
                            if kpi.disappeared:
                                st.warning("This KPI appeared in earlier calls but is absent in the most recent quarter.")
                            if kpi.values:
                                import plotly.express as px
                                import pandas as pd
                                df = pd.DataFrame([{"Quarter": v.quarter, "Value": v.value} for v in kpi.values if v.value])
                                if not df.empty:
                                    st.dataframe(df, hide_index=True)
                            st.caption(kpi.trend_note)
                else:
                    st.info("No company-specific KPIs identified in available transcripts.")
        except Exception as exc:
            error_card("KPI extraction error", str(exc))

    # ── Quality Flags ─────────────────────────────────────────────────────────
    with sub_tabs[3]:
        st.subheader("Quality-of-Earnings Flags")
        try:
            with st.spinner("Computing quality flags (pure math)…"):
                panel = get_quality_panel(ticker)
            status_labels = {"green": "OK", "yellow": "WATCH", "red": "FLAG"}
            st.markdown(f"**Overall: {panel.overall.title()}** — {panel.summary}")
            st.divider()
            for flag in panel.flags:
                flag_label = status_labels.get(flag.status, flag.status.upper())
                with st.expander(f"[{flag_label}] {flag.name}"):
                    st.caption(f"**Trigger:** {flag.trigger_condition}")
                    if flag.observed_value:
                        st.metric("Observed", flag.observed_value)
                    st.caption(f"**Threshold:** {flag.threshold}")
                    st.write(flag.explanation)
        except Exception as exc:
            error_card("Quality flags error", str(exc))

    # ── Positioning ───────────────────────────────────────────────────────────
    with sub_tabs[4]:
        st.subheader("Positioning — Short Interest, Insiders, Holders")
        try:
            pos = get_positioning(ticker)
            st.info(pos.synthesis)
            c1, c2 = st.columns(2)
            c1.metric("Short interest", f"{pos.short_interest_pct_float*100:.1f}%" if pos.short_interest_pct_float else "N/A")
            c2.metric("Days to cover", f"{pos.days_to_cover:.1f}" if pos.days_to_cover else "N/A")
            st.caption(f"Insider sentiment (6mo): **{pos.insider_net_sentiment}**")
            if pos.insider_transactions:
                import pandas as pd
                rows = [
                    {
                        "Name": t.name, "Role": t.role, "Type": t.transaction_type,
                        "Shares": t.shares, "Date": str(t.date),
                    }
                    for t in pos.insider_transactions[:10]
                ]
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        except Exception as exc:
            error_card("Positioning error", str(exc))


def tab_thesis_memo(ticker: str, settings: Dict) -> None:
    from analysis.calls import analyse_all_calls, synthesize_calls
    from analysis.delta import run_delta
    from analysis.expectations import reverse_dcf
    from analysis.kpis import extract_kpis
    from analysis.memo import build_memo_object, memo_to_markdown, stream_memo
    from analysis.positioning import get_positioning
    from analysis.quality import get_quality_panel
    from analysis.schemas import RunSnapshotData
    from analysis.thesis import get_red_team, stream_theses
    from data.market import get_fundamentals
    from ui.components import error_card, streaming_container

    fund = get_fundamentals(ticker)
    dcf = None
    call_delta = None
    quality = None
    positioning = None
    kpi_summaries: List[str] = []
    high_mat_news: List[str] = []

    # Read DCF settings set by the Analyst Mode tab (default if not yet visited)
    _dr = st.session_state.get("dcf_discount_rate", 10.0) / 100.0
    _tg = st.session_state.get("dcf_terminal_growth", 2.5) / 100.0
    _hz = st.session_state.get("dcf_horizon", 10)

    with st.spinner("Assembling thesis inputs…"):
        try:
            dcf = reverse_dcf(ticker, _dr, _tg, _hz)
        except Exception:
            pass
        try:
            call_data = analyse_all_calls(ticker, n=4)
            if call_data["summaries"]:
                call_delta = synthesize_calls(call_data["summaries"], call_data["sentiments"], ticker)
                kpis = extract_kpis(ticker, call_data)
                kpi_summaries = [f"{k.kpi_name}: {k.trend_note}" for k in kpis]
        except Exception:
            pass
        try:
            quality = get_quality_panel(ticker)
        except Exception:
            pass
        try:
            positioning = get_positioning(ticker)
        except Exception:
            pass
        try:
            from analysis.news_impact import classify_headlines, high_materiality
            impacts = classify_headlines(ticker, fund.name or ticker)
            high_mat_news = [f"{h.title} ({h.one_line_why})" for h in high_materiality(impacts)]
        except Exception:
            pass

    thesis_tab, redteam_tab, memo_tab = st.tabs(["Theses", "Devil's Advocate", "One-Page Memo"])

    # ── Theses ────────────────────────────────────────────────────────────────
    with thesis_tab:
        st.caption("Bull / base / bear scenarios with confirm + kill signals. Sonnet, streaming.")
        use_web_search = st.checkbox(
            "Augment with live web search",
            value=False,
            help="Fetches recent analyst upgrades, news, and investor discussion via Claude web search. Adds ~$0.02–0.05 per run, cached 6h.",
            key=f"web_search_{ticker}",
        )
        if st.button("Generate Theses", type="primary"):
            container = st.empty()
            try:
                web_ctx = ""
                if use_web_search:
                    from analysis.thesis import fetch_web_context
                    with st.spinner("Searching web for latest analyst views…"):
                        web_ctx = fetch_web_context(ticker, fund.name or ticker)
                    if web_ctx:
                        st.caption("Web search complete — live context added to thesis.")
                    else:
                        st.caption("Web search returned no results — proceeding without.")
                token_iter, chunks = stream_theses(
                    ticker, fund, dcf, call_delta, quality, positioning,
                    high_mat_news, kpi_summaries,
                    estimates_summary=web_ctx,
                )
                thesis_text = streaming_container(token_iter, container)
                st.session_state[f"thesis_text_{ticker}"] = thesis_text
                st.session_state[f"thesis_chunks_{ticker}"] = chunks
            except Exception as exc:
                error_card("Thesis generation error", str(exc))
        elif f"thesis_text_{ticker}" in st.session_state:
            st.markdown(st.session_state[f"thesis_text_{ticker}"])

    # ── Red team ──────────────────────────────────────────────────────────────
    with redteam_tab:
        st.caption("Devil's advocate: strongest counter-argument to the prevailing thesis.")
        if st.button("Run Devil's Advocate", type="secondary"):
            context = f"Ticker: {ticker}\n{fund.model_dump_json()[:1500]}"
            if call_delta:
                context += f"\nCall delta: {call_delta.what_changed_narrative[:500]}"
            try:
                with st.spinner("Red-teaming…"):
                    rt = get_red_team(ticker, context)
                if rt:
                    st.markdown(f"**Strongest counter-argument:** {rt.strongest_counterargument}")
                    st.markdown(f"**Most fragile assumption:** {rt.most_fragile_assumption}")
                    st.markdown(f"**What bulls are ignoring:** {rt.what_bulls_are_ignoring}")
                    st.markdown(f"**What bears are ignoring:** {rt.what_bears_are_ignoring}")
                    st.error(f"**Fastest falsifier:** {rt.fastest_falsifier}")
            except Exception as exc:
                error_card("Red team error", str(exc))

    # ── Memo ─────────────────────────────────────────────────────────────────
    with memo_tab:
        st.caption("One-page memo <600 words with inline citations. Export as Markdown.")
        if st.button("Generate Memo", type="primary"):
            thesis_text = st.session_state.get(f"thesis_text_{ticker}", "Thesis not yet generated.")
            kpis_for_memo = []
            try:
                from analysis.kpis import extract_kpis
                call_data = analyse_all_calls(ticker, n=4)
                kpis_for_memo = extract_kpis(ticker, call_data) if call_data["transcripts"] else []
            except Exception:
                pass

            container = st.empty()
            try:
                token_iter, footnotes = stream_memo(
                    ticker, fund.name or ticker, thesis_text,
                    fund, dcf, call_delta, quality, positioning, kpis_for_memo,
                )
                memo_text = streaming_container(token_iter, container)
                memo_obj = build_memo_object(ticker, fund.name or ticker, memo_text, footnotes)
                st.caption(f"Word count: {memo_obj.word_count}")
                if memo_obj.word_count > 600:
                    st.warning(f"Memo is {memo_obj.word_count} words — exceeds 600 word target.")

                md_export = memo_to_markdown(memo_obj)
                st.download_button(
                    "Download Memo (Markdown)",
                    md_export,
                    file_name=f"{ticker}_memo.md",
                    mime="text/markdown",
                )
            except Exception as exc:
                error_card("Memo generation error", str(exc))


def tab_peer_comps(ticker: str, settings: Dict) -> None:
    from analysis.peers import get_peer_comps, _DEFAULT_PEERS
    from ui.components import error_card
    import pandas as pd

    st.caption("Compare key ratios against sector peers. Edit the peer list then click Load.")

    default_peers = _DEFAULT_PEERS.get(ticker.upper(), [])
    peer_input = st.text_input(
        "Peer tickers (comma-separated)",
        value=", ".join(default_peers),
        key=f"peers_input_{ticker}",
        placeholder="e.g. AMD, INTC, QCOM",
    )

    if st.button("Load Peer Comps", type="primary", key=f"peers_load_{ticker}"):
        parsed = [t.strip().upper() for t in peer_input.split(",") if t.strip()]
        with st.spinner(f"Fetching fundamentals for {ticker} + {len(parsed)} peers…"):
            try:
                comps = get_peer_comps(ticker, parsed)
                st.session_state[f"peer_comps_{ticker}"] = comps
            except Exception as exc:
                error_card("Peer comps error", str(exc))
                return

    comps = st.session_state.get(f"peer_comps_{ticker}")
    if not comps:
        st.info("Enter peer tickers above and click Load Peer Comps.")
        return

    if comps.synthesis:
        st.info(comps.synthesis)

    # Build DataFrame
    def _pct(v):
        return f"{v*100:.1f}%" if v is not None else "—"
    def _x(v):
        return f"{v:.1f}x" if v is not None else "—"
    def _b(v):
        return f"${v/1e9:.1f}B" if v is not None else "—"

    rows = []
    for r in comps.peers:
        rows.append({
            "": "★ " + r.ticker if r.is_target else r.ticker,
            "Name": (r.name or "")[:25],
            "Mkt Cap": _b(r.market_cap),
            "P/E TTM": _x(r.pe_ttm),
            "Fwd P/E": _x(r.pe_forward),
            "EV/EBITDA": _x(r.ev_ebitda),
            "P/S": _x(r.price_to_sales),
            "Gross Margin": _pct(r.gross_margin),
            "Net Margin": _pct(r.net_margin),
            "Rev Growth": _pct(r.revenue_growth_yoy),
            "FCF Yield": _pct(r.fcf_yield),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, use_container_width=True)

    # Per-metric sparkline bars (cheapest/most expensive highlights)
    numeric_metrics = [
        ("P/E TTM", "pe_ttm", False),       # lower = cheaper
        ("Fwd P/E", "pe_forward", False),
        ("EV/EBITDA", "ev_ebitda", False),
        ("Rev Growth %", "revenue_growth_yoy", True),  # higher = better
        ("Net Margin %", "net_margin", True),
        ("FCF Yield %", "fcf_yield", True),
    ]
    st.divider()
    st.markdown("**Metric comparison across peers**")
    import plotly.graph_objects as go
    from ui.charts import _BASE_LAYOUT, _ACCENT, _GREEN, _RED, _MUTED

    cols = st.columns(3)
    for idx, (label, field, higher_is_better) in enumerate(numeric_metrics):
        values = [(r.ticker, getattr(r, field)) for r in comps.peers if getattr(r, field) is not None]
        if not values:
            continue
        tickers_list = [v[0] for v in values]
        vals = [v[1] for v in values]
        if "growth" in field or "margin" in field or "yield" in field:
            vals = [v * 100 for v in vals]
        target_val = next((v for t, v in zip(tickers_list, vals) if t == ticker.upper()), None)
        bar_colors = []
        for t, v in zip(tickers_list, vals):
            if t == ticker.upper():
                bar_colors.append(_ACCENT)
            else:
                bar_colors.append(f"rgba(59,74,107,0.30)")
        fig = go.Figure(go.Bar(x=tickers_list, y=vals, marker_color=bar_colors, name=label))
        suffix = "x" if "P/E" in label or "EBITDA" in label or "P/S" in label else "%"
        fig.update_layout(
            **{k: v for k, v in _BASE_LAYOUT.items() if k not in ("xaxis", "yaxis")},
            height=200,
            title=dict(text=label, font=dict(size=12, weight=600)),
            margin=dict(l=4, r=4, t=32, b=4),
            yaxis=dict(ticksuffix=suffix, gridcolor="rgba(26,26,26,0.06)", zeroline=False, showline=False),
            xaxis=dict(gridcolor="rgba(26,26,26,0.06)", zeroline=False, showline=False),
            showlegend=False,
        )
        with cols[idx % 3]:
            st.plotly_chart(fig, use_container_width=True, key=f"peer_bar_{ticker}_{field}")


def run_ticker_mode(ticker: str, settings: Dict) -> None:
    from data.market import get_fundamentals
    from ui.components import cost_footer

    # Brief header
    try:
        fund = get_fundamentals(ticker)
        price = fund.current_price
        price_str = f" · ${price:.2f}" if price else ""
        st.markdown(f"## {fund.name or ticker} ({ticker}){price_str}")
        st.caption(f"{fund.sector or '—'} · {fund.industry or '—'}")
    except Exception:
        st.markdown(f"## {ticker}")

    tabs = st.tabs([
        "Overview",
        "Financials",
        "Earnings Calls",
        "News",
        "Analyst Mode",
        "Peer Comps",
        "Thesis & Memo",
    ])

    with tabs[0]:
        tab_overview(ticker, settings)
    with tabs[1]:
        tab_financials(ticker, settings)
    with tabs[2]:
        tab_earnings_calls(ticker, settings)
    with tabs[3]:
        tab_news(ticker, settings)
    with tabs[4]:
        tab_analyst_mode(ticker, settings)
    with tabs[5]:
        tab_peer_comps(ticker, settings)
    with tabs[6]:
        tab_thesis_memo(ticker, settings)


# ═══════════════════════════════════════════════════════════════════════════════
# URL MODE TABS
# ═══════════════════════════════════════════════════════════════════════════════

def tab_hiring_intel(url: str, domain: str) -> None:
    from analysis.job_intel import extract_hiring_intel
    from ui.components import error_card
    import plotly.graph_objects as go
    from ui.charts import _BASE_LAYOUT, _ACCENT

    # Pull hiring signals from the company intel crawl (already cached)
    results_key = f"company_intel__{url}"
    ci = st.session_state.get(results_key)
    if not ci:
        st.info("Run Company Intel first — hiring data is extracted from the same crawl.")
        return

    # Collect all hiring signals from page intels stored in session
    hiring_signals_key = f"hiring_signals__{url}"
    if hiring_signals_key not in st.session_state:
        # Try to re-derive from the cached page_intels
        st.info("Hiring signals are collected during the Company Intel crawl. Re-run Company Intel to populate.")
        return

    hiring_signals = st.session_state[hiring_signals_key]
    with st.spinner("Clustering job postings by department…"):
        intel = extract_hiring_intel(domain, hiring_signals)

    dept_counts = intel.get("department_counts") or {}
    roadmap = intel.get("roadmap_signals") or []
    standout = intel.get("standout_roles") or []

    if not dept_counts and not roadmap:
        st.info("Insufficient hiring data found on this site.")
        return

    if roadmap:
        st.markdown("**What hiring reveals about strategy**")
        for signal in roadmap:
            st.markdown(f"- {signal}")

    if dept_counts:
        st.divider()
        st.markdown("**Open roles by department**")
        depts = list(dept_counts.keys())
        counts = [dept_counts[d] for d in depts]
        fig = go.Figure(go.Bar(
            x=depts, y=counts,
            marker_color=_ACCENT,
        ))
        fig.update_layout(
            **{k: v for k, v in _BASE_LAYOUT.items() if k not in ("xaxis", "yaxis")},
            height=280, showlegend=False,
            margin=dict(l=4, r=4, t=20, b=4),
            yaxis=dict(title="Roles", gridcolor="rgba(26,26,26,0.06)", zeroline=False, showline=False),
            xaxis=dict(gridcolor="rgba(26,26,26,0.06)", zeroline=False, showline=False),
        )
        st.plotly_chart(fig, use_container_width=True)

    if standout:
        st.divider()
        st.markdown("**Standout / unusual roles**")
        for r in standout:
            st.markdown(f"- {r}")


def tab_competitors(url: str, domain: str) -> None:
    from analysis.competitors import discover_competitors, build_competitive_comparison
    from ui.components import error_card

    st.caption("Auto-discovers 3-5 competitors then crawls each for a side-by-side feature comparison.")

    # We need a summary of the target product to find competitors
    results_key = f"company_intel__{url}"
    ci = st.session_state.get(results_key)
    company_summary = (ci or {}).get("synthesis_text", "") or domain

    if st.button("Discover & Compare Competitors", type="primary", key=f"comp_discover_{domain}"):
        comp_key = f"competitors__{domain}"
        with st.spinner("Identifying competitors (Sonnet)…"):
            competitors = discover_competitors(domain, company_summary)
        if not competitors:
            st.warning("Could not identify competitors automatically. Try running Company Intel first.")
            return
        st.caption(f"Identified: {', '.join(c.get('name', c['domain']) for c in competitors)}")
        with st.spinner(f"Crawling {len(competitors)} competitor sites…"):
            comparison = build_competitive_comparison(domain, company_summary, competitors)
        st.session_state[comp_key] = {"competitors": competitors, "comparison": comparison}

    comp_key = f"competitors__{domain}"
    comp_data = st.session_state.get(comp_key)
    if not comp_data:
        st.info("Click the button above to start competitor discovery.")
        return

    competitors = comp_data.get("competitors", [])
    comparison = comp_data.get("comparison", "")

    if competitors:
        st.markdown("**Competitors identified**")
        for c in competitors:
            st.markdown(f"- [{c.get('name', c['domain'])}](https://{c['domain']})")

    if comparison:
        st.divider()
        st.markdown(comparison)


def tab_reviews(url: str, domain: str) -> None:
    from analysis.review_sentiment import fetch_review_sentiment
    from ui.components import error_card

    st.caption("Searches G2, Capterra, and Product Hunt for customer reviews.")

    reviews_key = f"reviews__{domain}"
    if reviews_key not in st.session_state:
        if st.button("Fetch Customer Reviews", type="primary", key=f"reviews_btn_{domain}"):
            with st.spinner("Checking G2, Capterra, and Product Hunt…"):
                reviews = fetch_review_sentiment(domain)
            st.session_state[reviews_key] = reviews
    else:
        reviews = st.session_state[reviews_key]
        if not reviews:
            st.info("No review data found on G2, Capterra, or Product Hunt for this domain.")
            return

        for r in reviews:
            platform = r.get("platform", "Unknown")
            stars = r.get("star_rating")
            count = r.get("review_count")
            with st.expander(
                f"**{platform}** — {'★' * int(stars or 0)}{f' {stars:.1f}/5' if stars else ''}"
                f"{f' · {count:,} reviews' if count else ''}",
                expanded=True,
            ):
                summary = r.get("sentiment_summary")
                if summary:
                    st.info(summary)

                col1, col2 = st.columns(2)
                pros = r.get("top_pros") or []
                cons = r.get("top_cons") or []
                with col1:
                    if pros:
                        st.markdown("**Top pros**")
                        for p in pros:
                            st.markdown(f"+ {p}")
                with col2:
                    if cons:
                        st.markdown("**Top cons**")
                        for c in cons:
                            st.markdown(f"- {c}")

                use_cases = r.get("common_use_cases") or []
                if use_cases:
                    st.caption("Common use cases: " + " · ".join(use_cases))

    if reviews_key in st.session_state and not st.session_state.get(reviews_key):
        if st.button("Fetch Customer Reviews", type="primary", key=f"reviews_btn2_{domain}"):
            with st.spinner("Checking G2, Capterra, and Product Hunt…"):
                st.session_state[reviews_key] = fetch_review_sentiment(domain)
            st.rerun()


def run_url_mode(url: str, settings: Dict) -> None:
    domain = url.replace("https://", "").replace("http://", "").split("/")[0]
    st.markdown(f"## {domain}")

    tabs = st.tabs(["Company Intel", "Product Deep Dive", "Hiring Intel", "Competitors", "Reviews"])

    with tabs[0]:
        tab_company_intel(url, domain)
    with tabs[1]:
        tab_product_deep_dive(url, domain)
    with tabs[2]:
        tab_hiring_intel(url, domain)
    with tabs[3]:
        tab_competitors(url, domain)
    with tabs[4]:
        tab_reviews(url, domain)


def tab_company_intel(url: str, domain: str) -> None:
    """
    Runs the full URL pipeline on first call for this URL.
    Results are stored in st.session_state and re-rendered on subsequent reruns
    without re-invoking the pipeline (survives widget interactions).
    """
    from analysis.company import extract_page_intel, stream_company_intel
    from analysis.delta import run_delta
    from data.cache import get_last_run_snapshot
    from data.webintel import discover_urls, fetch_pages
    from ui.components import delta_card, error_card, streaming_container

    results_key = f"company_intel__{url}"

    # ── Render cached results from a previous run this session ────────────────
    if results_key in st.session_state:
        cached = st.session_state[results_key]
        if cached.get("error"):
            error_card("Company intel error", cached["error"])
            return
        last_snap = get_last_run_snapshot(url)
        delta_card(last_snap)
        if last_snap is None:
            st.success("✓ Baseline saved — delta card will appear on next run.")
        st.divider()
        st.caption(
            f"✓ {cached['urls_found']} URLs discovered · "
            f"{cached['pages_fetched']} pages fetched · "
            f"{cached['intels_extracted']} with intel"
        )
        st.markdown(cached["synthesis_text"])
        return

    # ── First run for this URL this session — execute pipeline ────────────────
    try:
        with st.spinner("Discovering URLs (sitemap → RSS → conventional paths)…"):
            urls = discover_urls(url)
        st.caption(f"Found {len(urls)} URLs in scope")

        budget = min(len(urls), 40)
        with st.spinner(f"Fetching {budget} pages…"):
            pages = fetch_pages(urls[:budget])
        st.caption(f"Fetched {len(pages)} pages")

        with st.spinner("Extracting intel page-by-page (Haiku)…"):
            page_intels = []
            prog = st.progress(0)
            for i, pg in enumerate(pages):
                intel = extract_page_intel(pg)
                if intel:
                    page_intels.append(intel)
                prog.progress((i + 1) / max(len(pages), 1))
            prog.empty()
        st.caption(f"Intel extracted from {len(page_intels)} pages")

        if not page_intels:
            st.warning("No intel could be extracted from the fetched pages. Check the URL or try again.")
            st.session_state[results_key] = {
                "error": "No extractable content found on any fetched page.",
                "urls_found": len(urls), "pages_fetched": len(pages), "intels_extracted": 0,
                "synthesis_text": "",
            }
            return

        container = st.empty()
        with st.spinner("Synthesising (Sonnet, streaming)…"):
            token_iter, snap_data = stream_company_intel(domain, page_intels)
            synthesis_text = streaming_container(token_iter, container)

        # Persist snapshot — guarded against empty content inside run_delta
        run_delta(url, snap_data)

        # Collect hiring signals for the Hiring Intel tab
        all_hiring_signals = [s for intel in page_intels for s in intel.hiring_signals]
        st.session_state[f"hiring_signals__{url}"] = all_hiring_signals

        # Store results in session_state so reruns don't re-execute pipeline
        st.session_state[results_key] = {
            "synthesis_text": synthesis_text,
            "urls_found": len(urls),
            "pages_fetched": len(pages),
            "intels_extracted": len(page_intels),
            "error": None,
        }

    except Exception as exc:
        import traceback
        detail = f"{exc}\n\n{traceback.format_exc()[-800:]}"
        st.session_state[results_key] = {
            "error": detail, "urls_found": 0, "pages_fetched": 0,
            "intels_extracted": 0, "synthesis_text": "",
        }
        error_card("Company intel error", str(exc))


def tab_product_deep_dive(url: str, domain: str) -> None:
    """
    Runs docs crawl + product explainer on first call.
    Falls back to marketing pages from the Company Intel crawl when no docs are found.
    Results stored in session_state and re-rendered on reruns.
    """
    from data.docs_crawler import discover_docs, fetch_docs_pages, harvest_images
    from analysis.product import explain_screenshots, stream_product_explainer
    from ui.components import error_card, streaming_container

    results_key = f"product_deep_dive__{url}"

    # ── Re-render cached results ───────────────────────────────────────────────
    if results_key in st.session_state:
        cached = st.session_state[results_key]
        if cached.get("error"):
            error_card("Product deep dive error", cached["error"])
            return

        source_label = "docs" if not cached.get("fallback_mode") else "marketing pages"
        st.caption(
            f"✓ {cached['docs_found']} {source_label} URLs · "
            f"{cached['pages_fetched']} pages fetched"
        )
        if cached.get("probe_log"):
            with st.expander("Docs discovery coverage report", expanded=False):
                for line in cached["probe_log"]:
                    st.text(line)
        st.markdown(cached["explainer_text"])
        if cached.get("images"):
            st.divider()
            st.markdown(f"**Screenshots** ({len(cached['images'])})")
            explained = cached.get("explained", [])
            cols = st.columns(3)
            for i, (img, expl) in enumerate(zip(cached["images"], explained)):
                with cols[i % 3]:
                    if img.get("url"):
                        st.image(img["url"], caption=expl.screen_name_guess if expl else "")
                    if expl:
                        st.caption(expl.what_this_reveals_about_the_product[:120])
        return

    # ── First run — execute pipeline ──────────────────────────────────────────
    try:
        with st.spinner("Discovering docs (llms.txt → docs subdomain → /docs path)…"):
            docs_urls, probe_log = discover_docs(url)

        fallback_mode = False
        source_tag = "DOC"
        caveat = ""

        if docs_urls:
            st.caption(f"Found {len(docs_urls)} docs URLs")
            budget = min(len(docs_urls), 25)
            with st.spinner(f"Fetching {budget} docs pages…"):
                doc_pages = fetch_docs_pages(docs_urls[:budget])
            st.caption(f"Fetched {len(doc_pages)} pages")
        else:
            # No docs found — fall back to company intel marketing pages
            fallback_mode = True
            source_tag = "PAGE"
            caveat = (
                "No public documentation found for this site. "
                "Analysis is derived from marketing pages (product/features/pricing) — "
                "API details and technical depth may be limited."
            )
            st.info(
                "No docs found. Falling back to marketing site pages. "
                "See coverage report for per-probe results."
            )
            from data.webintel import discover_urls, fetch_pages as fetch_web_pages
            with st.spinner("Loading marketing pages (cached from Company Intel run)…"):
                marketing_urls = discover_urls(url)
                marketing_pages = fetch_web_pages(marketing_urls[:40])
            # Prefer product-type pages; fall back to all pages
            _PRODUCT_KEYWORDS = ("product", "feature", "solution", "pricing",
                                  "platform", "capability", "about", "how-it-works")
            doc_pages = [
                p for p in marketing_pages
                if any(kw in p.get("url", "").lower() for kw in _PRODUCT_KEYWORDS)
            ] or marketing_pages[:15]
            docs_urls = [p["url"] for p in doc_pages]
            st.caption(f"Using {len(doc_pages)} marketing pages as fallback source")

        # Show coverage report always
        if probe_log:
            with st.expander("Docs discovery coverage report", expanded=(not docs_urls or fallback_mode)):
                for line in probe_log:
                    st.text(line)

        container = st.empty()
        spinner_msg = "Writing product explainer (Sonnet, streaming)…" if not fallback_mode else "Writing marketing-site explainer (Sonnet, streaming)…"
        with st.spinner(spinner_msg):
            token_iter = stream_product_explainer(
                domain, doc_pages, source_tag=source_tag, caveat=caveat
            )
            explainer_text = streaming_container(token_iter, container)

        with st.spinner("Harvesting screenshots…"):
            images = harvest_images(doc_pages)

        explained: List = []
        if images:
            st.divider()
            st.markdown(f"**Screenshots** ({len(images)} found — running vision analysis…)")
            explained = explain_screenshots(images)
            cols = st.columns(3)
            for i, (img, expl) in enumerate(zip(images, explained)):
                with cols[i % 3]:
                    if img.get("url"):
                        st.image(img["url"], caption=expl.screen_name_guess if expl else "")
                    if expl:
                        st.caption(expl.what_this_reveals_about_the_product[:120])
        else:
            st.caption("No suitable screenshots found.")

        st.session_state[results_key] = {
            "explainer_text": explainer_text,
            "docs_found": len(docs_urls),
            "pages_fetched": len(doc_pages),
            "probe_log": probe_log,
            "fallback_mode": fallback_mode,
            "images": images,
            "explained": explained,
            "error": None,
        }

    except Exception as exc:
        import traceback
        detail = f"{exc}\n\n{traceback.format_exc()[-800:]}"
        st.session_state[results_key] = {
            "error": detail, "docs_found": 0, "pages_fetched": 0,
            "probe_log": [], "fallback_mode": False,
            "images": [], "explained": [], "explainer_text": "",
        }
        error_card("Product deep dive error", str(exc))


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    from ui.components import inject_design_css
    inject_design_css()

    settings = {
        "force_refresh": st.session_state.get("force_refresh", False),
    }

    st.title("DeepDive Research")
    st.caption("Equity research (NVDA) · Product intelligence (sierra.ai)")

    col_input, col_btn = st.columns([5, 1])
    with col_input:
        user_input: str = st.text_input(
            label="Ticker or website",
            placeholder="NVDA  ·  or  ·  sierra.ai",
            label_visibility="collapsed",
            key="main_input",
        )
    with col_btn:
        st.write("")
        run_clicked = st.button("Research →", type="primary", use_container_width=True)

    if run_clicked and user_input.strip():
        mode, normalised = detect_mode(user_input)

        if mode == "ticker" and not validate_ticker(normalised):
            st.error(
                f"**'{normalised}'** doesn't look like a valid ticker. "
                "Use 1–5 letters for tickers, or include a dot (e.g. `sierra.ai`) for URL mode."
            )
            return

        # If the query changed, clear cached URL-mode results for the old query
        prev = st.session_state.get("active_normalised")
        if prev != normalised:
            for k in [k for k in st.session_state
                      if k.startswith(("company_intel__", "product_deep_dive__"))]:
                del st.session_state[k]

        st.session_state["active_normalised"] = normalised
        st.session_state["active_mode"] = mode

    # Render whatever is currently active — survives widget reruns via session_state
    active = st.session_state.get("active_normalised")
    if active:
        if st.session_state["active_mode"] == "ticker":
            run_ticker_mode(active, settings)
        else:
            run_url_mode(active, settings)

    # Footer
    st.divider()
    with st.expander("Session cost & cache", expanded=False):
        st.checkbox("Force refresh (bypass cache)", value=False, key="force_refresh")
        from ui.components import cost_footer
        cost_footer(st.session_state["session_id"])


if __name__ == "__main__":
    main()
