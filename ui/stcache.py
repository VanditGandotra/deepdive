"""Streamlit in-memory cache layer — @st.cache_data wrappers for expensive data fetches.

Eliminates redundant SQLite hits and re-computation within a Streamlit session.
TTLs match the underlying SQLite TTLs so both caches age together.
"""
from __future__ import annotations

import streamlit as st


# ── Market data ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=86_400, show_spinner=False)
def cached_fundamentals(ticker: str):
    from data.market import get_fundamentals
    return get_fundamentals(ticker)


@st.cache_data(ttl=3_600, show_spinner=False)
def cached_prices(ticker: str, period: str = "5y"):
    from data.market import get_prices
    return get_prices(ticker, period)


@st.cache_data(ttl=3_600, show_spinner=False)
def cached_short_interest(ticker: str):
    from data.market import get_short_interest
    return get_short_interest(ticker)


@st.cache_data(ttl=86_400, show_spinner=False)
def cached_beat_miss(ticker: str):
    from data.market import get_beat_miss_history
    return get_beat_miss_history(ticker)


@st.cache_data(ttl=86_400, show_spinner=False)
def cached_estimates(ticker: str):
    from data.market import get_estimates
    return get_estimates(ticker)


# ── Analysis ──────────────────────────────────────────────────────────────────

@st.cache_data(ttl=86_400, show_spinner=False)
def cached_quality_panel(ticker: str):
    from analysis.quality import get_quality_panel
    return get_quality_panel(ticker)


@st.cache_data(ttl=21_600, show_spinner=False)
def cached_calls(ticker: str, n: int = 4, cache_version: int = 2):
    # cache_version busts stale pickled CallSummary objects from before the signals field existed
    from analysis.calls import analyse_all_calls
    return analyse_all_calls(ticker, n=n)


@st.cache_data(ttl=86_400, show_spinner=False)
def cached_ratio_groups(ticker: str):
    from analysis.ratios import get_ratio_groups
    return get_ratio_groups(ticker)


@st.cache_data(ttl=21_600, show_spinner=False)
def cached_headlines(ticker: str, company_name: str):
    from analysis.news_impact import classify_headlines
    return classify_headlines(ticker, company_name)


@st.cache_data(ttl=86_400, show_spinner=False)
def cached_reverse_dcf(
    ticker: str,
    discount_rate: float = 0.10,
    terminal_growth: float = 0.025,
    horizon_years: int = 10,
):
    from analysis.expectations import reverse_dcf
    return reverse_dcf(
        ticker,
        discount_rate=discount_rate,
        terminal_growth=terminal_growth,
        horizon_years=horizon_years,
    )


@st.cache_data(ttl=3_600, show_spinner=False)
def cached_positioning(ticker: str):
    from analysis.positioning import get_positioning
    return get_positioning(ticker)


@st.cache_data(ttl=21_600, show_spinner=False)
def cached_kpis(ticker: str):
    from analysis.kpis import extract_kpis
    call_data = cached_calls(ticker, n=4)
    return extract_kpis(ticker, call_data) if call_data.get("transcripts") else []
