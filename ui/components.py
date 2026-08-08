"""Reusable Streamlit UI components: freshness badges, citation badges, delta card, cost footer."""
from __future__ import annotations

from typing import Any, Dict, Iterator, Optional

import streamlit as st

from data.cache import get_freshness, get_session_stats


# ═══════════════════════════════════════════════════════════════════════════════
# Design system CSS — injected once per page load
# ═══════════════════════════════════════════════════════════════════════════════

_CSS_INJECTED = False

_DESIGN_CSS = """
<style>
/* Typography scale */
h1 { font-size: 1.5rem !important; font-weight: 700 !important; letter-spacing: -0.02em; }
h2 { font-size: 1.1rem !important; font-weight: 600 !important; letter-spacing: -0.01em; }
h3 { font-size: 1rem !important; font-weight: 600 !important; }

/* Metric card component */
.dd-metric-card {
    padding: 0.75rem 1rem;
    background: var(--secondary-background-color);
    border-radius: 6px;
    border: 1px solid rgba(26,26,26,0.08);
}
.dd-metric-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #6B6B6B;
    margin-bottom: 0.2rem;
}
.dd-metric-value {
    font-size: 1.3rem;
    font-weight: 600;
    color: #1A1A1A;
    line-height: 1.2;
}
.dd-metric-value.positive { color: #2D6A4F; }
.dd-metric-value.negative { color: #B5292A; }
.dd-metric-context {
    font-size: 0.72rem;
    color: #6B6B6B;
    margin-top: 0.25rem;
}

/* 52-week range bar */
.dd-range-bar-track {
    height: 4px;
    background: rgba(59,74,107,0.15);
    border-radius: 2px;
    position: relative;
    margin: 4px 0;
}
.dd-range-bar-fill {
    height: 100%;
    background: #3B4A6B;
    border-radius: 2px;
    position: absolute;
    left: 0;
}
.dd-range-bar-dot {
    width: 10px;
    height: 10px;
    background: #3B4A6B;
    border: 2px solid #FAFAF8;
    border-radius: 50%;
    position: absolute;
    top: -3px;
    transform: translateX(-50%);
    box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}

/* Header strip */
.dd-header-strip {
    display: flex;
    gap: 2rem;
    align-items: baseline;
    padding: 0.75rem 0;
    border-bottom: 1px solid rgba(26,26,26,0.08);
    margin-bottom: 1rem;
}
.dd-price { font-size: 2rem; font-weight: 700; letter-spacing: -0.03em; }
.dd-change-pos { color: #2D6A4F; font-weight: 600; }
.dd-change-neg { color: #B5292A; font-weight: 600; }
.dd-header-label { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.06em; color: #6B6B6B; }
.dd-header-val { font-size: 0.9rem; font-weight: 600; }

/* Freshness badge */
.dd-freshness { font-size: 0.68rem; color: #9B9B9B; text-align: right; }

/* Bullet consensus bar */
.dd-consensus-bar {
    height: 6px;
    border-radius: 3px;
    display: flex;
    overflow: hidden;
    gap: 1px;
}
.dd-consensus-buy { background: #2D6A4F; }
.dd-consensus-hold { background: #9B7D3A; }
.dd-consensus-sell { background: #B5292A; }

/* Analyst target bar */
.dd-target-track {
    height: 4px;
    background: rgba(59,74,107,0.1);
    border-radius: 2px;
    position: relative;
    margin: 4px 0;
}
.dd-target-fill {
    height: 100%;
    background: linear-gradient(90deg, #B5292A 0%, #9B7D3A 40%, #2D6A4F 100%);
    border-radius: 2px;
    position: absolute;
}
.dd-target-dot {
    width: 8px; height: 8px;
    background: #3B4A6B;
    border: 2px solid #FAFAF8;
    border-radius: 50%;
    position: absolute;
    top: -2px;
    transform: translateX(-50%);
}

/* Section headers — override st.subheader sizing */
[data-testid="stMarkdownContainer"] h2 {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    text-transform: none;
    margin-top: 1.25rem;
    margin-bottom: 0.5rem;
}

/* Divider */
hr { border-color: rgba(26,26,26,0.08) !important; margin: 0.75rem 0 !important; }
</style>
"""


def inject_design_css() -> None:
    global _CSS_INJECTED
    if not _CSS_INJECTED:
        st.html(_DESIGN_CSS)
        _CSS_INJECTED = True


# ═══════════════════════════════════════════════════════════════════════════════
# Number formatter — single source of truth
# ═══════════════════════════════════════════════════════════════════════════════

def fmt_money(val: Optional[float], decimals: int = 2) -> str:
    """$4.32T / $128.8B / $38.71 / $0 — negative in parens."""
    if val is None:
        return "N/A"
    neg = val < 0
    v = abs(val)
    if v >= 1e12:
        s = f"${v/1e12:.{max(0,decimals-1)}f}T"
    elif v >= 1e9:
        s = f"${v/1e9:.1f}B"
    elif v >= 1e6:
        s = f"${v/1e6:.1f}M"
    else:
        s = f"${v:,.{decimals}f}"
    return f"({s})" if neg else s


def fmt_pct(val: Optional[float], decimals: int = 1) -> str:
    """23.4% / (5.2%) — one decimal by default."""
    if val is None:
        return "N/A"
    neg = val < 0
    s = f"{abs(val)*100:.{decimals}f}%"
    return f"({s})" if neg else s


def fmt_ratio(val: Optional[float], decimals: int = 1) -> str:
    """25.3x — one decimal."""
    if val is None:
        return "N/A"
    neg = val < 0
    s = f"{abs(val):.{decimals}f}x"
    return f"({s})" if neg else s


def fmt_number(val: Optional[float], decimals: int = 1) -> str:
    """Generic number with thousands separator."""
    if val is None:
        return "N/A"
    return f"{val:,.{decimals}f}"


# ═══════════════════════════════════════════════════════════════════════════════
# Metric card component
# ═══════════════════════════════════════════════════════════════════════════════

def metric_card(
    label: str,
    value: str,
    context: str = "",
    positive: Optional[bool] = None,
) -> None:
    """Render a styled metric card: label (muted upper), value (large), context (muted small)."""
    val_class = "dd-metric-value"
    if positive is True:
        val_class += " positive"
    elif positive is False:
        val_class += " negative"
    ctx_html = f'<div class="dd-metric-context">{context}</div>' if context else ""
    st.html(f"""
<div class="dd-metric-card">
  <div class="dd-metric-label">{label}</div>
  <div class="{val_class}">{value}</div>
  {ctx_html}
</div>
""")


# ═══════════════════════════════════════════════════════════════════════════════
# 52-week range bar
# ═══════════════════════════════════════════════════════════════════════════════

def week_52_bar(current: float, low: float, high: float) -> None:
    if high <= low:
        return
    pct = max(0.0, min(1.0, (current - low) / (high - low))) * 100
    st.html(f"""
<div style="font-size:0.68rem;color:#6B6B6B;display:flex;justify-content:space-between;margin-bottom:1px">
  <span>${low:,.2f}</span><span style="font-weight:600">{pct:.0f}% of 52-wk range</span><span>${high:,.2f}</span>
</div>
<div class="dd-range-bar-track">
  <div class="dd-range-bar-fill" style="width:{pct}%"></div>
  <div class="dd-range-bar-dot" style="left:{pct}%"></div>
</div>
""")


# ═══════════════════════════════════════════════════════════════════════════════
# Analyst target bar
# ═══════════════════════════════════════════════════════════════════════════════

def analyst_target_bar(current: float, low: float, high: float, mean: float) -> None:
    if high <= low:
        return
    pct_current = max(0.0, min(1.0, (current - low) / (high - low))) * 100
    pct_mean = max(0.0, min(1.0, (mean - low) / (high - low))) * 100
    upside = (mean / current - 1) * 100
    up_str = f"+{upside:.0f}%" if upside >= 0 else f"{upside:.0f}%"
    st.html(f"""
<div style="font-size:0.68rem;color:#6B6B6B;display:flex;justify-content:space-between;margin-bottom:1px">
  <span>${low:,.0f} low</span>
  <span style="font-weight:600">{up_str} to mean (${mean:,.0f})</span>
  <span>${high:,.0f} high</span>
</div>
<div class="dd-target-track">
  <div class="dd-target-fill" style="width:100%"></div>
  <div class="dd-target-dot" style="left:{pct_current}%" title="Current ${current:,.2f}"></div>
  <div class="dd-target-dot" style="left:{pct_mean}%;background:#9B7D3A" title="Mean target ${mean:,.0f}"></div>
</div>
""")


# ═══════════════════════════════════════════════════════════════════════════════
# Analyst consensus bar
# ═══════════════════════════════════════════════════════════════════════════════

def analyst_consensus_bar(buys: int, holds: int, sells: int) -> None:
    total = buys + holds + sells
    if total == 0:
        st.caption("No analyst ratings")
        return
    bp = buys / total * 100
    hp = holds / total * 100
    sp = sells / total * 100
    st.html(f"""
<div class="dd-consensus-bar">
  <div class="dd-consensus-buy" style="flex:{bp}"></div>
  <div class="dd-consensus-hold" style="flex:{hp}"></div>
  <div class="dd-consensus-sell" style="flex:{sp}"></div>
</div>
<div style="font-size:0.7rem;color:#6B6B6B;display:flex;gap:1rem;margin-top:3px">
  <span style="color:#2D6A4F;font-weight:600">{buys} Buy</span>
  <span style="color:#9B7D3A">{holds} Hold</span>
  <span style="color:#B5292A">{sells} Sell</span>
</div>
""")


# ═══════════════════════════════════════════════════════════════════════════════
# Freshness / misc
# ═══════════════════════════════════════════════════════════════════════════════

def freshness_badge(cache_key: str, label: str = "") -> None:
    info = get_freshness(cache_key)
    if not info:
        return
    badge = f"{label + ' · ' if label else ''}as of {info['age_description']}"
    if info["is_stale"]:
        st.warning(badge + " stale", icon=None)
    else:
        st.caption(f"<span class='dd-freshness'>{badge}</span>", unsafe_allow_html=True)


def citation_badge(tag: str, chunks: Dict[str, str], tooltip_max: int = 200) -> str:
    chunk = chunks.get(tag, "")
    preview = chunk[:tooltip_max].replace('"', "&quot;")
    return (
        f'<sup title="{preview}" style="'
        f'background:#3B4A6B;color:#fff;border-radius:3px;'
        f'padding:1px 4px;font-size:0.7em;cursor:help">{tag}</sup>'
    )


def delta_card(delta: Optional[Dict[str, Any]]) -> None:
    if delta is None:
        return
    since = delta.get("_run_at", "unknown date")
    bullets = delta.get("narrative_bullets") or []
    with st.container(border=True):
        st.markdown(f"**Since your last run** ({since})")
        if not bullets:
            st.caption("No material changes detected.")
        else:
            for b in bullets:
                st.markdown(f"- {b}")


def streaming_container(
    token_iter: Iterator[str],
    placeholder: Optional[Any] = None,
) -> str:
    container = placeholder or st.empty()
    full_text = ""
    for token in token_iter:
        full_text += token
        container.markdown(full_text + "▌")
    container.markdown(full_text)
    return full_text


def cost_footer(session_id: str) -> None:
    stats = get_session_stats(session_id)
    if stats["total_calls"] == 0:
        return

    total = stats["total_calls"]
    cached = stats["cached_calls"]
    hit_rate = (cached / total * 100) if total else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("LLM calls", total)
    col2.metric("Cache hit rate", f"{hit_rate:.0f}%")
    col3.metric(
        "Tokens",
        f"{(stats['tokens_input'] + stats['tokens_output']):,}",
        help=f"In: {stats['tokens_input']:,}  Out: {stats['tokens_output']:,}",
    )
    col4.metric("Est. cost", f"${stats['cost_est_usd']:.4f}")

    if stats.get("by_model"):
        with st.expander("Cost by model"):
            for model, mstats in stats["by_model"].items():
                st.caption(f"**{model}**: {mstats['calls']} calls · ${mstats['cost']:.4f}")


def error_card(title: str, detail: str, tab_only: bool = True) -> None:
    with st.container(border=True):
        st.error(f"**{title}**")
        st.caption(detail)
        if tab_only:
            st.caption("Other tabs are unaffected.")


def unavailable_tab(source: str, reason: str) -> None:
    st.warning(f"**{source} unavailable**")
    st.caption(reason)
    st.caption("Freshness badges for other tabs are still accurate.")
