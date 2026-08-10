"""Plotly chart builders. All return fig objects; callers do st.plotly_chart(fig)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analysis.schemas import PriceData, RatioHistory, ReverseDCFResult

# ── Shared design tokens ──────────────────────────────────────────────────────
_ACCENT = "#3B4A6B"          # deep indigo
_GREEN = "#2D6A4F"
_RED = "#B5292A"
_MUTED = "#6B6B6B"
_GRID = "rgba(26,26,26,0.06)"
_BG = "#FAFAF8"
_PAPER_BG = "#FAFAF8"

_BASE_LAYOUT = dict(
    font=dict(family="'Source Sans 3', 'Inter', sans-serif", size=12, color="#1A1A1A"),
    plot_bgcolor=_BG,
    paper_bgcolor=_PAPER_BG,
    margin=dict(l=8, r=8, t=36, b=8),
    xaxis=dict(
        gridcolor=_GRID, gridwidth=1, zeroline=False,
        showline=False, tickfont=dict(size=11),
    ),
    yaxis=dict(
        gridcolor=_GRID, gridwidth=1, zeroline=False,
        showline=False, tickfont=dict(size=11),
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)", borderwidth=0,
        font=dict(size=11),
    ),
    hoverlabel=dict(
        bgcolor="white", bordercolor=_ACCENT,
        font_size=12, font_family="'Source Sans 3', sans-serif",
    ),
)


def _base_fig(**layout_kwargs) -> go.Figure:
    fig = go.Figure()
    merged = {**_BASE_LAYOUT, **layout_kwargs}
    fig.update_layout(**merged)
    return fig


def apply_base_layout(fig: go.Figure, **overrides: Any) -> go.Figure:
    """Apply _BASE_LAYOUT to fig, with per-chart overrides winning on conflict.

    Nested dict values (xaxis, yaxis, margin, legend, etc.) are deep-merged so
    base defaults and per-chart customizations both survive. Prevents the
    'multiple values for keyword argument' TypeError that occurs when the
    _BASE_LAYOUT spread and explicit kwargs contain the same key (e.g. margin).
    """
    layout: Dict[str, Any] = {}
    for k, base_v in _BASE_LAYOUT.items():
        if k in overrides and isinstance(base_v, dict) and isinstance(overrides[k], dict):
            layout[k] = {**base_v, **overrides[k]}   # override keys win in nested dicts
        else:
            layout[k] = base_v
    for k, v in overrides.items():
        if k not in layout:
            layout[k] = v
    fig.update_layout(**layout)
    return fig


# ── Price candlestick ─────────────────────────────────────────────────────────

def price_candlestick(price_data: PriceData, title: str = "") -> go.Figure:
    bars = price_data.bars
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25], vertical_spacing=0.02,
    )
    dates = [b.date for b in bars]
    fig.add_trace(go.Candlestick(
        x=dates,
        open=[b.open for b in bars], high=[b.high for b in bars],
        low=[b.low for b in bars], close=[b.close for b in bars],
        name="Price",
        increasing_line_color=_GREEN, decreasing_line_color=_RED,
        increasing_fillcolor=_GREEN, decreasing_fillcolor=_RED,
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=dates, y=[b.volume for b in bars], name="Volume",
        marker_color=f"rgba(59,74,107,0.35)",
    ), row=2, col=1)
    apply_base_layout(fig,
        title=dict(text=title or f"{price_data.ticker}", font=dict(size=13, weight=600)),
        xaxis_rangeslider_visible=False,
        height=480,
        xaxis=dict(gridcolor=_GRID, zeroline=False, showline=False),
        yaxis=dict(gridcolor=_GRID, zeroline=False, showline=False),
        xaxis2=dict(gridcolor=_GRID, zeroline=False, showline=False),
        yaxis2=dict(gridcolor=_GRID, zeroline=False, showline=False),
    )
    return fig


# ── Ratio sparkline ───────────────────────────────────────────────────────────

def ratio_sparkline(ratio: RatioHistory) -> go.Figure:
    fig = _base_fig(height=180, margin=dict(l=4, r=4, t=24, b=4), title=None)
    if ratio.history:
        fig.add_trace(go.Scatter(
            y=ratio.history, mode="lines+markers",
            line=dict(color=_ACCENT, width=2),
            marker=dict(size=5, color=_ACCENT),
            name=ratio.name,
        ))
    if ratio.current is not None:
        fig.add_hline(y=ratio.current, line_dash="dot", line_color=_RED,
                      line_width=1.5, annotation_text="Now", annotation_font_size=10)
    if ratio.median_5y is not None:
        fig.add_hline(y=ratio.median_5y, line_dash="dash", line_color=_MUTED,
                      line_width=1, annotation_text="5Y Med", annotation_font_size=10)
    return fig


# ── Sentiment trend chart ─────────────────────────────────────────────────────

def sentiment_trend_chart(quarters: List[str], scores: List[float]) -> go.Figure:
    colors = [_GREEN if s >= 0 else _RED for s in scores]
    fig = _base_fig(height=220, title=dict(text="Quarterly Sentiment", font=dict(size=13)))
    fig.add_trace(go.Bar(x=quarters, y=scores, marker_color=colors, name="Overall"))
    fig.add_hline(y=0, line_color=_MUTED, line_width=1)
    fig.update_yaxes(range=[-1, 1])
    return fig


def sentiment_mini_sparkline(
    quarters: List[str],
    prepared_scores: List[float],
    qa_scores: List[float],
) -> go.Figure:
    """Compact two-line sentiment chart for the Overview cockpit."""
    fig = _base_fig(height=140, margin=dict(l=4, r=4, t=20, b=4))
    fig.add_trace(go.Scatter(
        x=quarters, y=prepared_scores, mode="lines+markers",
        line=dict(color=_ACCENT, width=1.5),
        marker=dict(size=4), name="Prepared",
    ))
    fig.add_trace(go.Scatter(
        x=quarters, y=qa_scores, mode="lines+markers",
        line=dict(color=_RED, width=1.5, dash="dot"),
        marker=dict(size=4), name="Q&A",
    ))
    fig.add_hline(y=0, line_color=_MUTED, line_width=0.8, line_dash="dot")
    fig.update_yaxes(range=[-1, 1])
    fig.update_layout(
        legend=dict(orientation="h", y=1.1, x=0, font=dict(size=10)),
        showlegend=True,
    )
    return fig


# ── Beat/Miss chart ───────────────────────────────────────────────────────────

def beat_miss_chart(
    quarters: List[str],
    eps_est: List[Optional[float]],
    eps_actual: List[Optional[float]],
    reactions: Optional[List[Optional[float]]] = None,
) -> go.Figure:
    rows = 2 if reactions else 1
    specs = [[{}]] * rows
    titles = ["EPS: Estimate vs Actual"]
    if reactions:
        titles.append("1-Day Reaction (%)")
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        subplot_titles=titles, row_heights=[0.65, 0.35] if reactions else [1.0],
        vertical_spacing=0.12,
    )
    fig.add_trace(go.Bar(x=quarters, y=eps_est, name="Estimate",
                         marker_color=f"rgba(59,74,107,0.5)"), row=1, col=1)
    fig.add_trace(go.Bar(x=quarters, y=eps_actual, name="Actual",
                         marker_color=_ACCENT), row=1, col=1)
    if reactions:
        r_colors = [_GREEN if (r or 0) >= 0 else _RED for r in reactions]
        fig.add_trace(go.Bar(x=quarters, y=reactions, marker_color=r_colors,
                             name="Reaction %"), row=2, col=1)
    apply_base_layout(fig, height=350, barmode="group")
    return fig


# ── EPS surprise bars ────────────────────────────────────────────────────────

def eps_surprise_bars(records: List[Dict[str, Any]]) -> go.Figure:
    """
    Bar chart of EPS surprise % per quarter.
    Green = beat, red = miss. Returns the figure; callers provide the summary line.
    """
    quarters = [r["period"] for r in records]
    surprises = [r.get("eps_surprise_pct") for r in records]
    eps_est = [r.get("eps_est") for r in records]
    eps_actual = [r.get("eps_actual") for r in records]

    colors = []
    for s in surprises:
        if s is None:
            colors.append(_MUTED)
        elif s >= 0:
            colors.append(_GREEN)
        else:
            colors.append(_RED)

    custom = []
    for i in range(len(records)):
        est = eps_est[i]
        act = eps_actual[i]
        sur = surprises[i]
        custom.append((
            f"Est: ${est:.2f}" if est is not None else "Est: N/A",
            f"Act: ${act:.2f}" if act is not None else "Act: N/A",
            f"{sur:+.1f}%" if sur is not None else "N/A",
        ))

    fig = _base_fig(
        height=240,
        title=dict(text="EPS Surprise % — last 8 quarters", font=dict(size=13, weight=600)),
        margin=dict(l=8, r=8, t=40, b=8),
    )
    fig.add_trace(go.Bar(
        x=quarters,
        y=surprises,
        marker_color=colors,
        name="EPS Surprise %",
        customdata=custom,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "%{customdata[0]}<br>"
            "%{customdata[1]}<br>"
            "Surprise: %{customdata[2]}<extra></extra>"
        ),
    ))
    fig.add_hline(y=0, line_color=_MUTED, line_width=1.2)
    fig.update_yaxes(ticksuffix="%")
    return fig


# ── Revenue bars ──────────────────────────────────────────────────────────────

def revenue_bars(
    quarters: List[str],
    revenues: List[Optional[float]],
    yoy_growth: Optional[List[Optional[float]]] = None,
) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=quarters, y=revenues, name="Revenue",
                         marker_color=_ACCENT), secondary_y=False)
    if yoy_growth:
        fig.add_trace(go.Scatter(
            x=quarters, y=yoy_growth, name="YoY %",
            line=dict(color=_RED, width=2), mode="lines+markers",
            marker=dict(size=5),
        ), secondary_y=True)
    apply_base_layout(fig, title=dict(text="Quarterly Revenue", font=dict(size=13)), height=330)
    return fig


# ── Reverse DCF — filled contour heatmap ─────────────────────────────────────

def dcf_contour_heatmap(
    dcf: ReverseDCFResult,
    revenue_0: float,
) -> go.Figure:
    """
    Filled contour heatmap: x=revenue CAGR (0-20%), y=FCF margin (0-40%),
    z=implied price. Diverging colorscale centered at current price.
    Bold contour line at current price annotated 'priced in'.
    """
    from analysis.expectations import _dcf_implied_price  # intentional private import

    cagr_pcts = np.linspace(0.0, 0.20, 40)
    margin_pcts = np.linspace(0.0, 0.40, 40)

    z = []
    for m in margin_pcts:
        row = []
        for c in cagr_pcts:
            p = _dcf_implied_price(
                revenue_0, float(c), float(m),
                dcf.horizon_years, dcf.discount_rate, dcf.terminal_growth,
                dcf.net_debt, dcf.shares_outstanding,
            )
            row.append(p if p is not None else 0.0)
        z.append(row)

    z_arr = np.array(z, dtype=float)
    current = dcf.current_price

    # Build a diverging colorscale centered exactly at current_price
    z_min = max(0.0, float(z_arr.min()))
    z_max = float(z_arr.max())
    if z_max <= z_min:
        z_max = z_min + 1.0

    def _center_pos(v: float, lo: float, hi: float) -> float:
        return max(0.0, min(1.0, (v - lo) / (hi - lo)))

    mid = _center_pos(current, z_min, z_max)
    # Muted red → white → muted green, centered at mid
    colorscale = [
        [0.0,  "rgba(181,41,42,0.75)"],
        [max(0.0, mid - 0.05), "rgba(230,180,180,0.5)"],
        [mid,  "rgba(240,239,233,1)"],
        [min(1.0, mid + 0.05), "rgba(180,215,195,0.5)"],
        [1.0,  "rgba(45,106,79,0.75)"],
    ]

    # Main heatmap
    x_pcts = cagr_pcts * 100
    y_pcts = margin_pcts * 100

    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=x_pcts, y=y_pcts, z=z_arr,
        colorscale=colorscale,
        zmin=z_min, zmax=z_max,
        contours=dict(coloring="heatmap", showlabels=False),
        line=dict(width=0.3, color="rgba(0,0,0,0.1)"),
        colorbar=dict(
            title=dict(text="Implied $", side="right", font=dict(size=11)),
            tickformat="$,.0f",
            thickness=12, len=0.6, x=1.01,
        ),
        hovertemplate=(
            "<b>CAGR %{x:.1f}% + Margin %{y:.1f}%</b><br>"
            "Implied: $%{z:,.0f}<br>"
            f"vs current: %{{customdata:.1f}}%<extra></extra>"
        ),
        customdata=np.where(current > 0, (z_arr / current - 1) * 100, 0),
        name="Implied price",
    ))

    # Bold "priced in" contour at current price
    # Clamp to grid range
    contour_at = float(np.clip(current, z_min * 1.01, z_max * 0.99))
    fig.add_trace(go.Contour(
        x=x_pcts, y=y_pcts, z=z_arr,
        contours=dict(
            start=contour_at, end=contour_at, size=0.001,
            coloring="lines",
            showlabels=True,
            labelfont=dict(size=11, color="white"),
        ),
        line=dict(width=3, color=_ACCENT),
        showscale=False,
        name="Priced in",
        hoverinfo="skip",
    ))

    apply_base_layout(fig,
        height=420,
        title=dict(text=f"What growth does ${current:,.2f} require?", font=dict(size=13, weight=600)),
        xaxis=dict(title=dict(text="Revenue CAGR %", font=dict(size=11)), gridcolor=_GRID, zeroline=False, showline=False),
        yaxis=dict(title=dict(text="Terminal FCF Margin %", font=dict(size=11)), gridcolor=_GRID, zeroline=False, showline=False),
    )

    # Annotation on the bold line
    fig.add_annotation(
        x=x_pcts[len(x_pcts) // 2],
        y=y_pcts[-4],
        text=f"<b>priced in by the market (${current:,.0f})</b>",
        showarrow=False,
        font=dict(size=10, color=_ACCENT),
        bgcolor="rgba(250,250,248,0.85)",
        bordercolor=_ACCENT,
        borderwidth=1,
        borderpad=3,
    )

    return fig
