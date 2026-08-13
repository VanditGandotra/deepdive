"""Portfolio mode UI — triggered by ?view=portfolio in query params."""
from __future__ import annotations
import io
from typing import Optional

import pandas as pd
import streamlit as st

from core.portfolio import EnrichmentResult, Holding, Portfolio, enrich_with_prices
from data.portfolio_store import (
    create_portfolio, delete_portfolio, get_holdings,
    get_portfolio_id, get_portfolio_names, save_holdings,
)


def _parse_csv(uploaded) -> Optional[pd.DataFrame]:
    """
    Parse broker CSV exports. Tries to auto-map columns.
    Tolerates junk header rows by skipping rows until we find recognizable headers.
    """
    content = uploaded.read().decode("utf-8", errors="replace")
    lines = content.splitlines()

    # Find header row (contains "ticker" or "symbol" or "shares")
    header_idx = 0
    for i, line in enumerate(lines):
        lower = line.lower()
        if any(kw in lower for kw in ["ticker", "symbol", "shares", "quantity"]):
            header_idx = i
            break

    df = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])))
    df.columns = [c.strip() for c in df.columns]

    # Column mapping
    col_map = {}
    for col in df.columns:
        lower = col.lower().replace(" ", "_").replace("-", "_")
        if any(k in lower for k in ["ticker", "symbol", "cusip"]):
            col_map["Ticker"] = col
        elif any(k in lower for k in ["shares", "quantity", "qty"]):
            col_map["Shares"] = col
        elif any(k in lower for k in ["cost", "price", "basis", "avg"]):
            col_map["Cost Basis"] = col
        elif any(k in lower for k in ["account", "acct"]):
            col_map["Account"] = col

    if "Ticker" not in col_map or "Shares" not in col_map:
        raise ValueError(f"Cannot find Ticker and Shares columns. Found: {list(df.columns)}")

    result = pd.DataFrame()
    result["Ticker"] = df[col_map["Ticker"]].astype(str).str.strip().str.upper()
    result["Shares"] = pd.to_numeric(df[col_map["Shares"]], errors="coerce").fillna(0)
    result["Cost Basis"] = (
        pd.to_numeric(df[col_map["Cost Basis"]], errors="coerce")
        if "Cost Basis" in col_map
        else None
    )
    result["Account"] = df[col_map["Account"]].astype(str) if "Account" in col_map else ""
    result["Notes"] = ""
    result["Cash"] = result["Ticker"].str.contains(r"CASH|USD|MMF", regex=True, na=False)

    return result.dropna(subset=["Ticker"])


def _save_from_df(portfolio_id: int, df: pd.DataFrame) -> None:
    holdings = []
    for _, row in df.iterrows():
        ticker = str(row.get("Ticker", "")).strip().upper()
        if not ticker:
            continue
        cost_val = row.get("Cost Basis")
        holdings.append({
            "ticker": ticker,
            "shares": float(row.get("Shares", 0) or 0),
            "cost_basis": float(cost_val) if pd.notna(cost_val) and cost_val is not None else None,
            "account": str(row.get("Account", "")).strip() or None,
            "notes": str(row.get("Notes", "")).strip() or None,
            "is_cash": bool(row.get("Cash", False)),
        })
    save_holdings(portfolio_id, holdings)


def _build_portfolio(portfolio_id: int, name: str) -> Portfolio:
    raw = get_holdings(portfolio_id)
    holdings = [Holding(
        ticker=h["ticker"],
        shares=h["shares"],
        cost_basis=h["cost_basis"],
        account=h["account"],
        notes=h["notes"],
        is_cash=bool(h["is_cash"]),
    ) for h in raw]
    pf = Portfolio(name=name, holdings=holdings)
    result: EnrichmentResult = enrich_with_prices(pf)
    if result.failed:
        st.warning(f"Could not fetch prices for: {', '.join(result.failed)}. Their market values show as $0.")
    return result.portfolio


def _render_portfolio_editor(portfolio_id: int, portfolio_name: str) -> Optional[Portfolio]:
    """Show st.data_editor for holdings. Returns Portfolio if user saved & analyzed, None otherwise."""
    raw = get_holdings(portfolio_id)

    default_cols = {
        "Ticker": pd.Series(dtype="str"),
        "Shares": pd.Series(dtype="float"),
        "Cost Basis": pd.Series(dtype="float"),
        "Account": pd.Series(dtype="str"),
        "Notes": pd.Series(dtype="str"),
        "Cash": pd.Series(dtype="bool"),
    }

    if raw:
        df = pd.DataFrame([{
            "Ticker": h["ticker"],
            "Shares": h["shares"],
            "Cost Basis": h["cost_basis"],
            "Account": h["account"] or "",
            "Notes": h["notes"] or "",
            "Cash": bool(h["is_cash"]),
        } for h in raw])
    else:
        df = pd.DataFrame(default_cols)
        # Add a cash row by default
        df = pd.concat([df, pd.DataFrame([{
            "Ticker": "CASH", "Shares": 0.0, "Cost Basis": None,
            "Account": "", "Notes": "", "Cash": True
        }])], ignore_index=True)

    edited = st.data_editor(
        df,
        num_rows="dynamic",
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", max_chars=5),
            "Shares": st.column_config.NumberColumn("Shares", min_value=0),
            "Cost Basis": st.column_config.NumberColumn("Cost Basis ($/share)", min_value=0),
            "Account": st.column_config.TextColumn("Account"),
            "Notes": st.column_config.TextColumn("Notes"),
            "Cash": st.column_config.CheckboxColumn("Cash"),
        },
        use_container_width=True,
        key=f"holdings_editor_{portfolio_id}",
    )

    # CSV upload
    uploaded = st.file_uploader("Or upload CSV (Ticker, Shares, Cost Basis, Account)", type="csv")
    if uploaded:
        try:
            df_upload = _parse_csv(uploaded)
            if df_upload is not None:
                st.markdown("**Preview (first 10 rows):**")
                st.dataframe(df_upload.head(10), hide_index=True)
                if st.button("Use this CSV", key="use_csv"):
                    edited = df_upload
        except Exception as exc:
            st.error(f"CSV parse error: {exc}")

    col_save, col_analyze, col_delete = st.columns([2, 2, 1])
    with col_save:
        if st.button("Save", key="save_holdings"):
            _save_from_df(portfolio_id, edited)
            st.success("Saved!")
            st.rerun()
    with col_analyze:
        analyze_clicked = st.button("Save & Analyze ->", type="primary", key="save_analyze")
        if analyze_clicked:
            _save_from_df(portfolio_id, edited)
            with st.spinner("Fetching market data..."):
                return _build_portfolio(portfolio_id, portfolio_name)
    with col_delete:
        if st.button("Delete portfolio", key="del_pf"):
            delete_portfolio(portfolio_id)
            for k in list(st.session_state.keys()):
                if k.startswith("_pf"):
                    del st.session_state[k]
            st.rerun()

    return None


def _render_portfolio_analysis(portfolio: Portfolio) -> None:
    st.markdown(f"### {portfolio.name}")
    st.caption(f"Total value: ${portfolio.total_value:,.0f}")

    # Main holdings table
    df = portfolio.to_dataframe()

    def _fmt(col, df):
        if col in ("Market Value", "Unrealized P&L"):
            return df[col].map(lambda x: f"${x:,.0f}" if pd.notna(x) else "—")
        if col in ("Unrealized P&L %", "Weight"):
            return df[col].map(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")
        if col == "Price":
            return df[col].map(lambda x: f"${x:,.2f}" if pd.notna(x) else "—")
        if col == "Cost Basis":
            return df[col].map(lambda x: f"${x:,.2f}" if pd.notna(x) else "—")
        return df[col]

    display_df = df.copy()
    for col in ["Market Value", "Unrealized P&L", "Unrealized P&L %", "Weight", "Price", "Cost Basis"]:
        if col in display_df.columns:
            display_df[col] = _fmt(col, df)

    st.dataframe(display_df, hide_index=True, use_container_width=True)

    # Sector concentration
    st.subheader("Sector Concentration")
    sector_conc = portfolio.sector_concentration()
    if sector_conc:
        try:
            import plotly.graph_objects as go
            fig = go.Figure(go.Pie(
                labels=list(sector_conc.keys()),
                values=[v * 100 for v in sector_conc.values()],
                textinfo="label+percent",
                hole=0.4,
            ))
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            for sector, weight in sector_conc.items():
                st.text(f"{sector}: {weight*100:.1f}%")

    # Launch full analysis
    st.divider()
    if st.button("Run Full Analysis (Bull/Base/Bear + Optimizer) ->", type="primary"):
        tickers = [h.ticker for h in portfolio.holdings if not h.is_cash]
        st.query_params["view"] = "portfolio_analysis"
        st.query_params["portfolio"] = portfolio.name
        st.query_params["tickers"] = ",".join(tickers)
        st.rerun()

    # Disclaimer
    st.caption("Research tooling only — not investment advice.")


def render_portfolio_page() -> None:
    st.title("Portfolio")

    names = get_portfolio_names()
    new_mode = st.session_state.get("_pf_new_mode", False)

    if not names or new_mode:
        st.subheader("Create portfolio")
        with st.form("new_pf_form"):
            new_name = st.text_input("Portfolio name", placeholder="My Portfolio")
            submitted = st.form_submit_button("Create")
            if submitted and new_name.strip():
                create_portfolio(new_name.strip())
                st.session_state["_pf_new_mode"] = False
                st.session_state["_pf_selected"] = new_name.strip()
                st.rerun()
        if names:
            if st.button("Cancel"):
                st.session_state["_pf_new_mode"] = False
                st.rerun()
        return

    # Portfolio selector
    selected_name = st.session_state.get("_pf_selected") or names[0]
    if selected_name not in names:
        selected_name = names[0]

    col_sel, col_new = st.columns([4, 1])
    with col_sel:
        selected_name = st.selectbox(
            "Portfolio", names, index=names.index(selected_name), key="pf_selector"
        )
        st.session_state["_pf_selected"] = selected_name
    with col_new:
        st.write("")
        if st.button("+ New", key="new_pf_btn"):
            st.session_state["_pf_new_mode"] = True
            st.rerun()

    portfolio_id = get_portfolio_id(selected_name)
    if portfolio_id is None:
        st.error(f"Portfolio '{selected_name}' not found.")
        return

    # Check if we have an analyzed portfolio in session state
    analyzed_key = f"_pf_analyzed_{selected_name}"

    portfolio = _render_portfolio_editor(portfolio_id, selected_name)
    if portfolio is not None:
        st.session_state[analyzed_key] = portfolio

    analyzed = st.session_state.get(analyzed_key)
    if analyzed:
        st.divider()
        _render_portfolio_analysis(analyzed)

    # Back to single-ticker link
    st.divider()
    if st.button("<- Single ticker mode"):
        st.query_params.clear()
        st.rerun()
