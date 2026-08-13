from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from core.schemas import Driver, Scenario, Source, TickerAnalysis
from core.cache import get_cached_analysis, save_analysis, ANALYSIS_VERSION

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = {
    "discount_rate": 0.10,
    "terminal_growth": 0.025,
    "horizon_years": 10,
    "n_calls": 4,
}


def analyze_ticker(
    ticker: str,
    config: Optional[Dict] = None,
    as_of: Optional[date] = None,
) -> TickerAnalysis:
    """Full headless analysis. No st.* calls. Uses SQLite cache."""
    ticker = ticker.upper().strip()
    cfg = {**_DEFAULT_CONFIG, **(config or {})}
    today = as_of or date.today()

    # Cache check
    cached = get_cached_analysis(ticker, today, ANALYSIS_VERSION)
    if cached is not None:
        return cached

    data_gaps: List[str] = []
    sources: List[Source] = []
    success_count = 0
    total_count = 0

    # --- Parallel fetch ---
    from data.market import get_fundamentals
    from analysis.quality import get_quality_panel
    from analysis.positioning import get_positioning
    from analysis.calls import analyse_all_calls
    from analysis.expectations import reverse_dcf
    from analysis.news_impact import classify_headlines

    results: Dict[str, Any] = {}

    def _run(name, fn, *args, **kwargs):
        try:
            return name, fn(*args, **kwargs), None
        except Exception as exc:
            logger.warning("analyze_ticker [%s] failed for %s: %s", name, ticker, exc)
            return name, None, str(exc)

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {
            pool.submit(_run, "fund", get_fundamentals, ticker): "fund",
            pool.submit(_run, "quality", get_quality_panel, ticker): "quality",
            pool.submit(_run, "positioning", get_positioning, ticker): "positioning",
            pool.submit(_run, "calls", analyse_all_calls, ticker, cfg["n_calls"]): "calls",
            pool.submit(_run, "dcf", reverse_dcf, ticker,
                        cfg["discount_rate"], cfg["terminal_growth"], cfg["horizon_years"]): "dcf",
        }
        for fut in as_completed(futures):
            name, val, err = fut.result()
            total_count += 1
            results[name] = val
            if err:
                data_gaps.append(f"{name}: {err}")
            else:
                success_count += 1

    fund = results.get("fund")
    call_data = results.get("calls") or {}

    # Headlines needs fund.name
    company_name = (fund.name if fund else None) or ticker
    try:
        headlines = classify_headlines(ticker, company_name)
        total_count += 1
        success_count += 1
    except Exception as exc:
        headlines = []
        data_gaps.append(f"headlines: {exc}")
        total_count += 1

    # Sequential (depend on call_data)
    from analysis.calls import synthesize_calls
    from analysis.kpis import extract_kpis

    call_delta = None
    kpis = []
    transcripts = call_data.get("transcripts", [])
    summaries = call_data.get("summaries", [])
    sentiments = call_data.get("sentiments", [])

    if summaries:
        try:
            call_delta = synthesize_calls(summaries, sentiments, ticker)
            total_count += 1
            success_count += 1
        except Exception as exc:
            data_gaps.append(f"call_synthesis: {exc}")
            total_count += 1
    else:
        data_gaps.append("no transcripts for call synthesis")

    if transcripts:
        try:
            kpis = extract_kpis(ticker, call_data)
            total_count += 1
            success_count += 1
        except Exception as exc:
            data_gaps.append(f"kpis: {exc}")
            total_count += 1
    else:
        data_gaps.append("no transcripts for KPI extraction")

    # Theses
    from analysis.thesis import get_structured_theses
    from analysis.news_impact import high_materiality

    theses = []
    if fund:
        high_mat = [f"{h.title} ({h.one_line_why})" for h in high_materiality(headlines)] if headlines else []
        kpi_sums = [f"{k.kpi_name}: {k.trend_note}" for k in kpis] if kpis else []
        try:
            theses = get_structured_theses(
                ticker, fund,
                dcf=results.get("dcf"),
                call_delta=call_delta,
                quality=results.get("quality"),
                positioning=results.get("positioning"),
                high_mat_news=high_mat,
                kpi_summaries=kpi_sums,
            )
            total_count += 1
            success_count += 1
        except Exception as exc:
            data_gaps.append(f"theses: {exc}")
            total_count += 1
    else:
        data_gaps.append("no fundamentals — theses skipped")

    # Convert Thesis → Scenario
    scenarios = []
    for t in theses:
        drivers = [Driver(name=d, value=0.0, unit="", direction="neutral") for d in (t.key_drivers or [])]
        scenarios.append(Scenario(
            scenario=t.scenario,
            price_target=t.implied_price_12mo,
            probability=t.probability_weight,
            horizon_years=1,
            drivers=drivers,
            narrative=t.narrative,
        ))

    # Sources
    sources.append(Source(label="yfinance fundamentals", url=None, retrieved_at=datetime.utcnow()))
    if transcripts:
        sources.append(Source(
            label=f"earnings transcripts ({len(transcripts)} quarters)",
            retrieved_at=datetime.utcnow(),
        ))

    # Flat fundamentals
    fcf_yield = None
    if fund and fund.fcf_ttm and fund.market_cap and fund.market_cap > 0:
        fcf_yield = fund.fcf_ttm / fund.market_cap

    analysis = TickerAnalysis(
        ticker=ticker,
        company_name=company_name,
        as_of=today,
        analysis_version=ANALYSIS_VERSION,
        current_price=fund.current_price if fund else None,
        market_cap=fund.market_cap if fund else None,
        sector=fund.sector if fund else None,
        industry=fund.industry if fund else None,
        pe_ttm=fund.pe_ttm if fund else None,
        pe_forward=fund.pe_forward if fund else None,
        ev_ebitda=fund.ev_ebitda if fund else None,
        price_to_sales=fund.price_to_sales if fund else None,
        gross_margin=fund.gross_margin if fund else None,
        net_margin=fund.net_margin if fund else None,
        revenue_growth_yoy=fund.revenue_growth_yoy if fund else None,
        fcf_yield=fcf_yield,
        debt_to_equity=fund.debt_to_equity if fund else None,
        roe=fund.roe if fund else None,
        fundamentals=fund,
        dcf=results.get("dcf"),
        call_delta=call_delta,
        quality=results.get("quality"),
        positioning=results.get("positioning"),
        headlines=headlines,
        kpis=kpis,
        scenarios=scenarios,
        confidence=round(success_count / max(total_count, 1), 2),
        data_gaps=data_gaps,
        sources=sources,
    )

    save_analysis(analysis)
    return analysis
