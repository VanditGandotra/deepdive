"""yfinance wrapper: prices, fundamentals, estimates, insiders, holders, short interest, news."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import yfinance as yf
import pandas as pd
import requests
from requests.adapters import HTTPAdapter

from config import TTL_FUNDAMENTALS, TTL_NEWS, TTL_PRICES
from data.cache import (get_cache_obj, record_freshness, set_cache_obj)
from data.resilience import retry, SourceUnavailable
from analysis.schemas import (
    AnalystTarget, Fundamentals, InsiderTransaction,
    InstitutionalHolder, NewsItem, PriceBar, PriceData, ShortInterest,
)

logger = logging.getLogger(__name__)

_PROVIDER_HEALTH: Dict[str, Any] = {}


def _yf_session() -> "requests.Session":
    """requests.Session with desktop User-Agent and 20s read timeout for every call."""
    session = requests.Session()
    session.headers["User-Agent"] = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    )
    adapter = HTTPAdapter()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    _orig_send = session.send
    def _send_with_timeout(*args, **kwargs):
        kwargs.setdefault("timeout", 20)
        return _orig_send(*args, **kwargs)
    session.send = _send_with_timeout  # type: ignore[method-assign]
    return session


def _check_info(info: Dict[str, Any], ticker: str) -> None:
    """Raise SourceUnavailable when Yahoo Finance returns a sparse dict (IP rate-limited)."""
    n = sum(1 for v in info.values() if v is not None)
    if n < 5:
        _PROVIDER_HEALTH[ticker] = {
            "status": "rate_limited",
            "source": "yfinance/yahoo",
            "detail": f"returned only {n} non-null fields — IP is rate-limited",
        }
        raise SourceUnavailable(
            f"Yahoo Finance returned only {n} fields for {ticker}. "
            "Shared cloud IPs are often rate-limited — try refreshing in ~30 seconds."
        )
    _PROVIDER_HEALTH[ticker] = {"status": "ok", "source": "yfinance/yahoo", "fields": n}


def get_provider_health() -> Dict[str, Any]:
    """Return the last-recorded market-data provider status per ticker."""
    return dict(_PROVIDER_HEALTH)


def _safe_float(val: Any) -> Optional[float]:
    try:
        f = float(val)
        return None if (f != f) else f  # NaN check
    except (TypeError, ValueError):
        return None


def _safe_int(val: Any) -> Optional[int]:
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


@retry(max_attempts=3, base_delay=2.0)
def get_prices(ticker: str, period: str = "5y") -> PriceData:
    cache_key = f"market:{ticker}:prices:{period}"
    cached = get_cache_obj(cache_key)
    if cached:
        return PriceData.model_validate(cached)

    yf_ticker = yf.Ticker(ticker, session=_yf_session())
    hist = yf_ticker.history(period=period, auto_adjust=True)
    if hist.empty:
        raise ValueError(f"No price data returned for {ticker}")

    bars = []
    for dt, row in hist.iterrows():
        bars.append(PriceBar(
            date=dt.date(),
            open=float(row["Open"]),
            high=float(row["High"]),
            low=float(row["Low"]),
            close=float(row["Close"]),
            volume=int(row.get("Volume", 0)),
        ))

    info = yf_ticker.fast_info
    currency = getattr(info, "currency", "USD") or "USD"
    result = PriceData(ticker=ticker, currency=currency, bars=bars)
    set_cache_obj(cache_key, result.model_dump(mode="json"), TTL_PRICES, source="yfinance")
    record_freshness(cache_key, "yfinance", TTL_PRICES)
    return result


@retry(max_attempts=3, base_delay=2.0)
def get_fundamentals(ticker: str) -> Fundamentals:
    cache_key = f"market:{ticker}:fundamentals"
    cached = get_cache_obj(cache_key)
    if cached:
        return Fundamentals.model_validate(cached)

    yf_ticker = yf.Ticker(ticker, session=_yf_session())
    info: Dict[str, Any] = yf_ticker.info or {}
    _check_info(info, ticker)

    # Fetch income statement + cashflow once; used for FCF, net_margin, interest_coverage
    fin: Optional[pd.DataFrame] = None
    cf: Optional[pd.DataFrame] = None
    try:
        fin = yf_ticker.financials
        if fin is not None and fin.empty:
            fin = None
    except Exception:
        pass
    try:
        cf = yf_ticker.cashflow
        if cf is not None and cf.empty:
            cf = None
    except Exception:
        pass

    # Derive FCF from most recent annual statement
    fcf = None
    try:
        if cf is not None:
            ocf_row = next((r for r in cf.index if "operating" in r.lower()), None)
            cap_row = next((r for r in cf.index if "capital" in r.lower() and "expenditure" in r.lower()), None)
            if ocf_row and cap_row:
                ocf_val = _safe_float(cf.loc[ocf_row].iloc[0]) or 0
                cap_val = _safe_float(cf.loc[cap_row].iloc[0]) or 0
                fcf = (ocf_val + cap_val) if cap_val < 0 else (ocf_val - cap_val)
    except Exception:
        pass

    # Net margin from most recent annual filing (avoids TTM pollution by one-time items)
    net_margin_annual: Optional[float] = None
    try:
        if fin is not None:
            rev_row = next((r for r in fin.index if r.lower() == "total revenue"), None)
            ni_row = next(
                (r for r in fin.index if r in ("Net Income", "Net Income Common Stockholders")),
                None,
            )
            if rev_row and ni_row:
                rev_v = _safe_float(fin.loc[rev_row].iloc[0])
                ni_v = _safe_float(fin.loc[ni_row].iloc[0])
                if rev_v and ni_v and rev_v > 0:
                    net_margin_annual = ni_v / rev_v
    except Exception:
        pass

    # Interest coverage = EBIT / interest expense (most recent annual)
    interest_coverage_calc: Optional[float] = None
    try:
        if fin is not None:
            ebit_row = next((r for r in fin.index if r in ("Operating Income", "EBIT")), None)
            int_row = next(
                (r for r in fin.index if "interest expense" in r.lower()
                 and "non operating" not in r.lower()),
                None,
            )
            if ebit_row and int_row:
                ebit_v = _safe_float(fin.loc[ebit_row].iloc[0])
                int_v = _safe_float(fin.loc[int_row].iloc[0])
                # Interest expense is often reported as negative in yfinance
                if ebit_v and int_v and int_v != 0:
                    interest_coverage_calc = ebit_v / abs(int_v)
    except Exception:
        pass

    result = Fundamentals(
        ticker=ticker,
        name=info.get("longName") or info.get("shortName"),
        sector=info.get("sector"),
        industry=info.get("industry"),
        market_cap=_safe_float(info.get("marketCap")),
        enterprise_value=_safe_float(info.get("enterpriseValue")),
        # Valuation
        pe_ttm=_safe_float(info.get("trailingPE")),
        pe_forward=_safe_float(info.get("forwardPE")),
        peg=_safe_float(info.get("pegRatio")),
        ev_ebitda=_safe_float(info.get("enterpriseToEbitda")),
        price_to_sales=_safe_float(info.get("priceToSalesTrailing12Months")),
        price_to_fcf=None,  # computed below
        # Profitability — use annual statement values, not TTM info dict (TTM can be
        # distorted by large one-time items such as asset sales or tax events)
        gross_margin=_safe_float(info.get("grossMargins")),
        operating_margin=_safe_float(info.get("operatingMargins")),
        net_margin=net_margin_annual if net_margin_annual is not None else _safe_float(info.get("profitMargins")),
        roe=_safe_float(info.get("returnOnEquity")),
        roic=_safe_float(info.get("returnOnAssets")),  # returnOnAssets (ROA) used as proxy; label in UI
        # Health
        current_ratio=_safe_float(info.get("currentRatio")),
        debt_to_equity=_safe_float(info.get("debtToEquity")),
        interest_coverage=interest_coverage_calc if interest_coverage_calc is not None else _safe_float(info.get("ebitdaMargins")),
        # Growth
        revenue_growth_yoy=_safe_float(info.get("revenueGrowth")),
        eps_growth_yoy=_safe_float(info.get("earningsGrowth")),
        # Raw
        revenue_ttm=_safe_float(info.get("totalRevenue")),
        ebitda_ttm=_safe_float(info.get("ebitda")),
        net_income_ttm=_safe_float(info.get("netIncomeToCommon")),
        fcf_ttm=fcf,
        total_debt=_safe_float(info.get("totalDebt")),
        cash=_safe_float(info.get("totalCash")),
        shares_outstanding=_safe_float(info.get("sharesOutstanding")),
        current_price=_safe_float(info.get("currentPrice") or info.get("regularMarketPrice")),
        beta=_safe_float(info.get("beta")),
        fetched_at=datetime.utcnow(),
        # Price context
        previous_close=_safe_float(info.get("previousClose") or info.get("regularMarketPreviousClose")),
        week_52_high=_safe_float(info.get("fiftyTwoWeekHigh")),
        week_52_low=_safe_float(info.get("fiftyTwoWeekLow")),
        # Analyst consensus (buy/hold/sell counts populated below from recommendations_summary)
        analyst_target_mean=_safe_float(info.get("targetMeanPrice")),
        analyst_target_low=_safe_float(info.get("targetLowPrice")),
        analyst_target_high=_safe_float(info.get("targetHighPrice")),
    )

    # Compute derived ratios
    if result.market_cap and result.fcf_ttm and result.fcf_ttm > 0:
        result.price_to_fcf = result.market_cap / result.fcf_ttm

    debt = result.total_debt or 0
    cash_val = result.cash or 0
    net_debt = debt - cash_val
    if result.ebitda_ttm and result.ebitda_ttm > 0:
        result.net_debt_ebitda = net_debt / result.ebitda_ttm

    # Day change %
    if result.current_price and result.previous_close and result.previous_close > 0:
        result.day_change_pct = (result.current_price - result.previous_close) / result.previous_close

    # Next earnings date
    try:
        cal = yf_ticker.calendar
        if cal is not None:
            earnings_raw = None
            if isinstance(cal, dict):
                earnings_raw = cal.get("Earnings Date") or cal.get("earningsDate")
                if isinstance(earnings_raw, (list, tuple)) and earnings_raw:
                    earnings_raw = earnings_raw[0]
            elif hasattr(cal, "get"):
                earnings_raw = cal.get("Earnings Date")
            if earnings_raw is not None:
                if hasattr(earnings_raw, "date"):
                    result.next_earnings_date = earnings_raw.date()
                else:
                    result.next_earnings_date = pd.Timestamp(earnings_raw).date()
    except Exception:
        pass

    # Analyst buy/hold/sell from recommendations_summary if available
    try:
        rec_sum = yf_ticker.recommendations_summary
        if rec_sum is not None and not rec_sum.empty:
            latest = rec_sum.iloc[0]
            result.analyst_buy_count = _safe_int(
                (latest.get("strongBuy", 0) or 0) + (latest.get("buy", 0) or 0)
            )
            result.analyst_hold_count = _safe_int(latest.get("hold", 0) or 0)
            result.analyst_sell_count = _safe_int(
                (latest.get("sell", 0) or 0) + (latest.get("strongSell", 0) or 0)
            )
    except Exception:
        pass

    set_cache_obj(cache_key, result.model_dump(mode="json"), TTL_FUNDAMENTALS, source="yfinance")
    record_freshness(cache_key, "yfinance", TTL_FUNDAMENTALS)
    return result


@retry(max_attempts=3, base_delay=2.0)
def get_estimates(ticker: str) -> Dict[str, Any]:
    cache_key = f"market:{ticker}:estimates"
    cached = get_cache_obj(cache_key)
    if cached:
        return cached

    yf_ticker = yf.Ticker(ticker, session=_yf_session())
    result: Dict[str, Any] = {}

    # EPS trend
    try:
        et = yf_ticker.eps_trend
        if et is not None and not et.empty:
            result["eps_trend"] = et.reset_index().to_dict(orient="records")
    except Exception as exc:
        logger.debug("eps_trend error for %s: %s", ticker, exc)

    # Revenue estimates
    try:
        re = yf_ticker.revenue_estimate
        if re is not None and not re.empty:
            result["revenue_estimate"] = re.reset_index().to_dict(orient="records")
    except Exception as exc:
        logger.debug("revenue_estimate error for %s: %s", ticker, exc)

    # Earnings history (beat/miss)
    try:
        eh = yf_ticker.earnings_history
        if eh is not None and not eh.empty:
            result["earnings_history"] = eh.reset_index().to_dict(orient="records")
    except Exception as exc:
        logger.debug("earnings_history error for %s: %s", ticker, exc)

    # Analyst price targets / recommendations
    try:
        recs = yf_ticker.recommendations
        if recs is not None and not recs.empty:
            result["recommendations"] = recs.reset_index().tail(20).to_dict(orient="records")
    except Exception as exc:
        logger.debug("recommendations error for %s: %s", ticker, exc)

    try:
        targets = yf_ticker.analyst_price_targets
        if targets is not None and not targets.empty:
            result["analyst_price_targets"] = targets.reset_index().to_dict(orient="records")
    except Exception as exc:
        logger.debug("analyst_price_targets error for %s: %s", ticker, exc)

    set_cache_obj(cache_key, result, TTL_FUNDAMENTALS, source="yfinance")
    record_freshness(cache_key, "yfinance", TTL_FUNDAMENTALS)
    return result


def _quarter_label(dt: Any) -> str:
    try:
        ts = pd.Timestamp(dt)
        q = (ts.month - 1) // 3 + 1
        return f"Q{q} '{str(ts.year)[2:]}"
    except Exception:
        return str(dt)[:7]


@retry(max_attempts=3, base_delay=2.0)
def get_beat_miss_history(ticker: str) -> List[Dict[str, Any]]:
    """Return last 8 quarters of EPS beat/miss data, chronological order."""
    from config import TTL_ESTIMATES
    cache_key = f"market:{ticker}:beat_miss"
    cached = get_cache_obj(cache_key)
    if cached:
        return cached

    yf_ticker = yf.Ticker(ticker, session=_yf_session())
    records: List[Dict[str, Any]] = []

    try:
        eh = yf_ticker.earnings_history
        if eh is not None and not eh.empty:
            eh = eh.sort_index(ascending=False).head(8)
            for dt, row in eh.iterrows():
                eps_est = _safe_float(row.get("epsEstimate"))
                eps_actual = _safe_float(row.get("epsActual"))
                surprise_pct = _safe_float(row.get("surprisePercent"))
                records.append({
                    "period": _quarter_label(dt),
                    "date": str(dt)[:10],
                    "eps_est": eps_est,
                    "eps_actual": eps_actual,
                    "eps_surprise_pct": surprise_pct,
                })
    except Exception as exc:
        logger.warning("beat_miss_history error for %s: %s", ticker, exc)

    records = list(reversed(records))  # oldest → newest left to right
    set_cache_obj(cache_key, records, TTL_ESTIMATES, source="yfinance")
    return records


@retry(max_attempts=3, base_delay=2.0)
def get_insiders(ticker: str) -> List[InsiderTransaction]:
    cache_key = f"market:{ticker}:insiders"
    cached = get_cache_obj(cache_key)
    if cached:
        return [InsiderTransaction.model_validate(r) for r in cached]

    yf_ticker = yf.Ticker(ticker, session=_yf_session())
    transactions: List[InsiderTransaction] = []
    try:
        df = yf_ticker.insider_transactions
        if df is not None and not df.empty:
            for _, row in df.iterrows():
                tx_date = row.get("startDate") or row.get("Date")
                if hasattr(tx_date, "date"):
                    tx_date = tx_date.date()
                elif tx_date is not None:
                    try:
                        tx_date = pd.Timestamp(tx_date).date()
                    except Exception:
                        tx_date = None
                transactions.append(InsiderTransaction(
                    name=str(row.get("Insider", row.get("insider", "Unknown"))),
                    role=str(row.get("Position", row.get("position", ""))),
                    transaction_type=str(row.get("Transaction", row.get("transaction", ""))),
                    shares=_safe_int(row.get("Shares", row.get("shares", 0))) or 0,
                    price=_safe_float(row.get("Value", row.get("startPrice"))),
                    value=_safe_float(row.get("Value")),
                    date=tx_date,
                ))
    except Exception as exc:
        logger.warning("insider_transactions error for %s: %s", ticker, exc)

    set_cache_obj(cache_key, [t.model_dump(mode="json") for t in transactions],
                  TTL_FUNDAMENTALS, source="yfinance")
    record_freshness(cache_key, "yfinance", TTL_FUNDAMENTALS)
    return transactions


@retry(max_attempts=3, base_delay=2.0)
def get_holders(ticker: str) -> List[InstitutionalHolder]:
    cache_key = f"market:{ticker}:holders"
    cached = get_cache_obj(cache_key)
    if cached:
        return [InstitutionalHolder.model_validate(r) for r in cached]

    yf_ticker = yf.Ticker(ticker, session=_yf_session())
    holders: List[InstitutionalHolder] = []
    try:
        df = yf_ticker.institutional_holders
        if df is not None and not df.empty:
            for _, row in df.head(15).iterrows():
                holders.append(InstitutionalHolder(
                    name=str(row.get("Holder", row.get("holder", "Unknown"))),
                    shares=_safe_int(row.get("Shares", row.get("shares", 0))) or 0,
                    pct_held=_safe_float(row.get("pctHeld", row.get("% Out"))),
                    date_reported=None,
                    change=_safe_int(row.get("Change", row.get("change"))),
                ))
    except Exception as exc:
        logger.warning("institutional_holders error for %s: %s", ticker, exc)

    set_cache_obj(cache_key, [h.model_dump(mode="json") for h in holders],
                  TTL_FUNDAMENTALS, source="yfinance")
    record_freshness(cache_key, "yfinance", TTL_FUNDAMENTALS)
    return holders


@retry(max_attempts=3, base_delay=2.0)
def get_short_interest(ticker: str) -> ShortInterest:
    cache_key = f"market:{ticker}:short_interest"
    cached = get_cache_obj(cache_key)
    if cached:
        return ShortInterest.model_validate(cached)

    yf_ticker = yf.Ticker(ticker, session=_yf_session())
    info = yf_ticker.info or {}
    result = ShortInterest(
        date=None,
        short_interest=_safe_int(info.get("sharesShort")),
        pct_float=_safe_float(info.get("shortPercentOfFloat")),
        days_to_cover=_safe_float(info.get("shortRatio")),
    )
    set_cache_obj(cache_key, result.model_dump(mode="json"), TTL_FUNDAMENTALS, source="yfinance")
    record_freshness(cache_key, "yfinance", TTL_FUNDAMENTALS)
    return result


@retry(max_attempts=3, base_delay=2.0)
def get_news(ticker: str, days: int = 30) -> List[NewsItem]:
    cache_key = f"market:{ticker}:news:{days}"
    cached = get_cache_obj(cache_key)
    if cached:
        return [NewsItem.model_validate(r) for r in cached]

    yf_ticker = yf.Ticker(ticker, session=_yf_session())
    items: List[NewsItem] = []
    cutoff = datetime.utcnow() - timedelta(days=days)

    try:
        raw_news = yf_ticker.news or []
        for n in raw_news:
            # yfinance ≥0.2.40 wraps articles in a 'content' dict
            if "content" in n and isinstance(n["content"], dict):
                c = n["content"]
                pub_raw = c.get("pubDate")
                pub_dt: Optional[datetime] = None
                if pub_raw:
                    try:
                        pub_dt = datetime.fromisoformat(pub_raw.replace("Z", "+00:00")).replace(tzinfo=None)
                    except ValueError:
                        pass
                if pub_dt and pub_dt < cutoff:
                    continue
                url = (c.get("canonicalUrl") or c.get("clickThroughUrl") or {}).get("url")
                provider = (c.get("provider") or {}).get("displayName")
                items.append(NewsItem(
                    title=c.get("title", ""),
                    source=provider,
                    published_at=pub_dt,
                    url=url,
                    snippet=c.get("summary") or c.get("description"),
                ))
            else:
                # Legacy format
                pub = n.get("providerPublishTime")
                pub_dt = datetime.utcfromtimestamp(pub) if pub else None
                if pub_dt and pub_dt < cutoff:
                    continue
                items.append(NewsItem(
                    title=n.get("title", ""),
                    source=n.get("publisher"),
                    published_at=pub_dt,
                    url=n.get("link"),
                    snippet=n.get("summary") or n.get("description"),
                ))
    except Exception as exc:
        logger.warning("yfinance news error for %s: %s", ticker, exc)

    set_cache_obj(cache_key, [i.model_dump(mode="json") for i in items],
                  TTL_NEWS, source="yfinance")
    record_freshness(cache_key, "yfinance", TTL_NEWS)
    return items
