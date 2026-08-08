"""News aggregator: yfinance (primary) + NewsAPI (supplement). Deduplicates by title."""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta
from typing import List

from config import NEWSAPI_KEY, TTL_NEWS
from data.cache import get_cache_obj, record_freshness, set_cache_obj
from data.market import get_news as yf_get_news
from data.resilience import retry
from analysis.schemas import NewsItem

logger = logging.getLogger(__name__)


def _title_hash(title: str) -> str:
    return hashlib.md5(title.lower().strip().encode()).hexdigest()


@retry(max_attempts=2, base_delay=2.0)
def _get_newsapi(ticker: str, company_name: str, days: int) -> List[NewsItem]:
    if not NEWSAPI_KEY:
        return []
    try:
        from newsapi import NewsApiClient  # type: ignore
        client = NewsApiClient(api_key=NEWSAPI_KEY)
        from_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        # Search by company name for better recall than bare ticker
        resp = client.get_everything(
            q=f'"{company_name}" OR "{ticker}"',
            from_param=from_date,
            language="en",
            sort_by="relevancy",
            page_size=50,
        )
        items: List[NewsItem] = []
        for a in (resp.get("articles") or []):
            pub = a.get("publishedAt")
            pub_dt = datetime.fromisoformat(pub.replace("Z", "+00:00")).replace(tzinfo=None) if pub else None
            items.append(NewsItem(
                title=a.get("title") or "",
                source=a.get("source", {}).get("name"),
                published_at=pub_dt,
                url=a.get("url"),
                snippet=a.get("description"),
            ))
        return items
    except Exception as exc:
        logger.warning("NewsAPI error for %s: %s", ticker, exc)
        return []


def get_news(
    ticker: str,
    company_name: str = "",
    days: int = 30,
) -> List[NewsItem]:
    """
    Return deduplicated news items for the past `days` days.
    yfinance provides the core feed; NewsAPI supplements if key is set.
    Items are sorted newest-first.
    """
    cache_key = f"news:{ticker.upper()}:{days}"
    cached = get_cache_obj(cache_key)
    if cached:
        return [NewsItem.model_validate(r) for r in cached]

    yf_items = yf_get_news(ticker, days=days)
    newsapi_items = _get_newsapi(ticker, company_name or ticker, days)

    # Deduplicate by title hash
    seen: set[str] = set()
    combined: List[NewsItem] = []
    for item in yf_items + newsapi_items:
        h = _title_hash(item.title)
        if h not in seen and item.title:
            seen.add(h)
            combined.append(item)

    # Sort newest-first
    combined.sort(
        key=lambda x: x.published_at or datetime.min,
        reverse=True,
    )

    set_cache_obj(cache_key, [i.model_dump(mode="json") for i in combined],
                  TTL_NEWS, source="yfinance+newsapi")
    record_freshness(cache_key, "news", TTL_NEWS)
    return combined
