"""Feature 8: Review site sentiment — G2 and Capterra scrape + Haiku extraction."""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

import httpx

import llm
from config import HAIKU, TTL_WEB_PAGES
from data.cache import get_cache_obj, set_cache_obj

logger = logging.getLogger(__name__)

_SYSTEM = llm.cached_system("""
You are a product analyst extracting structured review data from a review site page.
Return ONLY valid JSON with keys:
  platform (str), star_rating (float or null), review_count (int or null),
  top_pros (list of str, max 5), top_cons (list of str, max 5),
  common_use_cases (list of str, max 3), sentiment_summary (str, 1-2 sentences).
Extract only what is explicitly present in the text. Do not invent data.
""")

_REVIEW_PLATFORMS = [
    ("G2", "https://www.g2.com/products/{slug}/reviews"),
    ("Capterra", "https://www.capterra.com/p/{slug}/"),
    ("Product Hunt", "https://www.producthunt.com/products/{slug}"),
]

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; DeepDiveBot/1.0; research-only)",
    "Accept": "text/html,application/xhtml+xml",
}


def _domain_to_slug(domain: str) -> str:
    return domain.replace(".", "-").replace("_", "-").lower().split("/")[0]


def _fetch_html(url: str) -> Optional[str]:
    try:
        r = httpx.get(url, headers=_HEADERS, follow_redirects=True, timeout=10)
        if r.status_code == 200:
            # Strip tags and collapse whitespace
            text = re.sub(r"<[^>]+>", " ", r.text)
            text = re.sub(r"\s+", " ", text)
            return text[:8000]
        return None
    except Exception:
        return None


def _extract_review_data(platform: str, html_text: str, domain: str) -> Optional[Dict]:
    messages = [
        {
            "role": "user",
            "content": llm.text_block(
                f"Platform: {platform}\nProduct: {domain}\n\n"
                f"Page text (truncated):\n{html_text}\n\n"
                "Extract the review data and return JSON."
            ),
        }
    ]
    try:
        import json
        raw = llm.call(
            HAIKU, messages,
            system=_SYSTEM,
            prompt_version="v1",
            max_tokens=600,
        )
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            data["platform"] = platform
            return data
    except Exception as exc:
        logger.debug("review extract failed for %s/%s: %s", platform, domain, exc)
    return None


def fetch_review_sentiment(domain: str) -> List[Dict]:
    """
    Try G2, Capterra, and Product Hunt for the given domain.
    Returns a list of review dicts (one per platform that returned data).
    Cached for 7 days.
    """
    cache_key = f"reviews:{domain}"
    cached = get_cache_obj(cache_key)
    if cached:
        return cached

    slug = _domain_to_slug(domain)
    results: List[Dict] = []

    for platform_name, url_template in _REVIEW_PLATFORMS:
        url = url_template.format(slug=slug)
        html = _fetch_html(url)
        if not html:
            continue
        # Quick gate: skip if page is clearly empty/404
        if len(html) < 500:
            continue
        data = _extract_review_data(platform_name, html, domain)
        if data and (data.get("star_rating") or data.get("top_pros") or data.get("top_cons")):
            results.append(data)

    set_cache_obj(cache_key, results, TTL_WEB_PAGES)
    return results
