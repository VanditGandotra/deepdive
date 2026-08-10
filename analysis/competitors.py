"""Feature 7: Competitive discovery — identify competitors and do a side-by-side feature compare."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import llm
from config import HAIKU, SONNET, TTL_NEWS, TTL_WEB_PAGES
from data.cache import get_cache_obj, set_cache_obj

logger = logging.getLogger(__name__)

_SYSTEM_DISCOVER = llm.cached_system("""
You are a competitive intelligence analyst. Given a company description, identify its 3-5 closest
direct competitors. Return ONLY a JSON array of objects with keys:
  name (company name), domain (website domain, no https://)
Example: [{"name": "Competitor A", "domain": "competitor-a.com"}, ...]
No explanation. Only valid JSON array.
""")

_SYSTEM_COMPARE = llm.cached_system("""
You are a product analyst writing a competitive comparison for an internal strategy memo.
Given summaries of a target product and its competitors, write a structured comparison covering:
## Feature Parity
## Key Differentiators (target vs each competitor)
## Where Target Wins
## Where Competitors Win
## Positioning Summary
Be specific — name actual features. Under 500 words.
""")


def discover_competitors(domain: str, company_summary: str) -> List[Dict]:
    """
    Use Sonnet to identify 3-5 competitors for the given domain/company.
    Returns list of {name, domain}.
    """
    cache_key = f"competitors:discover:{domain}"
    cached = get_cache_obj(cache_key)
    if cached:
        return cached

    messages = [
        {
            "role": "user",
            "content": llm.text_block(
                f"Company: {domain}\n\nDescription:\n{company_summary[:1500]}\n\n"
                "Identify the 3-5 closest direct competitors. Return JSON array only."
            ),
        }
    ]
    try:
        import json, re
        raw = llm.call(
            SONNET, messages,
            system=_SYSTEM_DISCOVER,
            prompt_version="v1",
            max_tokens=400,
        )
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        result = json.loads(match.group()) if match else []
        # Validate structure
        result = [r for r in result if isinstance(r, dict) and "domain" in r][:5]
    except Exception as exc:
        logger.warning("competitor discovery failed for %s: %s", domain, exc)
        result = []

    set_cache_obj(cache_key, result, TTL_WEB_PAGES)
    return result


def _crawl_competitor(competitor_domain: str) -> Optional[str]:
    """Fetch top pages from a competitor and return a short text summary."""
    from data.webintel import discover_urls, fetch_pages
    from analysis.company import extract_page_intel

    cache_key = f"competitors:crawl:{competitor_domain}"
    cached = get_cache_obj(cache_key)
    if cached:
        return cached

    try:
        url = f"https://{competitor_domain}"
        urls = discover_urls(url)[:15]
        pages = fetch_pages(urls)
        intels = []
        for pg in pages[:10]:
            intel = extract_page_intel(pg)
            if intel:
                intels.append(intel)

        if not intels:
            return None

        # Build compact summary
        features = []
        for intel in intels:
            features.extend(intel.feature_claims[:3])
        customers = list({c for intel in intels for c in intel.named_customers})[:5]
        summary_parts = [f"Domain: {competitor_domain}"]
        if features:
            summary_parts.append("Features: " + "; ".join(features[:8]))
        if customers:
            summary_parts.append("Customers: " + ", ".join(customers))
        result = "\n".join(summary_parts)
        set_cache_obj(cache_key, result, TTL_WEB_PAGES)
        return result
    except Exception as exc:
        logger.warning("competitor crawl failed for %s: %s", competitor_domain, exc)
        return None


def build_competitive_comparison(
    target_domain: str,
    target_summary: str,
    competitor_list: List[Dict],
) -> str:
    """
    Crawl competitors in parallel and produce a Sonnet comparison narrative.
    Returns markdown text.
    """
    cache_key = f"competitors:compare:{target_domain}:{','.join(c.get('domain','') for c in competitor_list)}"
    cached = get_cache_obj(cache_key)
    if cached:
        return cached

    competitor_summaries: List[str] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_crawl_competitor, c["domain"]): c for c in competitor_list}
        for fut in as_completed(futures):
            comp = futures[fut]
            result = fut.result()
            if result:
                competitor_summaries.append(
                    f"### {comp.get('name', comp['domain'])}\n{result}"
                )

    if not competitor_summaries:
        return "Could not fetch competitor data."

    context = (
        f"## Target: {target_domain}\n{target_summary[:2000]}\n\n"
        + "\n\n".join(competitor_summaries)
    )
    messages = [
        {
            "role": "user",
            "content": [
                *llm.cached_content(context),
                llm.text_block(
                    f"Write the competitive comparison for {target_domain} vs the listed competitors."
                ),
            ],
        }
    ]
    try:
        result = llm.call(
            SONNET, messages,
            system=_SYSTEM_COMPARE,
            prompt_version="v1",
            max_tokens=1500,
        )
        set_cache_obj(cache_key, result, TTL_NEWS)
        return result
    except Exception as exc:
        logger.warning("competitive compare Sonnet call failed: %s", exc)
        return "Comparison generation failed."
