"""Feature 7: Competitive discovery — identify competitors and side-by-side comparison."""
from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import llm
from config import HAIKU, SONNET, TTL_NEWS, TTL_WEB_PAGES
from data.cache import get_cache_obj, set_cache_obj

logger = logging.getLogger(__name__)

_SYSTEM_DISCOVER = llm.cached_system("""
You are a competitive intelligence analyst. Given a company/product, identify its 3-5 closest
direct competitors. Return ONLY a JSON array of objects with keys:
  name (company/product name), domain (website domain, no https://)
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


def _domain_to_product_name(domain: str) -> str:
    bare = domain.split("/")[0].lower()
    parts = bare.split(".")
    if parts[0] == "www" and len(parts) > 1:
        parts = parts[1:]
    return parts[0].title()


def discover_competitors(domain: str, company_summary: str = "") -> Tuple[List[Dict], Dict]:
    """
    Identify 3-5 competitors for the given domain using web search + Sonnet.
    Returns (list of {name, domain}, diag_dict).
    Does NOT require company_summary — web search covers the gap.
    """
    cache_key = f"competitors_v2:discover:{domain}"
    cached = get_cache_obj(cache_key)
    if cached is not None:
        return cached, {"method": "cache"}

    product_name = _domain_to_product_name(domain)
    diag = {"domain": domain, "product_name": product_name}

    # Step 1: web search to find real competitors by name
    web_context = llm.web_search_synthesis(
        f'Who are the main competitors and alternatives to "{product_name}" ({domain})? '
        f'List the top 3-5 direct competitors with their website domains. '
        f'Include both established players and newer challengers.',
        cache_key=f"competitors_v2:search:{domain}",
        max_tokens=600,
    )
    diag["web_search_chars"] = len(web_context)
    logger.info("[competitors] web_search for %s: %d chars", domain, len(web_context))

    # Step 2: Sonnet structures the result into JSON
    context_for_llm = web_context or company_summary or f"Product: {product_name} at {domain}"
    messages = [{
        "role": "user",
        "content": llm.text_block(
            f"Product: {product_name} ({domain})\n\n"
            f"Research findings:\n{context_for_llm[:2000]}\n\n"
            "List the 3-5 closest direct competitors. Return JSON array only."
        ),
    }]
    try:
        raw = llm.call(
            SONNET, messages,
            system=_SYSTEM_DISCOVER,
            prompt_version="v2",
            max_tokens=400,
        )
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        result = json.loads(m.group()) if m else []
        result = [r for r in result if isinstance(r, dict) and "domain" in r][:5]
        diag["competitors_found"] = [r.get("name") for r in result]
    except Exception as exc:
        logger.warning("competitor discovery failed for %s: %s", domain, exc)
        result = []
        diag["error"] = str(exc)

    set_cache_obj(cache_key, result, TTL_WEB_PAGES)
    return result, diag


def _crawl_competitor(competitor_domain: str) -> Tuple[Optional[str], Dict]:
    """Fetch competitor site. Falls back to web_search if crawl is blocked."""
    from data.webintel import discover_urls, fetch_pages
    from analysis.company import extract_page_intel

    cache_key = f"competitors_v2:crawl:{competitor_domain}"
    cached = get_cache_obj(cache_key)
    if cached is not None:
        return cached, {"method": "cache"}

    product_name = _domain_to_product_name(competitor_domain)
    diag = {"domain": competitor_domain, "product_name": product_name}

    # Try direct crawl first
    try:
        url = f"https://{competitor_domain}"
        urls = discover_urls(url)[:15]
        pages = fetch_pages(urls)
        intels = []
        for pg in pages[:10]:
            intel = extract_page_intel(pg)
            if intel:
                intels.append(intel)

        if intels:
            features = []
            for intel in intels:
                features.extend(intel.feature_claims[:3])
            customers = list({c for intel in intels for c in intel.named_customers})[:5]
            parts = [f"Domain: {competitor_domain}"]
            if features:
                parts.append("Features: " + "; ".join(features[:8]))
            if customers:
                parts.append("Customers: " + ", ".join(customers))
            result = "\n".join(parts)
            diag["method"] = "crawl"
            diag["pages_crawled"] = len(intels)
            set_cache_obj(cache_key, result, TTL_WEB_PAGES)
            return result, diag
    except Exception as exc:
        diag["crawl_error"] = str(exc)
        logger.debug("competitor crawl failed for %s: %s", competitor_domain, exc)

    # Fallback: web search for competitor info
    diag["method"] = "web_search_fallback"
    search_result = llm.web_search_synthesis(
        f'What does "{product_name}" ({competitor_domain}) do? '
        f'Key features, pricing, target customers, main differentiators.',
        cache_key=f"competitors_v2:search_crawl:{competitor_domain}",
        max_tokens=500,
    )
    diag["search_chars"] = len(search_result)
    logger.info("[competitors] crawl fallback for %s: %d chars", competitor_domain, len(search_result))

    if search_result:
        result = f"Domain: {competitor_domain}\n{search_result}"
        set_cache_obj(cache_key, result, TTL_WEB_PAGES)
        return result, diag

    return None, diag


def build_competitive_comparison(
    target_domain: str,
    target_summary: str,
    competitor_list: List[Dict],
) -> Tuple[str, List[Dict]]:
    """
    Crawl competitors in parallel and produce a Sonnet comparison narrative.
    Returns (markdown_text, list_of_crawl_diags).
    """
    cache_key = (
        f"competitors_v2:compare:{target_domain}:"
        + ",".join(c.get("domain", "") for c in competitor_list)
    )
    cached = get_cache_obj(cache_key)
    if cached is not None:
        return cached, []

    product_name = _domain_to_product_name(target_domain)
    crawl_diags: List[Dict] = []
    competitor_summaries: List[str] = []

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(_crawl_competitor, c["domain"]): c
            for c in competitor_list
        }
        for fut in as_completed(futures):
            comp = futures[fut]
            try:
                result, diag = fut.result()
            except Exception as exc:
                result, diag = None, {"domain": comp.get("domain"), "error": str(exc)}
            diag["name"] = comp.get("name")
            crawl_diags.append(diag)
            logger.info("[competitors] crawl result for %s: %s", comp.get("domain"), diag)
            if result:
                competitor_summaries.append(
                    f"### {comp.get('name', comp['domain'])}\n{result}"
                )

    if not competitor_summaries:
        return "Could not fetch competitor data.", crawl_diags

    context = (
        f"## Target: {product_name} ({target_domain})\n"
        + (target_summary[:2000] if target_summary else f"See {target_domain}")
        + "\n\n"
        + "\n\n".join(competitor_summaries)
    )
    messages = [{
        "role": "user",
        "content": [
            *llm.cached_content(context),
            llm.text_block(
                f"Write the competitive comparison for {product_name} vs the listed competitors."
            ),
        ],
    }]
    try:
        result = llm.call(
            SONNET, messages,
            system=_SYSTEM_COMPARE,
            prompt_version="v1",
            max_tokens=3000,
            continue_on_truncation=True,
        )
        set_cache_obj(cache_key, result, TTL_NEWS)
        return result, crawl_diags
    except Exception as exc:
        logger.warning("competitive compare Sonnet call failed: %s", exc)
        return "Comparison generation failed.", crawl_diags
