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

_THIN_SEARCH_THRESHOLD = 50  # chars — below this, treat web search as effectively empty


def normalize_domain(raw: str) -> str:
    """Normalize a URL, domain, or bare hostname to a lowercase registrable domain.

    Strips scheme (https://), www prefix, path, query string, fragment, and
    trailing punctuation. Returns lowercase bare domain suitable for deduplication
    and use as a crawl target.

    Examples:
        "https://www.resolve.ai/product?ref=g" → "resolve.ai"
        "Resolve.AI"                            → "resolve.ai"
        "traversal.com/"                        → "traversal.com"
    """
    raw = raw.strip().lower()
    # Strip scheme
    raw = re.sub(r"^https?://", "", raw)
    # Strip auth (user:pass@)
    raw = raw.split("@")[-1]
    # Strip www.
    raw = re.sub(r"^www\.", "", raw)
    # Strip path, query, fragment — take only the host part
    raw = raw.split("/")[0].split("?")[0].split("#")[0]
    # Strip port
    raw = raw.split(":")[0]
    # Strip trailing dots
    raw = raw.rstrip(".")
    return raw


def _domain_to_product_name(domain: str) -> str:
    bare = normalize_domain(domain)
    parts = bare.split(".")
    return parts[0].title()


def resolve_name_to_domain(company_name: str) -> Optional[str]:
    """Resolve a plain company name (no dot) to a registrable domain via web search.

    Returns None if resolution fails so the caller can surface a clear error.
    """
    result = llm.web_search_synthesis(
        f'What is the official website of "{company_name}"? '
        "Reply with just the domain, e.g. example.com — no https:// or www.",
        max_tokens=120,
    )
    if not result:
        return None
    # Extract the first thing that looks like a domain in the response
    m = re.search(r"\b([a-z0-9][a-z0-9\-]*\.[a-z]{2,}(?:\.[a-z]{2,})?)\b", result.lower())
    return normalize_domain(m.group(1)) if m else None


def _validate_messages(messages: list) -> None:
    """Assert that every message has a content field the API will accept.

    Raises ValueError naming the bad field so the error surfaces before the
    round-trip to Anthropic (which would return a cryptic 400).
    """
    for i, msg in enumerate(messages):
        content = msg.get("content")
        if not isinstance(content, (str, list)):
            raise ValueError(
                f"messages[{i}].content must be str or list[dict], "
                f"got {type(content).__name__!r}. "
                "Single text_block() must be wrapped: content=[text_block(...)]"
            )
        if isinstance(content, list):
            for j, block in enumerate(content):
                if not isinstance(block, dict):
                    raise ValueError(
                        f"messages[{i}].content[{j}] must be dict, "
                        f"got {type(block).__name__!r}"
                    )


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

    # Treat near-empty search result as no result to avoid passing thin context to the LLM.
    if len(web_context) < _THIN_SEARCH_THRESHOLD:
        logger.warning(
            "[competitors] web search returned only %d chars for %s — treating as empty",
            len(web_context), domain,
        )
        web_context = ""
        diag["web_search_thin"] = True

    if not web_context and not company_summary:
        diag["error"] = (
            f"Web search returned no usable results ({diag['web_search_chars']} chars) "
            "and no company summary is available. Add competitors manually."
        )
        return [], diag

    # Step 2: Sonnet structures the result into JSON
    context_for_llm = web_context or company_summary or f"Product: {product_name} at {domain}"
    messages = [{
        "role": "user",
        # content must be a list — a bare text_block() dict causes a 400
        "content": [llm.text_block(
            f"Product: {product_name} ({domain})\n\n"
            f"Research findings:\n{context_for_llm[:2000]}\n\n"
            "List the 3-5 closest direct competitors. Return JSON array only."
        )],
    }]
    _validate_messages(messages)
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
        # Normalize domains returned by the LLM
        for r in result:
            r["domain"] = normalize_domain(r["domain"])
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

    ``competitor_list`` is source-agnostic — it may contain auto-discovered
    entries, manually added entries, or a mix. Each dict must have a ``domain``
    key; ``name`` and ``source`` are optional extras that are preserved in diags.

    Returns (markdown_text, list_of_crawl_diags).
    """
    cache_key = (
        f"competitors_v2:compare:{target_domain}:"
        + ",".join(sorted(c.get("domain", "") for c in competitor_list))
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
            diag["source"] = comp.get("source", "auto")
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
    _validate_messages(messages)
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
