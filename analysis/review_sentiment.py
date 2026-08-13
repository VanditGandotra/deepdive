"""Feature 8: Review site sentiment — G2, Product Hunt, Trustpilot, Capterra.

Fetch strategy per platform:
  G2         — web_search snippets only (Cloudflare-blocked, snippet has rating/count)
  Trustpilot — web_search snippets only (Cloudflare 403 on direct fetch)
  Product Hunt — official GraphQL API (PRODUCT_HUNT_TOKEN); Playwright fallback if
                 token absent (Playwright IS installed); skip cleanly if both unavailable
  Capterra   — direct HTTP with browser UA (less aggressively protected)
"""
from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Optional, Tuple

import httpx

import llm
from config import HAIKU, PRODUCT_HUNT_TOKEN, TTL_WEB_PAGES
from data.cache import get_cache_obj, set_cache_obj

logger = logging.getLogger(__name__)

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

_SYSTEM_EXTRACT = llm.cached_system("""
You are extracting structured review data from text (web search snippets, API responses,
or scraped pages). Return ONLY valid JSON with exactly these keys:
  platform (str), star_rating (float or null), review_count (int or null),
  top_pros (list[str] max 5), top_cons (list[str] max 5),
  common_use_cases (list[str] max 3), sentiment_summary (str 1-2 sentences).
Use null for fields not present. Return nothing except the JSON object.
""")

# ── Slug / name helpers ───────────────────────────────────────────────────────

def _domain_to_slug(domain: str) -> str:
    """cursor.com -> cursor, anthropic.com -> anthropic, notion.so -> notion"""
    bare = domain.split("/")[0].lower()
    parts = bare.split(".")
    if parts[0] == "www" and len(parts) > 1:
        parts = parts[1:]
    return parts[0]


def _domain_to_product_name(domain: str) -> str:
    return _domain_to_slug(domain).title()


# ── Body classification ───────────────────────────────────────────────────────

def _classify_body(status: int, body: str) -> str:
    """Return a diagnostic label for an HTTP response."""
    if status == 0:
        return "connection_error"
    if status == 403:
        low = body.lower()
        if "cloudflare" in low or "verifying" in low or "cf-browser" in low:
            return "cloudflare_403"
        return "forbidden_403"
    if status == 404:
        return "not_found_404"
    if status != 200:
        return f"http_{status}"
    low = body.lower()
    if "verifying your browser" in low or "cf-browser-verification" in low:
        return "cloudflare_challenge_200"
    if "dangling reference" in low or ("apollo_state" in body and '"data":{}' in body):
        return "js_rendered_shell"
    if "__next_data__" in low and len(body) < 1500:
        return "nextjs_ssr_empty"
    if len(body) < 300:
        return "empty_body"
    return "ok"


# ── LLM extraction ────────────────────────────────────────────────────────────

def _llm_extract(platform: str, product_name: str, content: str) -> Optional[Dict]:
    messages = [{
        "role": "user",
        "content": [llm.text_block(
            f"Platform: {platform}\nProduct: {product_name}\n\n"
            f"Content:\n{content}\n\nExtract review data as JSON."
        )],
    }]
    try:
        raw = llm.call(HAIKU, messages, system=_SYSTEM_EXTRACT, prompt_version="v1", max_tokens=600)
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            data = json.loads(m.group())
            data["platform"] = platform
            return data
    except Exception as exc:
        logger.debug("llm_extract failed for %s/%s: %s", platform, product_name, exc)
    return None


def _has_useful_data(d: Optional[Dict]) -> bool:
    """Require at least a numeric field (rating or count) — text-only results don't count."""
    if not d:
        return False
    return bool(d.get("star_rating") or d.get("review_count"))


# ── Platform fetchers ─────────────────────────────────────────────────────────

def _fetch_g2(slug: str, product_name: str) -> Tuple[Optional[Dict], Dict]:
    """
    G2 is Cloudflare-blocked on direct fetch.
    Strategy: web_search WITHOUT site: restriction — search broadly for the
    G2 rating, which appears on comparison sites, review aggregators, and in
    Google search snippets. Avoid site:g2.com which causes Claude to attempt
    fetching the blocked page.
    """
    diag = {"platform": "G2", "method": "web_search_snippets"}

    # Claude's web_search infrastructure can access G2 even though direct httpx
    # is Cloudflare-blocked. Use site: query to direct Claude's crawler to the
    # sellers page (g2.com/sellers/{slug}) which contains "rated X.X stars by N reviews".
    prompt = (
        f'Search: site:g2.com "{product_name}"\n\n'
        f"Fetch g2.com/sellers/{slug} or g2.com/products/{slug}/reviews "
        f"and extract the star rating (out of 5) and total review count for "
        f'"{product_name}". Also note top pros and cons if visible. Return JSON.'
    )
    search_result = llm.web_search_synthesis(
        prompt,
        cache_key=f"reviews:g2_v6:{slug}",
        max_tokens=800,
    )
    diag["search_chars"] = len(search_result)
    diag["search_preview"] = search_result[:400]

    if not search_result or len(search_result) < 50:
        diag["result"] = "search_insufficient"
        return None, diag

    data = _llm_extract("G2", product_name, search_result)
    if _has_useful_data(data):
        diag["result"] = "found"
        return data, diag

    diag["result"] = "genuinely_no_reviews" if len(search_result) > 200 else "search_insufficient"
    return None, diag


def _fetch_producthunt(slug: str, product_name: str) -> Tuple[Optional[Dict], Dict]:
    """
    Strategy order:
      1. Official GraphQL API (PRODUCT_HUNT_TOKEN env var)
      2. Playwright headless render (Playwright IS installed)
      3. web_search snippets
    """
    diag = {"platform": "Product Hunt", "slug": slug}

    # ── 1. Official API ──────────────────────────────────────────────────────
    if PRODUCT_HUNT_TOKEN:
        diag["method"] = "api"
        query = """
query GetProduct($slug: String!) {
  post(slug: $slug) {
    name
    tagline
    votesCount
    reviewsRating
    reviewsCount
    topics { edges { node { name } } }
  }
}"""
        try:
            r = httpx.post(
                "https://api.producthunt.com/v2/api/graphql",
                json={"query": query, "variables": {"slug": slug}},
                headers={
                    "Authorization": f"Bearer {PRODUCT_HUNT_TOKEN}",
                    "Content-Type": "application/json",
                },
                timeout=12,
            )
            diag["api_status"] = r.status_code
            diag["api_preview"] = r.text[:400]
            if r.status_code == 200:
                body = r.json()
                post = (body.get("data") or {}).get("post") or {}
                if post:
                    topics = [
                        e["node"]["name"]
                        for e in (post.get("topics") or {}).get("edges", [])
                    ]
                    data = {
                        "platform": "Product Hunt",
                        "star_rating": post.get("reviewsRating"),
                        "review_count": post.get("reviewsCount") or post.get("votesCount"),
                        "top_pros": [],
                        "top_cons": [],
                        "common_use_cases": topics[:3],
                        "sentiment_summary": post.get("tagline", ""),
                    }
                    if _has_useful_data(data):
                        diag["result"] = "found"
                        return data, diag
                diag["api_body"] = body
        except Exception as exc:
            diag["api_error"] = str(exc)
        diag["result_after_api"] = "api_failed_trying_fallback"
    else:
        diag["api_token"] = "no_token"

    # ── 2. Playwright headless render ────────────────────────────────────────
    try:
        from playwright.sync_api import sync_playwright
        diag["method"] = "playwright"
        url = f"https://www.producthunt.com/products/{slug}"
        diag["playwright_url"] = url
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            ctx = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                )
            )
            page = ctx.new_page()
            page.goto(url, wait_until="networkidle", timeout=25000)
            text = page.inner_text("body")
            browser.close()
        diag["playwright_chars"] = len(text)
        diag["playwright_preview"] = text[:400]
        body_cls = _classify_body(200, text)
        diag["playwright_body_cls"] = body_cls
        if body_cls == "ok" and len(text) > 300:
            data = _llm_extract("Product Hunt", product_name, text[:6000])
            if _has_useful_data(data):
                diag["result"] = "found"
                return data, diag
    except Exception as exc:
        diag["playwright_error"] = str(exc)

    # ── 3. web_search fallback ────────────────────────────────────────────────
    diag["method"] = diag.get("method", "") + "+web_search"
    prompt = (
        f'Search: site:producthunt.com/products/{slug}\n'
        f'Also search: "{product_name}" site:producthunt.com upvotes reviews\n\n'
        "From the search snippets (do NOT fetch pages), extract: "
        "upvote count, review rating, what users say. Return JSON."
    )
    sr = llm.web_search_synthesis(
        prompt, cache_key=f"reviews:ph_v3:{slug}", max_tokens=600,
    )
    diag["search_chars"] = len(sr)
    diag["search_preview"] = sr[:400]
    data = _llm_extract("Product Hunt", product_name, sr) if sr else None
    if _has_useful_data(data):
        diag["result"] = "found"
        return data, diag

    diag["result"] = "no_token" if not PRODUCT_HUNT_TOKEN else "genuinely_no_reviews"
    return None, diag


def _fetch_trustpilot(slug: str, product_name: str, domain: str) -> Tuple[Optional[Dict], Dict]:
    """
    Trustpilot returns Cloudflare 403 on direct fetch.
    Strategy: web_search snippet extraction only.
    Trustpilot TrustScore + review count appear in Google meta descriptions.
    """
    diag = {"platform": "Trustpilot", "method": "web_search_snippets"}

    prompt = (
        f'Search: "{product_name}" Trustpilot TrustScore rating\n'
        f'Also search: trustpilot.com/review/{domain}\n\n'
        f'I need the Trustpilot TrustScore (out of 5) and review count for '
        f'"{product_name}" ({domain}). '
        "Look at search result TITLES and SNIPPETS — Trustpilot pages often show "
        "'TrustScore X.X | N reviews' in the page title or meta description. "
        "Do NOT try to fetch any page — Trustpilot blocks bots. "
        "Report exactly what TrustScore and count appear in the snippets, "
        "plus any review themes mentioned, then return JSON."
    )
    sr = llm.web_search_synthesis(
        prompt, cache_key=f"reviews:tp_v3:{slug}", max_tokens=900,
    )
    diag["search_chars"] = len(sr)
    diag["search_preview"] = sr[:400]

    if not sr or len(sr) < 20:
        diag["result"] = "empty_search"
        return None, diag

    data = _llm_extract("Trustpilot", product_name, sr)
    if _has_useful_data(data):
        diag["result"] = "found"
        return data, diag

    diag["result"] = "genuinely_no_reviews" if len(sr) > 100 else "empty_search"
    return None, diag


def _fetch_capterra(slug: str, product_name: str) -> Tuple[Optional[Dict], Dict]:
    """Capterra: direct HTTP (less aggressively protected than G2/Trustpilot)."""
    url = f"https://www.capterra.com/p/{slug}/"
    diag = {"platform": "Capterra", "method": "direct", "url": url}

    try:
        r = httpx.get(url, headers=_BROWSER_HEADERS, follow_redirects=True, timeout=12)
        text = re.sub(r"<[^>]+>", " ", r.text)
        text = re.sub(r"\s+", " ", text).strip()
        diag["status"] = r.status_code
        diag["body_cls"] = _classify_body(r.status_code, text)
        diag["preview"] = text[:300]
        if diag["body_cls"] == "ok":
            data = _llm_extract("Capterra", product_name, text[:6000])
            if _has_useful_data(data):
                diag["result"] = "found"
                return data, diag
        diag["result"] = diag["body_cls"]
    except Exception as exc:
        diag["error"] = str(exc)
        diag["result"] = "connection_error"

    return None, diag


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_review_sentiment(domain: str) -> List[Dict]:
    """
    Returns list of review dicts (one per platform with useful data).
    Always includes '_all_diag' in result[0] for diagnostic display.
    Cached 7 days. Cache key v3 to bust old broken entries.
    """
    cache_key = f"reviews_v3:{domain}"
    cached = get_cache_obj(cache_key)
    if cached is not None:
        return cached

    slug = _domain_to_slug(domain)
    product_name = _domain_to_product_name(domain)

    logger.info("[reviews] domain=%s slug=%s product=%s", domain, slug, product_name)

    results: List[Dict] = []
    all_diag: List[Dict] = []

    for fetcher, args in [
        (_fetch_g2,          (slug, product_name)),
        (_fetch_producthunt, (slug, product_name)),
        (_fetch_trustpilot,  (slug, product_name, domain)),
        (_fetch_capterra,    (slug, product_name)),
    ]:
        try:
            data, diag = fetcher(*args)
        except Exception as exc:
            data, diag = None, {"platform": fetcher.__name__, "error": str(exc), "result": "exception"}
        logger.info("[reviews] %s", diag)
        all_diag.append(diag)
        if data:
            data.pop("_diag", None)
            results.append(data)

    if not results:
        out = [{"_diagnostic_only": True, "_all_diag": all_diag}]
        # Don't cache a total failure — allow retry on next visit
    else:
        results[0]["_all_diag"] = all_diag
        out = results
        set_cache_obj(cache_key, out, TTL_WEB_PAGES)
    return out
