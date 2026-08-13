"""Phase 9: Web intelligence — URL discovery, rate-limited fetching, text extraction."""
from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
import urllib.robotparser
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

import httpx
import trafilatura

from config import MAX_WEB_PAGES, TTL_WEB_PAGES, WEB_RATE_LIMIT
from data.cache import get_cache_obj, record_freshness, set_cache_obj

logger = logging.getLogger(__name__)

_CONVENTIONAL_PATHS = [
    "/blog", "/news", "/press", "/newsroom", "/customers", "/case-studies",
    "/case_studies", "/changelog", "/updates", "/about", "/pricing",
    "/product", "/features", "/enterprise", "/partners", "/resources",
    "/solutions", "/integrations",
]

_HEADERS = {
    "User-Agent": "DeepDiveBot/1.0 (research-only; contact vandit@deductive.ai)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

_MIN_INTERVAL = 1.0 / WEB_RATE_LIMIT  # seconds between requests
_last_req_at: float = 0.0
_rate_lock = threading.Lock()


def _rate_limit() -> None:
    global _last_req_at
    with _rate_lock:
        now = time.monotonic()
        wait = _MIN_INTERVAL - (now - _last_req_at)
        if wait > 0:
            time.sleep(wait)
        _last_req_at = time.monotonic()


def _safe_get(url: str, timeout: int = 12) -> Optional[httpx.Response]:
    try:
        _rate_limit()
        r = httpx.get(url, headers=_HEADERS, follow_redirects=True, timeout=timeout)
        return r if r.status_code == 200 else None
    except Exception as exc:
        logger.debug("GET %s → %s", url, exc)
        return None


# ── Robots ────────────────────────────────────────────────────────────────────

def _build_robots(origin: str) -> urllib.robotparser.RobotFileParser:
    rp = urllib.robotparser.RobotFileParser()
    r = _safe_get(urljoin(origin, "/robots.txt"))
    if r:
        rp.parse(r.text.splitlines())
    rp.set_url(urljoin(origin, "/robots.txt"))
    return rp


def _allowed(rp: urllib.robotparser.RobotFileParser, url: str) -> bool:
    try:
        return rp.can_fetch("*", url)
    except Exception:
        return True


# ── Sitemaps ──────────────────────────────────────────────────────────────────

def _parse_sitemap(url: str, seen: Set[str], budget: int) -> List[str]:
    if len(seen) >= budget or url in seen:
        return []
    seen.add(url)
    r = _safe_get(url)
    if not r:
        return []
    urls: List[str] = []
    try:
        from lxml import etree
        root = etree.fromstring(r.content)
        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        for idx_loc in root.findall(".//sm:sitemap/sm:loc", ns):
            if len(seen) >= budget:
                break
            child = (idx_loc.text or "").strip()
            if child:
                urls.extend(_parse_sitemap(child, seen, budget))
        for loc in root.findall(".//sm:url/sm:loc", ns):
            if len(seen) >= budget:
                break
            href = (loc.text or "").strip()
            if href and href not in seen:
                seen.add(href)
                urls.append(href)
    except Exception as exc:
        logger.debug("sitemap parse %s: %s", url, exc)
    return urls


def _discover_via_sitemap(origin: str, rp: urllib.robotparser.RobotFileParser) -> List[str]:
    seen: Set[str] = set()
    urls: List[str] = []
    # Prefer sitemaps declared in robots.txt
    for sm in (rp.site_maps() or []):
        urls.extend(_parse_sitemap(sm, seen, 500))
    if not urls:
        for path in ["/sitemap.xml", "/sitemap_index.xml", "/sitemap.xml.gz"]:
            found = _parse_sitemap(urljoin(origin, path), seen, 500)
            if found:
                urls.extend(found)
                break
    return urls


# ── RSS / Atom ────────────────────────────────────────────────────────────────

_FEED_HREF_RE = re.compile(
    r'<link[^>]*?(?:type=["\']application/(?:rss|atom)\+xml["\'][^>]*?href=["\']([^"\']+)["\']|'
    r'href=["\']([^"\']+)["\'][^>]*?type=["\']application/(?:rss|atom)\+xml["\'])',
    re.I | re.S,
)


def _find_feed_urls(html: str, origin: str) -> List[str]:
    urls: List[str] = []
    for m in _FEED_HREF_RE.finditer(html):
        href = m.group(1) or m.group(2) or ""
        if href:
            href = href if href.startswith("http") else urljoin(origin, href)
            urls.append(href)
    return list(set(urls))


def _parse_feed(feed_url: str, origin: str) -> List[str]:
    r = _safe_get(feed_url)
    if not r:
        return []
    urls: List[str] = []
    try:
        from lxml import etree
        root = etree.fromstring(r.content)
        # RSS
        for item in root.findall(".//item/link"):
            if item.text:
                urls.append(item.text.strip())
        # Atom
        ns = "{http://www.w3.org/2005/Atom}"
        for entry in root.findall(f".//{ns}entry"):
            link = entry.find(f"{ns}link")
            if link is not None:
                href = link.get("href", "")
                if href:
                    urls.append(href)
    except Exception:
        pass
    return [u for u in urls if urlparse(u).netloc == urlparse(origin).netloc]


# ── Internal link scraper (conventional paths) ────────────────────────────────

_HREF_RE = re.compile(r'href=["\'](/[^"\'?#]{3,})["\']', re.I)


def _scrape_internal_links(html: str, origin: str, limit: int = 50) -> List[str]:
    links: List[str] = []
    seen: Set[str] = set()
    for m in _HREF_RE.finditer(html):
        path = m.group(1)
        url = origin + path
        if url not in seen:
            seen.add(url)
            links.append(url)
        if len(links) >= limit:
            break
    return links


# ── Public API ────────────────────────────────────────────────────────────────

def discover_urls(base_url: str, max_pages: int = MAX_WEB_PAGES) -> List[str]:
    """
    Discover crawlable URLs for a site in priority order:
    sitemap → RSS feed → conventional paths → internal link scrape.
    Respects robots.txt. Deduplicates. Returns up to max_pages URLs.
    """
    cache_key = f"webintel:discover:{base_url}"
    cached = get_cache_obj(cache_key)
    if cached is not None:
        return cached

    parsed = urlparse(base_url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    rp = _build_robots(origin)

    collected: List[str] = []
    seen: Set[str] = set()

    def _add(url: str) -> bool:
        if url not in seen and _allowed(rp, url) and urlparse(url).netloc == parsed.netloc:
            seen.add(url)
            collected.append(url)
            return True
        return False

    # Root always first
    _add(base_url)

    # Sitemap (highest quality)
    for u in _discover_via_sitemap(origin, rp):
        _add(u)
        if len(collected) >= max_pages * 5:
            break

    # RSS / Atom feeds
    root_r = _safe_get(base_url)
    if root_r:
        for feed_url in _find_feed_urls(root_r.text, origin):
            for u in _parse_feed(feed_url, origin):
                _add(u)

    # Conventional paths — probe + one-level link scrape
    if len(collected) < max_pages:
        for path in _CONVENTIONAL_PATHS:
            full = origin + path
            if full not in seen:
                r = _safe_get(full)
                if r:
                    _add(full)
                    for lnk in _scrape_internal_links(r.text, origin, limit=30):
                        _add(lnk)
            if len(collected) >= max_pages * 3:
                break

    # Score & rank: prefer blog, news, customer, case-study, changelog URLs
    def _priority(url: str) -> int:
        low = url.lower()
        for kw in ["customer", "case-stud", "changelog", "blog", "news", "press"]:
            if kw in low:
                return 0
        for kw in ["product", "feature", "pricing", "about"]:
            if kw in low:
                return 1
        return 2

    ordered = sorted(collected, key=_priority)
    result = ordered[:max_pages]
    set_cache_obj(cache_key, result, ttl=TTL_WEB_PAGES)
    record_freshness("webintel_discover", base_url, TTL_WEB_PAGES)
    return result


def fetch_pages(urls: List[str]) -> List[Dict]:
    """
    Fetch and extract text from a list of URLs.
    Returns list of page dicts: {url, text, html_snippet, content_hash, status}.
    """
    pages: List[Dict] = []
    for url in urls:
        cache_key = f"webintel:page:{url}"
        cached = get_cache_obj(cache_key)
        if cached is not None:
            pages.append(cached)
            continue
        try:
            _rate_limit()
            r = httpx.get(url, headers=_HEADERS, follow_redirects=True, timeout=15)
            if r.status_code != 200:
                continue
            html = r.text
            text = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=True,
                no_fallback=False,
                favor_recall=True,
            ) or ""
            content_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
            page: Dict = {
                "url": url,
                "text": text[:8000],
                "html_snippet": html[:4000],
                "content_hash": content_hash,
                "status": r.status_code,
            }
            set_cache_obj(cache_key, page, ttl=TTL_WEB_PAGES)
            pages.append(page)
        except Exception as exc:
            logger.debug("fetch_pages %s: %s", url, exc)
    return pages
