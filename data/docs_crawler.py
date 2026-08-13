"""Phase 10: Documentation crawler — llms.txt, docs subdomain, image harvest."""
from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import httpx
import trafilatura

from config import MAX_DOCS_PAGES, MAX_IMAGES, TTL_WEB_PAGES, WEB_RATE_LIMIT
from data.cache import get_cache_obj, record_freshness, set_cache_obj

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": "DeepDiveBot/1.0 (research-only; contact vandit@deductive.ai)",
    "Accept": "text/html,application/xhtml+xml,text/plain,*/*;q=0.8",
}

_MIN_INTERVAL = 1.0 / WEB_RATE_LIMIT
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
    """Return the Response (any status) or None on network error / timeout."""
    try:
        _rate_limit()
        return httpx.get(url, headers=_HEADERS, follow_redirects=True, timeout=timeout)
    except Exception as exc:
        logger.debug("docs GET %s → %s", url, exc)
        return None


def _probe_status(r: Optional[httpx.Response]) -> str:
    if r is None:
        return "timeout/error"
    ct = r.headers.get("content-type", "")
    return f"{r.status_code} {ct.split(';')[0].strip()!r}"


def _is_valid_llms_content(text: str, content_type: str) -> bool:
    """Return True only if the response looks like a real llms.txt file, not SPA HTML."""
    # HTML content-type is always a SPA/catchall page
    if "text/html" in content_type:
        return False
    # Content that begins with HTML doctype/tags is SPA catchall despite non-HTML content-type
    stripped = text.lstrip()
    if stripped.startswith("<!DOCTYPE") or stripped.lower().startswith("<html"):
        return False
    return True


# ── llms.txt helpers ──────────────────────────────────────────────────────────

def _parse_llms_txt(text: str, origin: str) -> List[str]:
    """
    Extract URLs from llms.txt / llms-full.txt.
    Both formats: markdown links [title](url) and bare https:// lines.
    """
    urls: List[str] = []
    seen: Set[str] = set()
    for line in text.splitlines():
        line = line.strip()
        for m in re.finditer(r'\[.*?\]\((https?://[^)]+)\)', line):
            u = m.group(1)
            if u not in seen:
                seen.add(u)
                urls.append(u)
        if re.match(r'https?://', line):
            u = line.split()[0]
            if u not in seen:
                seen.add(u)
                urls.append(u)
    return urls


def _try_llms_txt(origin: str, probe_log: List[str]) -> List[str]:
    """Try llms-full.txt and llms.txt at origin. Rejects HTML SPA responses."""
    for path in ["/llms-full.txt", "/llms.txt"]:
        url = origin + path
        r = _safe_get(url)
        status = _probe_status(r)
        if r is None or r.status_code != 200:
            probe_log.append(f"{url}: {status}")
            continue
        ct = r.headers.get("content-type", "")
        if not _is_valid_llms_content(r.text, ct):
            probe_log.append(f"{url}: {status} — HTML/SPA catchall, skipped")
            continue
        urls = _parse_llms_txt(r.text, origin)
        if urls:
            probe_log.append(f"{url}: {status} — {len(urls)} URLs parsed ✓")
            logger.info("Found %d URLs via %s", len(urls), path)
            return urls
        probe_log.append(f"{url}: {status} — 0 URLs found (empty file)")
    return []


# ── Docs subdomain + conventional paths ──────────────────────────────────────

_DOCS_PATHS = [
    "/docs", "/documentation", "/guide", "/guides", "/manual",
    "/reference", "/api", "/api-reference", "/sdk", "/developer",
    "/developers", "/learn", "/tutorials", "/tutorial", "/handbook",
    "/help", "/support",
]

_HREF_RE = re.compile(r'href=["\']((?:https?://[^"\'?#]+|/[^"\'?#]{3,}))["\']', re.I)


def _scrape_doc_links(html: str, origin: str, limit: int = 100) -> List[str]:
    parsed_origin = urlparse(origin)
    links: List[str] = []
    seen: Set[str] = set()
    for m in _HREF_RE.finditer(html):
        href = m.group(1)
        if href.startswith("/"):
            href = origin + href
        u = urlparse(href)
        if u.netloc == parsed_origin.netloc or not u.netloc:
            full = href if href.startswith("http") else origin + href
            if full not in seen:
                seen.add(full)
                links.append(full)
        if len(links) >= limit:
            break
    return links


def _try_docs_subdomain(base_url: str, probe_log: List[str]) -> List[str]:
    parsed = urlparse(base_url)
    bare = parsed.netloc.lstrip("www.")
    docs_origin = f"{parsed.scheme}://docs.{bare}"

    # Try llms.txt at the docs subdomain first (before scraping the SPA)
    llms_urls = _try_llms_txt(docs_origin, probe_log)
    if llms_urls:
        return llms_urls

    # Then try scraping static links from the docs subdomain root
    r = _safe_get(docs_origin)
    status = _probe_status(r)
    if not r or r.status_code != 200:
        probe_log.append(f"{docs_origin}: {status}")
        return []

    links = _scrape_doc_links(r.text, docs_origin, limit=150)
    if links:
        probe_log.append(f"{docs_origin}: {status} — {len(links)} static links scraped ✓")
        logger.info("Found %d links on docs subdomain", len(links))
        return [docs_origin] + links
    probe_log.append(f"{docs_origin}: {status} — JS SPA, 0 static <a href> links found")
    return []


def _try_docs_paths(origin: str, probe_log: List[str]) -> List[str]:
    urls: List[str] = []
    seen: Set[str] = set()
    for path in _DOCS_PATHS:
        full = origin + path
        if full in seen:
            continue
        seen.add(full)
        r = _safe_get(full)
        status = _probe_status(r)
        if not r or r.status_code != 200:
            probe_log.append(f"{full}: {status}")
            continue
        probe_log.append(f"{full}: {status} — found, scraping links")
        urls.append(full)
        sub_links = _scrape_doc_links(r.text, origin, limit=50)
        for lnk in sub_links:
            if lnk not in seen:
                seen.add(lnk)
                urls.append(lnk)
        if len(urls) >= MAX_DOCS_PAGES * 4:
            break
    return urls


# ── Image harvesting ──────────────────────────────────────────────────────────

_IMG_TAG_RE = re.compile(
    r'<img[^>]+src=["\']([^"\'?]+)["\'][^>]*(?:width=["\'](\d+)["\'])?[^>]*>',
    re.I | re.S,
)
_EXCLUDE_PATTERNS = re.compile(
    r'(logo|icon|favicon|avatar|badge|button|arrow|chevron|sprite|'
    r'\.svg$|1x1|pixel|tracking)',
    re.I,
)


def _extract_images_from_html(html: str, page_url: str) -> List[Dict]:
    origin = f"{urlparse(page_url).scheme}://{urlparse(page_url).netloc}"
    images: List[Dict] = []
    for m in _IMG_TAG_RE.finditer(html):
        src = m.group(1)
        width_str = m.group(2) or "0"
        try:
            width = int(width_str)
        except ValueError:
            width = 0

        if _EXCLUDE_PATTERNS.search(src):
            continue
        if src.endswith(".svg") or src.endswith(".gif"):
            continue
        if not src.startswith("http"):
            src = urljoin(page_url, src)

        images.append({"url": src, "width": width, "source_page": page_url})
    return [img for img in images if img["width"] == 0 or img["width"] >= 300]


def harvest_images(pages: List[Dict], max_images: int = MAX_IMAGES) -> List[Dict]:
    """
    Extract screenshot-like images from fetched pages.
    Filters: ≥300px wide (when measurable), excludes SVG/logo/icon.
    Returns list of {url, width, source_page}.
    """
    seen_urls: Set[str] = set()
    results: List[Dict] = []
    for page in pages:
        html = page.get("html_snippet", "")
        page_url = page.get("url", "")
        for img in _extract_images_from_html(html, page_url):
            if img["url"] not in seen_urls:
                seen_urls.add(img["url"])
                results.append(img)
            if len(results) >= max_images:
                break
        if len(results) >= max_images:
            break
    return results[:max_images]


# ── Public API ────────────────────────────────────────────────────────────────

def discover_docs(base_url: str) -> Tuple[List[str], List[str]]:
    """
    Discover documentation URLs for a site.
    Priority: llms.txt (root) → llms.txt (docs subdomain) → docs subdomain scrape → /docs paths.

    Returns (urls, probe_log) where probe_log is a list of per-probe status strings.
    Cached; re-runs return cached urls with a fresh note in probe_log.
    """
    cache_key = f"docs:discover:{base_url}"
    cached = get_cache_obj(cache_key)
    if cached is not None:
        cached_urls = cached.get("urls", []) if isinstance(cached, dict) else cached
        log = cached.get("probe_log", ["(served from cache)"]) if isinstance(cached, dict) else ["(served from cache)"]
        return cached_urls, log

    parsed = urlparse(base_url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    probe_log: List[str] = []

    # 1. llms.txt at root
    urls = _try_llms_txt(origin, probe_log)

    # 2. llms.txt + scrape at docs subdomain
    if not urls:
        urls = _try_docs_subdomain(base_url, probe_log)

    # 3. Conventional /docs paths
    if not urls:
        urls = _try_docs_paths(origin, probe_log)

    # Deduplicate
    seen: Set[str] = set()
    deduped: List[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)

    result_urls = deduped[:MAX_DOCS_PAGES * 4]
    set_cache_obj(cache_key, {"urls": result_urls, "probe_log": probe_log}, ttl=TTL_WEB_PAGES)
    record_freshness("docs_discover", base_url, TTL_WEB_PAGES)
    return result_urls, probe_log


def fetch_docs_pages(urls: List[str]) -> List[Dict]:
    """
    Fetch and extract text from documentation URLs.
    Stores full HTML snippet for image harvesting.
    Returns list of page dicts: {url, text, html_snippet, content_hash}.
    """
    pages: List[Dict] = []
    for url in urls[:MAX_DOCS_PAGES]:
        cache_key = f"docs:page:{url}"
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
                "text": text[:10000],
                "html_snippet": html[:8000],
                "content_hash": content_hash,
            }
            set_cache_obj(cache_key, page, ttl=TTL_WEB_PAGES)
            pages.append(page)
        except Exception as exc:
            logger.debug("fetch_docs_pages %s: %s", url, exc)
    return pages
