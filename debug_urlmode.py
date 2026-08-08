"""Step 2: Run URL-mode discovery + fetch pipeline for sierra.ai, no Streamlit."""
import sys, logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

URL = "https://sierra.ai"

print(f"\n{'='*60}")
print(f"URL-mode pipeline test: {URL}")
print(f"{'='*60}\n")

# ── 1. robots.txt ─────────────────────────────────────────────────────────────
print("--- robots.txt ---")
import urllib.robotparser, httpx
rp = urllib.robotparser.RobotFileParser()
try:
    r = httpx.get(URL + "/robots.txt", timeout=10, follow_redirects=True)
    print(f"  Status: {r.status_code}")
    lines = r.text.splitlines()
    print(f"  Lines: {len(lines)}")
    for l in lines[:10]:
        print(f"  {l}")
    rp.parse(lines)
    sitemaps = rp.site_maps() or []
    print(f"  Sitemaps declared: {sitemaps}")
except Exception as e:
    print(f"  ERROR: {e}")

print()

# ── 2. Sitemap probe ──────────────────────────────────────────────────────────
print("--- Sitemap probe ---")
for path in ["/sitemap.xml", "/sitemap_index.xml"]:
    try:
        r = httpx.get(URL + path, timeout=10, follow_redirects=True)
        print(f"  {path}: HTTP {r.status_code}, {len(r.content)} bytes")
        if r.status_code == 200:
            print(f"  First 300 chars: {r.text[:300]!r}")
    except Exception as e:
        print(f"  {path}: ERROR {e}")

print()

# ── 3. llms.txt probe ─────────────────────────────────────────────────────────
print("--- llms.txt probe ---")
for path in ["/llms-full.txt", "/llms.txt"]:
    try:
        r = httpx.get(URL + path, timeout=10, follow_redirects=True)
        print(f"  {path}: HTTP {r.status_code}")
        if r.status_code == 200:
            lines = r.text.splitlines()
            print(f"  Lines: {len(lines)}, first 5: {lines[:5]}")
    except Exception as e:
        print(f"  {path}: ERROR {e}")

print()

# ── 4. Full discover_urls ─────────────────────────────────────────────────────
print("--- discover_urls() ---")
try:
    from data.webintel import discover_urls
    urls = discover_urls(URL, max_pages=40)
    print(f"  Found {len(urls)} URLs")
    print(f"  First 5 samples:")
    for u in urls[:5]:
        print(f"    {u}")
    print(f"  Last 5:")
    for u in urls[-5:]:
        print(f"    {u}")
except Exception as e:
    import traceback
    print(f"  ERROR: {e}")
    traceback.print_exc()

print()

# ── 5. fetch_pages ────────────────────────────────────────────────────────────
print("--- fetch_pages() (first 5 URLs only) ---")
try:
    from data.webintel import fetch_pages
    sample = urls[:5]
    pages = fetch_pages(sample)
    print(f"  Fetched {len(pages)}/{len(sample)} pages")
    for p in pages:
        print(f"  [{p['status']}] {p['url']} — {len(p['text'])} chars text, hash={p['content_hash']}")
        if p['text']:
            print(f"    Preview: {p['text'][:120]!r}")
except Exception as e:
    import traceback
    print(f"  ERROR: {e}")
    traceback.print_exc()

print()

# ── 6. extract_page_intel (first page only) ───────────────────────────────────
print("--- extract_page_intel() (first page only) ---")
try:
    from analysis.company import extract_page_intel
    intel = extract_page_intel(pages[0]) if pages else None
    if intel:
        print(f"  page_type: {intel.page_type}")
        print(f"  named_customers: {intel.named_customers[:5]}")
        print(f"  feature_claims: {intel.feature_claims[:3]}")
    else:
        print("  No intel extracted (text too short or Haiku call failed)")
except Exception as e:
    import traceback
    print(f"  ERROR: {e}")
    traceback.print_exc()

print()
print("Pipeline isolation test complete.")
