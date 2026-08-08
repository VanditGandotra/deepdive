"""EDGAR filings + XBRL facts via edgartools (primary) and direct SEC API (fallback)."""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import httpx

from config import (
    EDGAR_RATE_LIMIT, EDGAR_USER_AGENT, EDGAR_XBRL_FACTS_URL,
    TTL_FILINGS,
)
from data.cache import get_cache_obj, record_freshness, set_cache_obj
from data.resilience import retry, with_fallback

logger = logging.getLogger(__name__)
_last_edgar_request = 0.0

# One-time EDGAR identity setup
_identity_set = False


def _set_edgar_identity() -> None:
    global _identity_set
    if _identity_set:
        return
    try:
        from edgar import set_identity  # type: ignore
        set_identity(EDGAR_USER_AGENT)
        _identity_set = True
    except ImportError:
        logger.warning("edgartools not installed; EDGAR features limited to direct XBRL API")


def _rate_limit() -> None:
    global _last_edgar_request
    min_interval = 1.0 / EDGAR_RATE_LIMIT
    elapsed = time.time() - _last_edgar_request
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    _last_edgar_request = time.time()


# ── XBRL company facts (direct SEC API — always available) ───────────────────

@retry(max_attempts=3, base_delay=2.0, retryable=(httpx.HTTPError, httpx.TimeoutException))
def _fetch_xbrl_facts_direct(cik: str) -> Dict[str, Any]:
    _rate_limit()
    url = EDGAR_XBRL_FACTS_URL.format(cik=str(cik).zfill(10))
    with httpx.Client(timeout=30) as client:
        resp = client.get(url, headers={"User-Agent": EDGAR_USER_AGENT})
        resp.raise_for_status()
    return resp.json()


def _get_cik_direct(ticker: str) -> Optional[str]:
    """Resolve ticker → CIK via SEC company_tickers.json."""
    cache_key = f"edgar:cik:{ticker.upper()}"
    cached = get_cache_obj(cache_key)
    if cached:
        return str(cached)
    try:
        _rate_limit()
        with httpx.Client(timeout=15) as client:
            resp = client.get(
                "https://www.sec.gov/files/company_tickers.json",
                headers={"User-Agent": EDGAR_USER_AGENT},
            )
            resp.raise_for_status()
        tickers_map = resp.json()
        for _idx, entry in tickers_map.items():
            if entry.get("ticker", "").upper() == ticker.upper():
                cik = str(entry["cik_str"]).zfill(10)
                set_cache_obj(cache_key, cik, TTL_FILINGS, source="sec")
                return cik
    except Exception as exc:
        logger.warning("CIK lookup failed for %s: %s", ticker, exc)
    return None


def get_xbrl_facts(ticker: str) -> Optional[Dict[str, Any]]:
    """Return SEC XBRL company facts for ticker. Cached 30d."""
    cache_key = f"edgar:xbrl:{ticker.upper()}"
    cached = get_cache_obj(cache_key)
    if cached:
        return cached

    def _via_edgartools() -> Dict[str, Any]:
        _set_edgar_identity()
        from edgar import Company  # type: ignore
        _rate_limit()
        company = Company(ticker)
        cik = str(company.cik).zfill(10)
        return _fetch_xbrl_facts_direct(cik)

    def _via_direct() -> Dict[str, Any]:
        cik = _get_cik_direct(ticker)
        if not cik:
            raise ValueError(f"CIK not found for {ticker}")
        return _fetch_xbrl_facts_direct(cik)

    try:
        facts = with_fallback(_via_edgartools, _via_direct, label=f"XBRL:{ticker}")
        set_cache_obj(cache_key, facts, TTL_FILINGS, source="edgar")
        record_freshness(cache_key, "edgar", TTL_FILINGS)
        return facts
    except Exception as exc:
        logger.warning("XBRL facts unavailable for %s: %s", ticker, exc)
        return None


def extract_xbrl_metric(
    facts: Dict[str, Any],
    concept: str,
    namespace: str = "us-gaap",
    form: str = "10-K",
    n_periods: int = 5,
) -> List[Dict[str, Any]]:
    """Extract last N annual values of a concept from XBRL facts."""
    try:
        units = (
            facts.get("facts", {})
                 .get(namespace, {})
                 .get(concept, {})
                 .get("units", {})
        )
        usd = units.get("USD") or units.get("shares") or next(iter(units.values()), [])
        annual = [
            r for r in usd
            if r.get("form") == form and r.get("fp") == "FY"
        ]
        annual.sort(key=lambda r: r.get("end", ""), reverse=True)
        return annual[:n_periods]
    except Exception:
        return []


def compute_edgar_ttm(
    facts: Dict[str, Any],
    concept: str,
    namespace: str = "us-gaap",
) -> Optional[tuple]:
    """
    Compute TTM (trailing twelve months) value by summing the 4 most recent
    individual quarterly filings (~90-day duration each).

    Returns (value: float, ttm_end_date: str) or None if < 4 quarters found.
    """
    from datetime import date as _date

    try:
        units = (
            facts.get("facts", {})
                 .get(namespace, {})
                 .get(concept, {})
                 .get("units", {})
        )
        all_records = units.get("USD") or units.get("shares") or next(iter(units.values()), [])

        # Only 10-Q and 10-K filings carry period-level detail
        candidates = [r for r in all_records if r.get("form") in ("10-Q", "10-K")]

        # Deduplicate: for each (start, end) pair keep the latest-filed record
        best: Dict[tuple, Dict] = {}
        for r in candidates:
            key = (r.get("start", ""), r.get("end", ""))
            if key not in best or r.get("filed", "") > best[key].get("filed", ""):
                best[key] = r

        # Keep only single-quarter records: duration 60–120 days
        quarters = []
        for r in best.values():
            start_str = r.get("start")
            end_str = r.get("end")
            if not start_str or not end_str:
                continue
            try:
                s = _date.fromisoformat(start_str)
                e = _date.fromisoformat(end_str)
                if 60 <= (e - s).days <= 120 and r.get("val") is not None:
                    quarters.append((e, float(r["val"])))
            except (ValueError, TypeError):
                continue

        if len(quarters) < 4:
            return None

        # Sort newest-first, pick 4 most recent non-duplicate end dates
        quarters.sort(key=lambda x: x[0], reverse=True)
        selected: List[tuple] = []
        seen_ends: set = set()
        for end_d, val in quarters:
            if end_d not in seen_ends:
                selected.append((end_d, val))
                seen_ends.add(end_d)
            if len(selected) == 4:
                break

        if len(selected) < 4:
            return None

        ttm_value = sum(v for _, v in selected)
        ttm_end = max(e for e, _ in selected).isoformat()
        return (ttm_value, ttm_end)

    except Exception:
        return None


def build_xbrl_composite(
    facts: Dict[str, Any],
    slots: List[Dict],
) -> Dict[str, Any]:
    """
    Build a composite metric by summing component XBRL instant values.

    slots: list of {
        "aliases": ["TagA", "TagB", ...],   # try in order
        "namespace": "us-gaap",             # default
        "required": True,                   # if True and missing → incomplete
        "label": "friendly name",
    }

    Returns {
        "value": float,      # sum of all found components (None if any required missing)
        "end": str,          # balance-sheet date (all must match; None if mixed)
        "components": {label: value},
        "missing": [label],  # required components with no XBRL data
        "incomplete": bool,  # True if any required component is missing
        "date_mismatch": bool,
    }
    """
    found_components: Dict[str, float] = {}
    found_ends: List[str] = []
    missing_labels: List[str] = []

    for slot in slots:
        aliases = slot.get("aliases", [])
        ns = slot.get("namespace", "us-gaap")
        required = slot.get("required", True)
        label = slot.get("label") or aliases[0] if aliases else "unknown"

        slot_result: Optional[tuple] = None
        for alias in aliases:
            slot_result = extract_xbrl_instant(facts, alias, ns)
            if slot_result is not None:
                break

        if slot_result is None:
            if required:
                missing_labels.append(label)
            # Optional missing slots are silently skipped (add 0)
        else:
            val, end = slot_result
            found_components[label] = val
            found_ends.append(end)

    # Check date consistency across components
    unique_ends = list(dict.fromkeys(found_ends))  # preserve order, deduplicate
    date_mismatch = len(unique_ends) > 1
    consensus_end = unique_ends[0] if unique_ends else ""

    incomplete = len(missing_labels) > 0
    total = sum(found_components.values()) if found_components and not incomplete else None

    return {
        "value": total,
        "end": consensus_end,
        "components": found_components,
        "missing": missing_labels,
        "incomplete": incomplete,
        "date_mismatch": date_mismatch,
    }


def extract_xbrl_instant(
    facts: Dict[str, Any],
    concept: str,
    namespace: str = "us-gaap",
) -> Optional[tuple]:
    """
    Return the most recent point-in-time (balance-sheet / instant) value.

    Returns (value: float, end_date: str) or None.
    """
    try:
        units = (
            facts.get("facts", {})
                 .get(namespace, {})
                 .get(concept, {})
                 .get("units", {})
        )
        all_records = units.get("USD") or units.get("shares") or next(iter(units.values()), [])

        # Accept 10-Q and 10-K; prefer most recent end date + latest filed
        candidates = [
            r for r in all_records
            if r.get("form") in ("10-Q", "10-K", "10-K/A", "DEF 14A")
            and r.get("val") is not None
        ]
        if not candidates:
            return None

        # Sort by end date desc, then by filed date desc for ties
        candidates.sort(
            key=lambda r: (r.get("end", ""), r.get("filed", "")),
            reverse=True,
        )
        best = candidates[0]
        return (float(best["val"]), best.get("end", ""))

    except Exception:
        return None


# ── 10-K section extraction ───────────────────────────────────────────────────

@retry(max_attempts=3, base_delay=3.0)
def get_10k_sections(
    ticker: str,
    items: List[str] | None = None,
    max_chars_per_item: int = 20_000,
) -> Dict[str, str]:
    """
    Return a dict of {item_key: text} for the latest 10-K.
    items default: ["1", "1A", "7"]
    Falls back to an empty dict with a warning if edgartools is absent.

    edgartools TenK API:
      tenk.business            → Item 1
      tenk.management_discussion → Item 7
      tenk.get_item_with_part(part, item) → any item by part/item label
    """
    if items is None:
        items = ["1", "1A", "7"]

    cache_key = f"edgar:10k_sections:{ticker.upper()}:{'_'.join(items)}"
    cached = get_cache_obj(cache_key)
    if cached:
        return cached

    _set_edgar_identity()
    try:
        from edgar import Company  # type: ignore
    except ImportError:
        logger.warning("edgartools not installed; 10-K sections unavailable")
        return {}

    # item_key → (part, attribute_on_tenk_obj)
    # Fall back to get_item_with_part for anything not covered by named attributes
    _ITEM_MAP: Dict[str, tuple] = {
        "1":  ("Part I",  "business"),
        "1A": ("Part I",  None),
        "1B": ("Part I",  None),
        "7":  ("Part II", "management_discussion"),
        "7A": ("Part II", None),
    }

    try:
        _rate_limit()
        company = Company(ticker)
        filings = company.get_filings(form="10-K")
        if not filings:
            logger.warning("No 10-K filings found for %s", ticker)
            return {}

        filing = filings[0]
        _rate_limit()

        sections: Dict[str, str] = {}
        tenk = filing.obj()

        for item_key in items:
            part, attr = _ITEM_MAP.get(item_key, (None, None))
            content = None

            # Try named attribute first (fastest)
            if attr:
                content = getattr(tenk, attr, None)

            # Fall back to get_item_with_part
            if content is None and part:
                try:
                    content = tenk.get_item_with_part(part, f"Item {item_key}")
                except Exception:
                    pass

            # Last resort: scan full text
            if content is None:
                try:
                    full_text = str(filing.text or "")[:400_000]
                    label = f"ITEM {item_key}"
                    idx = full_text.upper().find(label)
                    if idx >= 0:
                        content = full_text[idx : idx + max_chars_per_item]
                except Exception:
                    pass

            if content:
                sections[f"Item {item_key}"] = str(content)[:max_chars_per_item]

        if sections:
            set_cache_obj(cache_key, sections, TTL_FILINGS, source="edgar")
            record_freshness(cache_key, "edgar", TTL_FILINGS)

        return sections

    except Exception as exc:
        logger.warning("get_10k_sections failed for %s: %s", ticker, exc)
        return {}


def get_filing_metadata(ticker: str, form: str = "10-K", limit: int = 5) -> List[Dict[str, Any]]:
    """Return metadata (date, accession_no, url) for recent filings."""
    cache_key = f"edgar:filings_meta:{ticker.upper()}:{form}:{limit}"
    cached = get_cache_obj(cache_key)
    if cached:
        return cached

    _set_edgar_identity()
    try:
        from edgar import Company  # type: ignore
        _rate_limit()
        company = Company(ticker)
        filings = company.get_filings(form=form)
        result = []
        for f in filings:
            result.append({
                "form": str(f.form),
                "filed_date": str(f.filing_date),
                "accession_no": str(getattr(f, "accession_no", "")),
            })
        set_cache_obj(cache_key, result, TTL_FILINGS, source="edgar")
        return result
    except Exception as exc:
        logger.warning("Filing metadata unavailable for %s: %s", ticker, exc)
        return []
