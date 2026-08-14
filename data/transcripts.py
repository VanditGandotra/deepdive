"""Multi-provider earnings transcript fetcher with SQLite cache.

Provider chain (priority order):
  1. RoicProvider      — roic.ai v3 API (ROIC_API_KEY)
  2. FmpProvider       — Financial Modeling Prep free tier (FMP_API_KEY, 250 req/day)
  3. FinnhubProvider   — Finnhub paid plan (FINNHUB_API_KEY, Basic+ required)
  4. DefeatBetaProvider — defeatbeta-api parquet (free, US-focused)
  5. ApiNinjasProvider — api-ninjas.com (API_NINJAS_PREMIUM=true)
  6. MotleyFoolProvider — web-scrape fallback via DuckDuckGo (always available)

Ticker normalisation: FMP, Finnhub, and MotleyFool all accept US-listed tickers
(ASML, TSM, etc.) directly. RoicProvider tries a broader exchange prefix list
including AMS and EURONEXT so foreign dual-listings are found.
"""
from __future__ import annotations

import logging
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import httpx

from config import (
    API_NINJAS_KEY, API_NINJAS_PREMIUM, API_NINJAS_TRANSCRIPT_URL,
    FINNHUB_API_KEY, FINNHUB_TRANSCRIPT_LIST_URL, FINNHUB_TRANSCRIPT_URL,
    FMP_API_KEY, FMP_TRANSCRIPT_URL,
    ROIC_API_KEY, ROIC_TRANSCRIPT_DETAIL_URL, ROIC_TRANSCRIPT_LIST_URL,
    TTL_TRANSCRIPTS,
)
from data.cache import get_cache_obj, record_freshness, set_cache_obj
from data.resilience import SourceUnavailable, retry

logger = logging.getLogger(__name__)

_TRANSCRIPT_HEALTH: Dict[str, List[Dict]] = {}

# Dual-class / renamed tickers whose transcripts are published under a different symbol.
# Map canonical → list of aliases to try (primary first when fetching).
_TICKER_ALIASES: Dict[str, List[str]] = {
    "GOOG":  ["GOOGL", "GOOG"],   # Alphabet non-voting → try voting class first
    "GOOGL": ["GOOGL", "GOOG"],
    "BRK.B": ["BRK-B", "BRKB", "BRK.B"],
    "BRK-B": ["BRK-B", "BRK.B", "BRKB"],
    "BRKB":  ["BRK-B", "BRKB"],
    "BRK.A": ["BRK-A", "BRKA", "BRK.A"],
    "BRK-A": ["BRK-A", "BRK.A", "BRKA"],
    "BRKA":  ["BRK-A", "BRKA"],
    "META":  ["META"],             # Formerly FB — already correct
}


def _tickers_to_try(ticker: str) -> List[str]:
    """Return [primary, *aliases] for a ticker, deduped and in priority order."""
    t = ticker.upper()
    known = _TICKER_ALIASES.get(t)
    if known:
        return list(dict.fromkeys(known))   # preserve order, remove dups
    # Generic dot/dash/no-suffix normalization (e.g. BF.B → BF-B)
    variants = [t]
    if "." in t:
        variants.append(t.replace(".", "-"))
        variants.append(t.replace(".", ""))
    elif "-" in t:
        variants.append(t.replace("-", "."))
        variants.append(t.replace("-", ""))
    return list(dict.fromkeys(variants))


def get_transcript_provider_outcomes(ticker: str) -> List[Dict]:
    """Per-provider outcomes from the most recent get_last_n_transcripts call for ticker."""
    return list(_TRANSCRIPT_HEALTH.get(ticker.upper(), []))


_MAX_ATTEMPTS = 8

# Q&A boundary: Operator line that announces the session opening
_QA_SPLIT_RE = re.compile(
    r"(?m)^(?:Operator|OPERATOR)\s*:.*?"
    r"(?:question[- ]and[- ]answer|Q&A|open.*?for question|"
    r"begin.*?question|floor.*?question|now take.*?question)",
    re.IGNORECASE,
)


# ── Exceptions ────────────────────────────────────────────────────────────────

class TranscriptRateLimited(Exception):
    """Raised on HTTP 429; carries retry-after seconds."""
    def __init__(self, retry_after: Optional[int] = None) -> None:
        self.retry_after = retry_after
        msg = f"Rate limited — retry in {retry_after}s" if retry_after else "Rate limited"
        super().__init__(msg)


# ── Normalised transcript schema ──────────────────────────────────────────────

class Transcript:
    """Provider-agnostic transcript container."""

    def __init__(
        self,
        ticker: str,
        year: int,
        quarter: int,
        date: Optional[str],
        prepared_remarks: str,
        qa_section: str,
        participants: List[str],
        source: str,
    ) -> None:
        self.ticker = ticker
        self.year = year
        self.quarter = quarter
        self.date = date
        self.prepared_remarks = prepared_remarks
        self.qa_section = qa_section
        self.participants = participants
        self.source = source

    @property
    def content(self) -> str:
        parts = [self.prepared_remarks]
        if self.qa_section:
            parts.append(self.qa_section)
        return "\n\n".join(p for p in parts if p)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "year": self.year,
            "quarter": self.quarter,
            "date": self.date,
            "prepared_remarks": self.prepared_remarks,
            "qa_section": self.qa_section,
            "participants": self.participants,
            "source": self.source,
            # legacy key for existing analysis code
            "content": self.content,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Transcript":
        return cls(
            ticker=d["ticker"],
            year=d["year"],
            quarter=d["quarter"],
            date=d.get("date"),
            prepared_remarks=d.get("prepared_remarks") or d.get("content", ""),
            qa_section=d.get("qa_section", ""),
            participants=d.get("participants", []),
            source=d.get("source", "unknown"),
        )


# ── Provider base ─────────────────────────────────────────────────────────────

class TranscriptProvider(ABC):
    name: str = "base"

    def available(self) -> bool:
        return True

    @abstractmethod
    def fetch(self, ticker: str, year: int, quarter: int) -> Optional[Transcript]:
        """Return Transcript or None if not found. Raise SourceUnavailable / TranscriptRateLimited on hard errors."""
        ...


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _split_transcript_text(text: str) -> Tuple[str, str]:
    """Split a plain-text transcript into (prepared_remarks, qa_section).

    Searches for an Operator line that announces the Q&A session; everything
    before it is prepared remarks, everything from it onward is Q&A.
    Returns (full_text, "") when no boundary is found.
    """
    match = _QA_SPLIT_RE.search(text)
    if match:
        cut = match.start()
        return text[:cut].strip(), text[cut:].strip()
    return text.strip(), ""


def _latest_likely_quarter() -> Tuple[int, int]:
    """Best-guess at the most recently-reported fiscal quarter.

    Assumes companies finish reporting within ~6 weeks of quarter end:
      Q1 (Jan-Mar) → available May+
      Q2 (Apr-Jun) → available Aug+
      Q3 (Jul-Sep) → available Nov+
      Q4 (Oct-Dec) → available Mar+ next year
    """
    now = datetime.utcnow()
    m, y = now.month, now.year
    if m >= 11:
        return y, 3
    if m >= 8:
        return y, 2
    if m >= 5:
        return y, 1
    return y - 1, 4


# ── DefeatBeta provider (primary free tier) ───────────────────────────────────

def _split_defeatbeta_df(df: Any, ticker: str, year: int, quarter: int, date: str, source: str) -> Transcript:
    """Split a defeatbeta transcript DataFrame into prepared_remarks / qa_section."""
    import pandas as pd  # noqa: F401 (needed for type checking)

    # Find Q&A boundary: first Operator paragraph containing "question"
    qa_start_idx: Optional[int] = None
    for idx, row in df.iterrows():
        if (str(row.get("speaker", "")).strip() == "Operator"
                and "question" in str(row.get("content", "")).lower()):
            qa_start_idx = idx
            break

    def concat(frame: Any) -> str:
        parts = []
        for _, r in frame.iterrows():
            speaker = str(r.get("speaker", "")).strip()
            content = str(r.get("content", "")).strip()
            if content:
                parts.append(f"{speaker}: {content}" if speaker else content)
        return "\n\n".join(parts)

    if qa_start_idx is not None:
        prepared_df = df.loc[:qa_start_idx - 1]
        qa_df = df.loc[qa_start_idx:]
    else:
        prepared_df = df
        qa_df = df.iloc[0:0]  # empty

    participants = [
        s for s in df["speaker"].unique().tolist()
        if s and str(s).strip().lower() not in ("operator", "")
    ]

    return Transcript(
        ticker=ticker.upper(),
        year=year,
        quarter=quarter,
        date=str(date) if date else None,
        prepared_remarks=concat(prepared_df),
        qa_section=concat(qa_df) if not qa_df.empty else "",
        participants=participants,
        source=source,
    )


class DefeatBetaProvider(TranscriptProvider):
    """Free parquet-backed provider via defeatbeta-api. No API key required."""
    name = "defeatbeta"

    def available(self) -> bool:
        try:
            from defeatbeta_api.data.ticker import Ticker  # noqa: F401
            return True
        except ImportError:
            return False

    def fetch(self, ticker: str, year: int, quarter: int) -> Optional[Transcript]:
        try:
            from defeatbeta_api.data.ticker import Ticker
            t = Ticker(ticker.upper())
            ects = t.earning_call_transcripts()

            lst = ects.get_transcripts_list()
            if lst.empty:
                return None

            mask = (lst["fiscal_year"] == year) & (lst["fiscal_quarter"] == quarter)
            matched = lst[mask]
            if matched.empty:
                return None

            report_date = matched.iloc[0].get("report_date")
            df = ects.get_transcript(year, quarter)
            if df is None or (hasattr(df, "empty") and df.empty):
                return None

            return _split_defeatbeta_df(df, ticker, year, quarter, report_date, self.name)

        except (SourceUnavailable, TranscriptRateLimited):
            raise
        except Exception as exc:
            logger.debug("DefeatBetaProvider %s %d Q%d: %s", ticker, year, quarter, exc)
            return None


# ── Roic.ai provider ──────────────────────────────────────────────────────────

# Broader exchange prefix list to catch foreign dual-listings (e.g. AMS:ASML)
_ROIC_EXCHANGES = ("NASDAQ", "NYSE", "TSX", "AMS", "EURONEXT", "LSE", "ETR", "HKEX")


class RoicProvider(TranscriptProvider):
    """roic.ai v3 earnings-call API. Requires ROIC_API_KEY. 5 req/min free tier."""
    name = "roic"
    _last_req: float = 0.0
    _min_interval: float = 60.0 / 5  # 5 req/min

    def available(self) -> bool:
        return bool(ROIC_API_KEY)

    def _throttle(self) -> None:
        elapsed = time.monotonic() - RoicProvider._last_req
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        RoicProvider._last_req = time.monotonic()

    def fetch(self, ticker: str, year: int, quarter: int) -> Optional[Transcript]:
        if not self.available():
            return None
        try:
            return self._fetch(ticker, year, quarter)
        except (SourceUnavailable, TranscriptRateLimited):
            raise
        except Exception as exc:
            logger.debug("RoicProvider %s %d Q%d: %s", ticker, year, quarter, exc)
            return None

    def _fetch(self, ticker: str, year: int, quarter: int) -> Optional[Transcript]:
        with httpx.Client(timeout=20) as client:
            ecall_id, report_date = self._find_ecall(client, ticker, year, quarter)
            if not ecall_id:
                return None
            content = self._fetch_detail(client, ecall_id, year, quarter)
            if not content:
                return None

        return Transcript(
            ticker=ticker.upper(),
            year=year,
            quarter=quarter,
            date=str(report_date) if report_date else None,
            prepared_remarks=content,
            qa_section="",
            participants=[],
            source=self.name,
        )

    def _find_ecall(
        self, client: httpx.Client, ticker: str, year: int, quarter: int
    ) -> Tuple[Optional[str], Optional[str]]:
        for exchange in _ROIC_EXCHANGES:
            self._throttle()
            resp = client.get(
                ROIC_TRANSCRIPT_LIST_URL,
                params={
                    "apikey": ROIC_API_KEY,
                    "identifier": f"{exchange}:{ticker.upper()}",
                    "limit": 12,
                },
            )
            if resp.status_code == 429:
                ra = resp.headers.get("Retry-After")
                raise TranscriptRateLimited(int(ra) if ra and ra.isdigit() else None)
            if resp.status_code == 401:
                raise SourceUnavailable("ROIC_API_KEY auth failed — check config")
            if resp.status_code != 200 or not resp.text.strip():
                continue
            try:
                items = resp.json()
            except Exception:
                continue
            if not items:
                continue
            for item in (items if isinstance(items, list) else [items]):
                fy = item.get("fiscal_year") or item.get("year")
                fq = item.get("fiscal_quarter") or item.get("quarter")
                if fy == year and fq == quarter:
                    ecall_id = item.get("id") or item.get("ecall_id")
                    date = item.get("date") or item.get("report_date")
                    return ecall_id, date
            # Company found on this exchange but not this quarter
            if items:
                return None, None
        return None, None

    def _fetch_detail(
        self, client: httpx.Client, ecall_id: str, year: int, quarter: int
    ) -> Optional[str]:
        self._throttle()
        url = ROIC_TRANSCRIPT_DETAIL_URL.format(ecall_id=ecall_id)
        resp = client.get(
            url,
            params={"apikey": ROIC_API_KEY, "fiscal_year": year, "fiscal_quarter": quarter},
        )
        if resp.status_code == 429:
            ra = resp.headers.get("Retry-After")
            raise TranscriptRateLimited(int(ra) if ra and ra.isdigit() else None)
        if resp.status_code != 200:
            return None
        try:
            detail = resp.json()
        except Exception:
            return None
        content = (
            detail.get("content")
            or detail.get("transcript")
            or detail.get("text")
            or ""
        )
        if isinstance(content, list):
            content = "\n\n".join(str(c) for c in content)
        return str(content) if content else None


# ── Financial Modeling Prep provider ─────────────────────────────────────────

class FmpProvider(TranscriptProvider):
    """FMP earnings call transcripts. Free tier: 250 req/day (financialmodelingprep.com).

    Covers US-listed tickers including foreign dual-listings like ASML and TSM.
    Add FMP_API_KEY to .env — free plan is sufficient.
    """
    name = "fmp"

    def available(self) -> bool:
        return bool(FMP_API_KEY)

    def fetch(self, ticker: str, year: int, quarter: int) -> Optional[Transcript]:
        if not self.available():
            return None
        try:
            return self._fetch(ticker, year, quarter)
        except (SourceUnavailable, TranscriptRateLimited):
            raise
        except Exception as exc:
            logger.debug("FmpProvider %s %d Q%d: %s", ticker, year, quarter, exc)
            return None

    def _fetch(self, ticker: str, year: int, quarter: int) -> Optional[Transcript]:
        url = FMP_TRANSCRIPT_URL.format(symbol=ticker.upper())
        with httpx.Client(timeout=20) as client:
            resp = client.get(
                url,
                params={"year": year, "quarter": quarter, "apikey": FMP_API_KEY},
            )
        if resp.status_code == 401:
            raise SourceUnavailable("FMP_API_KEY auth failed — check config")
        if resp.status_code == 429:
            ra = resp.headers.get("Retry-After")
            raise TranscriptRateLimited(int(ra) if ra and ra.isdigit() else None)
        if resp.status_code != 200:
            return None

        try:
            data = resp.json()
        except Exception:
            return None
        if not data or not isinstance(data, list):
            return None

        item = data[0]
        content = (item.get("content") or "").strip()
        if not content:
            return None

        prepared, qa = _split_transcript_text(content)
        return Transcript(
            ticker=ticker.upper(),
            year=int(item.get("year", year)),
            quarter=int(item.get("quarter", quarter)),
            date=item.get("date"),
            prepared_remarks=prepared,
            qa_section=qa,
            participants=[],
            source=self.name,
        )


# ── Finnhub provider ──────────────────────────────────────────────────────────

class FinnhubProvider(TranscriptProvider):
    """Finnhub earnings call transcripts. Requires FINNHUB_API_KEY on a paid plan.

    Transcripts are available on Finnhub's Basic plan (~$10/month) and above.
    finnhub.io/pricing
    """
    name = "finnhub"

    def available(self) -> bool:
        return bool(FINNHUB_API_KEY)

    def fetch(self, ticker: str, year: int, quarter: int) -> Optional[Transcript]:
        if not self.available():
            return None
        try:
            return self._fetch(ticker, year, quarter)
        except (SourceUnavailable, TranscriptRateLimited):
            raise
        except Exception as exc:
            logger.debug("FinnhubProvider %s %d Q%d: %s", ticker, year, quarter, exc)
            return None

    def _fetch(self, ticker: str, year: int, quarter: int) -> Optional[Transcript]:
        with httpx.Client(timeout=20) as client:
            list_resp = client.get(
                FINNHUB_TRANSCRIPT_LIST_URL,
                params={"symbol": ticker.upper(), "token": FINNHUB_API_KEY},
            )
            if list_resp.status_code == 403:
                raise SourceUnavailable("Finnhub transcripts require a paid plan — see finnhub.io/pricing")
            if list_resp.status_code == 401:
                raise SourceUnavailable("FINNHUB_API_KEY auth failed — check config")
            if list_resp.status_code == 429:
                raise TranscriptRateLimited()
            if list_resp.status_code != 200:
                return None

            transcripts_list = list_resp.json().get("transcripts") or []
            transcript_id: Optional[str] = None
            transcript_date: Optional[str] = None
            for item in transcripts_list:
                if item.get("year") == year and item.get("quarter") == quarter:
                    transcript_id = item.get("id")
                    transcript_date = item.get("date")
                    break
            if not transcript_id:
                return None

            detail_resp = client.get(
                FINNHUB_TRANSCRIPT_URL,
                params={"symbol": ticker.upper(), "id": transcript_id, "token": FINNHUB_API_KEY},
            )
            if detail_resp.status_code != 200:
                return None

        data = detail_resp.json()
        content_entries = data.get("content") or []
        if not content_entries:
            return None

        lines: List[str] = []
        for entry in content_entries:
            name = (entry.get("name") or "").strip()
            for speech in (entry.get("speech") or []):
                lines.append(f"{name}: {speech}" if name else str(speech))

        full_text = "\n\n".join(lines)
        prepared, qa = _split_transcript_text(full_text)
        participants = list({e.get("name") for e in content_entries if e.get("name")})

        return Transcript(
            ticker=ticker.upper(),
            year=year,
            quarter=quarter,
            date=str(transcript_date) if transcript_date else None,
            prepared_remarks=prepared,
            qa_section=qa,
            participants=participants,
            source=self.name,
        )


# ── API Ninjas provider (legacy / premium only) ───────────────────────────────

class ApiNinjasProvider(TranscriptProvider):
    """api-ninjas.com earnings transcript endpoint. Requires API_NINJAS_PREMIUM=true."""
    name = "api_ninjas"

    def available(self) -> bool:
        return bool(API_NINJAS_KEY and API_NINJAS_PREMIUM)

    def fetch(self, ticker: str, year: int, quarter: int) -> Optional[Transcript]:
        if not self.available():
            return None
        result = _fetch_transcript(ticker, year, quarter)
        if result is None:
            return None
        return Transcript(
            ticker=result["ticker"],
            year=result["year"],
            quarter=result["quarter"],
            date=result.get("date"),
            prepared_remarks=result.get("content", ""),
            qa_section="",
            participants=[],
            source=self.name,
        )


# ── Motley Fool web-scrape provider (last resort) ─────────────────────────────

class MotleyFoolProvider(TranscriptProvider):
    """Web-scrape fallback: searches DuckDuckGo for Motley Fool transcript pages.

    No API key required. Uses trafilatura for clean text extraction.
    Throttled to 3 s/request to avoid rate-limiting.
    This is intentionally last in the chain — structured API providers are faster
    and return cleaner text. MotleyFool covers tickers missed by all other providers.
    """
    name = "motleyfool"
    _last_req: float = 0.0
    _min_interval: float = 3.0  # polite rate limit
    _HEADERS: Dict[str, str] = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    def _throttle(self) -> None:
        elapsed = time.monotonic() - MotleyFoolProvider._last_req
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        MotleyFoolProvider._last_req = time.monotonic()

    def fetch(self, ticker: str, year: int, quarter: int) -> Optional[Transcript]:
        try:
            return self._fetch(ticker, year, quarter)
        except (SourceUnavailable, TranscriptRateLimited):
            raise
        except Exception as exc:
            logger.debug("MotleyFoolProvider %s %d Q%d: %s", ticker, year, quarter, exc)
            return None

    def _find_url(self, ticker: str, year: int, quarter: int) -> Optional[str]:
        """Search DuckDuckGo HTML for a Motley Fool transcript page."""
        from urllib.parse import unquote

        query = f'site:fool.com "{ticker.upper()}" "Q{quarter} {year}" earnings call transcript'
        self._throttle()
        with httpx.Client(timeout=15, headers=self._HEADERS, follow_redirects=True) as client:
            resp = client.get("https://html.duckduckgo.com/html/", params={"q": query})
        if resp.status_code != 200:
            return None

        found: List[str] = []
        # Direct href links
        for m in re.finditer(
            r"https?://(?:www\.)?fool\.com/earnings/call-transcripts/[^\s\"'<>&]+",
            resp.text,
        ):
            found.append(m.group(0).rstrip('/"'))
        # DuckDuckGo redirect-encoded links
        for m in re.finditer(r"uddg=([^&\"<>\s]+)", resp.text):
            decoded = unquote(m.group(1))
            if "fool.com/earnings/call-transcripts" in decoded:
                found.append(decoded.rstrip('/"'))

        if not found:
            return None
        # Prefer URLs that contain the year string
        year_matches = [u for u in found if str(year) in u]
        return (year_matches or found)[0]

    def _fetch(self, ticker: str, year: int, quarter: int) -> Optional[Transcript]:
        import trafilatura

        url = self._find_url(ticker, year, quarter)
        if not url:
            logger.debug("MotleyFoolProvider: no URL found for %s Q%d %d", ticker, quarter, year)
            return None

        logger.info("MotleyFoolProvider: fetching %s", url)
        self._throttle()
        with httpx.Client(timeout=25, headers=self._HEADERS, follow_redirects=True) as client:
            page = client.get(url)
        if page.status_code != 200:
            return None

        text = trafilatura.extract(page.text, include_comments=False, include_tables=False)
        if not text or len(text) < 500:
            return None

        prepared, qa = _split_transcript_text(text)
        return Transcript(
            ticker=ticker.upper(),
            year=year,
            quarter=quarter,
            date=None,
            prepared_remarks=prepared,
            qa_section=qa,
            participants=[],
            source=self.name,
        )


# ── Provider chain ────────────────────────────────────────────────────────────

_PROVIDERS: List[TranscriptProvider] = [
    RoicProvider(),
    FmpProvider(),
    FinnhubProvider(),
    DefeatBetaProvider(),
    ApiNinjasProvider(),
    MotleyFoolProvider(),   # always available; keep last (slowest, web scrape)
]


def get_available_provider_names() -> List[str]:
    """Names of providers that pass the available() check in the current environment."""
    return [p.name for p in _PROVIDERS if p.available()]


def _get_transcript_from_chain(ticker: str, year: int, quarter: int) -> Optional[Transcript]:
    symbols = _tickers_to_try(ticker)
    outcomes: List[Dict] = []
    result: Optional[Transcript] = None

    for provider in _PROVIDERS:
        if not provider.available():
            outcomes.append({"provider": provider.name, "status": "no_key"})
            continue

        found = False
        for sym in symbols:
            t0 = time.monotonic()
            try:
                t = provider.fetch(sym, year, quarter)
                latency_ms = int((time.monotonic() - t0) * 1000)
                if t is not None:
                    t.ticker = ticker.upper()  # normalize back to input ticker
                    outcomes.append({
                        "provider": provider.name,
                        "status": "ok",
                        "symbol_used": sym,
                        "latency_ms": latency_ms,
                        "chars": len(t.content),
                    })
                    result = t
                    found = True
                    break
                # sym returned nothing — try next alias
            except (SourceUnavailable, TranscriptRateLimited) as exc:
                latency_ms = int((time.monotonic() - t0) * 1000)
                outcomes.append({
                    "provider": provider.name,
                    "status": "unavailable",
                    "detail": str(exc),
                    "symbol_used": sym,
                    "latency_ms": latency_ms,
                })
                raise
            except Exception as exc:
                latency_ms = int((time.monotonic() - t0) * 1000)
                outcomes.append({
                    "provider": provider.name,
                    "status": "error",
                    "detail": str(exc),
                    "symbol_used": sym,
                    "latency_ms": latency_ms,
                })
                logger.debug("Provider %s failed for %s %d Q%d: %s", provider.name, sym, year, quarter, exc)
                break  # provider errored — don't try more aliases with same broken provider

        if found:
            break

        # Record not_found only if no outcome already recorded for this provider
        if not any(o["provider"] == provider.name for o in outcomes):
            outcomes.append({
                "provider": provider.name,
                "status": "not_found",
                "symbol_used": symbols[0] if symbols else ticker.upper(),
            })

    # Merge into per-ticker health dict (keep best status per provider across quarters)
    existing = {o["provider"]: o for o in _TRANSCRIPT_HEALTH.get(ticker.upper(), [])}
    for o in outcomes:
        pname = o["provider"]
        if pname not in existing or existing[pname].get("status") != "ok":
            existing[pname] = o
    _TRANSCRIPT_HEALTH[ticker.upper()] = list(existing.values())

    if result is not None:
        logger.info(
            "Transcript %s %d Q%d served by %s via %s (%d chars, %dms)",
            ticker, year, quarter, result.source,
            next((o.get("symbol_used", ticker) for o in outcomes if o.get("status") == "ok"), ticker),
            len(result.content),
            next((o.get("latency_ms", 0) for o in outcomes if o.get("status") == "ok"), 0),
        )
    return result


# ── Public API ────────────────────────────────────────────────────────────────

def get_transcript(ticker: str, year: int, quarter: int) -> Optional[Dict[str, Any]]:
    """Fetch a single earnings transcript. Returns None if not published."""
    cache_key = f"transcript:{ticker.upper()}:{year}:Q{quarter}"
    cached = get_cache_obj(cache_key)
    if cached:
        return cached

    result = _get_transcript_from_chain(ticker, year, quarter)
    if result is not None:
        d = result.to_dict()
        set_cache_obj(cache_key, d, TTL_TRANSCRIPTS, source=result.source)
        record_freshness(cache_key, result.source, TTL_TRANSCRIPTS)
        return d
    return None


def get_last_n_transcripts(ticker: str, n: int = 4) -> List[Dict[str, Any]]:
    """Walk backward through calendar quarters until n transcripts are collected
    or _MAX_ATTEMPTS quarters are exhausted.

    Starts from the most recently-reported quarter (not always Q4) so we don't
    waste attempts on future quarters that cannot have transcripts yet.

    SourceUnavailable / TranscriptRateLimited propagate immediately.
    Missing quarters (None) are skipped silently.
    """
    results: List[Dict[str, Any]] = []
    attempts = 0
    year, quarter = _latest_likely_quarter()

    while len(results) < n and attempts < _MAX_ATTEMPTS:
        attempts += 1
        try:
            t = get_transcript(ticker, year, quarter)
            if t:
                results.append(t)
        except (SourceUnavailable, TranscriptRateLimited):
            raise
        except Exception as exc:
            logger.debug("Transcript %s %d Q%d skipped: %s", ticker, year, quarter, exc)

        quarter -= 1
        if quarter == 0:
            quarter = 4
            year -= 1

    if not results:
        logger.info(
            "No transcripts found for %s after %d attempts; providers available: %s",
            ticker, attempts, ", ".join(get_available_provider_names()) or "none",
        )
    return results


def transcript_word_count(transcript: Dict[str, Any]) -> int:
    return len((transcript.get("content") or "").split())


# ── API Ninjas internals (kept for backward-compat with existing tests) ───────

def _check_key() -> None:
    if not API_NINJAS_KEY:
        raise SourceUnavailable("API_NINJAS_KEY not configured — Earnings Calls tab unavailable")


@retry(max_attempts=3, base_delay=2.0, retryable=(httpx.TimeoutException, httpx.NetworkError))
def _fetch_transcript(ticker: str, year: int, quarter: int) -> Optional[Dict[str, Any]]:
    """Low-level API Ninjas fetch. Used by ApiNinjasProvider and backward-compat tests."""
    _check_key()
    params = {"ticker": ticker.upper(), "year": year, "quarter": quarter}
    with httpx.Client(timeout=30) as client:
        resp = client.get(
            API_NINJAS_TRANSCRIPT_URL,
            params=params,
            headers={"X-Api-Key": API_NINJAS_KEY},
        )

    if resp.status_code == 401:
        raise SourceUnavailable("Transcript API auth failed — check API_NINJAS_KEY")

    if resp.status_code == 400:
        try:
            err_msg = (resp.json() if resp.text else {}).get("error", "")
        except Exception:
            err_msg = resp.text[:200]
        if "premium" in err_msg.lower():
            raise SourceUnavailable(
                "API Ninjas earnings transcript requires a premium subscription "
                "(api-ninjas.com/pricing) — the free tier does not include this endpoint"
            )
        return None

    if resp.status_code == 404:
        return None

    if resp.status_code == 429:
        ra_str = resp.headers.get("Retry-After")
        ra = int(ra_str) if ra_str and ra_str.isdigit() else None
        raise TranscriptRateLimited(ra)

    resp.raise_for_status()

    data = resp.json()
    if not data or (isinstance(data, list) and len(data) == 0):
        return None
    if isinstance(data, list):
        data = data[0]
    if not isinstance(data, dict):
        return None

    return {
        "ticker": ticker.upper(),
        "year": year,
        "quarter": quarter,
        "content": data.get("transcript") or data.get("content") or str(data),
        "date": data.get("date"),
    }
