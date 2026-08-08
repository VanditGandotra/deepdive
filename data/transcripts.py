"""Multi-provider earnings transcript fetcher with SQLite cache.

Provider chain (highest priority first):
  1. RoicProvider   — roic.ai v3 API (requires ROIC_API_KEY)
  2. DefeatBetaProvider — defeatbeta-api parquet data (free, no key)
  3. ApiNinjasProvider — api-ninjas.com (requires API_NINJAS_PREMIUM=true)
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import httpx

from config import (
    API_NINJAS_KEY, API_NINJAS_PREMIUM, API_NINJAS_TRANSCRIPT_URL,
    ROIC_API_KEY, ROIC_TRANSCRIPT_DETAIL_URL, ROIC_TRANSCRIPT_LIST_URL,
    TTL_TRANSCRIPTS,
)
from data.cache import get_cache_obj, record_freshness, set_cache_obj
from data.resilience import SourceUnavailable, retry

logger = logging.getLogger(__name__)

_CURRENT_YEAR = datetime.utcnow().year
_MAX_ATTEMPTS = 8


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


# ── DefeatBeta provider (primary free tier) ───────────────────────────────────

def _split_defeatbeta_df(df: Any, ticker: str, year: int, quarter: int, date: str, source: str) -> Transcript:
    """Split a defeatbeta transcript DataFrame into prepared_remarks / qa_section."""
    import pandas as pd

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
        for exchange in ("NASDAQ", "NYSE", "TSX"):
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
            # Found the company but not the quarter
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


# ── Provider chain ────────────────────────────────────────────────────────────

_PROVIDERS: List[TranscriptProvider] = [
    RoicProvider(),
    DefeatBetaProvider(),
    ApiNinjasProvider(),
]


def _get_transcript_from_chain(ticker: str, year: int, quarter: int) -> Optional[Transcript]:
    for provider in _PROVIDERS:
        if not provider.available():
            continue
        try:
            result = provider.fetch(ticker, year, quarter)
            if result is not None:
                logger.info(
                    "Transcript %s %d Q%d served by %s (%d chars)",
                    ticker, year, quarter, provider.name, len(result.content),
                )
                return result
        except (SourceUnavailable, TranscriptRateLimited):
            raise
        except Exception as exc:
            logger.debug("Provider %s failed for %s %d Q%d: %s", provider.name, ticker, year, quarter, exc)
    return None


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
    """
    Walk backward through calendar quarters until n transcripts are collected
    or _MAX_ATTEMPTS quarters are exhausted.

    SourceUnavailable / TranscriptRateLimited propagate immediately.
    Missing quarters (None) are skipped silently.
    """
    results: List[Dict[str, Any]] = []
    attempts = 0
    year = _CURRENT_YEAR
    quarter = 4

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

    if not results and attempts >= _MAX_ATTEMPTS:
        logger.info(
            "No transcripts found for %s after %d attempts (last checked %d Q%d)",
            ticker, attempts, year, quarter,
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
