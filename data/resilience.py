"""Retry decorator, fallback chains, and freshness tracking for external data sources."""
from __future__ import annotations

import functools
import logging
import re
import time
from typing import Any, Callable, Optional, Tuple, TypeVar

import httpx

logger = logging.getLogger(__name__)
T = TypeVar("T")

# Query-param names that may carry secrets
_SECRET_PARAMS = re.compile(
    r"([?&])(apikey|api_key|token|access_token|key|secret|password|passwd|auth)[^&]*",
    re.IGNORECASE,
)
# URLs with a scheme — redact to scheme://host/path only
_URL_PATTERN = re.compile(r"https?://[^\s\"'>]+")


def redact(text: str) -> str:
    """Scrub API keys and secrets from error strings before they reach the UI.

    Strips query-string params that carry secrets and replaces any URL's query
    string with '?<redacted>' so the host+path remain readable for debugging.
    """
    def _scrub_url(m: re.Match) -> str:
        url = m.group(0)
        # Strip query string entirely from URLs — show scheme://host/path only
        base = url.split("?")[0]
        if "?" in url:
            return base + "?<redacted>"
        return base

    return _URL_PATTERN.sub(_scrub_url, text)


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    retryable: Tuple[type, ...] = (Exception,),
    on_give_up: Optional[Callable[[Exception], None]] = None,
):
    """Exponential-backoff retry decorator. Respects Retry-After on 429s."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exc: Exception = RuntimeError("no attempts made")
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except retryable as exc:
                    last_exc = exc
                    if attempt == max_attempts - 1:
                        break
                    delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                    # Honour Retry-After header on rate-limit responses
                    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
                        ra = exc.response.headers.get("Retry-After")
                        if ra:
                            try:
                                delay = max(delay, float(ra))
                            except ValueError:
                                pass
                    logger.warning(
                        "%s attempt %d/%d failed: %s — retrying in %.1fs",
                        func.__qualname__, attempt + 1, max_attempts, exc, delay,
                    )
                    time.sleep(delay)
            if on_give_up:
                on_give_up(last_exc)
            raise last_exc

        return wrapper  # type: ignore[return-value]
    return decorator


def with_fallback(
    primary: Callable[[], T],
    fallback: Callable[[], T],
    label: str = "",
) -> T:
    """Try primary; on any exception, log and try fallback."""
    try:
        return primary()
    except Exception as exc:
        logger.warning("Primary%s failed (%s), using fallback", f" [{label}]" if label else "", exc)
        return fallback()


def with_fallback_chain(*fns: Callable[[], T], label: str = "") -> T:
    """Try each callable in sequence; return first success; re-raise last failure."""
    last_exc: Exception = RuntimeError("empty fallback chain")
    for i, fn in enumerate(fns):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "Fallback chain%s step %d/%d failed: %s",
                f" [{label}]" if label else "", i + 1, len(fns), exc,
            )
    raise last_exc


class PartialResult(Exception):
    """Raised when a source returns incomplete data; carries what was retrieved."""
    def __init__(self, message: str, partial: Any = None) -> None:
        super().__init__(message)
        self.partial = partial


class SourceUnavailable(Exception):
    """Raised when a data source is entirely unreachable after retries."""


import threading as _threading


class CircuitBreaker:
    """Opens after `failure_threshold` consecutive failures; cools for `cooldown_secs`."""

    def __init__(self, name: str, failure_threshold: int = 5, cooldown_secs: float = 300.0) -> None:
        self.name = name
        self._failures = 0
        self._opened_at: Optional[float] = None
        self._threshold = failure_threshold
        self._cooldown = cooldown_secs
        self._lock = _threading.Lock()

    @property
    def is_open(self) -> bool:
        with self._lock:
            if self._opened_at is None:
                return False
            return time.time() - self._opened_at < self._cooldown

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self._threshold:
                # Reset the cooldown timer on every probe failure so that a failed
                # half-open probe restarts the full cooldown rather than staying
                # permanently half-open.
                self._opened_at = time.time()
                logger.warning("CircuitBreaker [%s] opened after %d failures", self.name, self._failures)

    def state(self) -> str:
        with self._lock:
            if self._opened_at is None:
                return "closed"
            if time.time() - self._opened_at >= self._cooldown:
                return "half-open"
            return "open"


class TokenBucket:
    """Leaky token bucket: allows `rate` requests per second up to `capacity` burst."""

    def __init__(self, rate: float, capacity: float) -> None:
        self._rate = rate
        self._capacity = capacity
        self._tokens = float(capacity)
        self._last_refill = time.time()
        self._lock = _threading.Lock()

    def acquire(self, block: bool = True) -> bool:
        with self._lock:
            now = time.time()
            self._tokens = min(self._capacity, self._tokens + (now - self._last_refill) * self._rate)
            self._last_refill = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            if not block:
                return False
            wait = (1.0 - self._tokens) / self._rate
        time.sleep(wait)
        with self._lock:
            self._tokens = max(0.0, self._tokens - 1.0)
            return True
