"""Retry decorator, fallback chains, and freshness tracking for external data sources."""
from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, Optional, Tuple, TypeVar

import httpx

logger = logging.getLogger(__name__)
T = TypeVar("T")


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
