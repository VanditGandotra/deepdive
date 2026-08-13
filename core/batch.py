from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Dict, List, Optional

from core.analyze import analyze_ticker
from core.schemas import TickerAnalysis

logger = logging.getLogger(__name__)


class BatchStatus(str, Enum):
    QUEUED = "queued"
    FETCHING = "fetching"
    ANALYZING = "analyzing"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TickerState:
    ticker: str
    status: BatchStatus = BatchStatus.QUEUED
    result: Optional[TickerAnalysis] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    @property
    def elapsed(self) -> Optional[float]:
        if self.started_at is None:
            return None
        end = self.finished_at or time.time()
        return end - self.started_at


class BatchRunner:
    """
    Runs analyze_ticker() for multiple tickers concurrently.
    Thread-safe state store. Results populate as each ticker completes.

    NOTE: _cancel_event is checked at the top of _run_one — it only prevents
    starting new work; it does NOT interrupt an in-progress analyze_ticker call
    (which may be doing LLM/network calls). That is by design: cancellation is
    best-effort and graceful, not forceful.
    """

    def __init__(
        self,
        tickers: List[str],
        config: Optional[dict] = None,
        concurrency: int = 5,
        as_of: Optional[date] = None,
    ):
        self.config = config or {}
        self.as_of = as_of
        self.concurrency = concurrency
        self._cancel_event = threading.Event()
        self._lock = threading.Lock()
        self._states: Dict[str, TickerState] = {
            t.upper(): TickerState(ticker=t.upper()) for t in tickers
        }
        self._executor: Optional[ThreadPoolExecutor] = None

    @property
    def states(self) -> Dict[str, TickerState]:
        """Snapshot of current state (thread-safe copy)."""
        with self._lock:
            return dict(self._states)

    def _update(self, ticker: str, **kwargs) -> None:
        with self._lock:
            state = self._states[ticker]
            for k, v in kwargs.items():
                setattr(state, k, v)

    def _run_one(self, ticker: str) -> None:
        if self._cancel_event.is_set():
            self._update(ticker, status=BatchStatus.CANCELLED)
            return
        self._update(ticker, status=BatchStatus.FETCHING, started_at=time.time())
        try:
            self._update(ticker, status=BatchStatus.ANALYZING)
            result = analyze_ticker(ticker, config=self.config, as_of=self.as_of)
            self._update(ticker, status=BatchStatus.DONE, result=result, finished_at=time.time())
        except Exception as exc:
            logger.error("BatchRunner: %s failed: %s", ticker, exc, exc_info=True)
            self._update(ticker, status=BatchStatus.FAILED, error=str(exc), finished_at=time.time())

    def run(self) -> None:
        """Run all tickers. Blocks until all complete or cancel() is called."""
        tickers = list(self._states.keys())
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            self._executor = executor
            futures = {executor.submit(self._run_one, t): t for t in tickers}
            for fut in as_completed(futures):
                ticker = futures[fut]
                try:
                    fut.result()  # re-raise any uncaught exception (safety net)
                except Exception as exc:
                    logger.error("Unhandled batch future exception for %s: %s", ticker, exc)
            self._executor = None

    def cancel(self) -> None:
        """Signal all pending/running work to stop.

        Already-running analyze_ticker calls are not interrupted; only queued
        work that has not yet started will be marked CANCELLED.
        """
        self._cancel_event.set()
        if self._executor:
            self._executor.shutdown(wait=False, cancel_futures=True)

    @property
    def is_done(self) -> bool:
        return all(
            s.status in (BatchStatus.DONE, BatchStatus.FAILED, BatchStatus.CANCELLED)
            for s in self.states.values()
        )

    @property
    def progress(self) -> tuple[int, int]:
        """Returns (completed_count, total_count)."""
        states = self.states
        done = sum(
            1 for s in states.values()
            if s.status in (BatchStatus.DONE, BatchStatus.FAILED, BatchStatus.CANCELLED)
        )
        return done, len(states)

    def results(self) -> Dict[str, Optional[TickerAnalysis]]:
        """Returns {ticker: TickerAnalysis or None} for all finished tickers."""
        return {t: s.result for t, s in self.states.items()}
