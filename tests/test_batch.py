"""Tests for core/batch.py — no real API calls; analyze_ticker is mocked."""
from __future__ import annotations

import threading
import time
from datetime import date
from unittest.mock import patch, MagicMock

import pytest

from core.batch import BatchRunner, BatchStatus, TickerState
from core.schemas import TickerAnalysis, Scenario


# ── helpers ────────────────────────────────────────────────────────────────────

def _minimal_analysis(ticker: str = "TEST") -> TickerAnalysis:
    """Return the smallest valid TickerAnalysis (no API calls)."""
    return TickerAnalysis(
        ticker=ticker,
        as_of=date(2026, 1, 1),
        confidence=0.8,
    )


def _make_runner(tickers, **kwargs) -> BatchRunner:
    return BatchRunner(tickers, config={}, concurrency=kwargs.pop("concurrency", 5), **kwargs)


# ── test 1: all succeed ────────────────────────────────────────────────────────

class TestBatchAllSucceed:
    def test_batch_all_succeed(self) -> None:
        tickers = ["AAPL", "MSFT", "NVDA"]

        def fake_analyze(ticker, config=None, as_of=None):
            return _minimal_analysis(ticker)

        with patch("core.batch.analyze_ticker", side_effect=fake_analyze):
            runner = _make_runner(tickers)
            runner.run()

        states = runner.states
        assert len(states) == 3
        for ticker in [t.upper() for t in tickers]:
            assert states[ticker].status == BatchStatus.DONE
            assert states[ticker].result is not None
            assert states[ticker].error is None


# ── test 2: one fails ─────────────────────────────────────────────────────────

class TestBatchOneFails:
    def test_batch_one_fails(self) -> None:
        tickers = ["AAPL", "BADTICKER", "NVDA"]

        def fake_analyze(ticker, config=None, as_of=None):
            if ticker == "BADTICKER":
                raise ValueError("Symbol not found")
            return _minimal_analysis(ticker)

        with patch("core.batch.analyze_ticker", side_effect=fake_analyze):
            runner = _make_runner(tickers)
            runner.run()

        states = runner.states
        assert states["BADTICKER"].status == BatchStatus.FAILED
        assert "Symbol not found" in states["BADTICKER"].error
        assert states["AAPL"].status == BatchStatus.DONE
        assert states["NVDA"].status == BatchStatus.DONE


# ── test 3: cancel before run ─────────────────────────────────────────────────

class TestBatchCancel:
    def test_batch_cancel_before_run(self) -> None:
        """Cancel before run() starts — all tickers should end as CANCELLED."""
        tickers = ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL"]
        # Use a slow fake so cancellation races; but set cancel_event before run
        call_count = 0

        def slow_analyze(ticker, config=None, as_of=None):
            nonlocal call_count
            call_count += 1
            time.sleep(5)  # deliberately long
            return _minimal_analysis(ticker)

        with patch("core.batch.analyze_ticker", side_effect=slow_analyze):
            runner = _make_runner(tickers, concurrency=1)
            # Set cancel before run — _run_one checks it at the top
            runner.cancel()
            runner.run()

        states = runner.states
        for ticker in [t.upper() for t in tickers]:
            assert states[ticker].status == BatchStatus.CANCELLED, (
                f"{ticker} expected CANCELLED, got {states[ticker].status}"
            )


# ── test 4: progress tuple ────────────────────────────────────────────────────

class TestBatchProgress:
    def test_batch_progress_before_and_after(self) -> None:
        tickers = ["AAPL", "MSFT", "NVDA"]

        def fake_analyze(ticker, config=None, as_of=None):
            return _minimal_analysis(ticker)

        runner = _make_runner(tickers)
        # Before run: (0, n)
        assert runner.progress == (0, 3)

        with patch("core.batch.analyze_ticker", side_effect=fake_analyze):
            runner.run()

        # After run: (n, n)
        assert runner.progress == (3, 3)

    def test_batch_progress_with_failure(self) -> None:
        tickers = ["AAPL", "FAIL"]

        def fake_analyze(ticker, config=None, as_of=None):
            if ticker == "FAIL":
                raise RuntimeError("boom")
            return _minimal_analysis(ticker)

        with patch("core.batch.analyze_ticker", side_effect=fake_analyze):
            runner = _make_runner(tickers)
            runner.run()

        done, total = runner.progress
        assert total == 2
        assert done == 2  # both DONE and FAILED count as completed


# ── test 5: is_done ───────────────────────────────────────────────────────────

class TestBatchIsDone:
    def test_is_done_false_before_run(self) -> None:
        runner = _make_runner(["AAPL", "MSFT"])
        assert runner.is_done is False

    def test_is_done_true_after_successful_run(self) -> None:
        def fake_analyze(ticker, config=None, as_of=None):
            return _minimal_analysis(ticker)

        with patch("core.batch.analyze_ticker", side_effect=fake_analyze):
            runner = _make_runner(["AAPL", "MSFT"])
            runner.run()

        assert runner.is_done is True

    def test_is_done_true_with_mixed_terminal_states(self) -> None:
        """DONE + FAILED together still means is_done == True."""
        def fake_analyze(ticker, config=None, as_of=None):
            if ticker == "FAIL":
                raise RuntimeError("fail")
            return _minimal_analysis(ticker)

        with patch("core.batch.analyze_ticker", side_effect=fake_analyze):
            runner = _make_runner(["AAPL", "FAIL"])
            runner.run()

        assert runner.is_done is True

    def test_is_done_true_after_cancel(self) -> None:
        runner = _make_runner(["AAPL"])
        runner.cancel()

        def fake_analyze(ticker, config=None, as_of=None):
            return _minimal_analysis(ticker)

        with patch("core.batch.analyze_ticker", side_effect=fake_analyze):
            runner.run()

        assert runner.is_done is True


# ── test 6: concurrency limit ─────────────────────────────────────────────────

class TestBatchConcurrencyLimit:
    def test_concurrency_limit_respected(self) -> None:
        """At most `concurrency` workers should run simultaneously."""
        concurrency = 2
        tickers = ["A", "B", "C", "D", "E"]

        peak_concurrent = 0
        current_concurrent = 0
        sem_lock = threading.Lock()
        gate = threading.Barrier(1)  # just for signaling

        def fake_analyze(ticker, config=None, as_of=None):
            nonlocal peak_concurrent, current_concurrent
            with sem_lock:
                current_concurrent += 1
                if current_concurrent > peak_concurrent:
                    peak_concurrent = current_concurrent
            time.sleep(0.05)  # hold the "slot" briefly
            with sem_lock:
                current_concurrent -= 1
            return _minimal_analysis(ticker)

        with patch("core.batch.analyze_ticker", side_effect=fake_analyze):
            runner = _make_runner(tickers, concurrency=concurrency)
            runner.run()

        assert peak_concurrent <= concurrency, (
            f"Peak concurrent workers {peak_concurrent} exceeded limit {concurrency}"
        )
        # Sanity: all tickers completed
        for t in [x.upper() for x in tickers]:
            assert runner.states[t].status == BatchStatus.DONE


# ── test 7: TickerState.elapsed ───────────────────────────────────────────────

class TestTickerStateElapsed:
    def test_elapsed_none_before_start(self) -> None:
        state = TickerState(ticker="AAPL")
        assert state.started_at is None
        assert state.elapsed is None

    def test_elapsed_positive_while_running(self) -> None:
        state = TickerState(ticker="AAPL")
        state.started_at = time.time() - 1.0  # simulate 1s ago
        elapsed = state.elapsed
        assert elapsed is not None
        assert elapsed >= 1.0

    def test_elapsed_uses_finished_at_when_done(self) -> None:
        state = TickerState(ticker="AAPL")
        start = time.time() - 5.0
        finish = time.time() - 2.0
        state.started_at = start
        state.finished_at = finish
        elapsed = state.elapsed
        assert elapsed is not None
        assert abs(elapsed - (finish - start)) < 0.01

    def test_elapsed_through_full_batch_run(self) -> None:
        """Verify elapsed is set correctly after a completed batch run."""
        def fake_analyze(ticker, config=None, as_of=None):
            time.sleep(0.05)
            return _minimal_analysis(ticker)

        with patch("core.batch.analyze_ticker", side_effect=fake_analyze):
            runner = _make_runner(["AAPL"])
            runner.run()

        state = runner.states["AAPL"]
        assert state.started_at is not None
        assert state.finished_at is not None
        assert state.elapsed is not None
        assert state.elapsed > 0
        # elapsed should equal finished_at - started_at (since finished_at is set)
        expected = state.finished_at - state.started_at
        assert abs(state.elapsed - expected) < 0.01


# ── test 8: results() helper ──────────────────────────────────────────────────

class TestBatchResults:
    def test_results_returns_analysis_for_done(self) -> None:
        def fake_analyze(ticker, config=None, as_of=None):
            return _minimal_analysis(ticker)

        with patch("core.batch.analyze_ticker", side_effect=fake_analyze):
            runner = _make_runner(["AAPL", "MSFT"])
            runner.run()

        results = runner.results()
        assert set(results.keys()) == {"AAPL", "MSFT"}
        assert results["AAPL"] is not None
        assert results["MSFT"] is not None

    def test_results_none_for_failed(self) -> None:
        def fake_analyze(ticker, config=None, as_of=None):
            raise ValueError("fail")

        with patch("core.batch.analyze_ticker", side_effect=fake_analyze):
            runner = _make_runner(["FAIL"])
            runner.run()

        results = runner.results()
        assert results["FAIL"] is None


# ── test 9: ticker deduplication and normalization ────────────────────────────

class TestTickerNormalization:
    def test_tickers_uppercased(self) -> None:
        def fake_analyze(ticker, config=None, as_of=None):
            return _minimal_analysis(ticker)

        with patch("core.batch.analyze_ticker", side_effect=fake_analyze):
            runner = _make_runner(["aapl", "msft"])
            runner.run()

        states = runner.states
        assert "AAPL" in states
        assert "MSFT" in states
        assert "aapl" not in states

    def test_duplicate_tickers_deduplicated(self) -> None:
        """Same ticker given twice should result in only one state entry."""
        runner = _make_runner(["AAPL", "AAPL"])
        assert len(runner.states) == 1
        assert "AAPL" in runner.states
