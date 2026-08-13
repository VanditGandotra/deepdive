#!/usr/bin/env python3
"""
CLI batch analysis harness.
Usage:
    python -m cli.batch MSFT NVDA ASML
    python -m cli.batch MSFT NVDA --concurrency 3
    python -m cli.batch MSFT NVDA --discount-rate 0.12 --terminal-growth 0.03
"""
from __future__ import annotations

import argparse
import sys
import time
import threading
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.batch import BatchRunner, BatchStatus


_STATUS_ICONS = {
    BatchStatus.QUEUED:    "⏳",
    BatchStatus.FETCHING:  "🔍",
    BatchStatus.ANALYZING: "🧠",
    BatchStatus.DONE:      "✅",
    BatchStatus.FAILED:    "❌",
    BatchStatus.CANCELLED: "🚫",
}


def _print_status(runner: BatchRunner) -> None:
    states = runner.states
    done, total = runner.progress
    print(f"\r[{done}/{total}] ", end="")
    for ticker, state in sorted(states.items()):
        icon = _STATUS_ICONS.get(state.status, "?")
        elapsed = f" {state.elapsed:.0f}s" if state.elapsed else ""
        print(f"{icon} {ticker}{elapsed}  ", end="")
    print("", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepDive batch analysis")
    parser.add_argument("tickers", nargs="+", help="Ticker symbols to analyze")
    parser.add_argument("--concurrency", type=int, default=3, help="Max parallel analyses")
    parser.add_argument("--discount-rate", type=float, default=0.10)
    parser.add_argument("--terminal-growth", type=float, default=0.025)
    parser.add_argument("--horizon-years", type=int, default=10)
    args = parser.parse_args()

    tickers = [t.upper().strip().lstrip("$") for t in args.tickers]
    tickers = list(dict.fromkeys(tickers))  # dedupe, preserve order

    config = {
        "discount_rate": args.discount_rate,
        "terminal_growth": args.terminal_growth,
        "horizon_years": args.horizon_years,
    }

    print(f"Analyzing {len(tickers)} ticker(s): {', '.join(tickers)}")
    print(f"Concurrency: {args.concurrency}  Config: {config}")
    print()

    runner = BatchRunner(tickers, config=config, concurrency=args.concurrency)

    t = threading.Thread(target=runner.run, daemon=True)
    t.start()

    while not runner.is_done:
        _print_status(runner)
        time.sleep(1)
    _print_status(runner)
    print()

    t.join()

    # Print results summary
    print("\n=== Results ===\n")
    for ticker, state in sorted(runner.states.items()):
        if state.status == BatchStatus.DONE and state.result:
            r = state.result
            price = f"${r.current_price:.2f}" if r.current_price else "N/A"
            er1y = f"{r.expected_return_1y*100:.1f}%" if r.expected_return_1y is not None else "N/A"
            gap_ct = len(r.data_gaps)
            print(f"  {ticker:6s} | {price:10s} | Exp return 1y: {er1y:8s} | Confidence: {r.confidence:.0%} | {gap_ct} gaps")
            if r.scenarios:
                for s in r.scenarios:
                    pt = f"${s.price_target:.2f}" if s.price_target else "N/A"
                    print(f"           {s.scenario:5s} p={s.probability:.0%} target={pt}")
        elif state.status == BatchStatus.FAILED:
            print(f"  {ticker:6s} | FAILED: {state.error}")
        elif state.status == BatchStatus.CANCELLED:
            print(f"  {ticker:6s} | CANCELLED")
    print()

    failed = sum(1 for s in runner.states.values() if s.status == BatchStatus.FAILED)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
