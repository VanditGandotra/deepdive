"""
Phase 11: Eval runner CLI.

Usage:
    python evals/run_evals.py                       # run all fast (non-LLM) checks
    python evals/run_evals.py --feature delta        # run only delta checks
    python evals/run_evals.py --llm                 # include LLM-marked tests (costs money)
    python evals/run_evals.py --batch               # run LLM evals via Batch API (async)
    python evals/run_evals.py --report              # generate markdown report
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).parent.parent
GOLDEN_DIR = Path(__file__).parent / "golden"
REPORTS_DIR = Path(__file__).parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


# ── Feature → test node mapping ───────────────────────────────────────────────

FEATURE_MAP: Dict[str, List[str]] = {
    "schema":      ["evals/checks.py::TestSchemaConformance"],
    "transcript":  ["evals/checks.py::TestTranscriptExtraction"],
    "headlines":   ["evals/checks.py::TestHeadlineClassification"],
    "grounding":   ["evals/checks.py::TestGroundingAudit"],
    "delta":       ["evals/checks.py::TestDeltaDeterminism"],
    "math":        ["evals/checks.py::TestReverseDCFMath"],
    "cost":        ["evals/checks.py::TestCostRegression"],
    "cache":       ["evals/checks.py::TestCacheLayer"],
    "charts":      ["evals/checks.py::TestChartRendering"],
    "providers":   ["evals/checks.py::TestTranscriptProviders"],
}

ALL_FEATURES = list(FEATURE_MAP.keys())
FAST_FEATURES = ["schema", "delta", "math", "cost", "cache", "charts", "providers"]


# ── Pytest runner ─────────────────────────────────────────────────────────────

def _run_pytest(
    nodes: List[str],
    include_llm: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    cmd = [sys.executable, "-m", "pytest"] + nodes + ["--tb=short"]
    if verbose:
        cmd.append("-v")
    if not include_llm:
        cmd += ["-m", "not llm"]

    start = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    elapsed = time.monotonic() - start

    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "elapsed_s": round(elapsed, 2),
        "passed": result.returncode == 0,
    }


# ── Batch API eval harness ────────────────────────────────────────────────────

def _run_batch_evals() -> Dict[str, Any]:
    """
    Submit LLM-backed eval requests via Batch API.
    Returns batch_id and instructions for polling.
    """
    try:
        import llm as llm_mod
        from config import HAIKU
    except ImportError as e:
        return {"error": f"Import failed: {e}"}

    golden_transcript = json.loads((GOLDEN_DIR / "nvda_transcript.json").read_text())
    transcript_text = golden_transcript["transcript"]

    # Build batch requests
    batch_requests = []

    # Request 1: guidance extraction
    batch_requests.append({
        "custom_id": "guidance_extraction",
        "model": HAIKU,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        f"Extract guidance items from this earnings transcript.\n\n{transcript_text[:3000]}"
                    )},
                ],
            }
        ],
        "max_tokens": 1000,
    })

    # Request 2: headline classification (first 10)
    golden_headlines = json.loads((GOLDEN_DIR / "headlines_labeled.json").read_text())
    headlines_10 = golden_headlines["headlines"][:10]
    hl_text = "\n".join(f"{i+1}. {h['title']}" for i, h in enumerate(headlines_10))
    batch_requests.append({
        "custom_id": "headline_classification",
        "model": HAIKU,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        f"Classify each headline for NVIDIA as: direction (positive/negative/neutral/mixed) "
                        f"and materiality (high/medium/low).\n\n{hl_text}\n\n"
                        "Return JSON array: [{\"title\": ..., \"direction\": ..., \"materiality\": ...}]"
                    )},
                ],
            }
        ],
        "max_tokens": 800,
    })

    batch_id = llm_mod.batch_create(batch_requests)
    return {
        "batch_id": batch_id,
        "submitted_at": datetime.utcnow().isoformat(),
        "request_count": len(batch_requests),
        "poll_command": f"python evals/run_evals.py --poll-batch {batch_id}",
    }


def _poll_batch(batch_id: str) -> Dict[str, Any]:
    """Poll a batch and score the results."""
    try:
        import llm as llm_mod
    except ImportError as e:
        return {"error": str(e)}

    print(f"Polling batch {batch_id}…")
    results = llm_mod.batch_poll(batch_id, timeout=3600, poll_interval=30)

    scores: Dict[str, Any] = {}
    golden_headlines = json.loads((GOLDEN_DIR / "headlines_labeled.json").read_text())
    hl_golden = {h["title"]: h["direction"] for h in golden_headlines["headlines"][:10]}

    for custom_id, message in results:
        if message is None:
            scores[custom_id] = {"status": "failed"}
            continue
        text = "".join(b.text for b in message.content if b.type == "text")

        if custom_id == "headline_classification":
            try:
                preds = json.loads(text)
                correct = sum(1 for p in preds if hl_golden.get(p.get("title")) == p.get("direction"))
                scores[custom_id] = {
                    "status": "ok",
                    "accuracy": correct / len(preds) if preds else 0,
                    "correct": correct,
                    "total": len(preds),
                }
            except json.JSONDecodeError:
                scores[custom_id] = {"status": "parse_error", "raw": text[:200]}

        elif custom_id == "guidance_extraction":
            has_revenue = "revenue" in text.lower() or "37.5" in text
            has_margin = "margin" in text.lower() or "73" in text
            scores[custom_id] = {
                "status": "ok",
                "revenue_found": has_revenue,
                "margin_found": has_margin,
                "recall_score": (int(has_revenue) + int(has_margin)) / 2,
            }

    return scores


# ── Markdown report ───────────────────────────────────────────────────────────

def _generate_report(results: Dict[str, Any]) -> str:
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# DeepDive Eval Report — {now}\n",
        "| Feature | Result | Elapsed | Notes |",
        "|---------|--------|---------|-------|",
    ]
    for feature, res in results.items():
        icon = "✅" if res.get("passed") else "❌"
        elapsed = f"{res.get('elapsed_s', 0):.1f}s"
        # Extract pass/fail count from stdout
        stdout = res.get("stdout", "")
        summary = ""
        for line in stdout.splitlines():
            if "passed" in line or "failed" in line or "error" in line:
                summary = line.strip()[:80]
                break
        lines.append(f"| {feature} | {icon} | {elapsed} | {summary} |")

    lines += [
        "",
        "## Full Output",
    ]
    for feature, res in results.items():
        lines += [
            f"\n### {feature}",
            "```",
            res.get("stdout", "")[-2000:],
            "```",
        ]

    lines.append("\n_Generated by DeepDive eval runner._")
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="DeepDive eval runner")
    parser.add_argument("--feature", nargs="+", choices=ALL_FEATURES,
                        help="Run specific feature evals only")
    parser.add_argument("--llm", action="store_true",
                        help="Include LLM-backed tests (costs money)")
    parser.add_argument("--batch", action="store_true",
                        help="Submit LLM evals via Batch API and exit")
    parser.add_argument("--poll-batch", metavar="BATCH_ID",
                        help="Poll and score a batch eval by ID")
    parser.add_argument("--report", action="store_true",
                        help="Write markdown report after run")
    parser.add_argument("--fast", action="store_true",
                        help="Run only fast (non-LLM) checks (default)")
    args = parser.parse_args()

    # Batch mode
    if args.batch:
        result = _run_batch_evals()
        print(json.dumps(result, indent=2))
        return

    if args.poll_batch:
        scores = _poll_batch(args.poll_batch)
        print(json.dumps(scores, indent=2))
        return

    # Determine which features to run
    if args.feature:
        features = args.feature
    elif args.llm:
        features = ALL_FEATURES
    else:
        features = FAST_FEATURES

    # Run
    all_results: Dict[str, Any] = {}
    overall_pass = True

    print(f"\n{'='*60}")
    print(f"DeepDive Eval Suite — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Features: {', '.join(features)}")
    print(f"LLM tests: {'yes' if args.llm else 'no (use --llm to enable)'}")
    print(f"{'='*60}\n")

    for feature in features:
        nodes = FEATURE_MAP[feature]
        print(f"▶ Running: {feature}")
        result = _run_pytest(nodes, include_llm=args.llm)
        all_results[feature] = result
        icon = "✅" if result["passed"] else "❌"
        print(f"  {icon} {feature} — {result['elapsed_s']}s\n")
        if not result["passed"]:
            overall_pass = False
            print(result["stdout"][-1500:])

    # Summary
    passed = sum(1 for r in all_results.values() if r["passed"])
    total = len(all_results)
    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} passed")
    print(f"{'='*60}\n")

    # Report
    if args.report:
        md = _generate_report(all_results)
        report_path = REPORTS_DIR / f"eval_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.md"
        report_path.write_text(md)
        print(f"Report written: {report_path}")

    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
