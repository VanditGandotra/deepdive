"""Import smoke tests and mode routing tests.

These catch Crash-A class bugs (import-time failures) and Crash-B class
bugs (mode string mismatches) in under a second each.
"""
from __future__ import annotations

import importlib
import pytest


# ── Import smoke tests ─────────────────────────────────────────────────────────

IMPORTABLE_MODULES = [
    # data layer
    "data.cache",
    "data.portfolio_store",
    "data.market",
    # core layer
    "core.schemas",
    "core.batch",
    "core.portfolio",
    "core.optimizer",
    "core.montecarlo",
    # ui layer
    "ui.modes",
    "ui.portfolio_ui",
    "ui.portfolio_analysis_ui",
    "ui.ticker_drillthrough_ui",
    # cli
    "cli.batch",
]


@pytest.mark.parametrize("module", IMPORTABLE_MODULES)
def test_module_imports_cleanly(module: str) -> None:
    """Every listed module must import without raising."""
    importlib.import_module(module)


# ── Mode routing tests ─────────────────────────────────────────────────────────

from ui.modes import Mode, detect, from_any, from_label, label_index, display_labels, LABELS


@pytest.mark.parametrize("member", list(Mode))
def test_detect_portfolio_view(member: Mode) -> None:
    """Every Mode member must appear in the display labels and have a valid index."""
    assert member in LABELS
    idx = label_index(member)
    assert 0 <= idx < len(display_labels())


def test_detect_portfolio_mode_from_view() -> None:
    assert detect("portfolio", False) == Mode.PORTFOLIO
    assert detect("portfolio_analysis", False) == Mode.PORTFOLIO


def test_detect_single_stock_default() -> None:
    assert detect("", False) == Mode.SINGLE


def test_detect_ignores_stale_tickers_param() -> None:
    """?tickers= in URL (old batch bookmark) must not crash — falls back to SINGLE."""
    assert detect("", True) == Mode.SINGLE
    assert detect("ticker", False) == Mode.SINGLE


def test_unknown_mode_falls_back_to_default() -> None:
    """Stale bookmarks or typos must never raise — they fall back to SINGLE."""
    assert from_any("garbage") == Mode.SINGLE
    assert from_any("") == Mode.SINGLE
    assert from_any("Batch") == Mode.SINGLE           # legacy short form → SINGLE
    assert from_any("Batch Analysis") == Mode.SINGLE   # legacy display label → SINGLE


def test_label_index_never_raises() -> None:
    """label_index() must return a valid index for every Mode member."""
    labels = display_labels()
    for m in Mode:
        idx = label_index(m)
        assert labels[idx] == LABELS[m]


def test_from_label_roundtrip() -> None:
    """Every display label must round-trip back to its Mode."""
    for mode, label in LABELS.items():
        assert from_label(label) == mode


def test_from_label_unknown_falls_back() -> None:
    assert from_label("nonsense") == Mode.SINGLE
