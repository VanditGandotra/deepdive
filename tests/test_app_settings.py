"""Tests for app.py DCF slider defaults."""
from __future__ import annotations

import ast
import pathlib

import pytest


def _parse_slider_defaults(source: str) -> dict:
    """Extract default values for named sliders from the app source via AST."""
    tree = ast.parse(source)
    defaults: dict = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (func.attr if isinstance(func, ast.Attribute) else
                func.id if isinstance(func, ast.Name) else None)
        if name != "slider":
            continue
        # positional args: label, min, max, value, step
        if len(node.args) < 4:
            continue
        label_node = node.args[0]
        value_node = node.args[3]
        label = label_node.value if isinstance(label_node, ast.Constant) else None
        value = value_node.value if isinstance(value_node, ast.Constant) else None
        if label and value is not None:
            defaults[label] = value
    return defaults


_APP_SOURCE = (pathlib.Path(__file__).parent.parent / "app.py").read_text()
_SLIDER_DEFAULTS = _parse_slider_defaults(_APP_SOURCE)


def test_discount_rate_default_is_10_pct() -> None:
    """Discount rate slider default must be 10.0 (percent scale), giving 0.10 after /100."""
    default_pct = _SLIDER_DEFAULTS.get("Discount rate")
    assert default_pct is not None, "Discount rate slider not found in app.py"
    assert default_pct == pytest.approx(10.0), (
        f"Expected 10.0 (renders as '10.0%'), got {default_pct}. "
        "Using a raw decimal like 0.10 here would show as '0.1%'."
    )
    assert default_pct / 100.0 == pytest.approx(0.10)


def test_terminal_growth_default_is_2_5_pct() -> None:
    """Terminal growth slider default must be 2.5 (percent scale)."""
    default_pct = _SLIDER_DEFAULTS.get("Terminal growth")
    assert default_pct is not None, "Terminal growth slider not found in app.py"
    assert default_pct == pytest.approx(2.5)
    assert default_pct / 100.0 == pytest.approx(0.025)
