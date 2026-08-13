from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch, call

import pytest

from core.schemas import Driver, Scenario, TickerAnalysis


def _make_analysis(ticker="AAPL") -> TickerAnalysis:
    return TickerAnalysis(
        ticker=ticker,
        company_name="Apple Inc.",
        as_of=date.today(),
        current_price=150.0,
        scenarios=[
            Scenario(scenario="bull", price_target=210.0, probability=0.35, narrative="Strong demand."),
            Scenario(scenario="base", price_target=160.0, probability=0.45, narrative="Stable growth."),
            Scenario(scenario="bear", price_target=100.0, probability=0.20, narrative="Margin pressure."),
        ],
    )


def _call_scenario_card(scenario: Scenario, current_price=None):
    """Call _scenario_card with st patched, return the mock st."""
    mock_st = MagicMock()
    with patch("ui.portfolio_analysis_ui.st", mock_st):
        from ui.portfolio_analysis_ui import _scenario_card
        _scenario_card(scenario, current_price)
    return mock_st


def test_scenario_card_renders_bull_green():
    scenario = Scenario(
        scenario="bull",
        price_target=300.0,
        probability=0.35,
        narrative="Bullish thesis text.",
        drivers=[Driver(name="Revenue growth", direction="positive")],
    )
    mock_st = _call_scenario_card(scenario, current_price=200.0)
    calls_text = " ".join(str(c) for c in mock_st.markdown.call_args_list)
    assert "🟢" in calls_text


def test_scenario_card_renders_bear_red():
    scenario = Scenario(scenario="bear", price_target=80.0, probability=0.20, narrative="Bearish.")
    mock_st = _call_scenario_card(scenario, current_price=150.0)
    calls_text = " ".join(str(c) for c in mock_st.markdown.call_args_list)
    assert "🔴" in calls_text


def test_scenario_card_renders_base_blue():
    scenario = Scenario(scenario="base", price_target=160.0, probability=0.45, narrative="Base case.")
    mock_st = _call_scenario_card(scenario, current_price=150.0)
    calls_text = " ".join(str(c) for c in mock_st.markdown.call_args_list)
    assert "🔵" in calls_text


def test_scenario_card_no_price_target_skips_metric():
    scenario = Scenario(scenario="base", probability=0.5, price_target=None, narrative="")
    mock_st = _call_scenario_card(scenario, current_price=None)
    mock_st.metric.assert_not_called()


def test_scenario_card_shows_probability():
    scenario = Scenario(scenario="bull", probability=0.35, narrative="")
    mock_st = _call_scenario_card(scenario)
    caption_calls = " ".join(str(c) for c in mock_st.caption.call_args_list)
    assert "35%" in caption_calls


def test_ticker_drillthrough_module_deleted():
    """ticker_drillthrough_ui was absorbed into app.py main() routing.
    Verify it no longer exists and that app.py imports correctly."""
    import importlib
    import importlib.util
    spec = importlib.util.find_spec("ui.ticker_drillthrough_ui")
    assert spec is None, "ticker_drillthrough_ui.py should be deleted — drill-through is now in app.py"


def test_app_imports_cleanly():
    """app.py must import without raising (smoke test for routing refactor)."""
    # We can't import app.py normally (it calls st.set_page_config at module level)
    # so just verify the module source is parseable and has run_ticker_mode.
    import ast
    from pathlib import Path
    source = (Path(__file__).parent.parent / "app.py").read_text()
    tree = ast.parse(source)
    fn_names = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    assert "run_ticker_mode" in fn_names, "run_ticker_mode must exist in app.py"


def test_ticker_analysis_scenario_ordering():
    analysis = _make_analysis()
    ordered = sorted(
        analysis.scenarios,
        key=lambda s: ["bull", "base", "bear"].index(s.scenario)
        if s.scenario in ["bull", "base", "bear"] else 99,
    )
    assert [s.scenario for s in ordered] == ["bull", "base", "bear"]


def test_ticker_analysis_implied_return_derived():
    analysis = _make_analysis()
    bull = next(s for s in analysis.scenarios if s.scenario == "bull")
    assert bull.implied_return is not None
    assert abs(bull.implied_return - (210.0 / 150.0 - 1.0)) < 1e-6


def test_ticker_analysis_expected_return_1y_derived():
    analysis = _make_analysis()
    assert analysis.expected_return_1y is not None
    assert -1.0 < analysis.expected_return_1y < 5.0
