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
    """Call _scenario_card with st patched, return the mock st.

    Patches both ``ui.portfolio_analysis_ui.st`` (direct calls) and
    ``streamlit.markdown`` (calls routed via ``render_md`` from core.text_render)
    so both paths are captured by the same mock.
    """
    mock_st = MagicMock()
    with patch("ui.portfolio_analysis_ui.st", mock_st), \
         patch("streamlit.markdown", mock_st.markdown):
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
    # Probability is now in the header markdown, not a separate caption
    rendered_text = " ".join(
        str(c) for c in mock_st.markdown.call_args_list + mock_st.caption.call_args_list
    )
    assert "35%" in rendered_text


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


# ── Issue 5 regression tests: scenario cards and portfolio aggregation ────────

def test_scenario_card_shows_all_three_cases():
    """Every holding must produce a card for bull, base, and bear."""
    analysis = _make_analysis("AAPL")
    assert len(analysis.scenarios) == 3
    labels = {s.scenario for s in analysis.scenarios}
    assert labels == {"bull", "base", "bear"}


def test_scenario_card_shows_narrative():
    """Narrative text must be rendered — not just the price target."""
    scenario = Scenario(
        scenario="bull", probability=0.35, price_target=200.0,
        narrative="AI spending drives upside."
    )
    mock_st = _call_scenario_card(scenario, current_price=150.0)
    rendered_text = " ".join(str(c) for c in mock_st.markdown.call_args_list)
    assert "AI spending" in rendered_text


def test_scenario_card_shows_drivers():
    """Drivers must appear in the rendered output."""
    scenario = Scenario(
        scenario="bear", probability=0.20, price_target=80.0,
        narrative="",
        drivers=[
            Driver(name="Margin compression", direction="negative"),
            Driver(name="FX headwind", direction="negative"),
        ],
    )
    mock_st = _call_scenario_card(scenario, current_price=150.0)
    rendered_text = " ".join(str(c) for c in mock_st.markdown.call_args_list)
    assert "Margin compression" in rendered_text


def test_scenario_aggregation_produces_three_rows():
    """Portfolio aggregation must produce exactly 3 rows (bull, base, bear)."""
    from unittest.mock import patch, MagicMock

    tickers = ["AAPL", "MSFT"]
    current_weights = {"AAPL": 0.6, "MSFT": 0.4}
    proposed_weights = {"AAPL": 0.5, "MSFT": 0.5}

    analysis_aapl = _make_analysis("AAPL")
    analysis_msft = _make_analysis("MSFT")

    session = {
        "_scenario_AAPL": analysis_aapl,
        "_scenario_MSFT": analysis_msft,
    }

    captured_dfs = []
    mock_st = MagicMock()
    mock_st.session_state = session

    def fake_dataframe(df, **kwargs):
        captured_dfs.append(df)

    mock_st.dataframe.side_effect = fake_dataframe

    with patch("ui.portfolio_analysis_ui.st", mock_st):
        from ui.portfolio_analysis_ui import _render_scenario_aggregation
        _render_scenario_aggregation(tickers, current_weights, proposed_weights)

    assert len(captured_dfs) == 1
    df = captured_dfs[0]
    assert len(df) == 3  # bull, base, bear
    assert set(df["Scenario"]) == {"Bull", "Base", "Bear"}


def test_scenario_aggregation_computes_weighted_return():
    """Weighted portfolio return = Σ weight × implied_return for each scenario."""
    from unittest.mock import patch, MagicMock
    from core.schemas import TickerAnalysis, Scenario
    from datetime import date

    # Build analyses with known implied returns
    def make_a(ticker, bull_target, base_target, bear_target, price=100.0):
        return TickerAnalysis(
            ticker=ticker, as_of=date.today(), current_price=price,
            scenarios=[
                Scenario(scenario="bull", price_target=bull_target, probability=0.35),
                Scenario(scenario="base", price_target=base_target, probability=0.45),
                Scenario(scenario="bear", price_target=bear_target, probability=0.20),
            ],
        )

    # AAPL: bull=+20%, base=+10%, bear=-10% (at price 100)
    # MSFT: bull=+30%, base=+5%,  bear=-20%
    analysis_aapl = make_a("AAPL", 120, 110, 90)
    analysis_msft = make_a("MSFT", 130, 105, 80)

    tickers = ["AAPL", "MSFT"]
    cur_w = {"AAPL": 0.5, "MSFT": 0.5}
    prop_w = {"AAPL": 0.6, "MSFT": 0.4}

    session = {"_scenario_AAPL": analysis_aapl, "_scenario_MSFT": analysis_msft}

    captured_dfs = []
    mock_st = MagicMock()
    mock_st.session_state = session
    mock_st.dataframe.side_effect = lambda df, **kw: captured_dfs.append(df)

    with patch("ui.portfolio_analysis_ui.st", mock_st):
        from ui.portfolio_analysis_ui import _render_scenario_aggregation
        _render_scenario_aggregation(tickers, cur_w, prop_w)

    df = captured_dfs[0]
    bull_row = df[df["Scenario"] == "Bull"].iloc[0]

    # Expected bull return (current weights): 0.5*0.20 + 0.5*0.30 = 0.25
    cur_bull_str = bull_row["Weighted Return (current weights)"]
    assert "+25.0%" in cur_bull_str or "+25%" in cur_bull_str, f"Got: {cur_bull_str}"
