"""Tests for core/schemas.py — pure logic, no Streamlit, no API calls."""
from __future__ import annotations

import pytest
from datetime import date

from core.schemas import Driver, Scenario, Source, TickerAnalysis


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_analysis(**kwargs) -> TickerAnalysis:
    defaults = dict(ticker="TEST", as_of=date(2026, 1, 1))
    defaults.update(kwargs)
    return TickerAnalysis(**defaults)


# ── import sanity ─────────────────────────────────────────────────────────────

class TestImport:
    def test_import_without_streamlit(self) -> None:
        """core.schemas must import cleanly with no st.* side-effects."""
        from core import TickerAnalysis as TA, analyze_ticker  # noqa: F401
        assert TA is TickerAnalysis

    def test_ticker_analysis_instantiation(self) -> None:
        ta = _make_analysis()
        assert ta.ticker == "TEST"
        assert ta.analysis_version == "v1"


# ── probability normalization ─────────────────────────────────────────────────

class TestProbabilityNormalization:
    def test_already_normalized_unchanged(self) -> None:
        scenarios = [
            Scenario(scenario="bull", probability=0.3, price_target=200.0),
            Scenario(scenario="base", probability=0.5, price_target=150.0),
            Scenario(scenario="bear", probability=0.2, price_target=100.0),
        ]
        ta = _make_analysis(scenarios=scenarios, current_price=130.0)
        total = sum(s.probability for s in ta.scenarios)
        assert abs(total - 1.0) < 1e-5

    def test_unnormalized_probabilities_are_scaled(self) -> None:
        scenarios = [
            Scenario(scenario="bull", probability=0.6, price_target=200.0),
            Scenario(scenario="base", probability=1.0, price_target=150.0),
            Scenario(scenario="bear", probability=0.4, price_target=100.0),
        ]
        ta = _make_analysis(scenarios=scenarios, current_price=130.0)
        total = sum(s.probability for s in ta.scenarios)
        assert abs(total - 1.0) < 1e-5, f"Expected probabilities summing to 1, got {total}"

    def test_single_scenario_gets_probability_1(self) -> None:
        scenarios = [Scenario(scenario="base", probability=0.7, price_target=150.0)]
        ta = _make_analysis(scenarios=scenarios, current_price=130.0)
        assert abs(ta.scenarios[0].probability - 1.0) < 1e-5

    def test_empty_scenarios_no_error(self) -> None:
        ta = _make_analysis(scenarios=[])
        assert ta.scenarios == []


# ── implied_return derivation ─────────────────────────────────────────────────

class TestImpliedReturn:
    def test_implied_return_computed_from_price_target(self) -> None:
        scenarios = [
            Scenario(scenario="bull", probability=0.5, price_target=200.0),
            Scenario(scenario="bear", probability=0.5, price_target=100.0),
        ]
        ta = _make_analysis(scenarios=scenarios, current_price=100.0)
        bull = next(s for s in ta.scenarios if s.scenario == "bull")
        bear = next(s for s in ta.scenarios if s.scenario == "bear")
        assert bull.implied_return == pytest.approx(1.0)   # 200/100 - 1
        assert bear.implied_return == pytest.approx(0.0)   # 100/100 - 1

    def test_implied_return_none_when_no_price_target(self) -> None:
        scenarios = [Scenario(scenario="base", probability=1.0, price_target=None)]
        ta = _make_analysis(scenarios=scenarios, current_price=100.0)
        assert ta.scenarios[0].implied_return is None

    def test_implied_return_none_when_no_current_price(self) -> None:
        scenarios = [Scenario(scenario="base", probability=1.0, price_target=150.0)]
        ta = _make_analysis(scenarios=scenarios, current_price=None)
        assert ta.scenarios[0].implied_return is None

    def test_implied_return_negative_for_downside(self) -> None:
        scenarios = [Scenario(scenario="bear", probability=1.0, price_target=80.0)]
        ta = _make_analysis(scenarios=scenarios, current_price=100.0)
        assert ta.scenarios[0].implied_return == pytest.approx(-0.2)


# ── expected return derivation ────────────────────────────────────────────────

class TestExpectedReturnDerivation:
    def _ta_with_scenarios(self, current_price: float, targets: list) -> TickerAnalysis:
        """Builds TickerAnalysis with equal-prob scenarios at given price targets."""
        p = 1.0 / len(targets)
        scenarios = [
            Scenario(scenario=label, probability=p, price_target=tgt)
            for label, tgt in zip(["bull", "base", "bear"], targets)
        ]
        return _make_analysis(scenarios=scenarios, current_price=current_price)

    def test_expected_return_1y_symmetric(self) -> None:
        # 50% chance of +100%, 50% chance of 0%  → expected 1y = 0.5 * 1.0 + 0.5 * 0.0 = 0.5
        scenarios = [
            Scenario(scenario="bull", probability=0.5, price_target=200.0),
            Scenario(scenario="bear", probability=0.5, price_target=100.0),
        ]
        ta = _make_analysis(scenarios=scenarios, current_price=100.0)
        assert ta.expected_return_1y == pytest.approx(0.5, abs=1e-5)

    def test_expected_return_longer_horizons_annualized(self) -> None:
        # +100% total return annualized over 3y = (2.0)^(1/3) - 1 ≈ 0.2599
        scenarios = [Scenario(scenario="bull", probability=1.0, price_target=200.0)]
        ta = _make_analysis(scenarios=scenarios, current_price=100.0)
        expected_3y = (2.0) ** (1.0 / 3) - 1.0
        assert ta.expected_return_3y == pytest.approx(expected_3y, abs=1e-5)

    def test_no_expected_return_without_price(self) -> None:
        scenarios = [Scenario(scenario="base", probability=1.0, price_target=150.0)]
        ta = _make_analysis(scenarios=scenarios, current_price=None)
        assert ta.expected_return_1y is None
        assert ta.expected_return_3y is None
        assert ta.expected_return_5y is None

    def test_no_expected_return_without_scenarios(self) -> None:
        ta = _make_analysis(current_price=100.0)
        assert ta.expected_return_1y is None

    def test_all_three_horizons_computed(self) -> None:
        scenarios = [Scenario(scenario="base", probability=1.0, price_target=150.0)]
        ta = _make_analysis(scenarios=scenarios, current_price=100.0)
        assert ta.expected_return_1y is not None
        assert ta.expected_return_3y is not None
        assert ta.expected_return_5y is not None
        # 1y return > 3y annualized return > 5y annualized (since total_return > 0)
        assert ta.expected_return_1y > ta.expected_return_3y > ta.expected_return_5y


# ── data_gaps and sources ─────────────────────────────────────────────────────

class TestMetadataFields:
    def test_data_gaps_stored(self) -> None:
        ta = _make_analysis(data_gaps=["no transcripts", "dcf failed"])
        assert "no transcripts" in ta.data_gaps

    def test_sources_stored(self) -> None:
        s = Source(label="yfinance", url=None)
        ta = _make_analysis(sources=[s])
        assert ta.sources[0].label == "yfinance"

    def test_confidence_clamped(self) -> None:
        with pytest.raises(Exception):
            _make_analysis(confidence=1.5)

    def test_confidence_default(self) -> None:
        ta = _make_analysis()
        assert ta.confidence == 0.5


# ── Driver and Scenario models ────────────────────────────────────────────────

class TestDriverAndScenario:
    def test_driver_defaults(self) -> None:
        d = Driver(name="Revenue growth")
        assert d.direction == "neutral"
        assert d.value == 0.0

    def test_scenario_probability_bounds(self) -> None:
        with pytest.raises(Exception):
            Scenario(scenario="bull", probability=1.5)
        with pytest.raises(Exception):
            Scenario(scenario="bull", probability=-0.1)
