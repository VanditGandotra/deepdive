"""Tests for reconcile.py: period alignment, concept selection, composites, math."""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from analysis.reconcile import (
    _diff_note,
    _edgar_flow_ttm,
    _edgar_instant,
    _pct_diff,
    _SANITY_THRESHOLD,
    _STALE_DAYS,
    _stale_note,
    get_reconciliation,
)
from data.edgar import build_xbrl_composite, compute_edgar_ttm, extract_xbrl_instant


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_quarterly_record(start: str, end: str, val: float, filed: str = "2025-01-01") -> Dict:
    return {"start": start, "end": end, "val": val, "form": "10-Q", "filed": filed}


def _make_facts(concept: str, records: list, namespace: str = "us-gaap") -> Dict[str, Any]:
    return {
        "facts": {
            namespace: {
                concept: {
                    "units": {
                        "USD": records,
                    }
                }
            }
        }
    }


def _make_share_facts(concept: str, records: list, namespace: str = "dei") -> Dict[str, Any]:
    return {
        "facts": {
            namespace: {
                concept: {
                    "units": {
                        "shares": records,
                    }
                }
            }
        }
    }


# ── Unit tests: compute_edgar_ttm ─────────────────────────────────────────────

class TestComputeEdgarTTM:

    def _quarterly(self, start: str, end: str, val: float, filed: str = "2025-01-01") -> Dict:
        return _make_quarterly_record(start, end, val, filed)

    def test_four_quarters_sums_correctly(self) -> None:
        records = [
            self._quarterly("2024-10-01", "2024-12-31", 100.0),
            self._quarterly("2024-07-01", "2024-09-30", 90.0),
            self._quarterly("2024-04-01", "2024-06-30", 80.0),
            self._quarterly("2024-01-01", "2024-03-31", 70.0),
        ]
        facts = _make_facts("NetIncomeLoss", records)
        result = compute_edgar_ttm(facts, "NetIncomeLoss")
        assert result is not None
        value, end_date = result
        assert value == pytest.approx(340.0)
        assert end_date == "2024-12-31"

    def test_fewer_than_four_returns_none(self) -> None:
        records = [
            self._quarterly("2024-10-01", "2024-12-31", 100.0),
            self._quarterly("2024-07-01", "2024-09-30", 90.0),
            self._quarterly("2024-04-01", "2024-06-30", 80.0),
        ]
        facts = _make_facts("NetIncomeLoss", records)
        assert compute_edgar_ttm(facts, "NetIncomeLoss") is None

    def test_annual_records_excluded(self) -> None:
        """Annual (365-day) records should not count as quarters."""
        records = [
            {"start": "2024-01-01", "end": "2024-12-31", "val": 400.0, "form": "10-K", "filed": "2025-01-01"},
            self._quarterly("2024-10-01", "2024-12-31", 100.0),
            self._quarterly("2024-07-01", "2024-09-30", 90.0),
            self._quarterly("2024-04-01", "2024-06-30", 80.0),
        ]
        # Only 3 true quarters → should return None
        facts = _make_facts("Revenues", records)
        assert compute_edgar_ttm(facts, "Revenues") is None

    def test_deduplication_keeps_latest_filed(self) -> None:
        """Two records for same (start, end) → keep the one with the latest filed date."""
        records = [
            self._quarterly("2024-10-01", "2024-12-31", 999.0, filed="2025-01-01"),  # stale
            self._quarterly("2024-10-01", "2024-12-31", 105.0, filed="2025-02-15"),  # latest
            self._quarterly("2024-07-01", "2024-09-30", 90.0),
            self._quarterly("2024-04-01", "2024-06-30", 80.0),
            self._quarterly("2024-01-01", "2024-03-31", 70.0),
        ]
        facts = _make_facts("NetIncomeLoss", records)
        result = compute_edgar_ttm(facts, "NetIncomeLoss")
        assert result is not None
        value, _ = result
        assert value == pytest.approx(345.0)  # 105 + 90 + 80 + 70

    def test_missing_concept_returns_none(self) -> None:
        facts = _make_facts("SomeOtherConcept", [])
        assert compute_edgar_ttm(facts, "NetIncomeLoss") is None


# ── Unit tests: extract_xbrl_instant ─────────────────────────────────────────

class TestExtractXbrlInstant:

    def test_returns_most_recent_value(self) -> None:
        records = [
            {"end": "2024-09-30", "val": 15_000_000.0, "form": "10-Q", "filed": "2024-11-01"},
            {"end": "2024-12-31", "val": 16_000_000.0, "form": "10-Q", "filed": "2025-02-01"},
        ]
        facts = _make_share_facts("EntityCommonStockSharesOutstanding", records, namespace="dei")
        result = extract_xbrl_instant(facts, "EntityCommonStockSharesOutstanding", namespace="dei")
        assert result is not None
        val, end = result
        assert val == pytest.approx(16_000_000.0)
        assert end == "2024-12-31"

    def test_returns_none_when_no_records(self) -> None:
        facts = _make_share_facts("EntityCommonStockSharesOutstanding", [], namespace="dei")
        assert extract_xbrl_instant(facts, "EntityCommonStockSharesOutstanding", namespace="dei") is None


# ── Unit tests: _pct_diff ────────────────────────────────────────────────────

class TestPctDiff:
    def test_basic(self) -> None:
        assert _pct_diff(110.0, 100.0) == pytest.approx(0.10)

    def test_zero_denominator(self) -> None:
        assert _pct_diff(100.0, 0.0) is None

    def test_none_inputs(self) -> None:
        assert _pct_diff(None, 100.0) is None
        assert _pct_diff(100.0, None) is None


# ── Unit tests: period-gate note ─────────────────────────────────────────────

class TestDiffNote:

    def test_stale_edgar_flagged_in_note(self) -> None:
        old_date = (date.today() - timedelta(days=_STALE_DAYS + 30)).isoformat()
        note = _diff_note("revenue", 0.05, old_date)
        assert "lag" in note.lower() or "ago" in note.lower()

    def test_fresh_edgar_no_stale_warning(self) -> None:
        fresh_date = (date.today() - timedelta(days=30)).isoformat()
        note = _diff_note("revenue", 0.05, fresh_date)
        assert "lag" not in note.lower()

    def test_agree_within_2pct(self) -> None:
        note = _diff_note("revenue", 0.01, "")
        assert "agree" in note.lower()

    def test_sanity_gate_triggers(self) -> None:
        fresh_date = (date.today() - timedelta(days=30)).isoformat()
        note = _diff_note("revenue", _SANITY_THRESHOLD + 0.01, fresh_date)
        assert "pipeline error" in note.lower() or "⚠" in note


# ── Unit tests: build_xbrl_composite ─────────────────────────────────────────

class TestBuildXbrlComposite:

    def _make_instant_facts(self, tag: str, val: float, end: str, ns: str = "us-gaap") -> Dict:
        return {
            "facts": {
                ns: {
                    tag: {
                        "units": {
                            "USD": [
                                {"end": end, "val": val, "form": "10-Q", "filed": "2025-01-01"},
                            ]
                        }
                    }
                }
            }
        }

    def _merge_facts(self, *fact_dicts) -> Dict:
        merged: Dict = {"facts": {"us-gaap": {}, "dei": {}}}
        for fd in fact_dicts:
            for ns, concepts in fd.get("facts", {}).items():
                merged["facts"].setdefault(ns, {}).update(concepts)
        return merged

    def test_complete_composite_sums_correctly(self) -> None:
        facts = self._merge_facts(
            self._make_instant_facts("CashAndCashEquivalentsAtCarryingValue", 40.0, "2024-12-31"),
            self._make_instant_facts("MarketableSecuritiesCurrent", 20.0, "2024-12-31"),
        )
        result = build_xbrl_composite(facts, [
            {"label": "Cash", "aliases": ["CashAndCashEquivalentsAtCarryingValue"], "required": True},
            {"label": "ShortTermInvestments", "aliases": ["MarketableSecuritiesCurrent", "ShortTermInvestments"], "required": False},
        ])
        assert not result["incomplete"]
        assert not result["date_mismatch"]
        assert result["value"] == pytest.approx(60.0)
        assert result["end"] == "2024-12-31"
        assert result["components"]["Cash"] == pytest.approx(40.0)
        assert result["components"]["ShortTermInvestments"] == pytest.approx(20.0)

    def test_missing_required_slot_marks_incomplete(self) -> None:
        facts = self._make_instant_facts("CashAndCashEquivalentsAtCarryingValue", 40.0, "2024-12-31")
        result = build_xbrl_composite(facts, [
            {"label": "Cash", "aliases": ["CashAndCashEquivalentsAtCarryingValue"], "required": True},
            {"label": "LongTermDebtCurrent", "aliases": ["LongTermDebtCurrent"], "required": True},
        ])
        assert result["incomplete"]
        assert "LongTermDebtCurrent" in result["missing"]
        assert result["value"] is None

    def test_missing_optional_slot_is_skipped(self) -> None:
        facts = self._make_instant_facts("CashAndCashEquivalentsAtCarryingValue", 40.0, "2024-12-31")
        result = build_xbrl_composite(facts, [
            {"label": "Cash", "aliases": ["CashAndCashEquivalentsAtCarryingValue"], "required": True},
            {"label": "ShortTermInvestments", "aliases": ["ShortTermInvestments"], "required": False},
        ])
        assert not result["incomplete"]
        assert result["value"] == pytest.approx(40.0)

    def test_date_mismatch_detected(self) -> None:
        facts = self._merge_facts(
            self._make_instant_facts("CashAndCashEquivalentsAtCarryingValue", 40.0, "2024-12-31"),
            self._make_instant_facts("MarketableSecuritiesCurrent", 20.0, "2024-09-30"),  # different date
        )
        result = build_xbrl_composite(facts, [
            {"label": "Cash", "aliases": ["CashAndCashEquivalentsAtCarryingValue"], "required": True},
            {"label": "ShortTermInvestments", "aliases": ["MarketableSecuritiesCurrent"], "required": False},
        ])
        assert result["date_mismatch"]

    def test_alias_fallback_used(self) -> None:
        """If primary alias missing, should try secondary."""
        facts = self._make_instant_facts("ShortTermInvestments", 25.0, "2024-12-31")
        result = build_xbrl_composite(facts, [
            {"label": "STI", "aliases": ["MarketableSecuritiesCurrent", "ShortTermInvestments"], "required": True},
        ])
        assert not result["incomplete"]
        assert result["value"] == pytest.approx(25.0)


# ── Integration smoke-test: get_reconciliation structure ─────────────────────

class TestGetReconciliationStructure:

    def test_returns_five_metrics(self) -> None:
        """get_reconciliation always returns exactly 5 MetricReconciliation rows."""
        from analysis.schemas import Fundamentals

        mock_fund = Fundamentals(
            ticker="TEST",
            revenue_ttm=100.0,
            net_income_ttm=10.0,
            shares_outstanding=5e9,
            total_debt=20.0,
            cash=30.0,
        )

        q_records = [
            {"start": "2024-10-01", "end": "2024-12-31", "val": 30.0, "form": "10-Q", "filed": "2025-01-01"},
            {"start": "2024-07-01", "end": "2024-09-30", "val": 25.0, "form": "10-Q", "filed": "2024-10-15"},
            {"start": "2024-04-01", "end": "2024-06-30", "val": 22.0, "form": "10-Q", "filed": "2024-07-15"},
            {"start": "2024-01-01", "end": "2024-03-31", "val": 23.0, "form": "10-Q", "filed": "2024-04-15"},
        ]
        instant_records = [
            {"end": "2024-12-31", "val": 5.0e9, "form": "10-Q", "filed": "2025-01-01"},
        ]
        mock_facts = {
            "facts": {
                "us-gaap": {
                    "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": q_records}},
                    "NetIncomeLoss": {"units": {"USD": q_records}},
                    "LongTermDebt": {"units": {"USD": instant_records}},
                    "CashCashEquivalentsAndShortTermInvestments": {"units": {"USD": instant_records}},
                },
                "dei": {
                    "EntityCommonStockSharesOutstanding": {"units": {"shares": instant_records}},
                },
            }
        }

        with patch("analysis.reconcile.get_fundamentals", return_value=mock_fund), \
             patch("analysis.reconcile.get_xbrl_facts", return_value=mock_facts):
            result = get_reconciliation("TEST")

        assert len(result) == 5
        metrics = {r.metric for r in result}
        assert metrics == {"revenue", "net_income", "shares", "total_debt", "cash"}

    def test_no_edgar_data_returns_yfinance_only(self) -> None:
        from analysis.schemas import Fundamentals

        mock_fund = Fundamentals(ticker="TEST", revenue_ttm=100.0)

        with patch("analysis.reconcile.get_fundamentals", return_value=mock_fund), \
             patch("analysis.reconcile.get_xbrl_facts", return_value=None):
            result = get_reconciliation("TEST")

        assert len(result) == 5
        for r in result:
            assert r.edgar_value is None
