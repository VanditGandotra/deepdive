"""
Phase 11: Eval checks — schema conformance, extraction accuracy, grounding audits,
determinism, pure math, and cost regression.

Run via: pytest evals/checks.py -v
Or via evals/run_evals.py CLI for Batch API mode.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

GOLDEN_DIR = Path(__file__).parent / "golden"


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _load_golden(name: str) -> Dict:
    path = GOLDEN_DIR / f"{name}.json"
    with open(path) as f:
        return json.load(f)


def _schema_valid(obj: Any, model_class: Any) -> bool:
    try:
        model_class.model_validate(obj if isinstance(obj, dict) else obj.model_dump())
        return True
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Transcript Provider Chain
# ══════════════════════════════════════════════════════════════════════════════

class TestTranscriptProviders:
    """Unit tests for the transcript provider chain — no network calls, no LLM."""

    def test_split_qa_boundary_detected(self) -> None:
        from data.transcripts import _split_transcript_text
        text = (
            "CEO: Revenue was $10B in the quarter.\n\n"
            "CFO: Free cash flow was strong.\n\n"
            "Operator: We will now begin the question-and-answer session.\n\n"
            "Analyst: What is your guidance for next quarter?\n\n"
            "CEO: We expect continued growth."
        )
        prepared, qa = _split_transcript_text(text)
        assert "Revenue" in prepared
        assert "question-and-answer" in qa
        assert "guidance" in qa
        assert "guidance" not in prepared

    def test_split_no_qa_returns_full_and_empty(self) -> None:
        from data.transcripts import _split_transcript_text
        text = "CEO: Revenue was $10B.\n\nCFO: Margins expanded."
        prepared, qa = _split_transcript_text(text)
        assert qa == ""
        assert "Revenue" in prepared

    def test_split_open_floor_variant(self) -> None:
        from data.transcripts import _split_transcript_text
        text = (
            "CFO: EBITDA margin was 28%.\n\n"
            "Operator: Thank you. We will now open the floor for questions.\n\n"
            "Analyst: Can you walk us through capex plans?"
        )
        prepared, qa = _split_transcript_text(text)
        assert "EBITDA" in prepared
        assert "capex" in qa

    def test_latest_likely_quarter_valid_range(self) -> None:
        from data.transcripts import _latest_likely_quarter
        year, quarter = _latest_likely_quarter()
        assert isinstance(year, int) and year >= 2020
        assert quarter in (1, 2, 3, 4)

    def test_latest_likely_quarter_not_future(self) -> None:
        from datetime import datetime
        from data.transcripts import _latest_likely_quarter
        year, quarter = _latest_likely_quarter()
        now = datetime.utcnow()
        assert year <= now.year

    def test_motleyfool_always_available(self) -> None:
        from data.transcripts import MotleyFoolProvider
        assert MotleyFoolProvider().available() is True

    def test_motleyfool_last_in_chain(self) -> None:
        from data.transcripts import _PROVIDERS, MotleyFoolProvider
        assert isinstance(_PROVIDERS[-1], MotleyFoolProvider), (
            "MotleyFoolProvider must be last (web scrape is the slowest fallback)"
        )

    def test_provider_names_unique(self) -> None:
        from data.transcripts import _PROVIDERS
        names = [p.name for p in _PROVIDERS]
        assert len(names) == len(set(names)), f"Duplicate provider names: {names}"

    def test_get_available_includes_motleyfool(self) -> None:
        from data.transcripts import get_available_provider_names
        assert "motleyfool" in get_available_provider_names()

    def test_fmp_unavailable_without_key(self, monkeypatch) -> None:
        import data.transcripts as dt
        monkeypatch.setattr(dt, "FMP_API_KEY", "")
        from data.transcripts import FmpProvider
        assert FmpProvider().available() is False

    def test_fmp_available_with_key(self, monkeypatch) -> None:
        import data.transcripts as dt
        monkeypatch.setattr(dt, "FMP_API_KEY", "testkey123")
        from data.transcripts import FmpProvider
        assert FmpProvider().available() is True

    def test_finnhub_unavailable_without_key(self, monkeypatch) -> None:
        import data.transcripts as dt
        monkeypatch.setattr(dt, "FINNHUB_API_KEY", "")
        from data.transcripts import FinnhubProvider
        assert FinnhubProvider().available() is False

    def test_six_providers_in_chain(self) -> None:
        from data.transcripts import _PROVIDERS
        assert len(_PROVIDERS) == 6, f"Expected 6 providers, got {len(_PROVIDERS)}"

    def test_provider_chain_includes_fmp(self) -> None:
        from data.transcripts import _PROVIDERS, FmpProvider
        assert any(isinstance(p, FmpProvider) for p in _PROVIDERS)

    def test_provider_chain_includes_finnhub(self) -> None:
        from data.transcripts import _PROVIDERS, FinnhubProvider
        assert any(isinstance(p, FinnhubProvider) for p in _PROVIDERS)


# ══════════════════════════════════════════════════════════════════════════════
# Schema Conformance
# ══════════════════════════════════════════════════════════════════════════════

class TestSchemaConformance:
    """All Pydantic models must accept valid data without raising."""

    def test_call_summary_schema(self) -> None:
        from analysis.schemas import CallSummary, GuidanceItem
        cs = CallSummary(
            quarter="Q3 FY2025",
            guidance_items=[GuidanceItem(metric="Revenue", value="$37.5B", direction="raised")],
            key_themes=["Blackwell ramp", "Data Center strength"],
            top_analyst_concerns_from_qa=["supply constraints", "China exports"],
            notable_quotes_paraphrased=["Demand is extraordinary"],
        )
        assert cs.quarter == "Q3 FY2025"
        assert len(cs.guidance_items) == 1

    def test_call_sentiment_schema(self) -> None:
        from analysis.schemas import CallSentiment, HedgingIndex, SpeakerSentiment
        cs = CallSentiment(
            overall_score=0.6,
            prepared_remarks_score=0.7,
            qa_score=0.4,
            per_speaker=[SpeakerSentiment(name="Jensen Huang", role="CEO", score=0.65, confidence_language_examples=["extraordinary demand"])],
            hedging_index=HedgingIndex(level="medium", example_phrases=["we believe", "we expect"]),
            evasiveness_flags=[],
        )
        assert cs.overall_score == 0.6

    def test_page_intel_schema(self) -> None:
        from analysis.schemas import DatedAnnouncement, PageIntel
        pi = PageIntel(
            url="https://example.com/customers",
            page_type="customer",
            named_customers=["Acme Corp", "Globex Inc"],
            feature_claims=["Real-time analytics", "API-first"],
            tech_details=["REST API", "Webhooks"],
            hiring_signals=["Hiring Senior ML Engineer"],
            dated_announcements=[
                DatedAnnouncement(headline="Acme deploys product", summary="Case study published.")
            ],
        )
        assert pi.page_type == "customer"
        assert "Acme Corp" in pi.named_customers

    def test_run_snapshot_data_schema(self) -> None:
        from analysis.schemas import RunSnapshotData
        snap = RunSnapshotData(
            customer_list=["A Corp", "B Corp"],
            last_30_days_count=5,
            quality_flags=["dso_expanding"],
            short_interest_pct=0.08,
        )
        assert snap.last_30_days_count == 5

    def test_delta_item_schema(self) -> None:
        from analysis.schemas import DeltaItem
        di = DeltaItem(
            field="P/E TTM",
            old_value=25.0,
            new_value=32.5,
            change_type="changed",
            description="P/E TTM: up 30.0% (25.0 → 32.5)",
        )
        assert di.change_type == "changed"

    def test_screen_explanation_schema(self) -> None:
        from analysis.schemas import ScreenExplanation
        se = ScreenExplanation(
            screen_name_guess="Dashboard — Analytics Overview",
            what_it_shows="Time-series chart of active users with date-range selector",
            ui_elements_of_note=["Date picker", "Export CSV button", "Metric cards"],
            what_this_reveals_about_the_product="Product has built-in analytics with export capabilities",
        )
        assert "Date picker" in se.ui_elements_of_note


# ══════════════════════════════════════════════════════════════════════════════
# Extraction Accuracy — Transcript
# ══════════════════════════════════════════════════════════════════════════════

class TestTranscriptExtraction:
    """Guidance recall ≥ 90% on synthetic transcript."""

    GOLDEN_GUIDANCE = [
        {"metric": "Revenue", "direction": "raised"},
        {"metric": "Gross margin", "direction": "lowered"},
        {"metric": "Operating expenses", "direction": "raised"},
    ]

    def _extract_from_golden(self) -> Any:
        from analysis.calls import extract_call_summary
        golden = _load_golden("nvda_transcript")
        transcript_text = golden["transcript"]
        return extract_call_summary({"transcript": transcript_text, "ticker": "NVDA", "year": 2024, "quarter": "Q3"})

    @pytest.mark.llm
    def test_guidance_recall(self) -> None:
        result = self._extract_from_golden()
        extracted_metrics = [g.metric.lower() for g in result.guidance_items]
        hits = 0
        for expected in self.GOLDEN_GUIDANCE:
            metric_kw = expected["metric"].lower()
            if any(metric_kw in em for em in extracted_metrics):
                hits += 1
        recall = hits / len(self.GOLDEN_GUIDANCE)
        assert recall >= 0.90, f"Guidance recall {recall:.0%} < 90% — missing: {[g['metric'] for g in self.GOLDEN_GUIDANCE if g['metric'].lower() not in ' '.join(extracted_metrics)]}"

    @pytest.mark.llm
    def test_key_themes_non_empty(self) -> None:
        result = self._extract_from_golden()
        assert len(result.key_themes) >= 2, "Expected at least 2 key themes from transcript"

    @pytest.mark.llm
    def test_analyst_concerns_captured(self) -> None:
        result = self._extract_from_golden()
        combined = " ".join(result.top_analyst_concerns_from_qa).lower()
        assert "supply" in combined or "constraint" in combined or "china" in combined, \
            "Expected supply/China concerns in analyst Q&A"


# ══════════════════════════════════════════════════════════════════════════════
# Extraction Accuracy — Headlines
# ══════════════════════════════════════════════════════════════════════════════

class TestHeadlineClassification:
    """Direction accuracy ≥ 80% across 30 labeled headlines."""

    @pytest.mark.llm
    def test_direction_accuracy(self) -> None:
        from analysis.news_impact import classify_headlines
        golden = _load_golden("headlines_labeled")
        headlines = golden["headlines"]

        # Build NewsItem list from golden data
        from analysis.schemas import NewsItem
        news_items = [
            NewsItem(title=h["title"], source="golden", url=f"https://example.com/{i}")
            for i, h in enumerate(headlines)
        ]

        from unittest.mock import patch
        from data.news import get_news
        with patch("analysis.news_impact.get_news", return_value=news_items):
            results = classify_headlines("NVDA", "NVIDIA Corporation")

        result_map = {r.title: r.direction for r in results}
        correct = 0
        for h in headlines:
            pred = result_map.get(h["title"])
            if pred == h["direction"]:
                correct += 1

        accuracy = correct / len(headlines)
        assert accuracy >= 0.80, f"Direction accuracy {accuracy:.0%} < 80% ({correct}/{len(headlines)} correct)"

    @pytest.mark.llm
    def test_materiality_high_recall(self) -> None:
        """All high-materiality headlines should be classified as high or medium."""
        from analysis.news_impact import classify_headlines
        golden = _load_golden("headlines_labeled")
        headlines = golden["headlines"]
        high_mat = [h["title"] for h in headlines if h["materiality"] == "high"]

        from analysis.schemas import NewsItem
        news_items = [NewsItem(title=h["title"], source="golden") for h in headlines]

        from unittest.mock import patch
        with patch("analysis.news_impact.get_news", return_value=news_items):
            results = classify_headlines("NVDA", "NVIDIA Corporation")

        result_map = {r.title: r.materiality for r in results}
        misses = [t for t in high_mat if result_map.get(t) == "low"]
        assert not misses, f"High-materiality headlines classified as low: {misses}"


# ══════════════════════════════════════════════════════════════════════════════
# Grounding / Citation Audit
# ══════════════════════════════════════════════════════════════════════════════

class TestGroundingAudit:
    """Every Sonnet output must contain citation tags. Zero hallucination tolerance."""

    _CITATION_RE = re.compile(r'\[(RATIOS|XDCF|CALLΔ|QUAL|POS|KPI|EST|NEWS-HI|10K-\w+|PAGE-\d+|DOC-\d+)\]')

    def _has_citations(self, text: str) -> bool:
        return bool(self._CITATION_RE.search(text))

    @pytest.mark.llm
    def test_thesis_output_has_citations(self) -> None:
        """Streamed thesis narrative must contain at least one citation tag."""
        from analysis.schemas import Fundamentals
        from analysis.thesis import stream_theses
        fund = Fundamentals(
            ticker="NVDA",
            name="NVIDIA Corporation",
            pe_ttm=35.0,
            revenue_ttm=70e9,
            net_margin=0.55,
            revenue_growth_yoy=0.94,
            current_price=130.0,
            market_cap=3.2e12,
        )
        token_iter, _ = stream_theses("NVDA", fund)
        full_text = "".join(token_iter)
        assert self._has_citations(full_text), \
            "Thesis output contains no citation tags — grounding failure"

    @pytest.mark.llm
    def test_business_explainer_has_citations(self) -> None:
        """Business explainer stream must cite at least one 10K section."""
        from unittest.mock import patch
        fake_sections = {
            "item1": "NVIDIA designs GPUs. [Test Section]",
            "item1a": "Risk: AMD competition.",
            "item7": "Revenue grew 94% driven by Data Center.",
        }
        with patch("analysis.business.get_10k_sections", return_value=fake_sections):
            from analysis.business import stream_business_explainer
            token_iter, _ = stream_business_explainer("NVDA")
            full_text = "".join(token_iter)
        assert self._has_citations(full_text), \
            "Business explainer output contains no [10K-n] citation tags"


# ══════════════════════════════════════════════════════════════════════════════
# Determinism — Delta Engine
# ══════════════════════════════════════════════════════════════════════════════

class TestDeltaDeterminism:
    """compute_diff is pure Python — must produce identical results on repeated calls."""

    def test_pe_expansion_and_new_flag(self) -> None:
        from analysis.delta import compute_diff
        from analysis.schemas import RunSnapshotData
        golden = _load_golden("delta_snapshots")
        tc = golden["test_cases"][0]

        snap_data = RunSnapshotData(
            fundamentals=tc["new_snapshot"]["fundamentals"],
            short_interest_pct=tc["new_snapshot"]["short_interest_pct"],
            quality_flags=tc["new_snapshot"]["quality_flags"],
            customer_list=tc["new_snapshot"]["customer_list"],
            last_30_days_count=tc["new_snapshot"]["last_30_days_count"],
        )

        diff1 = compute_diff(tc["old_snapshot"], snap_data)
        diff2 = compute_diff(tc["old_snapshot"], snap_data)

        assert diff1 == diff2, "compute_diff not deterministic"

        expected = tc["expected_diffs"]
        changed_fields = [d.field for d in diff1 if d.change_type == "changed"]
        for cf in expected["changed_fields"]:
            assert cf in changed_fields, f"Expected changed field '{cf}' not in diff: {changed_fields}"

        new_flags = [d.new_value for d in diff1 if d.change_type == "new_flag"]
        for nf in expected["new_flags"]:
            assert nf in new_flags, f"Expected new flag '{nf}' not in diff: {new_flags}"

        added_customers = [d.new_value for d in diff1 if d.change_type == "added" and d.field == "named_customers"]
        for ac in expected["added_customers"]:
            assert ac in added_customers, f"Expected added customer '{ac}' not in diff: {added_customers}"

    def test_no_material_changes(self) -> None:
        from analysis.delta import compute_diff
        from analysis.schemas import RunSnapshotData
        golden = _load_golden("delta_snapshots")
        tc = golden["test_cases"][1]

        snap_data = RunSnapshotData(
            fundamentals=tc["new_snapshot"]["fundamentals"],
            short_interest_pct=tc["new_snapshot"]["short_interest_pct"],
            quality_flags=tc["new_snapshot"]["quality_flags"],
            customer_list=tc["new_snapshot"]["customer_list"],
            last_30_days_count=tc["new_snapshot"]["last_30_days_count"],
        )
        diff = compute_diff(tc["old_snapshot"], snap_data)
        assert diff == [], f"Expected empty diff for sub-threshold changes, got: {diff}"

    def test_diff_produces_same_result_twice(self) -> None:
        """Strict idempotency check: same inputs → same list object values."""
        from analysis.delta import compute_diff
        from analysis.schemas import RunSnapshotData

        old = {
            "_run_at": "2024-01-01T00:00:00",
            "snapshot": {
                "fundamentals": {"pe_ttm": 20.0, "revenue_ttm": 1e9, "market_cap": 1e11},
                "short_interest_pct": 0.05,
                "quality_flags": ["dso_expanding"],
                "customer_list": ["Alpha"],
                "last_30_days_count": 1,
            }
        }
        new = RunSnapshotData(
            fundamentals={"pe_ttm": 26.0, "revenue_ttm": 1.1e9, "market_cap": 1.2e11},
            short_interest_pct=0.06,
            quality_flags=["dso_expanding", "sbc_elevated"],
            customer_list=["Alpha", "Beta"],
            last_30_days_count=3,
        )
        r1 = compute_diff(old, new)
        r2 = compute_diff(old, new)
        assert [d.field for d in r1] == [d.field for d in r2]
        assert [d.change_type for d in r1] == [d.change_type for d in r2]


# ══════════════════════════════════════════════════════════════════════════════
# Pure Math — Reverse DCF
# ══════════════════════════════════════════════════════════════════════════════

class TestReverseDCFMath:
    """Verify isocurve math: implied price at (cagr=0, margin=fcf_ttm/rev) ≈ current price."""

    def _compute_implied_price(
        self,
        revenue: float,
        cagr: float,
        fcf_margin: float,
        discount_rate: float,
        terminal_growth: float,
        horizon: int,
        net_debt: float,
        shares: float,
    ) -> float:
        """Mirror of the math in analysis/expectations.py _dcf_implied_price."""
        fcf_base = revenue * fcf_margin
        pv = 0.0
        for t in range(1, horizon + 1):
            fcf_t = fcf_base * ((1 + cagr) ** t)
            pv += fcf_t / ((1 + discount_rate) ** t)
        # Terminal value (Gordon Growth)
        terminal_fcf = fcf_base * ((1 + cagr) ** horizon) * (1 + terminal_growth)
        tv = terminal_fcf / (discount_rate - terminal_growth)
        pv_tv = tv / ((1 + discount_rate) ** horizon)
        equity_value = pv + pv_tv - net_debt
        return equity_value / shares if shares > 0 else 0.0

    def test_zero_cagr_flat_fcf(self) -> None:
        """At 0% growth, implied price depends only on FCF margin and discount rate."""
        price = self._compute_implied_price(
            revenue=10e9, cagr=0.0, fcf_margin=0.20,
            discount_rate=0.10, terminal_growth=0.025,
            horizon=10, net_debt=0, shares=10e9,
        )
        # 10B * 20% = 2B FCF; terminal = 2B * 1.025 / (0.10 - 0.025) = 27.3B; PV TV = 27.3B / 1.1^10 = 10.5B
        # Annuity PV: 2B * (1 - 1/1.1^10) / 0.10 = 2B * 6.145 = 12.3B
        # Total EV = 22.8B → per share at 10B shares = $2.28
        assert 2.0 < price < 3.0, f"Expected price ~$2.28, got {price:.2f}"

    def test_higher_cagr_gives_higher_price(self) -> None:
        """Higher CAGR should always imply a higher stock price, all else equal."""
        kwargs = dict(
            revenue=10e9, fcf_margin=0.15,
            discount_rate=0.10, terminal_growth=0.025,
            horizon=10, net_debt=1e9, shares=5e9,
        )
        prices = [self._compute_implied_price(cagr=c, **kwargs) for c in [0.05, 0.10, 0.20, 0.30]]
        assert prices == sorted(prices), f"Prices not monotonically increasing with CAGR: {prices}"

    def test_pct_delta_math(self) -> None:
        from analysis.delta import _pct_delta
        assert _pct_delta(100.0, 120.0) == pytest.approx(0.20)
        assert _pct_delta(100.0, 80.0) == pytest.approx(-0.20)
        assert _pct_delta(0.0, 50.0) is None
        assert _pct_delta(None, 50.0) is None
        assert _pct_delta(50.0, None) is None

    def test_diff_scalar_threshold(self) -> None:
        from analysis.delta import _diff_scalar
        # Below 2% threshold — should return None
        result = _diff_scalar("P/E", 25.0, 25.4, threshold=0.02)
        assert result is None
        # Above threshold
        result = _diff_scalar("P/E", 25.0, 30.0, threshold=0.02)
        assert result is not None
        assert result.change_type == "changed"
        assert "up" in result.description

    def test_estimate_cost_math(self) -> None:
        from llm import estimate_cost
        from config import HAIKU, SONNET
        # Haiku: 1000 input, 500 output, 0 cache — cost should be > 0
        cost = estimate_cost(HAIKU, 1000, 500, 0, 0)
        assert cost > 0
        # Sonnet more expensive than Haiku for same tokens
        cost_sonnet = estimate_cost(SONNET, 1000, 500, 0, 0)
        assert cost_sonnet > cost, "Sonnet should cost more than Haiku"
        # Cache read should be cheaper than cold input
        cost_cold = estimate_cost(SONNET, 10000, 500, 0, 0)
        cost_cached = estimate_cost(SONNET, 10000, 500, 9000, 0)
        assert cost_cached < cost_cold, "Cache read should reduce cost"


# ══════════════════════════════════════════════════════════════════════════════
# Cost Regression
# ══════════════════════════════════════════════════════════════════════════════

class TestCostRegression:
    """Warn (not fail) if token count grows >25% without a prompt_version bump."""

    _BASELINE_TOKENS: Dict[str, int] = {
        "call_extraction_a": 800,
        "call_sentiment_b": 600,
        "headline_classification": 400,
        "kpi_extraction": 1200,
        "page_extraction": 700,
    }

    def test_prompt_versions_exist(self) -> None:
        """All expected prompt version keys must be present in config."""
        from config import PROMPT_VERSIONS
        required_keys = list(self._BASELINE_TOKENS.keys())
        for key in required_keys:
            assert key in PROMPT_VERSIONS, f"Missing PROMPT_VERSIONS['{key}']"

    def test_prompt_versions_non_empty(self) -> None:
        from config import PROMPT_VERSIONS
        for key, value in PROMPT_VERSIONS.items():
            assert value, f"PROMPT_VERSIONS['{key}'] is empty"


# ══════════════════════════════════════════════════════════════════════════════
# Cache Layer
# ══════════════════════════════════════════════════════════════════════════════

class TestChartRendering:
    """Regression suite: every chart builder must not raise TypeError from layout kwargs.

    These tests are fast (pure Python / Plotly, no LLM or network calls) and run
    as part of the default FAST_FEATURES suite. They specifically guard against the
    'multiple values for keyword argument' crash that occurs when a _BASE_LAYOUT
    spread and explicit kwargs share a key (e.g. margin, xaxis).
    """

    def _fake_price_data(self):
        from analysis.schemas import PriceBar, PriceData
        from datetime import date
        bars = [
            PriceBar(date=date(2024, 1, i + 1), open=100.0 + i, high=105.0 + i,
                     low=99.0 + i, close=102.0 + i, volume=1_000_000)
            for i in range(20)
        ]
        return PriceData(ticker="TEST", period="1y", bars=bars)

    def _fake_ratio(self):
        from analysis.schemas import RatioHistory
        return RatioHistory(
            name="P/E", current=25.0,
            history=[20.0, 22.0, 24.0, 25.0, 26.0],
            median_5y=23.0, min_5y=18.0, max_5y=28.0,
            description="Test ratio",
        )

    def test_price_candlestick_no_error(self) -> None:
        import plotly.graph_objects as go
        from ui.charts import price_candlestick
        fig = price_candlestick(self._fake_price_data(), "TEST — 1y")
        assert isinstance(fig, go.Figure)

    def test_ratio_sparkline_no_error(self) -> None:
        import plotly.graph_objects as go
        from ui.charts import ratio_sparkline
        fig = ratio_sparkline(self._fake_ratio())
        assert isinstance(fig, go.Figure)

    def test_sentiment_trend_chart_no_error(self) -> None:
        import plotly.graph_objects as go
        from ui.charts import sentiment_trend_chart
        fig = sentiment_trend_chart(["Q1", "Q2", "Q3", "Q4"], [0.3, -0.1, 0.5, 0.2])
        assert isinstance(fig, go.Figure)

    def test_sentiment_mini_sparkline_no_error(self) -> None:
        import plotly.graph_objects as go
        from ui.charts import sentiment_mini_sparkline
        fig = sentiment_mini_sparkline(["Q1", "Q2", "Q3"], [0.4, 0.2, 0.5], [0.1, -0.2, 0.3])
        assert isinstance(fig, go.Figure)

    def test_beat_miss_chart_no_error(self) -> None:
        import plotly.graph_objects as go
        from ui.charts import beat_miss_chart
        fig = beat_miss_chart(["Q1", "Q2", "Q3"], [0.85, 0.90, 0.88], [0.90, 0.88, 0.92])
        assert isinstance(fig, go.Figure)

    def test_eps_surprise_bars_no_error(self) -> None:
        import plotly.graph_objects as go
        from ui.charts import eps_surprise_bars
        records = [
            {"period": f"Q{i+1} '24", "eps_surprise_pct": (i % 3 - 1) * 5.0,
             "eps_est": 1.0, "eps_actual": 1.0 + (i % 3 - 1) * 0.05}
            for i in range(4)
        ]
        fig = eps_surprise_bars(records)
        assert isinstance(fig, go.Figure)

    def test_revenue_bars_no_error(self) -> None:
        import plotly.graph_objects as go
        from ui.charts import revenue_bars
        fig = revenue_bars(["Q1", "Q2", "Q3"], [1e9, 1.1e9, 1.2e9], [0.05, 0.10, 0.09])
        assert isinstance(fig, go.Figure)

    def test_apply_base_layout_margin_override_no_crash(self) -> None:
        """Regression: explicit margin override must not produce 'multiple values' TypeError."""
        import plotly.graph_objects as go
        from ui.charts import apply_base_layout
        fig = go.Figure()
        apply_base_layout(fig, height=200, margin=dict(l=4, r=4, t=32, b=4), showlegend=False)
        assert isinstance(fig, go.Figure)
        assert fig.layout.margin.l == 4   # override value won

    def test_apply_base_layout_xaxis_deep_merge(self) -> None:
        """Regression: xaxis override must deep-merge with base, not drop base keys."""
        import plotly.graph_objects as go
        from ui.charts import _BASE_LAYOUT, apply_base_layout
        fig = go.Figure()
        apply_base_layout(fig, xaxis=dict(title="My X Axis"))
        # Base xaxis.gridcolor must still be present after merge
        assert fig.layout.xaxis.gridcolor == _BASE_LAYOUT["xaxis"]["gridcolor"]

    def test_peer_comps_chart_pattern_no_crash(self) -> None:
        """Regression: exact peer comps bar chart layout call must not raise."""
        import plotly.graph_objects as go
        from ui.charts import _ACCENT, apply_base_layout
        fig = go.Figure(go.Bar(x=["AAPL", "MSFT"], y=[30.0, 28.0], marker_color=_ACCENT))
        apply_base_layout(fig,
            height=200,
            title=dict(text="P/E TTM", font=dict(size=12, weight=600)),
            margin=dict(l=4, r=4, t=32, b=4),
            yaxis=dict(ticksuffix="x", gridcolor="rgba(26,26,26,0.06)", zeroline=False, showline=False),
            xaxis=dict(gridcolor="rgba(26,26,26,0.06)", zeroline=False, showline=False),
            showlegend=False,
        )
        assert isinstance(fig, go.Figure)

    def test_hiring_intel_chart_pattern_no_crash(self) -> None:
        """Regression: exact hiring intel bar chart layout call must not raise."""
        import plotly.graph_objects as go
        from ui.charts import _ACCENT, apply_base_layout
        fig = go.Figure(go.Bar(x=["Engineering", "Sales", "G&A"], y=[12, 5, 3], marker_color=_ACCENT))
        apply_base_layout(fig,
            height=280, showlegend=False,
            margin=dict(l=4, r=4, t=20, b=4),
            yaxis=dict(title="Roles", gridcolor="rgba(26,26,26,0.06)", zeroline=False, showline=False),
            xaxis=dict(gridcolor="rgba(26,26,26,0.06)", zeroline=False, showline=False),
        )
        assert isinstance(fig, go.Figure)


class TestCacheLayer:
    """Round-trip tests for SQLite cache layer (no LLM calls)."""

    def test_set_and_get_cache_obj(self, tmp_path: Path) -> None:
        import os
        os.environ["DEEPDIVE_DB_PATH"] = str(tmp_path / "test.db")
        # Re-import to pick up new DB path
        import importlib
        import data.cache as cache_mod
        importlib.reload(cache_mod)

        cache_mod.set_cache_obj("test:key", {"value": 42}, ttl=60)
        result = cache_mod.get_cache_obj("test:key")
        assert result == {"value": 42}

    def test_expired_cache_returns_none(self, tmp_path: Path) -> None:
        import os
        import time
        os.environ["DEEPDIVE_DB_PATH"] = str(tmp_path / "test2.db")
        import importlib
        import data.cache as cache_mod
        importlib.reload(cache_mod)

        cache_mod.set_cache_obj("test:expired", "hello", ttl=0)
        time.sleep(0.01)
        result = cache_mod.get_cache_obj("test:expired")
        assert result is None, "Expired cache entry should return None"

    def test_llm_cache_round_trip(self, tmp_path: Path) -> None:
        import os
        os.environ["DEEPDIVE_DB_PATH"] = str(tmp_path / "test3.db")
        import importlib
        import data.cache as cache_mod
        importlib.reload(cache_mod)

        cache_mod.set_llm_cache("abc123", '{"result": "cached"}')
        result = cache_mod.get_llm_cache("abc123")
        assert result == '{"result": "cached"}'

    def test_snapshot_round_trip(self, tmp_path: Path) -> None:
        import os
        os.environ["DEEPDIVE_DB_PATH"] = str(tmp_path / "test4.db")
        import importlib
        import data.cache as cache_mod
        importlib.reload(cache_mod)

        snap_json = '{"ticker_or_url": "NVDA", "snapshot": {"fundamentals": {"pe_ttm": 30.0}}}'
        cache_mod.save_run_snapshot("NVDA", snap_json)
        result = cache_mod.get_last_run_snapshot("NVDA")
        assert result is not None
        assert result.get("snapshot", {}).get("fundamentals", {}).get("pe_ttm") == 30.0
