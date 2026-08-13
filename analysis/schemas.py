"""All Pydantic v2 data models for DeepDive."""
from __future__ import annotations

from collections import Counter
from datetime import date, datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


# ═══════════════════════════════════════════════════════════════════════════════
# Market Data
# ═══════════════════════════════════════════════════════════════════════════════

class PriceBar(BaseModel):
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int

class PriceData(BaseModel):
    ticker: str
    currency: str = "USD"
    bars: List[PriceBar] = []
    fetched_at: datetime = Field(default_factory=datetime.utcnow)

class Estimate(BaseModel):
    period: str
    revenue_estimate: Optional[float] = None
    eps_estimate: Optional[float] = None
    eps_actual: Optional[float] = None
    revenue_actual: Optional[float] = None
    surprise_pct: Optional[float] = None

class AnalystTarget(BaseModel):
    analyst: Optional[str] = None
    firm: Optional[str] = None
    action: Optional[str] = None
    rating: Optional[str] = None
    price_target: Optional[float] = None
    date: Optional[date] = None

class InsiderTransaction(BaseModel):
    name: str
    role: Optional[str] = None
    transaction_type: str
    shares: int
    price: Optional[float] = None
    value: Optional[float] = None
    date: Optional[date] = None

class InstitutionalHolder(BaseModel):
    name: str
    shares: int
    pct_held: Optional[float] = None
    date_reported: Optional[date] = None
    change: Optional[int] = None

class ShortInterest(BaseModel):
    date: Optional[date] = None
    short_interest: Optional[int] = None
    pct_float: Optional[float] = None
    days_to_cover: Optional[float] = None

class Fundamentals(BaseModel):
    ticker: str
    name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[float] = None
    enterprise_value: Optional[float] = None
    # Valuation
    pe_ttm: Optional[float] = None
    pe_forward: Optional[float] = None
    peg: Optional[float] = None
    ev_ebitda: Optional[float] = None
    price_to_sales: Optional[float] = None
    price_to_fcf: Optional[float] = None
    # Profitability
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None
    roe: Optional[float] = None
    roic: Optional[float] = None
    # Health
    current_ratio: Optional[float] = None
    debt_to_equity: Optional[float] = None
    net_debt_ebitda: Optional[float] = None
    interest_coverage: Optional[float] = None
    # Growth
    revenue_growth_yoy: Optional[float] = None
    eps_growth_yoy: Optional[float] = None
    # Raw financials
    revenue_ttm: Optional[float] = None
    ebitda_ttm: Optional[float] = None
    net_income_ttm: Optional[float] = None
    fcf_ttm: Optional[float] = None
    total_debt: Optional[float] = None
    cash: Optional[float] = None
    shares_outstanding: Optional[float] = None
    current_price: Optional[float] = None
    beta: Optional[float] = None
    fetched_at: Optional[datetime] = None
    # Price context (for Overview header strip)
    previous_close: Optional[float] = None
    day_change_pct: Optional[float] = None
    week_52_high: Optional[float] = None
    week_52_low: Optional[float] = None
    next_earnings_date: Optional[date] = None
    # Analyst consensus
    analyst_buy_count: Optional[int] = None
    analyst_hold_count: Optional[int] = None
    analyst_sell_count: Optional[int] = None
    analyst_target_mean: Optional[float] = None
    analyst_target_low: Optional[float] = None
    analyst_target_high: Optional[float] = None


# ═══════════════════════════════════════════════════════════════════════════════
# Reconciliation
# ═══════════════════════════════════════════════════════════════════════════════

class MetricReconciliation(BaseModel):
    metric: str
    yfinance_value: Optional[float] = None
    edgar_value: Optional[float] = None
    diff_pct: Optional[float] = None
    note: str = ""
    canonical: Literal["edgar", "yfinance", "na"] = "edgar"
    # Composite breakdown: tag → value (for auditable UI expansion)
    components: Optional[Dict[str, float]] = None
    # "composition incomplete: missing ShortTermInvestments" etc.
    composite_note: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# Business Explainer
# ═══════════════════════════════════════════════════════════════════════════════

class RevenueSegment(BaseModel):
    name: str
    description: str
    pct_of_revenue: Optional[float] = None
    how_it_makes_money: str

class BusinessProfile(BaseModel):
    what_they_do: str
    revenue_segments: List[RevenueSegment] = []
    moat: str
    key_customers_geography: str
    top_risks: List[str] = []
    recent_strategic_shifts: str
    citations: Optional[Dict[str, str]] = None


# ═══════════════════════════════════════════════════════════════════════════════
# Earnings Calls
# ═══════════════════════════════════════════════════════════════════════════════

class EarningsSignal(BaseModel):
    topic: str                                   # e.g. "Revenue guidance", "Gross margin"
    signal: Literal["positive", "neutral", "negative"]
    confidence: Literal["high", "medium", "low"] = "medium"
    evidence: str = ""                           # brief quote or paraphrase (≤ 30 words)
    rationale: str = ""                          # why positive/neutral/negative vs prior expectations


class GuidanceItem(BaseModel):
    metric: str
    value: Optional[str] = None
    prior_value: Optional[str] = None
    direction: Literal["raised", "lowered", "maintained", "initiated", "withdrawn", "n/a"] = "n/a"

class CompetitiveMention(BaseModel):
    competitor: str
    context: str

class CallSummary(BaseModel):
    quarter: str
    guidance_items: List[GuidanceItem] = []
    key_themes: List[str] = []
    top_analyst_concerns_from_qa: List[str] = []
    notable_quotes_paraphrased: List[str] = []
    one_time_items_mentioned: Optional[str] = None
    competitive_mentions: List[CompetitiveMention] = []
    signals: List[EarningsSignal] = []
    signal_overall: Literal["positive", "neutral", "negative"] = "neutral"

    @model_validator(mode="after")
    def _derive_signal_overall(self) -> "CallSummary":
        if not self.signals:
            return self
        counts: Counter = Counter(s.signal for s in self.signals)
        top_signal, top_count = counts.most_common(1)[0]
        # Require strict majority; ties → neutral
        if top_count > len(self.signals) / 2:
            self.signal_overall = top_signal
        else:
            self.signal_overall = "neutral"
        return self

class EvasivenessFlag(BaseModel):
    analyst_question_topic: str
    why_answer_seemed_indirect: str

class HedgingIndex(BaseModel):
    level: Literal["low", "medium", "high"]
    example_phrases: List[str] = []

class SpeakerSentiment(BaseModel):
    name: str
    role: str
    score: float = Field(ge=-1, le=1)
    confidence_language_examples: List[str] = []

class CallSentiment(BaseModel):
    overall_score: float = Field(ge=-1, le=1)
    prepared_remarks_score: float = Field(ge=-1, le=1)
    qa_score: float = Field(ge=-1, le=1)
    per_speaker: List[SpeakerSentiment] = []
    hedging_index: HedgingIndex
    evasiveness_flags: List[EvasivenessFlag] = []
    superlative_density_note: Optional[str] = None

class GuidanceChange(BaseModel):
    metric: str
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    change_description: str

class CallDelta(BaseModel):
    guidance_changes: List[GuidanceChange] = []
    new_topics: List[str] = []
    dropped_topics: List[str] = []
    tone_trajectory_across_4q: str
    sentiment_trend: List[float] = []
    qa_vs_prepared_gap_trend: str
    recurring_analyst_pressure_points: List[str] = []
    what_changed_narrative: str


# ═══════════════════════════════════════════════════════════════════════════════
# News / Headlines
# ═══════════════════════════════════════════════════════════════════════════════

class NewsItem(BaseModel):
    title: str
    source: Optional[str] = None
    published_at: Optional[datetime] = None
    url: Optional[str] = None
    snippet: Optional[str] = None

class HeadlineImpact(BaseModel):
    title: str
    source: Optional[str] = None
    published_at: Optional[datetime] = None
    url: Optional[str] = None
    category: Literal[
        "earnings", "product", "regulatory", "legal", "macro",
        "analyst_action", "M&A", "management", "other"
    ]
    direction: Literal["positive", "negative", "neutral", "mixed"]
    materiality: Literal["high", "medium", "low"]
    one_line_why: str


# ═══════════════════════════════════════════════════════════════════════════════
# Ratios
# ═══════════════════════════════════════════════════════════════════════════════

class RatioHistory(BaseModel):
    name: str
    current: Optional[float] = None
    min_5y: Optional[float] = None
    median_5y: Optional[float] = None
    max_5y: Optional[float] = None
    history: List[Optional[float]] = []
    description: str = ""
    not_meaningful: bool = False
    not_meaningful_reason: Optional[str] = None

class RatioGroup(BaseModel):
    group: str
    ratios: List[RatioHistory] = []


# ═══════════════════════════════════════════════════════════════════════════════
# Analyst Mode — Reverse DCF
# ═══════════════════════════════════════════════════════════════════════════════

class ReverseDCFPoint(BaseModel):
    revenue_cagr: float
    fcf_margin: float
    implied_price: float
    matches_current: bool = False

class ReverseDCFResult(BaseModel):
    current_price: float
    shares_outstanding: float
    net_debt: float
    discount_rate: float
    terminal_growth: float
    horizon_years: int
    headline: str
    isocurve_points: List[ReverseDCFPoint] = []
    sensitivity_table: List[Dict[str, float]] = []


# ═══════════════════════════════════════════════════════════════════════════════
# KPIs
# ═══════════════════════════════════════════════════════════════════════════════

class KpiValue(BaseModel):
    quarter: str
    value: Optional[str] = None
    source: str

class KpiSeries(BaseModel):
    kpi_name: str
    definition_as_company_uses_it: str
    values: List[KpiValue] = []
    trend_note: str = ""
    disappeared: bool = False


# ═══════════════════════════════════════════════════════════════════════════════
# Quality of Earnings
# ═══════════════════════════════════════════════════════════════════════════════

class QualityFlag(BaseModel):
    name: str
    status: Literal["green", "yellow", "red"]
    trigger_condition: str
    observed_value: Optional[str] = None
    threshold: str
    explanation: str

class QualityPanel(BaseModel):
    flags: List[QualityFlag] = []
    overall: Literal["clean", "mixed", "concerning"] = "clean"
    summary: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# Positioning
# ═══════════════════════════════════════════════════════════════════════════════

class PositioningSummary(BaseModel):
    short_interest_pct_float: Optional[float] = None
    days_to_cover: Optional[float] = None
    insider_net_sentiment: Literal["net_buying", "net_selling", "mixed", "minimal"] = "minimal"
    insider_transactions: List[InsiderTransaction] = []
    top_holder_changes: List[InstitutionalHolder] = []
    synthesis: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# Thesis & Memo
# ═══════════════════════════════════════════════════════════════════════════════

class Thesis(BaseModel):
    scenario: Literal["bull", "base", "bear"]
    narrative: str
    key_drivers: List[str] = []
    implied_price_12mo: Optional[float] = None
    probability_weight: float = Field(ge=0, le=1)
    confirm_signals: List[str] = []
    kill_signals: List[str] = []

class RedTeam(BaseModel):
    strongest_counterargument: str
    most_fragile_assumption: str
    what_bulls_are_ignoring: str
    what_bears_are_ignoring: str
    fastest_falsifier: str

class Memo(BaseModel):
    ticker: str
    company_name: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    setup: str
    thesis_summary: str
    variant_vs_consensus: str
    key_drivers: List[str] = []
    kpis_to_watch: List[str] = []
    risks_and_falsifiers: List[str] = []
    positioning_note: str = ""
    word_count: int = 0
    footnotes: Dict[str, str] = {}


# ═══════════════════════════════════════════════════════════════════════════════
# Delta Engine
# ═══════════════════════════════════════════════════════════════════════════════

class DeltaItem(BaseModel):
    field: str
    old_value: Any = None
    new_value: Any = None
    change_type: Literal["added", "removed", "changed", "new_flag", "cleared_flag"]
    description: str

class DeltaNarrative(BaseModel):
    prior_run_at: datetime
    current_run_at: datetime
    items: List[DeltaItem] = []
    narrative_bullets: List[str] = []

class RunSnapshotData(BaseModel):
    fundamentals: Optional[Dict[str, Any]] = None
    estimates: Optional[Dict[str, Any]] = None
    sentiment_scores: Optional[List[float]] = None
    quality_flags: Optional[List[str]] = None
    short_interest_pct: Optional[float] = None
    kpi_values: Optional[Dict[str, Any]] = None
    # URL mode
    page_hashes: Optional[Dict[str, str]] = None
    customer_list: Optional[List[str]] = None
    feature_claims: Optional[List[str]] = None
    last_30_days_count: Optional[int] = None

class RunSnapshot(BaseModel):
    id: Optional[int] = None
    ticker_or_url: str
    run_at: datetime = Field(default_factory=datetime.utcnow)
    snapshot: RunSnapshotData = Field(default_factory=RunSnapshotData)


# ═══════════════════════════════════════════════════════════════════════════════
# URL Mode — Company Intel
# ═══════════════════════════════════════════════════════════════════════════════

class DatedAnnouncement(BaseModel):
    date: Optional[date] = None
    headline: str
    summary: str

class PageIntel(BaseModel):
    url: str
    page_type: Literal[
        "blog", "news", "customer", "case_study", "changelog",
        "pricing", "careers", "about", "press", "docs", "other"
    ] = "other"
    publish_date: Optional[date] = None
    dated_announcements: List[DatedAnnouncement] = []
    named_customers: List[str] = []
    feature_claims: List[str] = []
    tech_details: List[str] = []
    hiring_signals: List[str] = []

class NamedCustomer(BaseModel):
    name: str
    source_url: str

class LastThirtyDaysItem(BaseModel):
    date: Optional[date] = None
    event_type: str
    headline: str
    summary: str
    source_url: str

class CompanyIntel(BaseModel):
    what_they_sell: str
    target_customer_icp: str
    named_customers: List[NamedCustomer] = []
    feature_inventory: List[str] = []
    pricing_model: Optional[str] = None
    positioning_summary: str
    hiring_roadmap_signals: List[str] = []
    last_30_days: List[LastThirtyDaysItem] = []


# ═══════════════════════════════════════════════════════════════════════════════
# Product Deep Dive
# ═══════════════════════════════════════════════════════════════════════════════

class CoreConcept(BaseModel):
    name: str
    explanation: str
    why_it_matters: str

class FeatureEntry(BaseModel):
    feature: str
    what_it_does: str
    docs_url: Optional[str] = None

class ProductExplainer(BaseModel):
    what_it_is_one_paragraph: str
    core_concepts: List[CoreConcept] = []
    mental_model_summary: str
    zero_to_value_workflow: str
    feature_inventory: List[FeatureEntry] = []
    integrations: List[str] = []
    api_surface_summary: Optional[str] = None
    plans_and_limits_if_documented: Optional[str] = None
    changelog_velocity_note: Optional[str] = None

class ScreenExplanation(BaseModel):
    screen_name_guess: str
    what_it_shows: str
    ui_elements_of_note: List[str] = []
    what_this_reveals_about_the_product: str


# ═══════════════════════════════════════════════════════════════════════════════
# Peer Comps
# ═══════════════════════════════════════════════════════════════════════════════

class PeerRow(BaseModel):
    ticker: str
    name: Optional[str] = None
    market_cap: Optional[float] = None
    pe_ttm: Optional[float] = None
    pe_forward: Optional[float] = None
    ev_ebitda: Optional[float] = None
    price_to_sales: Optional[float] = None
    gross_margin: Optional[float] = None
    net_margin: Optional[float] = None
    revenue_growth_yoy: Optional[float] = None
    fcf_yield: Optional[float] = None
    is_target: bool = False  # True for the stock being analyzed

class PeerComps(BaseModel):
    target_ticker: str
    peers: List[PeerRow] = []
    synthesis: str = ""  # Sonnet one-paragraph narrative


# ═══════════════════════════════════════════════════════════════════════════════
# LLM Observability
# ═══════════════════════════════════════════════════════════════════════════════

class LLMCallRecord(BaseModel):
    id: Optional[int] = None
    model: str
    prompt_version: str
    input_hash: str
    tokens_input: int
    tokens_output: int
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    cost_est_usd: float
    was_cached: bool
    session_id: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)

class SessionStats(BaseModel):
    total_calls: int = 0
    cached_calls: int = 0
    tokens_input: int = 0
    tokens_output: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_est_usd: float = 0.0
    llm_cache_hits: int = 0
    llm_cache_misses: int = 0
    db_cache_hits: int = 0
    db_cache_misses: int = 0


# ═══════════════════════════════════════════════════════════════════════════════
# Freshness
# ═══════════════════════════════════════════════════════════════════════════════

class FreshnessRecord(BaseModel):
    source: str
    key: str
    last_fetched_at: datetime
    ttl_seconds: int

    @property
    def is_stale(self) -> bool:
        age = (datetime.utcnow() - self.last_fetched_at).total_seconds()
        return age > self.ttl_seconds

    @property
    def age_description(self) -> str:
        age_secs = (datetime.utcnow() - self.last_fetched_at).total_seconds()
        if age_secs < 3600:
            return f"{int(age_secs / 60)}m ago"
        if age_secs < 86400:
            return f"{int(age_secs / 3600)}h ago"
        return f"{int(age_secs / 86400)}d ago"
