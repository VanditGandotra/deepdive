"""Phase 3: Sentiment helpers — re-exports and per-call sentiment aggregation."""
from __future__ import annotations

from typing import List, Optional

from analysis.schemas import CallSentiment
# Pass B is implemented in calls.py; this module provides aggregation utilities.
from analysis.calls import extract_call_sentiment  # noqa: F401 (re-export)


def aggregate_sentiment_scores(sentiments: List[Optional[CallSentiment]]) -> List[Optional[float]]:
    """Return list of overall sentiment scores oldest→newest (None for missing quarters)."""
    return [s.overall_score if s else None for s in sentiments]


def sentiment_label(score: Optional[float]) -> str:
    if score is None:
        return "—"
    if score >= 0.4:
        return "Positive"
    if score >= 0.1:
        return "Slightly positive"
    if score >= -0.1:
        return "Neutral"
    if score >= -0.4:
        return "Slightly negative"
    return "Negative"


def prepared_qa_gap(sentiment: CallSentiment) -> float:
    """Positive = management more positive in prepared remarks than Q&A (a yellow flag)."""
    return sentiment.prepared_remarks_score - sentiment.qa_score
