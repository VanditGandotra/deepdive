"""Phase 4: Headline impact classification — one batched Haiku call for the full list."""
from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional

import llm
from analysis.schemas import HeadlineImpact, NewsItem
from config import HAIKU, PROMPT_VERSIONS
from data.news import get_news

logger = logging.getLogger(__name__)

_SYSTEM = llm.cached_system("""
You are a financial news analyst classifying headlines for a professional investor.
For each headline, classify based ONLY on the title and snippet provided — never invent article content.

Categories:
  earnings        = EPS beats/misses, guidance, revenue surprises
  product         = new products, launches, recalls, bugs, partnerships
  regulatory      = SEC, FDA, FTC, DOJ, compliance, fines, antitrust
  legal           = lawsuits, settlements, patents
  macro           = interest rates, inflation, tariffs, FX, macro data
  analyst_action  = upgrades, downgrades, price target changes
  M&A             = mergers, acquisitions, divestitures, spin-offs
  management      = CEO/CFO changes, board, activism
  other           = anything else

Direction: positive / negative / neutral / mixed

Materiality calibration (be generous — news that professionals would act on):
  high   = Could meaningfully move the stock. Examples:
           - Earnings beat/miss + guidance raise/cut
           - Major regulatory action, antitrust charge, or settlement >$1B
           - M&A announcement, acquisition, or strategic divestiture
           - New CEO/CFO or activist investor taking a stake
           - Major product launch or platform-changing partnership
           - Analyst consensus shift (multiple upgrades or target raises in one day)
           - Macro event directly affecting the company's core business
  medium = Noteworthy but not decisive: single analyst action, minor product news,
           routine legal filing, macro commentary
  low    = Background noise: general sector commentary, minor personnel changes,
           conference presentations without new disclosures

When in doubt, lean toward 'medium' rather than 'low'. Reserve 'low' for items
that would not appear on a professional investor's morning briefing.
""")


def classify_headlines(
    ticker: str,
    company_name: str = "",
    days: int = 30,
) -> List[HeadlineImpact]:
    """
    Classify all headlines in a single batched Haiku call.
    Returns list sorted by materiality (high→low) then recency.
    """
    cache_key = f"news_impact:{ticker.upper()}:{days}"
    from data.cache import get_cache_obj, set_cache_obj
    from config import TTL_NEWS

    cached = get_cache_obj(cache_key)
    if cached:
        return [HeadlineImpact.model_validate(r) for r in cached]

    news_items = get_news(ticker, company_name or ticker, days=days)
    if not news_items:
        return []

    # Build numbered article list for the prompt
    articles_text = "\n".join(
        f"[{i}] {item.title} | {item.source or 'unknown'} | "
        f"{item.published_at.strftime('%Y-%m-%d') if item.published_at else 'no date'} | "
        f"{(item.snippet or '')[:120]}"
        for i, item in enumerate(news_items)
    )

    messages = [
        {
            "role": "user",
            "content": [
                llm.text_block(
                    f"Classify these {len(news_items)} headlines for {ticker} ({company_name}).\n\n"
                    f"{articles_text}\n\n"
                    "Return a JSON array named 'results' where each element has: "
                    "index (int), category, direction, materiality, one_line_why. "
                    "Use ONLY the title and snippet for classification."
                ),
            ],
        }
    ]

    # Use JSON mode via schema-less call with explicit instruction
    raw_json = llm.call(
        HAIKU, messages,
        system=_SYSTEM,
        prompt_version=PROMPT_VERSIONS["headline_classification"],
        max_tokens=3000,
        skip_llm_cache=False,
    )

    # Parse JSON output and match back to original items
    results: List[HeadlineImpact] = []
    try:
        text = raw_json if isinstance(raw_json, str) else str(raw_json)
        # Strip markdown code fences if present
        if "```" in text:
            import re as _re
            text = _re.sub(r"```(?:json)?\s*", "", text).strip()

        classifications: list = []
        # Strategy 1: bare array
        a_start = text.find("[")
        a_end = text.rfind("]") + 1
        if a_start >= 0 and a_end > a_start:
            try:
                classifications = json.loads(text[a_start:a_end])
            except json.JSONDecodeError:
                pass
        # Strategy 2: object with "results" key
        if not classifications:
            o_start = text.find("{")
            o_end = text.rfind("}") + 1
            if o_start >= 0 and o_end > o_start:
                obj = json.loads(text[o_start:o_end])
                classifications = obj.get("results", obj.get("classifications", []))

        for cls in classifications:
            idx = cls.get("index", -1)
            if 0 <= idx < len(news_items):
                item = news_items[idx]
                results.append(HeadlineImpact(
                    title=item.title,
                    source=item.source,
                    published_at=item.published_at,
                    url=item.url,
                    category=cls.get("category", "other"),
                    direction=cls.get("direction", "neutral"),
                    materiality=cls.get("materiality", "low"),
                    one_line_why=cls.get("one_line_why", ""),
                ))
    except Exception as exc:
        logger.warning("Headline JSON parse failed: %s — falling back to unclassified list", exc)
        # Return items with default classification rather than empty
        for item in news_items:
            results.append(HeadlineImpact(
                title=item.title, source=item.source, published_at=item.published_at,
                url=item.url, category="other", direction="neutral", materiality="low",
                one_line_why="Classification unavailable",
            ))

    # Sort: high→medium→low, then newest first
    priority = {"high": 0, "medium": 1, "low": 2}
    results.sort(key=lambda r: (
        priority.get(r.materiality, 2),
        -(r.published_at.timestamp() if r.published_at else 0),
    ))

    set_cache_obj(cache_key, [r.model_dump(mode="json") for r in results], TTL_NEWS)
    return results


def high_materiality(impacts: List[HeadlineImpact]) -> List[HeadlineImpact]:
    return [h for h in impacts if h.materiality == "high"]
