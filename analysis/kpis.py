"""Phase 5: Non-GAAP KPI extraction from transcripts + MD&A. Haiku over cached context."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import llm
from analysis.schemas import KpiSeries, KpiValue
from config import HAIKU, PROMPT_VERSIONS
from data.cache import get_cache_obj, set_cache_obj
from config import TTL_LLM

logger = logging.getLogger(__name__)

_SYSTEM = llm.cached_system("""
You are a financial analyst specialising in non-GAAP metric extraction from earnings calls.
Your job is to identify the 3-5 KPIs that THIS company's management tracks most carefully —
not generic KPIs, but the ones that appear repeatedly across calls and drive their narrative.
Examples: ARR, NRR, DAU/MAU, Same-Store Sales, ADR, RevPAR, Backlog, Win Rate, etc.

For each KPI:
- Extract its exact definition as the company uses it (not a textbook definition)
- Pull per-quarter values (quarter label + value as a string with units)
- Cite the source transcript using its tag
- Mark disappeared=true if a KPI that appeared in earlier calls vanishes in the most recent one
""")


def extract_kpis(
    ticker: str,
    call_data: Dict,  # output of calls.analyse_all_calls()
    md_and_a_text: str = "",
) -> List[KpiSeries]:
    """Extract company-specific KPIs from transcripts + MD&A."""
    cache_key = f"kpis:{ticker.upper()}:{PROMPT_VERSIONS['kpi_extraction']}"
    cached = get_cache_obj(cache_key)
    if cached:
        return [KpiSeries.model_validate(r) for r in cached]

    transcripts = call_data.get("transcripts", [])
    chunk_tags = call_data.get("chunk_tags", {})
    if not transcripts:
        return []

    # Build multi-quarter context with citation tags
    context_parts = []
    for i, t in enumerate(reversed(transcripts)):  # oldest first
        tag = f"[T-Q{i+1}]"
        content = (t.get("content") or "")[:8000]
        context_parts.append(f'<transcript id="{tag}" quarter="Q{t.get("quarter")} {t.get("year")}">\n{content}\n</transcript>')

    if md_and_a_text:
        context_parts.append(f'<mda id="[MD&A]">\n{md_and_a_text[:6000]}\n</mda>')

    context = "\n\n".join(context_parts)

    messages = [
        {
            "role": "user",
            "content": [
                *llm.cached_content(context),
                llm.text_block(
                    f"Extract the 3-5 most important company-specific KPIs for {ticker}. "
                    "For each KPI, provide per-quarter values across all transcripts provided. "
                    "Return a JSON array of KpiSeries objects with fields: "
                    "kpi_name, definition_as_company_uses_it, values (array of {quarter, value, source}), "
                    "trend_note, disappeared (bool). "
                    "Use ONLY data from the transcripts above."
                ),
            ],
        }
    ]

    raw = llm.call(
        HAIKU, messages,
        system=_SYSTEM,
        prompt_version=PROMPT_VERSIONS["kpi_extraction"],
        max_tokens=3000,
    )

    # Parse JSON array from response
    import json
    results: List[KpiSeries] = []
    try:
        text = raw if isinstance(raw, str) else str(raw)
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            raw_list = json.loads(text[start:end])
            for item in raw_list:
                # Normalise values
                values = []
                for v in item.get("values", []):
                    values.append(KpiValue(
                        quarter=str(v.get("quarter", "")),
                        value=str(v.get("value", "")) if v.get("value") is not None else None,
                        source=str(v.get("source", "[T-Q1]")),
                    ))
                results.append(KpiSeries(
                    kpi_name=item.get("kpi_name", "Unknown KPI"),
                    definition_as_company_uses_it=item.get("definition_as_company_uses_it", ""),
                    values=values,
                    trend_note=item.get("trend_note", ""),
                    disappeared=bool(item.get("disappeared", False)),
                ))
    except Exception as exc:
        logger.warning("KPI JSON parse failed for %s: %s", ticker, exc)

    set_cache_obj(cache_key, [r.model_dump(mode="json") for r in results], TTL_LLM)
    return results
