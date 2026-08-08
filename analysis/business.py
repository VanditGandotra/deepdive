"""Phase 3: 10-K business explainer. Sonnet, streamed, citation-tagged."""
from __future__ import annotations

import logging
from typing import Dict, Iterator, Optional, Tuple

import llm
from analysis.schemas import BusinessProfile
from config import PROMPT_VERSIONS, SONNET
from data.edgar import get_10k_sections

logger = logging.getLogger(__name__)

# Citation IDs for each 10-K section
CHUNK_LABELS = {
    "Item 1":  "[10K-1]",
    "Item 1A": "[10K-1A]",
    "Item 7":  "[10K-7]",
}

_SYSTEM = llm.cached_system("""
You are a senior equity analyst reading SEC 10-K filings for institutional investors.

CITATION RULES — MANDATORY, NO EXCEPTIONS:
- Every factual claim MUST carry the source tag immediately after: [10K-1], [10K-1A], or [10K-7]
- Every number MUST carry its source tag: "Revenue was $X [10K-7]"
- If data is absent from the provided sections, write "data unavailable" — never estimate or hallucinate
- Tags go inline, right after the claim, not in footnotes

STYLE:
- Write as if briefing a PM who has 60 seconds to understand the business
- Avoid jargon; explain how each segment actually earns money
- Be specific: name products, geographies, customers where disclosed
- Top risks: pick the 3 that could most materially impair the thesis
""")


def _build_context(sections: Dict[str, str]) -> Tuple[str, Dict[str, str]]:
    """Assemble annotated context block and return (context_text, chunks_map)."""
    chunks: Dict[str, str] = {}
    parts = []
    for section_name, label in CHUNK_LABELS.items():
        text = sections.get(section_name, "")
        if text:
            chunks[label] = text[:3000]  # Store first 3000 chars for citation tooltip
            parts.append(f'<source id="{label}">\n{text[:15000]}\n</source>')
    return "\n\n".join(parts), chunks


def get_business_profile(ticker: str) -> Optional[BusinessProfile]:
    """Structured extraction (no streaming). Returns BusinessProfile or None."""
    sections = get_10k_sections(ticker)
    if not sections:
        return None

    context, chunks = _build_context(sections)
    messages = [
        {
            "role": "user",
            "content": [
                *llm.cached_content(context),
                llm.text_block(
                    "Using ONLY the source documents above (each tagged with its ID), "
                    "extract a BusinessProfile. Tag every claim with its source ID. "
                    f"Ticker: {ticker}"
                ),
            ],
        }
    ]
    try:
        profile: BusinessProfile = llm.call(
            SONNET, messages,
            system=_SYSTEM,
            schema=BusinessProfile,
            prompt_version=PROMPT_VERSIONS["business_explainer"],
            max_tokens=4096,
        )
        profile.citations = chunks
        return profile
    except Exception as exc:
        logger.warning("Business explainer failed for %s: %s", ticker, exc)
        return None


def stream_business_explainer(ticker: str) -> Tuple[Iterator[str], Dict[str, str]]:
    """
    Stream the business explainer narrative to the UI.
    Returns (token_iterator, chunks_map).
    The caller should render tokens then optionally do structured extraction separately.
    """
    sections = get_10k_sections(ticker)
    if not sections:
        return iter(["No 10-K sections available for this ticker."]), {}

    context, chunks = _build_context(sections)
    messages = [
        {
            "role": "user",
            "content": [
                *llm.cached_content(context),
                llm.text_block(
                    f"Write a comprehensive business overview for {ticker} using ONLY the source documents above. "
                    "Structure your response with these sections:\n"
                    "## What They Do\n## Revenue Segments\n## Competitive Moat\n## Key Customers & Geography\n"
                    "## Top 3 Risks\n## Recent Strategic Shifts\n\n"
                    "Tag every factual claim with its source ID ([10K-1], [10K-1A], [10K-7])."
                ),
            ],
        }
    ]
    token_iter = llm.call(SONNET, messages, system=_SYSTEM, mode="stream", max_tokens=4096)
    return token_iter, chunks
