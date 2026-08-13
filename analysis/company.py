"""Phase 9: Company intelligence — per-page extraction (Haiku) + synthesis (Sonnet)."""
from __future__ import annotations

import logging
from datetime import date
from typing import Dict, Iterator, List, Optional, Tuple

import llm
from analysis.schemas import (
    CompanyIntel, LastThirtyDaysItem, NamedCustomer,
    PageIntel, RunSnapshotData,
)
from config import HAIKU, PROMPT_VERSIONS, SONNET

logger = logging.getLogger(__name__)

_SYSTEM_EXTRACT = llm.cached_system("""
You extract structured company intelligence from a single web page.
Rules:
- named_customers: only real company/org names explicitly mentioned as customers or users.
- feature_claims: product capability statements ("supports X", "enables Y", "built-in Z").
- tech_details: specific technologies, APIs, integrations, infrastructure details.
- hiring_signals: job roles or team expansions mentioned.
- dated_announcements: concrete events with a discernible date (blog post date, press release date).
- page_type: classify as one of: blog, news, customer, case_study, changelog, pricing, careers, about, press, docs, other.
Be conservative — only include facts explicitly stated in the text.
""")

_SYSTEM_SYNTH = llm.cached_system("""
You are a competitive intelligence analyst synthesising web crawl data about a company.
Write for a B2B buyer or investor who needs to quickly understand what the company does,
who they sell to, and what momentum signals exist.
Citation tags MANDATORY on every claim: [PAGE-n] where n matches the source page index.
Structure your output with these sections:
## What They Sell
## Target Customer (ICP)
## Named Customers
## Feature Inventory
## Pricing Model
## Positioning vs Alternatives
## Hiring & Roadmap Signals
## Last 30 Days
""")


def extract_page_intel(page: Dict) -> Optional[PageIntel]:
    """Run Haiku structured extraction on a single fetched page dict."""
    text = (page.get("text") or "").strip()
    if len(text) < 100:
        return None

    url = page.get("url", "")
    messages = [
        {
            "role": "user",
            "content": [
                llm.text_block(
                    f"URL: {url}\n\n"
                    f"Page text (truncated to 6000 chars):\n{text[:6000]}\n\n"
                    "Extract all company intelligence fields."
                ),
            ],
        }
    ]
    try:
        return llm.call(
            HAIKU, messages,
            system=_SYSTEM_EXTRACT,
            schema=PageIntel,
            prompt_version=PROMPT_VERSIONS["page_extraction"],
            max_tokens=1500,
        )
    except Exception as exc:
        logger.warning("extract_page_intel %s: %s", url, exc)
        return PageIntel(url=url)


def stream_company_intel(
    domain: str,
    page_intels: List[PageIntel],
) -> Tuple[Iterator[str], RunSnapshotData]:
    """
    Stream Sonnet company synthesis. Also builds a RunSnapshotData for the delta engine.
    Returns (token_iterator, snapshot_data).
    """
    # Aggregate customer list and feature claims across all pages
    all_customers: List[str] = []
    all_features: List[str] = []
    last_30_items: List[str] = []
    today = date.today()

    context_parts: List[str] = []
    for i, pi in enumerate(page_intels):
        tag = f"[PAGE-{i+1}]"
        parts = [f"URL: {pi.url}", f"Type: {pi.page_type}"]
        if pi.named_customers:
            parts.append(f"Named customers: {', '.join(pi.named_customers)}")
            all_customers.extend(pi.named_customers)
        if pi.feature_claims:
            parts.append(f"Features: {'; '.join(pi.feature_claims[:5])}")
            all_features.extend(pi.feature_claims[:5])
        if pi.tech_details:
            parts.append(f"Tech: {'; '.join(pi.tech_details[:3])}")
        if pi.hiring_signals:
            parts.append(f"Hiring: {'; '.join(pi.hiring_signals[:3])}")
        if pi.dated_announcements:
            for ann in pi.dated_announcements[:3]:
                ann_text = f"{ann.date or ''}: {ann.headline} — {ann.summary[:100]}"
                parts.append(f"Event: {ann_text}")
                # Count last-30-days events
                if ann.date and (today - ann.date).days <= 30:
                    last_30_items.append(ann.headline)
        context_parts.append(f'<source id="{tag}">\n' + "\n".join(parts) + "\n</source>")

    context = "\n\n".join(context_parts)
    messages = [
        {
            "role": "user",
            "content": [
                *llm.cached_content(context),
                llm.text_block(
                    f"Synthesise company intelligence for {domain}. "
                    "Use the section structure from your instructions. "
                    "Cite every fact with its [PAGE-n] tag. "
                    "Be specific — actual customer names, actual feature names, actual pricing tiers."
                ),
            ],
        }
    ]

    # Snapshot data for delta engine
    snap = RunSnapshotData(
        customer_list=list(dict.fromkeys(all_customers))[:50],
        feature_claims=list(dict.fromkeys(all_features))[:30],
        last_30_days_count=len(last_30_items),
        page_hashes={pi.url: pi.url[-16:] for pi in page_intels},
    )

    token_iter = llm.call(
        SONNET, messages,
        system=_SYSTEM_SYNTH,
        mode="stream",
        max_tokens=6000,
    )
    return token_iter, snap


def get_company_intel(
    domain: str,
    page_intels: List[PageIntel],
) -> CompanyIntel:
    """Non-streaming structured company intel (for programmatic use / evals)."""
    all_customers: List[str] = []
    all_features: List[str] = []
    context_parts: List[str] = []

    for i, pi in enumerate(page_intels):
        tag = f"[PAGE-{i+1}]"
        parts = [f"URL: {pi.url}", f"Type: {pi.page_type}"]
        if pi.named_customers:
            parts.append(f"Customers: {', '.join(pi.named_customers)}")
            all_customers.extend(pi.named_customers)
        if pi.feature_claims:
            parts.append(f"Features: {'; '.join(pi.feature_claims[:5])}")
            all_features.extend(pi.feature_claims[:5])
        if pi.hiring_signals:
            parts.append(f"Hiring: {'; '.join(pi.hiring_signals[:3])}")
        if pi.dated_announcements:
            for ann in pi.dated_announcements[:2]:
                parts.append(f"Event: {ann.date or ''}: {ann.headline}")
        context_parts.append(f'<source id="{tag}">\n' + "\n".join(parts) + "\n</source>")

    context = "\n\n".join(context_parts)
    messages = [
        {
            "role": "user",
            "content": [
                *llm.cached_content(context),
                llm.text_block(
                    f"Return structured company intelligence for {domain}."
                ),
            ],
        }
    ]

    _SYSTEM_STRUCT = llm.cached_system("""
You extract structured company intelligence. Return only the requested schema fields.
named_customers: list of real company names explicitly named as customers.
feature_inventory: specific product capabilities.
pricing_model: pricing description if found.
hiring_roadmap_signals: team/role expansions.
last_30_days: recent dated events.
""")

    try:
        return llm.call(
            SONNET, messages,
            system=_SYSTEM_STRUCT,
            schema=CompanyIntel,
            prompt_version=PROMPT_VERSIONS["company_synthesis"],
            max_tokens=2500,
        )
    except Exception as exc:
        logger.warning("get_company_intel %s: %s", domain, exc)
        return CompanyIntel(
            what_they_sell="",
            target_customer_icp="",
            named_customers=[NamedCustomer(name=n, source_url="") for n in all_customers[:10]],
            feature_inventory=list(dict.fromkeys(all_features))[:15],
            positioning_summary="",
        )
