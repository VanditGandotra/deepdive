"""Feature 6: Job postings intelligence — cluster hiring by function, surface roadmap signals."""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

import llm
from config import HAIKU, TTL_NEWS
from data.cache import get_cache_obj, set_cache_obj

logger = logging.getLogger(__name__)

_SYSTEM = llm.cached_system("""
You are a competitive intelligence analyst reading a company's job postings to infer strategy.
Given a list of job titles and raw hiring signals, output:
1. A JSON object with department buckets (Engineering, AI/ML, Sales, Marketing, G&A, Product, Operations, Other)
   and the count of roles in each.
2. A "roadmap_signals" list: 3-5 bullet points revealing what the hiring pattern implies
   about product direction, expansion plans, or strategic priorities.
3. A "standout_roles" list: specific role titles that are unusual or reveal something interesting.

Return ONLY valid JSON with keys: department_counts (dict), roadmap_signals (list of str), standout_roles (list of str).
""")


def extract_hiring_intel(domain: str, hiring_signals: List[str]) -> Dict:
    """
    Cluster hiring signals by department and extract roadmap implications.
    hiring_signals: flat list of strings from page_intels
    """
    cache_key = f"hiring_intel:{domain}"
    cached = get_cache_obj(cache_key)
    if cached:
        return cached

    if not hiring_signals:
        return {"department_counts": {}, "roadmap_signals": [], "standout_roles": []}

    signals_text = "\n".join(f"- {s}" for s in hiring_signals[:120])
    messages = [
        {
            "role": "user",
            "content": [
                llm.text_block(
                    f"Company: {domain}\n\nJob postings / hiring signals:\n{signals_text}\n\n"
                    "Return the JSON analysis."
                ),
            ],
        }
    ]
    try:
        import json
        raw = llm.call(
            HAIKU, messages,
            system=_SYSTEM,
            prompt_version="v1",
            max_tokens=2000,
            continue_on_truncation=True,
        )
        # Parse JSON from response
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        result = json.loads(match.group()) if match else {}
    except Exception as exc:
        logger.warning("job_intel extract failed for %s: %s", domain, exc)
        result = {}

    result.setdefault("department_counts", {})
    result.setdefault("roadmap_signals", [])
    result.setdefault("standout_roles", [])

    set_cache_obj(cache_key, result, TTL_NEWS)
    return result
