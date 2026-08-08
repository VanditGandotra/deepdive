"""Phase 7: Delta engine — deterministic field diff + Sonnet narrative card."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import llm
from analysis.schemas import DeltaItem, DeltaNarrative, RunSnapshot, RunSnapshotData
from config import PROMPT_VERSIONS, SONNET
from data.cache import get_last_run_snapshot, save_run_snapshot

logger = logging.getLogger(__name__)

_SYSTEM = llm.cached_system("""
You are a systematic equity analyst producing a concise 'what changed' brief for a portfolio manager.
Write 3-6 bullet points, most material first. Be specific: name the metric, direction, and magnitude.
If nothing is material, say "No material changes detected." Use plain English, no jargon.
""")


# ── Snapshot I/O ──────────────────────────────────────────────────────────────

def save_snapshot(
    ticker_or_url: str,
    snapshot_data: RunSnapshotData,
) -> None:
    snap = RunSnapshot(ticker_or_url=ticker_or_url, snapshot=snapshot_data)
    save_run_snapshot(ticker_or_url, snap.model_dump_json())


def load_last_snapshot(ticker_or_url: str) -> Optional[Dict]:
    return get_last_run_snapshot(ticker_or_url)


# ── Deterministic diff ────────────────────────────────────────────────────────

def _pct_delta(old: Optional[float], new: Optional[float]) -> Optional[float]:
    if old is None or new is None or old == 0:
        return None
    return (new - old) / abs(old)


def _diff_scalar(
    field: str,
    old: Optional[float],
    new: Optional[float],
    threshold: float = 0.02,
) -> Optional[DeltaItem]:
    delta = _pct_delta(old, new)
    if delta is None:
        return None
    if abs(delta) < threshold:
        return None
    direction = "up" if delta > 0 else "down"
    return DeltaItem(
        field=field,
        old_value=old,
        new_value=new,
        change_type="changed",
        description=f"{field}: {direction} {abs(delta)*100:.1f}% ({old} → {new})",
    )


def _diff_list(
    field: str,
    old_list: Optional[List[str]],
    new_list: Optional[List[str]],
) -> List[DeltaItem]:
    items: List[DeltaItem] = []
    old_set = set(old_list or [])
    new_set = set(new_list or [])
    for added in new_set - old_set:
        items.append(DeltaItem(field=field, new_value=added, change_type="added",
                               description=f"New in {field}: {added}"))
    for removed in old_set - new_set:
        items.append(DeltaItem(field=field, old_value=removed, change_type="removed",
                               description=f"Removed from {field}: {removed}"))
    return items


def _diff_flags(
    old_flags: Optional[List[str]],
    new_flags: Optional[List[str]],
) -> List[DeltaItem]:
    items: List[DeltaItem] = []
    old_set = set(old_flags or [])
    new_set = set(new_flags or [])
    for flag in new_set - old_set:
        items.append(DeltaItem(field="quality_flag", new_value=flag,
                               change_type="new_flag",
                               description=f"New quality flag triggered: {flag}"))
    for flag in old_set - new_set:
        items.append(DeltaItem(field="quality_flag", old_value=flag,
                               change_type="cleared_flag",
                               description=f"Quality flag cleared: {flag}"))
    return items


def compute_diff(
    old_snap: Dict[str, Any],
    new_snap_data: RunSnapshotData,
) -> List[DeltaItem]:
    """Deterministic diff between old snapshot dict and new RunSnapshotData."""
    items: List[DeltaItem] = []
    old_data = old_snap.get("snapshot", {}) if isinstance(old_snap.get("snapshot"), dict) else {}
    old_fund = old_data.get("fundamentals") or {}
    new_fund = new_snap_data.fundamentals or {}

    for field, key in [
        ("P/E TTM", "pe_ttm"),
        ("Revenue TTM", "revenue_ttm"),
        ("Market Cap", "market_cap"),
    ]:
        item = _diff_scalar(field, old_fund.get(key), new_fund.get(key))
        if item:
            items.append(item)

    # Short interest
    si_item = _diff_scalar(
        "Short Interest %",
        old_data.get("short_interest_pct"),
        new_snap_data.short_interest_pct,
        threshold=0.10,  # 10% relative change threshold for SI
    )
    if si_item:
        items.append(si_item)

    # Quality flags
    items.extend(_diff_flags(
        old_data.get("quality_flags"),
        new_snap_data.quality_flags,
    ))

    # URL mode: customer list, pages
    items.extend(_diff_list("named_customers", old_data.get("customer_list"), new_snap_data.customer_list))

    last_30_old = old_data.get("last_30_days_count") or 0
    last_30_new = new_snap_data.last_30_days_count or 0
    if last_30_new > last_30_old:
        items.append(DeltaItem(
            field="last_30_days_events",
            old_value=last_30_old, new_value=last_30_new,
            change_type="changed",
            description=f"New company updates: {last_30_new - last_30_old} more events vs last run",
        ))

    return items


# ── Sonnet delta narrative (small call) ───────────────────────────────────────

def build_delta_narrative(
    ticker_or_url: str,
    diff_items: List[DeltaItem],
    prior_run_at: str,
) -> DeltaNarrative:
    if not diff_items:
        return DeltaNarrative(
            prior_run_at=datetime.fromisoformat(prior_run_at) if prior_run_at else datetime.utcnow(),
            current_run_at=datetime.utcnow(),
            items=[],
            narrative_bullets=["No material changes detected since last run."],
        )

    diff_text = "\n".join(f"- {item.description}" for item in diff_items)
    messages = [
        {
            "role": "user",
            "content": [
                llm.text_block(
                    f"Summarise these changes for {ticker_or_url} vs the prior run ({prior_run_at}):\n\n"
                    f"{diff_text}\n\n"
                    "Return 3-6 concise bullets, most material first. "
                    "Each bullet: one sentence, concrete metric + direction + magnitude."
                ),
            ],
        }
    ]
    try:
        narrative_text = llm.call(
            SONNET, messages,
            system=_SYSTEM,
            prompt_version=PROMPT_VERSIONS["delta_narrative"],
            max_tokens=500,
        )
        bullets = [
            line.lstrip("•-* ").strip()
            for line in narrative_text.split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]
        bullets = [b for b in bullets if b][:6]
    except Exception as exc:
        logger.warning("Delta narrative Sonnet call failed: %s", exc)
        bullets = [item.description for item in diff_items[:5]]

    return DeltaNarrative(
        prior_run_at=datetime.fromisoformat(prior_run_at) if prior_run_at else datetime.utcnow(),
        current_run_at=datetime.utcnow(),
        items=diff_items,
        narrative_bullets=bullets,
    )


def _is_meaningful(snapshot: RunSnapshotData) -> bool:
    """Guard: only write snapshots that have at least some content."""
    return bool(
        snapshot.fundamentals
        or snapshot.customer_list
        or snapshot.feature_claims
        or snapshot.quality_flags
        or snapshot.sentiment_scores
        or snapshot.kpi_values
    )


def run_delta(
    ticker_or_url: str,
    new_snapshot: RunSnapshotData,
) -> Optional[DeltaNarrative]:
    """
    Compare new_snapshot against the last stored run.
    Saves new snapshot ONLY if it has meaningful content.
    Returns DeltaNarrative, or None if no prior run / snapshot not worth saving.
    """
    if not _is_meaningful(new_snapshot):
        logger.info("run_delta: skipping empty snapshot for %s", ticker_or_url)
        return None

    old_snap = load_last_snapshot(ticker_or_url)
    save_snapshot(ticker_or_url, new_snapshot)

    if old_snap is None:
        return None  # First run — baseline saved, diff on next run

    prior_run_at = old_snap.get("_run_at", "")
    diff_items = compute_diff(old_snap, new_snapshot)
    return build_delta_narrative(ticker_or_url, diff_items, prior_run_at)
