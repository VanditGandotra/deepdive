"""Mode constants — single source of truth for app routing."""
from __future__ import annotations

from enum import StrEnum


class Mode(StrEnum):
    SINGLE = "single"
    BATCH = "batch"
    PORTFOLIO = "portfolio"


# Display labels — can be renamed freely without breaking URLs or session state.
LABELS: dict[Mode, str] = {
    Mode.SINGLE: "Single Stock",
    Mode.BATCH: "Batch Analysis",
    Mode.PORTFOLIO: "Portfolio",
}

_LABEL_ORDER = [Mode.SINGLE, Mode.BATCH, Mode.PORTFOLIO]

# Map any legacy or display-label string to a Mode.
_LEGACY: dict[str, Mode] = {
    # Old short forms written by the original _sidebar_nav()
    "Batch": Mode.BATCH,
    "Single Stock": Mode.SINGLE,
    "Portfolio": Mode.PORTFOLIO,
    # Current display labels (so LABELS values also work as keys)
    "Batch Analysis": Mode.BATCH,
    # Machine values are handled by StrEnum.__eq__ directly
}


def from_any(value: str, default: Mode = Mode.SINGLE) -> Mode:
    """Coerce any string to a Mode. Falls back to default for unknown values."""
    if value in Mode.__members__.values():       # "single", "batch", "portfolio"
        return Mode(value)
    return _LEGACY.get(value, default)


def detect(view: str, has_tickers: bool) -> Mode:
    """Derive the current top-level mode from query params."""
    if view in ("portfolio", "portfolio_analysis"):
        return Mode.PORTFOLIO
    if has_tickers or view == "ticker":
        return Mode.BATCH
    return Mode.SINGLE


def label_index(mode: Mode) -> int:
    """Zero-based index of mode in the display-label list (for st.radio index=)."""
    return _LABEL_ORDER.index(mode)


def display_labels() -> list[str]:
    """Ordered list of display labels for st.radio options=."""
    return [LABELS[m] for m in _LABEL_ORDER]


def from_label(label: str, default: Mode = Mode.SINGLE) -> Mode:
    """Map a display label back to a Mode (for reading st.radio return value)."""
    by_label = {v: k for k, v in LABELS.items()}
    return by_label.get(label, default)
