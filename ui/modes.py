"""Mode constants — single source of truth for app routing."""
from __future__ import annotations

from enum import StrEnum


class Mode(StrEnum):
    SINGLE = "single"
    PORTFOLIO = "portfolio"


# Display labels — can be renamed freely without breaking URLs or session state.
LABELS: dict[Mode, str] = {
    Mode.SINGLE: "Single Stock",
    Mode.PORTFOLIO: "Portfolio",
}

_LABEL_ORDER = [Mode.SINGLE, Mode.PORTFOLIO]

# Map any legacy or display-label string to a Mode.
_LEGACY: dict[str, Mode] = {
    "Single Stock": Mode.SINGLE,
    "Portfolio": Mode.PORTFOLIO,
    # Former batch labels — stale bookmarks fall back to SINGLE gracefully.
    "Batch": Mode.SINGLE,
    "Batch Analysis": Mode.SINGLE,
    "batch": Mode.SINGLE,
}


def from_any(value: str, default: Mode = Mode.SINGLE) -> Mode:
    """Coerce any string to a Mode. Falls back to default for unknown values."""
    if value in Mode.__members__.values():       # "single", "portfolio"
        return Mode(value)
    return _LEGACY.get(value, default)


def detect(view: str, has_tickers: bool) -> Mode:
    """Derive the current top-level mode from query params."""
    if view in ("portfolio", "portfolio_analysis"):
        return Mode.PORTFOLIO
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
