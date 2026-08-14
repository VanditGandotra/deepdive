"""Sanitize LLM/model-generated text before rendering as Streamlit markdown.

Dollar signs in prose (e.g. "$73.3B") are valid KaTeX delimiters — every pair
opens and closes a math span, collapsing spaces and switching to serif italic.
This module escapes them at the render boundary so plain dollar amounts display
as plain prose and intentional markdown (bold, headers, lists, links) is
preserved.
"""
from __future__ import annotations

import re
from typing import Any

# Placeholder used internally to protect already-escaped \$ during the pass.
# Must be a sequence that cannot appear in normal text or markdown.
_DOLLAR_PLACEHOLDER = "\x00DOLLAR\x00"


def sanitize_markdown(text: str) -> str:
    """Escape markdown characters that corrupt LLM prose when rendered via KaTeX.

    Guarantees:
    - All bare ``$`` are escaped to ``\\$`` (KaTeX math delimiter suppressed).
    - Already-escaped ``\\$`` are left alone (idempotent).
    - ``$$`` double-dollar sequences are both escaped (``\\$\\$``).
    - Unbalanced single backticks on a line are neutralized with ``&#96;``
      so a stray backtick cannot open an inline-code span that runs to EOL.
    - Content inside fenced code blocks (``` or ~~~) is untouched.
    - Intentional markdown (``**bold**``, ``# headers``, ``- lists``,
      ``[links](url)``) is preserved.
    - ``sanitize_markdown(sanitize_markdown(x)) == sanitize_markdown(x)``
    """
    if not isinstance(text, str):
        return "" if text is None else str(text)
    if not text:
        return text

    # Process line-by-line so code-fence tracking controls both transformations.
    # Content inside ``` or ~~~ fenced blocks is left completely unchanged.
    lines = text.split("\n")
    result_lines: list[str] = []
    in_code_fence = False

    for line in lines:
        stripped = line.strip()

        # Toggle fence state on opening/closing fence markers.
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_code_fence = not in_code_fence
            result_lines.append(line)
            continue

        # Inside a fenced block — leave content untouched.
        if in_code_fence:
            result_lines.append(line)
            continue

        # ── Step 1: Escape bare dollar signs ─────────────────────────────────
        # Protect already-escaped \$ so we don't double-escape them.
        line = line.replace(r"\$", _DOLLAR_PLACEHOLDER)
        line = line.replace("$", r"\$")
        line = line.replace(_DOLLAR_PLACEHOLDER, r"\$")

        # ── Step 2: Neutralize unbalanced single backticks ───────────────────
        # A stray lone backtick opens an inline-code span that can run to EOL,
        # rendering everything after it as green monospace.
        # (?<![`]) / (?![`]) exclude runs of two or more backticks.
        lone_ticks = re.findall(r"(?<!`)`(?!`)", line)
        if len(lone_ticks) % 2 == 1:
            # Find and neutralize the last lone backtick.
            pos = len(line) - 1
            while pos >= 0:
                if line[pos] == "`":
                    before_tick = pos > 0 and line[pos - 1] == "`"
                    after_tick = pos < len(line) - 1 and line[pos + 1] == "`"
                    if not before_tick and not after_tick:
                        line = line[:pos] + "&#96;" + line[pos + 1:]
                        break
                pos -= 1

        result_lines.append(line)

    return "\n".join(result_lines)


def render_md(text: str, **kwargs: Any) -> None:
    """``st.markdown`` with ``sanitize_markdown`` applied.

    Use this wrapper everywhere LLM-generated prose, narrative text, or
    interpolated financial strings are rendered. Drop-in replacement for
    ``st.markdown`` — all keyword arguments are forwarded unchanged.
    """
    import streamlit as st  # lazy import keeps core/ free of UI at import time
    st.markdown(sanitize_markdown(text), **kwargs)
