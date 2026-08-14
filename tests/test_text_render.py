"""Tests for core/text_render.py — sanitize_markdown() correctness and idempotency.

Real strings from the screenshots are used as fixtures to ensure the specific
regression (KaTeX eating dollar-sign pairs in narrative columns) is covered.
"""
from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

import pytest

from core.text_render import sanitize_markdown


# ── Helpers ───────────────────────────────────────────────────────────────────

def _has_unescaped_dollar(text: str) -> bool:
    """Return True if text contains a bare $ not preceded by backslash."""
    return bool(re.search(r"(?<!\\)\$", text))


def _single_backtick_count(text: str) -> int:
    """Count lone single backticks (not part of `` or ``` runs)."""
    return len(re.findall(r"(?<!`)`(?!`)", text))


# ── Dollar-sign escaping (the primary KaTeX regression) ───────────────────────

class TestDollarEscaping:
    def test_single_dollar_amount(self):
        text = "$73.3B TTM free cash flow"
        result = sanitize_markdown(text)
        assert r"\$73.3B" in result
        assert not _has_unescaped_dollar(result)

    def test_two_dollar_amounts_same_line(self):
        """Two bare $ on the same line create a KaTeX math span — both must be escaped."""
        text = "FCF of $73.3B TTM [RATIOS], and $175–185B 2026 CapEx"
        result = sanitize_markdown(text)
        assert r"\$73.3B" in result
        assert r"\$175" in result
        assert not _has_unescaped_dollar(result)

    def test_write_down_and_headwind(self):
        text = "$4.5B write-down, $8B total headwind"
        result = sanitize_markdown(text)
        assert r"\$4.5B" in result
        assert r"\$8B" in result
        assert not _has_unescaped_dollar(result)

    def test_fy_revenue_projection(self):
        text = "$20B+ in FY2026 revenues"
        result = sanitize_markdown(text)
        assert r"\$20B+" in result
        assert not _has_unescaped_dollar(result)

    def test_double_dollar(self):
        """$$ (double-dollar) triggers KaTeX display math — both must be escaped."""
        text = "$$10B market cap"
        result = sanitize_markdown(text)
        assert r"\$\$10B" in result
        assert not _has_unescaped_dollar(result)

    def test_already_escaped_not_doubled(self):
        r"""A pre-escaped \$ must be preserved, not double-escaped."""
        text = r"revenue of \$73.3B"
        result = sanitize_markdown(text)
        assert result == text
        assert result.count(r"\$") == 1

    def test_mixed_escaped_and_bare(self):
        r"""Mix of already-escaped \$ and bare $ — only bare ones get escaped."""
        text = r"guidance \$73B–$80B range"
        result = sanitize_markdown(text)
        assert result.count(r"\$") == 2
        assert not _has_unescaped_dollar(result)

    def test_no_dollars_unchanged(self):
        text = "Revenue grew 12% YoY with strong free cash flow generation."
        result = sanitize_markdown(text)
        assert result == text


# ── Backtick neutralisation ───────────────────────────────────────────────────

class TestBacktickNeutralisation:
    def test_stray_single_backtick(self):
        """One lone backtick on a line → even count after sanitization."""
        text = "revenue grew 12% YoY, margin expanded `by 200bps"
        result = sanitize_markdown(text)
        assert _single_backtick_count(result) % 2 == 0

    def test_balanced_backticks_untouched(self):
        """Balanced inline code span must survive."""
        text = "See the `config.yaml` for details."
        result = sanitize_markdown(text)
        assert "`config.yaml`" in result

    def test_double_backtick_untouched(self):
        """Double-backtick inline code (``foo``) is not affected."""
        text = "Use ``pip install`` to install."
        result = sanitize_markdown(text)
        assert "``pip install``" in result

    def test_code_fence_content_untouched(self):
        """Content inside a fenced code block must not be modified."""
        text = "```\n$73.3B raw dollar\n```"
        result = sanitize_markdown(text)
        # Dollar inside the fence should be left alone
        assert "$73.3B" in result

    def test_dollar_outside_fence_escaped(self):
        """Dollar outside code fence is escaped; inside is not."""
        text = "prose $10B\n```\n$raw\n```\nmore $20B"
        result = sanitize_markdown(text)
        lines = result.split("\n")
        assert r"\$10B" in lines[0]
        assert lines[2] == "$raw"          # inside fence: untouched
        assert r"\$20B" in lines[4]


# ── Idempotency ───────────────────────────────────────────────────────────────

class TestIdempotency:
    def test_dollar_idempotent(self):
        text = "FCF of $73.3B TTM, $175B CapEx"
        first = sanitize_markdown(text)
        second = sanitize_markdown(first)
        assert first == second

    def test_backtick_idempotent(self):
        text = "revenue grew `by 200bps unexpectedly"
        first = sanitize_markdown(text)
        second = sanitize_markdown(first)
        assert first == second

    def test_combined_idempotent(self):
        text = "FCF of $73.3B TTM [RATIOS], and $175–185B 2026 CapEx, margin `grew"
        first = sanitize_markdown(text)
        second = sanitize_markdown(first)
        assert first == second

    def test_already_clean_idempotent(self):
        text = "No dollars and balanced `code` spans here."
        first = sanitize_markdown(text)
        second = sanitize_markdown(first)
        assert first == second


# ── Intentional markdown preserved ───────────────────────────────────────────

class TestMarkdownPreserved:
    def test_bold_preserved(self):
        text = "**Strong growth** at $73.3B — remarkable quarter."
        result = sanitize_markdown(text)
        assert "**Strong growth**" in result

    def test_italic_preserved(self):
        text = "_Significant_ margin expansion worth $4.5B."
        result = sanitize_markdown(text)
        assert "_Significant_" in result

    def test_header_preserved(self):
        text = "## Bull Case\n\nRevenue of $100B expected."
        result = sanitize_markdown(text)
        assert result.startswith("## Bull Case")

    def test_list_items_preserved(self):
        text = "- Margin expansion\n- Revenue at $73.3B\n- FCF growth"
        result = sanitize_markdown(text)
        assert result.startswith("- Margin expansion")
        assert "- Revenue at" in result

    def test_markdown_link_preserved(self):
        text = "See [NVDA earnings]( https://example.com) for $30B guidance."
        result = sanitize_markdown(text)
        assert "[NVDA earnings]" in result
        assert not _has_unescaped_dollar(result)


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_string(self):
        assert sanitize_markdown("") == ""

    def test_none_returns_empty(self):
        assert sanitize_markdown(None) == ""  # type: ignore[arg-type]

    def test_non_string_coerced(self):
        result = sanitize_markdown(42)  # type: ignore[arg-type]
        assert isinstance(result, str)

    def test_only_dollar_sign(self):
        result = sanitize_markdown("$")
        assert result == r"\$"
        assert not _has_unescaped_dollar(result)

    def test_multiline_dollar(self):
        text = "Bull: $73.3B upside\nBase: $60B flat\nBear: $45B downside"
        result = sanitize_markdown(text)
        assert not _has_unescaped_dollar(result)
        assert result.count(r"\$") == 3


# ── render_md smoke test ──────────────────────────────────────────────────────

class TestRenderMd:
    def test_render_md_escapes_dollars(self):
        """render_md must not forward bare $ to st.markdown."""
        captured: list[str] = []
        with patch("streamlit.markdown", side_effect=lambda t, **kw: captured.append(t)):
            from core.text_render import render_md
            render_md("Revenue grew $73.3B this quarter")

        assert len(captured) == 1
        assert r"\$73.3B" in captured[0]
        assert not _has_unescaped_dollar(captured[0])

    def test_render_md_passes_kwargs(self):
        """Keyword arguments are forwarded to st.markdown unchanged."""
        mock_md = MagicMock()
        with patch("streamlit.markdown", mock_md):
            from core.text_render import render_md
            render_md("text", unsafe_allow_html=True)

        mock_md.assert_called_once()
        _, kwargs = mock_md.call_args
        assert kwargs.get("unsafe_allow_html") is True
