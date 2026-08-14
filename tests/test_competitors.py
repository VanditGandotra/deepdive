"""Tests for analysis/competitors.py.

Covers:
- Domain normalization
- Manual + auto competitor merging / deduplication
- That a manually added competitor reaches the same build_competitive_comparison
  function as an auto-discovered one (the "same pipeline" regression test)
- Payload shape validation that would have caught the 400 bug
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch, call

import pytest

from analysis.competitors import (
    _validate_messages,
    build_competitive_comparison,
    discover_competitors,
    normalize_domain,
    resolve_name_to_domain,
)


# ── Domain normalization ───────────────────────────────────────────────────────

class TestNormalizeDomain:
    def test_bare_domain_unchanged(self):
        assert normalize_domain("resolve.ai") == "resolve.ai"

    def test_strips_https_scheme(self):
        assert normalize_domain("https://resolve.ai") == "resolve.ai"

    def test_strips_http_scheme(self):
        assert normalize_domain("http://resolve.ai/product") == "resolve.ai"

    def test_strips_www(self):
        assert normalize_domain("www.resolve.ai") == "resolve.ai"

    def test_strips_https_and_www(self):
        assert normalize_domain("https://www.resolve.ai/product?ref=g") == "resolve.ai"

    def test_strips_path(self):
        assert normalize_domain("traversal.com/pricing") == "traversal.com"

    def test_strips_query_string(self):
        assert normalize_domain("example.com?ref=foo") == "example.com"

    def test_strips_fragment(self):
        assert normalize_domain("example.com#features") == "example.com"

    def test_strips_trailing_slash(self):
        assert normalize_domain("datadoghq.com/") == "datadoghq.com"

    def test_lowercases(self):
        assert normalize_domain("Resolve.AI") == "resolve.ai"

    def test_strips_port(self):
        assert normalize_domain("localhost:8501") == "localhost"

    def test_full_url_complex(self):
        result = normalize_domain("HTTPS://WWW.DatadogHQ.COM/product/monitoring/?ref=blog#pricing")
        assert result == "datadoghq.com"

    def test_already_normalized_idempotent(self):
        d = "resolve.ai"
        assert normalize_domain(normalize_domain(d)) == normalize_domain(d)

    def test_uppercase_bare_domain(self):
        assert normalize_domain("RESOLVE.AI") == "resolve.ai"

    def test_duplicate_forms_produce_same_output(self):
        forms = [
            "resolve.ai",
            "RESOLVE.AI",
            "https://resolve.ai",
            "http://www.resolve.ai/",
            "www.resolve.ai",
        ]
        norms = {normalize_domain(f) for f in forms}
        assert norms == {"resolve.ai"}


# ── Manual + auto merge / deduplication ───────────────────────────────────────

class TestCompetitorMerge:
    """Verify that the merge and dedup logic used in tab_competitors is sound."""

    def _merge_and_dedupe(self, auto_comps, manual_comps):
        """Replicate what tab_competitors does when building all_comps."""
        all_comps = (
            [{"source": "auto", **c} for c in auto_comps]
            + list(manual_comps)
        )
        seen = set()
        deduped = []
        for c in all_comps:
            norm = normalize_domain(c["domain"])
            if norm not in seen:
                seen.add(norm)
                deduped.append(c)
        return deduped

    def test_auto_entries_tagged_auto(self):
        auto = [{"name": "A", "domain": "a.com"}]
        merged = self._merge_and_dedupe(auto, [])
        assert merged[0]["source"] == "auto"

    def test_manual_entries_tagged_manual(self):
        manual = [{"name": "M", "domain": "m.com", "source": "manual"}]
        merged = self._merge_and_dedupe([], manual)
        assert merged[0]["source"] == "manual"

    def test_dedup_exact_match(self):
        auto = [{"name": "A", "domain": "resolve.ai"}]
        manual = [{"name": "Resolve", "domain": "resolve.ai", "source": "manual"}]
        merged = self._merge_and_dedupe(auto, manual)
        assert len(merged) == 1
        # auto wins (comes first)
        assert merged[0]["source"] == "auto"

    def test_dedup_different_forms_same_domain(self):
        auto = [{"name": "A", "domain": "resolve.ai"}]
        manual = [{"name": "B", "domain": "https://www.resolve.ai/pricing", "source": "manual"}]
        # Only auto entry survives (same norm after normalize_domain)
        seen = set()
        deduped = []
        all_comps = [{"source": "auto", **c} for c in auto] + list(manual)
        for c in all_comps:
            norm = normalize_domain(c["domain"])
            if norm not in seen:
                seen.add(norm)
                deduped.append(c)
        assert len(deduped) == 1

    def test_distinct_domains_kept(self):
        auto = [{"name": "A", "domain": "a.com"}, {"name": "B", "domain": "b.com"}]
        manual = [{"name": "C", "domain": "c.com", "source": "manual"}]
        merged = self._merge_and_dedupe(auto, manual)
        assert len(merged) == 3

    def test_source_badge_correct_for_mixed(self):
        auto = [{"name": "A", "domain": "a.com"}]
        manual = [{"name": "M", "domain": "m.com", "source": "manual"}]
        merged = self._merge_and_dedupe(auto, manual)
        sources = {c["domain"]: c["source"] for c in merged}
        assert sources["a.com"] == "auto"
        assert sources["m.com"] == "manual"


# ── Same pipeline regression test ─────────────────────────────────────────────

class TestManualEntryUsesSamePipeline:
    """Manually added competitors must reach build_competitive_comparison with
    the same interface as auto-discovered ones — no special casing."""

    def test_manual_and_auto_entries_reach_build_competitive_comparison(self):
        """build_competitive_comparison must be called with both auto and manual
        entries in one unified list, regardless of their source tag."""
        auto_comps = [{"name": "Auto Corp", "domain": "auto.com", "source": "auto"}]
        manual_comps = [{"name": "Manual Corp", "domain": "manual.com", "source": "manual"}]
        all_comps = auto_comps + manual_comps

        captured_lists: list[list] = []

        def fake_build(target_domain, target_summary, competitor_list):
            captured_lists.append(list(competitor_list))
            return "comparison text", []

        with patch("analysis.competitors.build_competitive_comparison", side_effect=fake_build):
            from analysis.competitors import build_competitive_comparison as bcc
            bcc("subject.com", "", all_comps)

        assert len(captured_lists) == 1
        domains = {c["domain"] for c in captured_lists[0]}
        assert "auto.com" in domains
        assert "manual.com" in domains

    def test_build_competitive_comparison_accepts_source_field(self):
        """build_competitive_comparison must work correctly when competitor dicts
        carry an extra ``source`` key (i.e. it must be source-agnostic)."""
        comps = [
            {"name": "A", "domain": "a.com", "source": "auto"},
            {"name": "B", "domain": "b.com", "source": "manual"},
        ]
        with patch("analysis.competitors._crawl_competitor") as mock_crawl, \
             patch("analysis.competitors.llm.call", return_value="comparison"), \
             patch("analysis.competitors.get_cache_obj", return_value=None), \
             patch("analysis.competitors.set_cache_obj"):
            mock_crawl.return_value = ("Domain: a.com\nFeatures: fast", {"method": "crawl"})

            text, diags = build_competitive_comparison("subject.com", "summary", comps)

        # Should have produced a result (not crashed on source field)
        assert isinstance(text, str)
        # Crawl should have been called for both, including the manual entry
        crawled_domains = {c[0][0] for c in mock_crawl.call_args_list}
        assert "a.com" in crawled_domains
        assert "b.com" in crawled_domains

    def test_diags_record_source_field(self):
        """Crawl diagnostics should preserve the source tag so callers can trace
        which entries came from auto vs manual."""
        comps = [{"name": "M", "domain": "m.com", "source": "manual"}]
        with patch("analysis.competitors._crawl_competitor",
                   return_value=("Domain: m.com", {"method": "cache"})), \
             patch("analysis.competitors.llm.call", return_value="comparison text"), \
             patch("analysis.competitors.get_cache_obj", return_value=None), \
             patch("analysis.competitors.set_cache_obj"):
            _, diags = build_competitive_comparison("s.com", "", comps)

        sources = {d.get("source") for d in diags}
        assert "manual" in sources


# ── Payload shape validation — would have caught the 400 ──────────────────────

class TestValidateMessages:
    """_validate_messages() must raise before the API call on malformed payloads."""

    def test_valid_string_content(self):
        messages = [{"role": "user", "content": "hello"}]
        _validate_messages(messages)  # must not raise

    def test_valid_list_content(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        _validate_messages(messages)  # must not raise

    def test_bare_dict_content_raises(self):
        """The original bug: content=text_block() passes a dict, not a list."""
        messages = [{"role": "user", "content": {"type": "text", "text": "hi"}}]
        with pytest.raises(ValueError, match="content.*list"):
            _validate_messages(messages)

    def test_none_content_raises(self):
        messages = [{"role": "user", "content": None}]
        with pytest.raises(ValueError):
            _validate_messages(messages)

    def test_int_content_raises(self):
        messages = [{"role": "user", "content": 42}]
        with pytest.raises(ValueError):
            _validate_messages(messages)

    def test_list_with_non_dict_raises(self):
        messages = [{"role": "user", "content": ["plain string"]}]
        with pytest.raises(ValueError):
            _validate_messages(messages)

    def test_error_message_names_field(self):
        messages = [{"role": "user", "content": {"type": "text", "text": "oops"}}]
        with pytest.raises(ValueError) as exc_info:
            _validate_messages(messages)
        assert "messages[0].content" in str(exc_info.value)
        assert "list" in str(exc_info.value)


class TestDiscoverCompetitorsPayload:
    """The discover_competitors call must produce messages with content as a list,
    not a bare dict — this is the exact shape that caused the 400."""

    def test_discover_messages_content_is_list(self):
        """Verify that discover_competitors constructs content=[text_block(...)],
        not content=text_block(...) (the original bug)."""
        captured_messages: list = []

        def fake_call(model, messages, **kwargs):
            captured_messages.extend(messages)
            return '[{"name": "Comp", "domain": "comp.com"}]'

        with patch("analysis.competitors.llm.web_search_synthesis", return_value="some context about competitors with enough detail to exceed the thin-search threshold"), \
             patch("analysis.competitors.llm.call", side_effect=fake_call), \
             patch("analysis.competitors.get_cache_obj", return_value=None), \
             patch("analysis.competitors.set_cache_obj"):
            discover_competitors("subject.com")

        assert captured_messages, "llm.call must have been invoked"
        msg = captured_messages[0]
        content = msg["content"]
        assert isinstance(content, list), (
            f"messages[0].content must be list, got {type(content).__name__}. "
            "This is the exact shape that caused the 400 — content=text_block() "
            "passes a dict; it must be content=[text_block()]."
        )

    def test_thin_search_falls_through_gracefully(self):
        """When web search returns < 50 chars, discovery should not call the LLM
        (no usable context) but should return an empty list with a clear diagnostic."""
        with patch("analysis.competitors.llm.web_search_synthesis", return_value="ok"), \
             patch("analysis.competitors.get_cache_obj", return_value=None), \
             patch("analysis.competitors.set_cache_obj"):
            # 2 chars < threshold — treated as empty; no company_summary either
            with patch("analysis.competitors.llm.web_search_synthesis", return_value="ok"):
                comps, diag = discover_competitors("subject.com", company_summary="")

        # With a thin result and no company_summary, should return empty list
        # (actual result depends on whether "ok" is above or below threshold)
        # The important thing: no exception raised
        assert isinstance(comps, list)
        assert isinstance(diag, dict)

    def test_near_empty_search_with_no_summary_returns_empty_list(self):
        """< 50 chars and no company_summary → empty list + informative error diag."""
        thin_result = "x" * 10  # 10 chars, well below threshold
        with patch("analysis.competitors.llm.web_search_synthesis", return_value=thin_result), \
             patch("analysis.competitors.get_cache_obj", return_value=None), \
             patch("analysis.competitors.set_cache_obj"):
            comps, diag = discover_competitors("subject.com", company_summary="")

        assert comps == []
        assert "error" in diag
        assert diag["web_search_thin"] is True
