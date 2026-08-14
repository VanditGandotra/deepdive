"""Phase 3: Deep earnings-call intelligence — Passes A (extraction), B (sentiment), C (synthesis)."""
from __future__ import annotations

import logging
from typing import Dict, Iterator, List, Optional, Tuple

import llm
from analysis.schemas import CallDelta, CallSentiment, CallSummary, HedgingIndex
from config import HAIKU, PROMPT_VERSIONS, SONNET
from data.transcripts import get_available_provider_names, get_last_n_transcripts

logger = logging.getLogger(__name__)

# ── Shared system prompts (1h TTL cache) ─────────────────────────────────────

_SYSTEM_EXTRACT = llm.cached_system("""
You are a financial analyst extracting structured data from earnings call transcripts.
Extract ONLY what is explicitly stated. Do NOT infer or estimate.

For guidance items, capture exact wording. Mark direction as:
  raised=management explicitly raised prior guidance
  lowered=management explicitly lowered prior guidance
  maintained=management confirmed prior guidance unchanged
  initiated=new guidance item this quarter
  withdrawn=guidance was pulled with no replacement
  n/a=not applicable or direction unclear

For signals (5–8 items covering key topics discussed), assess each RELATIVE TO PRIOR EXPECTATIONS —
meaning relative to consensus estimates, management's own prior guidance, or the market's implied
trajectory. A beat vs expectations is "positive" even if the metric declined year-over-year. A
miss vs expectations is "negative" even if the metric grew.

signal="positive": result/commentary explicitly beat expectations or management raised outlook vs prior
signal="neutral":  in-line with expectations, or directionally mixed with no clear net surprise
signal="negative": missed expectations or management cut/withdrew guidance vs prior

IMPORTANT signal rules:
- Management tone (confident, cautious, evasive) is NOT a signal — tone belongs in Pass B.
  Signals must be grounded in explicit data, metrics, or guidance language from the transcript.
- neutral is the DEFAULT. Set signal="neutral" whenever evidence is mixed, ambiguous, or thin.
  Do not manufacture positive or negative readings just to fill signal slots.
- Missing data is not neutral: omit the signal entirely rather than guessing.

confidence: "high" if supported by explicit numbers or direct guidance language,
            "medium" if directional language only (e.g. "above expectations" without quantifying),
            "low" if speculative or inferred — use sparingly
evidence: brief direct quote or paraphrase from the transcript (max 30 words)
rationale: one sentence explaining why positive/neutral/negative vs prior expectations
""")

_SYSTEM_SENTIMENT = llm.cached_system("""
You are a quantitative analyst assessing tone and language signals in earnings calls.
Score sentiment on a scale of -1 (very negative) to +1 (very positive) based on:
- Word choice: confident/hedged, optimistic/cautious
- Specificity: concrete numbers vs vague qualifiers
- Defensiveness: direct vs evasive answers to analyst questions
Evasiveness = analyst asked a specific question; management answered a different question or gave a
non-answer. Flag it only when the pivot is clear.
""")

_SYSTEM_SYNTHESIS = llm.cached_system("""
You are a senior portfolio manager synthesizing 4 consecutive quarters of earnings calls for a single company.
Your job is to identify what CHANGED, not what was said. Focus on:
- Guidance trajectory: raises → holds → cuts
- Topics management STOPPED mentioning vs new topics that appeared
- Whether management's tone in prepared remarks diverges from Q&A (a red flag)
- Recurring analyst pressure points that management has not satisfactorily addressed
Be direct and contrarian where warranted. Distinguish signal from noise.
""")


# ── Pass A: Structured extraction (Haiku per transcript) ─────────────────────

def extract_call_summary(transcript: Dict) -> Optional[CallSummary]:
    content = transcript.get("content", "")
    if not content.strip():
        return None
    quarter_label = f"Q{transcript.get('quarter', '?')} {transcript.get('year', '?')}"
    tag = f"[T-{quarter_label.replace(' ', '')}]"

    messages = [
        {
            "role": "user",
            "content": [
                *llm.cached_content(f"<transcript id=\"{tag}\">\n{content[:25000]}\n</transcript>"),
                llm.text_block(
                    f"Extract the CallSummary for {quarter_label}. "
                    "Set quarter field to the quarter label shown above. "
                    "Include 5–8 signals covering the most important topics discussed "
                    "(revenue, margins, guidance, demand, key metrics). "
                    "Assess each signal relative to prior expectations, not year-over-year change."
                ),
            ],
        }
    ]
    try:
        return llm.call(
            HAIKU, messages,
            system=_SYSTEM_EXTRACT,
            schema=CallSummary,
            prompt_version=PROMPT_VERSIONS["call_extraction_a"],
            max_tokens=2048,
        )
    except Exception as exc:
        logger.warning("Pass A failed for %s: %s", quarter_label, exc)
        return None


# ── Pass B: Sentiment analysis (Haiku per transcript) ────────────────────────

def extract_call_sentiment(transcript: Dict) -> Optional[CallSentiment]:
    content = transcript.get("content", "")
    if not content.strip():
        return None
    quarter_label = f"Q{transcript.get('quarter', '?')} {transcript.get('year', '?')}"
    tag = f"[T-{quarter_label.replace(' ', '')}]"

    # Split prepared remarks vs Q&A for separate scoring
    lower = content.lower()
    qa_start = -1
    for marker in ["question-and-answer", "question and answer", "q&a session", "open for questions"]:
        idx = lower.find(marker)
        if idx >= 0:
            qa_start = idx
            break

    if qa_start > 0:
        prepared = content[:qa_start]
        qa_section = content[qa_start:]
    else:
        prepared = content[:len(content) // 2]
        qa_section = content[len(content) // 2:]

    context = (
        f"<transcript id=\"{tag}\">\n"
        f"<prepared_remarks>\n{prepared[:12000]}\n</prepared_remarks>\n"
        f"<qa_section>\n{qa_section[:12000]}\n</qa_section>\n"
        "</transcript>"
    )

    messages = [
        {
            "role": "user",
            "content": [
                *llm.cached_content(context),
                llm.text_block(
                    f"Analyse sentiment and language signals for {quarter_label}. "
                    "Score prepared_remarks_score from the <prepared_remarks> section, "
                    "qa_score from the <qa_section>, and overall_score as a weighted average (60% prepared, 40% QA). "
                    "For per_speaker, include CEO and CFO if identifiable. "
                    "Flag evasiveness only for Q&A section where analyst question is clearly sidestepped."
                ),
            ],
        }
    ]
    try:
        return llm.call(
            HAIKU, messages,
            system=_SYSTEM_SENTIMENT,
            schema=CallSentiment,
            prompt_version=PROMPT_VERSIONS["call_sentiment_b"],
            max_tokens=2048,
        )
    except Exception as exc:
        logger.warning("Pass B failed for %s: %s", quarter_label, exc)
        # Return a minimal sentinel rather than None so we don't break the synthesis
        return CallSentiment(
            overall_score=0.0,
            prepared_remarks_score=0.0,
            qa_score=0.0,
            per_speaker=[],
            hedging_index=HedgingIndex(level="medium", example_phrases=[]),
            evasiveness_flags=[],
        )


# ── Pass C: Cross-quarter synthesis (Sonnet, streamed) ───────────────────────

def _build_synthesis_context(
    summaries: List[Optional[CallSummary]],
    sentiments: List[Optional[CallSentiment]],
) -> str:
    parts = []
    for i, (s, sent) in enumerate(zip(summaries, sentiments)):
        tag = f"[T-Q{i+1}]"
        parts.append(f"<call id=\"{tag}\" order=\"{i+1}_oldest_first\">")
        if s:
            guidance_text = "; ".join(
                f"{g.metric}: {g.value} ({g.direction})" for g in s.guidance_items
            ) or "none disclosed"
            parts.append(f"  Quarter: {s.quarter}")
            parts.append(f"  Key themes: {'; '.join(s.key_themes)}")
            parts.append(f"  Guidance: {guidance_text}")
            parts.append(f"  Analyst concerns: {'; '.join(s.top_analyst_concerns_from_qa)}")
            if s.competitive_mentions:
                parts.append(f"  Competitive: {'; '.join(m.competitor for m in s.competitive_mentions)}")
        if sent:
            parts.append(f"  Sentiment: overall={sent.overall_score:.2f} "
                         f"prepared={sent.prepared_remarks_score:.2f} "
                         f"qa={sent.qa_score:.2f}")
            if sent.evasiveness_flags:
                parts.append(f"  Evasive topics: {'; '.join(f.analyst_question_topic for f in sent.evasiveness_flags)}")
        parts.append("</call>")
    return "\n".join(parts)


def synthesize_calls(
    summaries: List[Optional[CallSummary]],
    sentiments: List[Optional[CallSentiment]],
    ticker: str,
) -> Optional[CallDelta]:
    """Pass C: structured synthesis. Use for delta card and Thesis tab inputs."""
    context = _build_synthesis_context(summaries, sentiments)
    messages = [
        {
            "role": "user",
            "content": [
                *llm.cached_content(context),
                llm.text_block(
                    f"Synthesize the 4-quarter earnings call evolution for {ticker}. "
                    "Calls are ordered oldest→newest (Q1=oldest). "
                    "Focus on what changed, what management stopped talking about, "
                    "and whether prepared-remarks tone diverges from Q&A. "
                    "sentiment_trend should list [Q1_score, Q2_score, Q3_score, Q4_score] from oldest to newest."
                ),
            ],
        }
    ]
    try:
        return llm.call(
            SONNET, messages,
            system=_SYSTEM_SYNTHESIS,
            schema=CallDelta,
            prompt_version=PROMPT_VERSIONS["call_synthesis_c"],
            max_tokens=3000,
        )
    except Exception as exc:
        logger.warning("Pass C failed for %s: %s", ticker, exc)
        return None


def stream_call_synthesis(
    summaries: List[Optional[CallSummary]],
    sentiments: List[Optional[CallSentiment]],
    ticker: str,
) -> Iterator[str]:
    """Stream the call synthesis narrative for the UI."""
    context = _build_synthesis_context(summaries, sentiments)
    messages = [
        {
            "role": "user",
            "content": [
                *llm.cached_content(context),
                llm.text_block(
                    f"Write a narrative 'What Changed' analysis for {ticker}'s last 4 earnings calls. "
                    "Structure as:\n"
                    "## Guidance Trajectory\n## Topics Management Stopped Discussing\n"
                    "## New Themes Emerging\n## Prepared Remarks vs Q&A Divergence\n"
                    "## Recurring Analyst Pressure Points\n## Bottom Line\n\n"
                    "Calls are tagged [T-Q1] (oldest) to [T-Q4] (newest). Cite them."
                ),
            ],
        }
    ]
    return llm.call(SONNET, messages, system=_SYSTEM_SYNTHESIS, mode="stream", max_tokens=5000)


# ── Orchestrator ──────────────────────────────────────────────────────────────

def analyse_all_calls(ticker: str, n: int = 4) -> Dict:
    """
    Run Passes A+B for the last n transcripts.
    Returns {"transcripts", "summaries", "sentiments", "chunk_tags"}.
    """
    providers_tried = get_available_provider_names()
    transcripts = get_last_n_transcripts(ticker, n=n)
    if not transcripts:
        return {"transcripts": [], "summaries": [], "sentiments": [], "chunk_tags": {}, "providers_tried": providers_tried}

    summaries: List[Optional[CallSummary]] = []
    sentiments: List[Optional[CallSentiment]] = []
    chunk_tags: Dict[str, str] = {}

    for i, t in enumerate(reversed(transcripts)):  # oldest first
        tag = f"[T-Q{i+1}]"
        chunk_tags[tag] = f"Q{t.get('quarter')} {t.get('year')} transcript (first 500 chars): {t.get('content','')[:500]}"
        summaries.append(extract_call_summary(t))
        sentiments.append(extract_call_sentiment(t))

    return {
        "transcripts": list(reversed(transcripts)),
        "summaries": summaries,
        "sentiments": sentiments,
        "chunk_tags": chunk_tags,
        "providers_tried": providers_tried,
    }
