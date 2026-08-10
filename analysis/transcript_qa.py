"""
Transcript Q&A: answer analyst questions over cached earnings call transcripts.
No vector DB — uses sliding-window keyword retrieval + Sonnet answer with citations.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

import llm
from config import SONNET

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 1200   # characters per chunk
_CHUNK_OVERLAP = 200
_MAX_CHUNKS = 8      # max chunks to pass to Sonnet per answer

_SYSTEM = llm.cached_system("""
You are a research assistant answering questions over earnings call transcripts.
Rules:
1. Answer ONLY from the provided transcript excerpts — never from general knowledge.
2. Every factual claim must include a citation: e.g. [Q3 '24] or [Q2 '24 – Q&A].
3. If the transcripts do not contain enough information to answer, say so explicitly.
4. Quote management directly where possible (use quotation marks).
5. Keep answers concise — 2-4 sentences max unless complexity demands more.
""")


def _chunk_transcript(text: str) -> List[str]:
    """Split transcript into overlapping character chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + _CHUNK_SIZE
        chunks.append(text[start:end])
        start += _CHUNK_SIZE - _CHUNK_OVERLAP
    return chunks


def _keyword_score(chunk: str, query: str) -> float:
    """Simple keyword overlap score (case-insensitive)."""
    words = re.findall(r"\b\w{3,}\b", query.lower())
    if not words:
        return 0.0
    chunk_lower = chunk.lower()
    hits = sum(1 for w in words if w in chunk_lower)
    return hits / len(words)


def _retrieve_chunks(
    query: str,
    transcripts: List[Dict],
    max_chunks: int = _MAX_CHUNKS,
) -> List[Tuple[str, str]]:
    """
    Return (chunk_text, quarter_label) pairs most relevant to the query.
    Simple keyword retrieval — good enough for transcript search without embedding costs.
    """
    candidates: List[Tuple[float, str, str]] = []

    for t in transcripts:
        content = (t.get("content") or "").strip()
        if not content:
            continue
        label = f"Q{t.get('quarter')} '{str(t.get('year', ''))[-2:]}"
        chunks = _chunk_transcript(content)
        for chunk in chunks:
            score = _keyword_score(chunk, query)
            candidates.append((score, chunk, label))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return [(c, l) for _, c, l in candidates[:max_chunks] if _ > 0.0]


def answer_question(
    query: str,
    transcripts: List[Dict],
) -> str:
    """
    Answer a free-form question over the provided transcripts.
    Returns the answer text with inline citations.
    """
    if not transcripts:
        return "No transcripts loaded — cannot answer."
    if not query.strip():
        return "Please enter a question."

    retrieved = _retrieve_chunks(query, transcripts)
    if not retrieved:
        return "The transcripts don't appear to contain information relevant to that question."

    context_parts = []
    for i, (chunk, label) in enumerate(retrieved):
        context_parts.append(
            f'<excerpt id="{label}" seq="{i+1}">\n{chunk}\n</excerpt>'
        )
    context = "\n\n".join(context_parts)

    messages = [
        {
            "role": "user",
            "content": [
                *llm.cached_content(context),
                llm.text_block(f"Question: {query}"),
            ],
        }
    ]

    try:
        return llm.call(
            SONNET, messages,
            system=_SYSTEM,
            prompt_version="v1",
            max_tokens=800,
        )
    except Exception as exc:
        logger.warning("transcript_qa failed: %s", exc)
        return f"Answer generation failed: {exc}"
