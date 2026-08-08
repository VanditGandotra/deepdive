"""Phase 10: Product explainer from docs (Sonnet streaming) + screenshot vision (Haiku)."""
from __future__ import annotations

import base64
import logging
from typing import Dict, Iterator, List, Optional

import httpx

import llm
from analysis.schemas import ScreenExplanation
from config import HAIKU, PROMPT_VERSIONS, SONNET

logger = logging.getLogger(__name__)

_SYSTEM_EXPLAINER = llm.cached_system("""
You are a technical writer producing a world-class product explainer from documentation.
Your reader is a sophisticated buyer or developer who has NOT used this product before.
Rules:
1. Build an accurate mental model — explain the abstraction layers.
2. "Zero to value" workflow: the minimal sequence to get something working.
3. Feature inventory: bullet list of named features/capabilities from the docs.
4. Integrations: list all third-party systems mentioned.
5. API surface: describe endpoints / SDK patterns if docs cover them.
6. Pricing / plans / limits: only include if explicitly documented.
7. Changelog velocity: note cadence / recency of recent changes if a changelog is included.
8. Use concrete examples from the docs — not made-up ones.
Write in plain English. Structure clearly.
""")

_SYSTEM_VISION = llm.cached_system("""
You analyse product screenshots for a B2B software intelligence report.
For each screenshot: identify what screen it shows, key UI elements, and what it reveals
about the product's capabilities, user workflow, or data model.
Be specific — name actual labels, menus, columns, and actions you can see.
""")


def stream_product_explainer(
    domain: str,
    doc_pages: List[Dict],
    source_tag: str = "DOC",
    caveat: str = "",
) -> Iterator[str]:
    """
    Stream a structured product explainer synthesised from pages.

    source_tag: prefix for citation tags — "DOC" for docs pages, "PAGE" for marketing pages.
    caveat: prepended to the output when using a fallback source (e.g. marketing site).
    """
    if not doc_pages:
        def _empty() -> Iterator[str]:
            yield f"No pages were found or fetched for {domain}."
        return _empty()

    context_parts: List[str] = []
    for i, page in enumerate(doc_pages):
        text = (page.get("text") or "").strip()
        if not text:
            continue
        tag = f"[{source_tag}-{i+1}]"
        snippet = text[:4000]
        context_parts.append(
            f'<source id="{tag}" url="{page.get("url", "")}">\n{snippet}\n</source>'
        )

    if not context_parts:
        def _empty2() -> Iterator[str]:
            yield f"Pages for {domain} contained no extractable text."
        return _empty2()

    context = "\n\n".join(context_parts)

    instruction = (
        f"Write the product explainer for {domain}.\n\n"
        "Use this structure:\n"
        "## What It Is\n"
        "## Core Concepts\n"
        "## Mental Model\n"
        "## Zero to Value Workflow\n"
        "## Feature Inventory\n"
        "## Integrations\n"
        "## API Surface\n"
        "## Plans & Limits\n"
        "## Changelog Velocity\n\n"
        f"Cite every fact with its [{source_tag}-n] source tag."
    )
    if caveat:
        instruction = (
            f"**Note:** {caveat}\n\n"
            "Where documentation is absent, note the gap explicitly rather than speculating.\n\n"
            + instruction
        )

    messages = [
        {
            "role": "user",
            "content": [
                *llm.cached_content(context),
                llm.text_block(instruction),
            ],
        }
    ]

    # If using marketing-site fallback, prefix the streamed output with the caveat banner
    if caveat:
        def _with_banner() -> Iterator[str]:
            yield f"> ⚠️ {caveat}\n\n"
            yield from llm.call(
                SONNET, messages,
                system=_SYSTEM_EXPLAINER,
                mode="stream",
                max_tokens=4096,
            )
        return _with_banner()

    return llm.call(
        SONNET, messages,
        system=_SYSTEM_EXPLAINER,
        mode="stream",
        max_tokens=4096,
    )


def explain_screenshots(images: List[Dict]) -> List[Optional[ScreenExplanation]]:
    """
    Run Haiku vision on a list of image dicts ({url, ...}).
    Returns one ScreenExplanation per image (or None on failure).
    """
    results: List[Optional[ScreenExplanation]] = []
    for img in images:
        img_url = img.get("url", "")
        try:
            image_data, media_type = _fetch_image(img_url)
        except Exception as exc:
            logger.debug("Image fetch %s: %s", img_url, exc)
            results.append(None)
            continue

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                    llm.text_block(
                        "Analyse this product screenshot. "
                        "Return the structured ScreenExplanation fields."
                    ),
                ],
            }
        ]
        try:
            expl = llm.call(
                HAIKU, messages,
                system=_SYSTEM_VISION,
                schema=ScreenExplanation,
                prompt_version=PROMPT_VERSIONS["screen_explanation"],
                max_tokens=600,
            )
            results.append(expl)
        except Exception as exc:
            logger.warning("explain_screenshots %s: %s", img_url, exc)
            results.append(None)
    return results


def _fetch_image(url: str) -> tuple[str, str]:
    """Download image and return (base64_data, media_type)."""
    headers = {"User-Agent": "DeepDiveBot/1.0 (research-only; contact vandit@deductive.ai)"}
    r = httpx.get(url, headers=headers, follow_redirects=True, timeout=15)
    r.raise_for_status()
    content_type = r.headers.get("content-type", "image/png").split(";")[0].strip()
    # Normalise to supported types
    if content_type not in ("image/jpeg", "image/png", "image/gif", "image/webp"):
        content_type = "image/png"
    data = base64.b64encode(r.content).decode()
    return data, content_type
