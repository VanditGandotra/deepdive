"""
LLM gateway: model tiering, prompt caching (1h TTL), streaming, batch,
structured outputs via tool-use, content-hash caching, usage logging.

call(model, messages, *, system, schema, mode, prompt_version, max_tokens)
  mode="realtime" → Pydantic model | str
  mode="stream"   → Iterator[str]
  mode="batch"    → batch_id str
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from typing import Any, Iterator, Optional, Type, Union

import anthropic
from pydantic import BaseModel, ValidationError

from config import COST_PER_MTOK, HAIKU, SONNET
from data.cache import get_llm_cache, log_llm_call, set_llm_cache

logger = logging.getLogger(__name__)

# ── Singleton client ──────────────────────────────────────────────────────────
_client: Optional[anthropic.Anthropic] = None
_client_lock = threading.Lock()
_thread_local = threading.local()


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = anthropic.Anthropic()
    return _client


def set_session_id(sid: str) -> None:
    _thread_local.session_id = sid


def _get_session_id() -> str:
    return getattr(_thread_local, "session_id", "")


# ── Cost estimation ───────────────────────────────────────────────────────────

def estimate_cost(
    model: str,
    tokens_input: int,
    tokens_output: int,
    cache_read: int = 0,
    cache_write: int = 0,
) -> float:
    rates = COST_PER_MTOK.get(model, COST_PER_MTOK["default"])
    billed_input = max(0, tokens_input - cache_read - cache_write)
    return (
        billed_input        * rates["input"]       +
        cache_read          * rates["cache_read"]  +
        cache_write         * rates["cache_write"] +
        tokens_output       * rates["output"]
    ) / 1_000_000


# ── Content-hash ──────────────────────────────────────────────────────────────

def _content_hash(
    model: str,
    prompt_version: str,
    messages: list,
    system: Any,
) -> str:
    payload = {
        "model": model,
        "pv": prompt_version,
        "msgs": messages,
        "sys": system,
    }
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ── System prompt helpers ─────────────────────────────────────────────────────

def cached_system(text: str, ttl: str = "1h") -> list[dict]:
    """Wrap a system prompt with 1h prompt-cache TTL."""
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral", "ttl": ttl}}]


def cached_content(text: str) -> list[dict]:
    """Wrap a user content block with ephemeral prompt-cache."""
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def text_block(text: str) -> dict:
    return {"type": "text", "text": text}


# ── Structured output helpers ─────────────────────────────────────────────────

def _build_tool(schema: Type[BaseModel]) -> tuple[list, dict]:
    """Return (tools list, tool_choice dict) for forced structured output."""
    raw_schema = schema.model_json_schema()
    raw_schema.pop("title", None)
    tools = [{
        "name": "structured_output",
        "description": f"Return data conforming to the {schema.__name__} schema.",
        "input_schema": raw_schema,
    }]
    tool_choice = {"type": "tool", "name": "structured_output"}
    return tools, tool_choice


def _extract_tool_output(response: Any, schema: Type[BaseModel]) -> BaseModel:
    for block in response.content:
        if block.type == "tool_use":
            return schema.model_validate(block.input)
    raise ValueError(f"No tool_use block in response for schema {schema.__name__}")


# ── Main gateway ──────────────────────────────────────────────────────────────

def call(
    model: str,
    messages: list,
    *,
    system: Optional[Union[str, list]] = None,
    schema: Optional[Type[BaseModel]] = None,
    mode: str = "realtime",
    prompt_version: str = "v1",
    max_tokens: int = 4096,
    temperature: float = 0.0,
    skip_llm_cache: bool = False,
    continue_on_truncation: bool = False,
    max_continuations: int = 3,
) -> Any:
    """
    Unified LLM call entry point.

    system: str → wrapped in a single text block (no caching)
            list → passed as-is (use cached_system() to add cache_control)
    schema: Type[BaseModel] → structured output via tool use
    mode:   "realtime" | "stream" | "batch"
    continue_on_truncation: when True, auto-continues if stop_reason=="max_tokens"
                            (text-only non-streaming calls only). Truncated
                            responses are NOT written to the LLM cache.
    """
    if mode == "stream":
        return _stream(model, messages, system=system, max_tokens=max_tokens, temperature=temperature)

    if mode == "batch":
        return _submit_batch(model, messages, system=system, schema=schema, max_tokens=max_tokens)

    # ── realtime ──────────────────────────────────────────────────────────────
    sys_param = _normalise_system(system)
    content_hash = _content_hash(model, prompt_version, messages, sys_param)

    if not skip_llm_cache:
        cached = get_llm_cache(content_hash)
        if cached is not None:
            log_llm_call(model, prompt_version, content_hash, 0, 0, 0, 0, 0.0, True, _get_session_id())
            if schema:
                try:
                    return schema.model_validate_json(cached)
                except (ValidationError, Exception):
                    pass  # fall through to live call on parse error
            else:
                return cached

    client = _get_client()
    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if sys_param is not None:
        kwargs["system"] = sys_param
    if temperature != 0.0:
        kwargs["temperature"] = temperature
    if schema:
        tools, tool_choice = _build_tool(schema)
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice

    response = client.messages.create(**kwargs)

    usage = response.usage
    tokens_in    = usage.input_tokens
    tokens_out   = usage.output_tokens
    cache_read   = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_write  = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cost         = estimate_cost(model, tokens_in, tokens_out, cache_read, cache_write)

    log_llm_call(model, prompt_version, content_hash, tokens_in, tokens_out,
                 cache_read, cache_write, cost, False, _get_session_id())

    if schema:
        try:
            result = _extract_tool_output(response, schema)
        except (ValueError, ValidationError) as exc:
            # One retry without cache
            logger.warning("Schema parse failed (%s), retrying once without tool_choice", exc)
            kwargs_retry = {**kwargs}
            kwargs_retry.pop("tool_choice", None)
            kwargs_retry["tools"][0]["description"] += " Output ONLY valid JSON."
            resp2 = client.messages.create(**kwargs_retry)
            result = _extract_tool_output(resp2, schema)
        value_str = result.model_dump_json()
        set_llm_cache(content_hash, value_str)
        return result

    # ── text response with optional auto-continuation ─────────────────────────
    text = "".join(b.text for b in response.content if b.type == "text")
    stop_reason = response.stop_reason

    if continue_on_truncation and stop_reason == "max_tokens":
        logger.warning(
            "Response truncated (max_tokens=%d) for model=%s — continuing (up to %d times)",
            max_tokens, model, max_continuations,
        )
        cont_messages = list(messages) + [
            {"role": "assistant", "content": text},
            {"role": "user", "content": [text_block("Please continue from where you left off.")]},
        ]
        for i in range(max_continuations):
            cont_kwargs = {**kwargs, "messages": cont_messages}
            cont_resp = client.messages.create(**cont_kwargs)
            cont_text = "".join(b.text for b in cont_resp.content if b.type == "text")
            text += cont_text
            stop_reason = cont_resp.stop_reason
            if stop_reason != "max_tokens":
                break
            logger.warning("Still truncated after continuation %d/%d", i + 1, max_continuations)
            cont_messages = cont_messages + [
                {"role": "assistant", "content": cont_text},
                {"role": "user", "content": [text_block("Please continue.")]},
            ]
        if stop_reason == "max_tokens":
            logger.error("Response still truncated after %d continuations", max_continuations)

    if stop_reason == "end_turn":
        set_llm_cache(content_hash, text)
    else:
        logger.warning(
            "NOT caching response with stop_reason=%s (model=%s, prompt_version=%s)",
            stop_reason, model, prompt_version,
        )

    return text


# ── Streaming ─────────────────────────────────────────────────────────────────

def _normalise_system(system: Optional[Union[str, list]]) -> Optional[list]:
    if system is None:
        return None
    if isinstance(system, str):
        return [{"type": "text", "text": system}]
    return system


def _stream(
    model: str,
    messages: list,
    *,
    system: Optional[Union[str, list]] = None,
    max_tokens: int = 4096,
    temperature: float = 0.0,
) -> Iterator[str]:
    """Yields text tokens. Logs usage after stream completes.
    If stop_reason is 'max_tokens', emits a visible truncation marker instead of
    silently cutting off mid-sentence.
    """
    client = _get_client()
    sys_param = _normalise_system(system)
    kwargs: dict = {"model": model, "max_tokens": max_tokens, "messages": messages}
    if sys_param:
        kwargs["system"] = sys_param
    if temperature != 0.0:
        kwargs["temperature"] = temperature

    with client.messages.stream(**kwargs) as stream:
        yield from stream.text_stream
        final = stream.get_final_message()
        usage = final.usage
        tokens_in   = usage.input_tokens
        tokens_out  = usage.output_tokens
        cache_read  = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cost        = estimate_cost(model, tokens_in, tokens_out, cache_read, cache_write)
        log_llm_call(model, "stream", "stream", tokens_in, tokens_out,
                     cache_read, cache_write, cost, False, _get_session_id())
        if final.stop_reason == "max_tokens":
            logger.warning(
                "Streaming response truncated at max_tokens=%d for model=%s",
                max_tokens, model,
            )
            yield (
                "\n\n---\n> ⚠️ **Response truncated** — hit the output token limit "
                f"({max_tokens} tokens). Increase `max_tokens` in the calling function "
                "or reduce the input context.\n"
            )


# ── Web search synthesis ─────────────────────────────────────────────────────

_WEB_SEARCH_TOOL: list = [{"type": "web_search_20250305", "name": "web_search"}]


def web_search_synthesis(
    prompt: str,
    *,
    cache_key: str = "",
    max_turns: int = 4,
    max_tokens: int = 1500,
) -> str:
    """
    Run a web-search-augmented synthesis using Claude's built-in web search.
    Returns the final text response. Cached by cache_key if provided (TTL=6h).
    Falls back to empty string on any error so callers can degrade gracefully.
    """
    from config import TTL_NEWS
    from data.cache import get_cache_obj, set_cache_obj

    if cache_key:
        cached = get_cache_obj(cache_key)
        if cached:
            return cached

    client = _get_client()
    messages: list = [{"role": "user", "content": prompt}]
    result_text = ""

    for _ in range(max_turns):
        try:
            response = client.messages.create(
                model=SONNET,
                max_tokens=max_tokens,
                tools=_WEB_SEARCH_TOOL,
                messages=messages,
            )
        except Exception as exc:
            logger.warning("web_search_synthesis API error: %s", exc)
            break

        for block in response.content:
            if getattr(block, "type", None) == "text":
                result_text = block.text

        if response.stop_reason == "end_turn":
            break

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            # For server-side web_search, the tool result is handled by Anthropic;
            # we pass back any tool_result blocks that arrived in the response.
            tool_results = []
            for block in response.content:
                block_type = getattr(block, "type", "")
                if block_type == "tool_use":
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": getattr(block, "content", "") or "",
                    })
            if tool_results:
                messages.append({"role": "user", "content": tool_results})
            else:
                break
        else:
            break

    if cache_key and result_text:
        set_cache_obj(cache_key, result_text, TTL_NEWS)

    usage_est = estimate_cost(SONNET, sum(
        getattr(m, "input_tokens", 0) for m in []
    ), 0)
    logger.info("web_search_synthesis done: %d chars", len(result_text))
    return result_text


# ── Batch API ─────────────────────────────────────────────────────────────────

def _submit_batch(
    model: str,
    messages: list,
    *,
    system: Optional[Union[str, list]] = None,
    schema: Optional[Type[BaseModel]] = None,
    max_tokens: int = 4096,
) -> str:
    """Submit single-item batch. Returns batch_id."""
    client = _get_client()
    params: dict = {"model": model, "max_tokens": max_tokens, "messages": messages}
    sys_param = _normalise_system(system)
    if sys_param:
        params["system"] = sys_param
    if schema:
        tools, tool_choice = _build_tool(schema)
        params["tools"] = tools
        params["tool_choice"] = tool_choice

    batch = client.messages.batches.create(requests=[{
        "custom_id": f"req-{int(time.time() * 1000)}",
        "params": params,
    }])
    return batch.id


def batch_create(requests: list[dict]) -> str:
    """
    Create a multi-request batch.
    Each request dict: {custom_id, model, messages, max_tokens, system?, schema?}
    Returns batch_id.
    """
    client = _get_client()
    batch_reqs = []
    for req in requests:
        schema = req.pop("schema", None)
        custom_id = req.pop("custom_id", f"req-{int(time.time() * 1000)}")
        sys_param = _normalise_system(req.pop("system", None))
        params = dict(req)
        if sys_param:
            params["system"] = sys_param
        if schema:
            tools, tool_choice = _build_tool(schema)
            params["tools"] = tools
            params["tool_choice"] = tool_choice
        batch_reqs.append({"custom_id": custom_id, "params": params})

    batch = client.messages.batches.create(requests=batch_reqs)
    return batch.id


def batch_poll(
    batch_id: str,
    timeout: int = 7200,
    poll_interval: int = 30,
) -> list[tuple[str, Optional[Any]]]:
    """
    Poll a batch until complete or timeout.
    Returns [(custom_id, message_or_None), ...].
    """
    client = _get_client()
    deadline = time.time() + timeout
    while time.time() < deadline:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            break
        time.sleep(poll_interval)

    results = []
    for item in client.messages.batches.results(batch_id):
        if item.result.type == "succeeded":
            results.append((item.custom_id, item.result.message))
        else:
            logger.warning("Batch item %s: %s", item.custom_id, item.result.type)
            results.append((item.custom_id, None))
    return results


def batch_extract_schema(
    message: Any,
    schema: Type[BaseModel],
) -> Optional[BaseModel]:
    """Extract structured output from a batch message response."""
    try:
        return _extract_tool_output(message, schema)
    except (ValueError, ValidationError) as exc:
        logger.warning("batch_extract_schema failed: %s", exc)
        return None
