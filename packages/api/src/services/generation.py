# This project was developed with assistance from AI tools.
"""Generation service -- calls the LLM via the MaaS /v1/chat/completions endpoint."""

import asyncio
import logging
import re

import httpx

from ..core.config import settings

logger = logging.getLogger(__name__)

GENERATION_TIMEOUT = 60.0  # seconds

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=GENERATION_TIMEOUT)
    return _client
_MAX_RETRIES = 3
_RETRY_BACKOFF = 2.0  # seconds, doubles each attempt
_DEBUG_ERROR_SNIPPET_LEN = 500
_DEFAULT_MAX_TOKENS = 2048
_MIN_RETRY_MAX_TOKENS = 512
_MIN_RETRY_CHUNKS = 4


def _summarize_upstream_error(response: httpx.Response) -> str:
    """Best-effort extract a short message from an OpenAI-style error body."""
    text = (response.text or "").strip()
    try:
        data = response.json()
    except Exception:
        return text[:_DEBUG_ERROR_SNIPPET_LEN] + (
            "..." if len(text) > _DEBUG_ERROR_SNIPPET_LEN else ""
        )

    err = data.get("error")
    if isinstance(err, dict):
        for key in ("message", "detail", "msg"):
            val = err.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()[:_DEBUG_ERROR_SNIPPET_LEN]
    if isinstance(err, str) and err.strip():
        return err.strip()[:_DEBUG_ERROR_SNIPPET_LEN]

    detail = data.get("detail")
    if isinstance(detail, str) and detail.strip():
        return detail.strip()[:_DEBUG_ERROR_SNIPPET_LEN]

    return text[:_DEBUG_ERROR_SNIPPET_LEN] + ("..." if len(text) > _DEBUG_ERROR_SNIPPET_LEN else "")


SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based on the provided context. "
    "Use only the information from the context to answer. If the context does not contain "
    "enough information to answer the question, say so clearly.\n\n"
    "RESPONSE RULE:\n"
    "Answer using the provided context and ONLY the provided context.\n"
    "- Lead with the direct answer, then provide supporting detail.\n"
    "- Do NOT add any information, citations, item numbers, rule numbers, section "
    "references, or form names from your own knowledge.\n"
    "- If a specific reference does not appear in the context text, do not include it. "
    "Describe the requirement in general terms instead.\n"
    "- Organize by topic, not by source document.\n"
    "- Cite the source document name and page number when they appear in the context.\n"
    "- Summarize key points rather than copying long passages verbatim.\n"
    "- If the context does not contain enough information, say so clearly."
)


def _build_context_block(chunks: list[dict]) -> str:
    """Format retrieved chunks into a context block for the prompt."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk["source_document"]
        page = chunk.get("page_number")
        header = f"[Source {i}: {source}"
        if page:
            header += f", page {page}"
        header += "]"
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n".join(parts)


def _build_generation_payload(
    *,
    question: str,
    chunks: list[dict],
    model_name: str,
    system_prompt: str | None,
    attempt: int,
    base_max_tokens: int | None = None,
) -> tuple[dict, int, int]:
    """Build request payload, shrinking context and max_tokens on retries."""
    if attempt <= 0:
        prompt_chunks = chunks
    else:
        # Keep the highest-ranked chunks and reduce prompt size after each transport failure.
        drop_per_retry = max(1, len(chunks) // 4)
        keep = max(_MIN_RETRY_CHUNKS, len(chunks) - (drop_per_retry * attempt))
        prompt_chunks = chunks[:keep]

    context = _build_context_block(prompt_chunks)
    user_message = f"Context:\n{context}\n\nQuestion: {question}"
    cap = base_max_tokens if base_max_tokens is not None else _DEFAULT_MAX_TOKENS
    max_tokens = max(_MIN_RETRY_MAX_TOKENS, cap // (2**attempt))
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    return payload, len(prompt_chunks), max_tokens


def _strip_reasoning_blocks(answer: str) -> str:
    """Remove model-visible reasoning blocks from final answers.

    Strips paired short/long ``think`` / ``redacted_thinking`` blocks and
    drops any text before the last orphan closing tag so automated metrics
    score only the answer body.
    """
    text = (answer or "").strip()

    for pattern in (
        r"<think\b[^>]*>.*?</think>\s*",
        r"<redacted_thinking\b[^>]*>.*?</redacted_thinking>\s*",
    ):
        text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)

    lower = text.lower()
    for closing in ("</think>", "</redacted_thinking>"):
        if closing in lower:
            end = lower.rfind(closing) + len(closing)
            text = text[end:]
            lower = text.lower()

    return text.strip()


async def generate_answer(
    question: str,
    chunks: list[dict],
    model_name: str,
    system_prompt: str | None = None,
    max_tokens: int | None = None,
) -> dict:
    """Generate an answer using the specified model with retrieved context.

    Args:
        question: The user's question.
        chunks: Retrieved context chunks from the retrieval service.
        model_name: Name of the model to use (e.g. granite-3.1-8b-instruct).
        system_prompt: Optional override for the system prompt. When provided
            (e.g. from an evaluation profile), replaces the default prompt.
        max_tokens: Optional cap on completion tokens (defaults to a conservative
            baseline; profiles may set a higher limit for structured regulatory answers).

    Returns:
        Dict with 'answer', 'model', 'usage' keys.
    """
    model_cfg = settings.get_model_config(model_name)
    if not model_cfg["token"]:
        msg = f"No API token configured for model {model_name}."
        return {
            "answer": msg,
            "model": model_name,
            "usage": None,
            "error": msg,
        }

    url = f"{model_cfg['endpoint']}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {model_cfg['token']}",
        "Content-Type": "application/json",
        "Connection": "close",
    }

    last_error: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        payload, chunk_count, max_tokens = _build_generation_payload(
            question=question,
            chunks=chunks,
            model_name=model_name,
            system_prompt=system_prompt,
            attempt=attempt,
            base_max_tokens=max_tokens,
        )
        if attempt > 0:
            logger.info(
                "Generation retry %d/%d for model %r with %d chunks and max_tokens=%d",
                attempt + 1,
                _MAX_RETRIES,
                model_name,
                chunk_count,
                max_tokens,
            )
        try:
            client = _get_client()
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()

            data = response.json()
            answer = data["choices"][0]["message"]["content"]
            answer = _strip_reasoning_blocks(answer)
            usage = data.get("usage")

            return {
                "answer": answer,
                "model": model_name,
                "usage": usage,
                "error": None,
            }

        except httpx.HTTPStatusError as e:
            detail = _summarize_upstream_error(e.response)
            logger.error(
                "Generation API HTTP %s for model %r: %s",
                e.response.status_code,
                model_name,
                detail or "(empty body)",
            )
            msg = f"Model {model_name} returned an error: {e.response.status_code}"
            if settings.DEBUG and detail:
                msg = f"{msg}. {detail}"
            return {
                "answer": msg,
                "model": model_name,
                "usage": None,
                "error": msg,
            }
        except httpx.TransportError as e:
            last_error = e
            wait = _RETRY_BACKOFF * (2**attempt)
            logger.warning(
                "Generation attempt %d/%d failed (transport: %s)",
                attempt + 1,
                _MAX_RETRIES,
                e,
            )
            if attempt < _MAX_RETRIES - 1:
                logger.info("Retrying generation in %.1fs", wait)
                await asyncio.sleep(wait)
        except Exception as e:
            logger.error("Generation request failed: %s", e)
            return {
                "answer": f"Failed to generate answer: {e}",
                "model": model_name,
                "usage": None,
                "error": str(e),
            }

    error_msg = f"Failed to generate answer after {_MAX_RETRIES} retries: {last_error}"
    logger.error("Generation failed after %d retries: %s", _MAX_RETRIES, last_error)
    return {
        "answer": error_msg,
        "model": model_name,
        "usage": None,
        "error": error_msg,
    }
