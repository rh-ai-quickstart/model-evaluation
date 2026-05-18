# This project was developed with assistance from AI tools.
"""Safety service -- input/output filtering via Llama Guard on MaaS."""

import logging
from dataclasses import dataclass

import httpx

from ..core.config import settings

logger = logging.getLogger(__name__)

SAFETY_TIMEOUT = 30.0  # seconds -- guard models are fast

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=SAFETY_TIMEOUT)
    return _client


@dataclass
class SafetyResult:
    """Result of a safety check."""

    is_safe: bool
    category: str | None = None
    raw_response: str | None = None


def _safety_configured() -> bool:
    """Return True if a safety model is configured and enabled."""
    return bool(settings.SAFETY_MODEL_NAME) and settings.SAFETY_ENABLED


async def check_input_safety(text: str) -> SafetyResult:
    """Check whether user input is safe to process.

    Returns SafetyResult(is_safe=True) if the safety model is not
    configured or if the check fails (graceful degradation).
    """
    if not _safety_configured():
        return SafetyResult(is_safe=True)
    return await _call_guard(text, role="user")


async def check_output_safety(text: str) -> SafetyResult:
    """Check whether model output is safe to return.

    Returns SafetyResult(is_safe=True) if the safety model is not
    configured or if the check fails (graceful degradation).
    """
    if not _safety_configured():
        return SafetyResult(is_safe=True)
    return await _call_guard(text, role="assistant")


async def _call_guard(text: str, role: str) -> SafetyResult:
    """Call the Llama Guard model to classify text safety.

    Llama Guard responds with "safe" or "unsafe\\n<category>".
    On any failure, returns is_safe=True to avoid blocking the pipeline.
    """
    model_cfg = settings.get_model_config(settings.SAFETY_MODEL_NAME)
    if not model_cfg["token"]:
        logger.warning("No API token for safety model, skipping check")
        return SafetyResult(is_safe=True)

    url = f"{model_cfg['endpoint']}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {model_cfg['token']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.SAFETY_MODEL_NAME,
        "messages": [{"role": role, "content": text}],
        "temperature": 0.0,
        "max_tokens": 100,
    }

    try:
        client = _get_client()
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()

        data = response.json()
        output = data["choices"][0]["message"]["content"].strip().lower()

        if output.startswith("safe"):
            return SafetyResult(is_safe=True, raw_response=output)

        # Llama Guard returns "unsafe\n<category>" (e.g. "unsafe\nS1")
        category = None
        if "\n" in output:
            category = output.split("\n", 1)[1].strip() or None
        return SafetyResult(is_safe=False, category=category, raw_response=output)

    except httpx.TimeoutException:
        logger.warning("Safety check timed out, proceeding without filtering")
        return SafetyResult(is_safe=True)
    except httpx.HTTPStatusError as e:
        logger.warning(
            "Safety model returned HTTP %s, proceeding without filtering",
            e.response.status_code,
        )
        return SafetyResult(is_safe=True)
    except Exception as e:
        logger.warning("Safety check failed (%s), proceeding without filtering", e)
        return SafetyResult(is_safe=True)
