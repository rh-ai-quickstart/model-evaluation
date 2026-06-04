"""Fetches model metadata from the LiteLLM admin API."""

import logging
import time

import httpx

from ..core.config import settings

logger = logging.getLogger(__name__)

_TIMEOUT = 15.0
_CACHE_TTL = 300  # 5 minutes

_client: httpx.AsyncClient | None = None
_cache: list[dict] | None = None
_cache_time: float = 0.0


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=_TIMEOUT)
    return _client


async def fetch_model_metadata() -> list[dict]:
    """Fetch model metadata from LiteLLM admin API with in-memory caching."""
    global _cache, _cache_time

    if not settings.LITELLM_ADMIN_URL:
        return []

    if _cache is not None and (time.monotonic() - _cache_time) < _CACHE_TTL:
        return _cache

    url = f"{settings.LITELLM_ADMIN_URL.rstrip('/')}/api/v1/models"
    headers = {"Authorization": f"Bearer {settings.api_token_bare}"}

    try:
        resp = await _get_client().get(url, params={"page": 1, "limit": 50}, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        models = []
        for item in data.get("data", []):
            models.append({
                "id": item.get("id", ""),
                "name": item.get("name", ""),
                "context_length": item.get("contextLength"),
                "max_tokens": item.get("maxTokens"),
                "pricing": item.get("pricing"),
                "capabilities": item.get("capabilities", []),
                "tpm": item.get("tpm"),
                "rpm": item.get("rpm"),
                "supports_vision": item.get("supportsVision"),
                "supports_function_calling": item.get("supportsFunctionCalling"),
                "supports_embeddings": item.get("supportsEmbeddings"),
            })

        _cache = models
        _cache_time = time.monotonic()
        return models
    except (httpx.HTTPStatusError, httpx.TransportError) as exc:
        logger.warning("Failed to fetch LiteLLM model metadata: %s", exc)
        return _cache if _cache is not None else []
