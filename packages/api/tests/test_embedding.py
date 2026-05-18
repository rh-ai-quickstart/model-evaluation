# This project was developed with assistance from AI tools.
"""Tests for embedding service."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.embedding import MAX_EMBED_CHARS, MAX_EMBED_WORDS, _truncate, generate_embeddings


@pytest.fixture(autouse=True)
def _reset_settings():
    """Ensure settings are restored after each test."""
    from src.core.config import settings

    original_token = settings.API_TOKEN
    original_maas = settings.MAAS_ENDPOINT
    original_embed = settings.EMBEDDING_MODEL
    yield
    settings.API_TOKEN = original_token
    settings.MAAS_ENDPOINT = original_maas
    settings.EMBEDDING_MODEL = original_embed


def test_truncate_char_cap_for_pdf_style_globs():
    """Long unbroken strings must be shortened (word count alone is not enough)."""
    blob = "x" * 10_000
    out = _truncate(blob)
    assert len(out) <= MAX_EMBED_CHARS


def test_truncate_word_cap():
    """Many short words must be capped at MAX_EMBED_WORDS."""
    words = "w " * (MAX_EMBED_WORDS + 200)
    out = _truncate(words.strip())
    assert len(out.split()) <= MAX_EMBED_WORDS


def test_returns_none_when_no_token():
    """Should skip embeddings when API_TOKEN is empty."""
    from src.core.config import settings

    settings.API_TOKEN = ""

    import asyncio

    result = asyncio.run(generate_embeddings(["hello"]))
    assert result.vectors is None
    assert result.error is not None
    assert "API_TOKEN" in result.error


def test_returns_embeddings_on_success():
    """Should return embeddings from the API response."""
    from src.core.config import settings

    settings.API_TOKEN = "test-token"
    settings.MAAS_ENDPOINT = "https://example.com"
    settings.EMBEDDING_MODEL = "test-embed-model"

    # Both texts fit in one batch, so the API receives a single POST.
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "data": [
            {"embedding": [0.1, 0.2, 0.3], "index": 0},
            {"embedding": [0.4, 0.5, 0.6], "index": 1},
        ]
    }

    import asyncio

    with patch("src.services.embedding.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = asyncio.run(generate_embeddings(["hello", "world"]))

    assert result.vectors == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    assert result.error is None


def test_retries_on_rate_limit_then_succeeds():
    """Should retry on 429 with exponential backoff and succeed on next attempt."""
    from src.core.config import settings

    settings.API_TOKEN = "test-token"
    settings.MAAS_ENDPOINT = "https://example.com"
    settings.EMBEDDING_MODEL = "test-embed-model"

    rate_limit_response = MagicMock()
    rate_limit_response.status_code = 429
    rate_limit_response.text = "Rate limit exceeded"

    ok_response = MagicMock()
    ok_response.status_code = 200
    ok_response.raise_for_status = MagicMock()
    ok_response.json.return_value = {"data": [{"embedding": [0.1, 0.2], "index": 0}]}

    import asyncio

    with (
        patch("src.services.embedding.httpx.AsyncClient") as mock_client_cls,
        patch("src.services.embedding.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
    ):
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=[rate_limit_response, ok_response])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = asyncio.run(generate_embeddings(["hello"]))

    assert result.vectors == [[0.1, 0.2]]
    assert result.error is None
    assert mock_client.post.call_count == 2
    mock_sleep.assert_called_once()


def test_rate_limit_retries_exhausted():
    """Should return None per-entry when all retries for a batch are exhausted."""
    from src.core.config import settings

    settings.API_TOKEN = "test-token"
    settings.MAAS_ENDPOINT = "https://example.com"
    settings.EMBEDDING_MODEL = "test-embed-model"

    import httpx as httpx_mod

    rate_limit_response = MagicMock()
    rate_limit_response.status_code = 429
    rate_limit_response.text = "Rate limit exceeded"
    rate_limit_response.reason_phrase = "Too Many Requests"
    rate_limit_response.raise_for_status.side_effect = httpx_mod.HTTPStatusError(
        "Rate limited", request=MagicMock(), response=rate_limit_response
    )

    non_raising_429 = MagicMock()
    non_raising_429.status_code = 429

    import asyncio

    from src.services.embedding import RATE_LIMIT_RETRIES

    responses = [non_raising_429] * RATE_LIMIT_RETRIES + [rate_limit_response]

    with (
        patch("src.services.embedding.httpx.AsyncClient") as mock_client_cls,
        patch("src.services.embedding.asyncio.sleep", new_callable=AsyncMock),
    ):
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=responses)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = asyncio.run(generate_embeddings(["hello"]))

    # Single batch, single text -- all failed, so entire result is None
    assert result.vectors is None
    assert result.error is not None
    assert "failed" in result.error.lower()


def test_returns_none_on_api_error():
    """Should return None per-entry when the API returns an error."""
    from src.core.config import settings

    settings.API_TOKEN = "test-token"
    settings.MAAS_ENDPOINT = "https://example.com"
    settings.EMBEDDING_MODEL = "test-embed-model"

    import asyncio

    import httpx

    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Server Error", request=MagicMock(), response=mock_response
    )

    with patch("src.services.embedding.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = asyncio.run(generate_embeddings(["hello"]))

    # Single batch, single text -- all failed, so entire result is None
    assert result.vectors is None
    assert result.error is not None
    assert "failed" in result.error.lower()


def test_partial_batch_failure_preserves_successful_embeddings():
    """Should return successful embeddings even when some batches fail."""
    from src.core.config import settings

    settings.API_TOKEN = "test-token"
    settings.MAAS_ENDPOINT = "https://example.com"
    settings.EMBEDDING_MODEL = "test-embed-model"

    import asyncio

    import httpx

    ok_response = MagicMock()
    ok_response.status_code = 200
    ok_response.raise_for_status = MagicMock()
    ok_response.json.return_value = {"data": [{"embedding": [0.1, 0.2], "index": 0}]}

    error_response = MagicMock()
    error_response.status_code = 500
    error_response.text = "Internal Server Error"
    error_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Server Error", request=MagicMock(), response=error_response
    )

    # Patch BATCH_SIZE to 1 so each text is its own batch
    with (
        patch("src.services.embedding.httpx.AsyncClient") as mock_client_cls,
        patch("src.services.embedding.BATCH_SIZE", 1),
    ):
        mock_client = AsyncMock()
        # First batch succeeds, second fails
        mock_client.post = AsyncMock(side_effect=[ok_response, error_response])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = asyncio.run(generate_embeddings(["hello", "world"]))

    assert result.vectors is not None
    assert len(result.vectors) == 2
    assert result.vectors[0] == [0.1, 0.2]
    assert result.vectors[1] is None
    assert result.error is not None
    assert "1/2 batches failed" in result.error
