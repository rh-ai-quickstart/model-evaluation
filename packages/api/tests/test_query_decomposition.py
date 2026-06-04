"""Tests for query decomposition service."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.query_decomposition import (
    _decomposition_cache,
    decompose_query,
)

# A question long enough (8+ words) with broad signals to pass the gate
BROAD_QUESTION = (
    "What are the key requirements for ETF regulatory compliance across all documents?"
)

# A narrow question (short, no broad signals) that should be skipped
NARROW_QUESTION = "What is Rule 6c-11?"


@pytest.fixture(autouse=True)
def _reset_settings_and_cache():
    """Ensure settings and cache are restored after each test."""
    from src.core.config import settings

    original_token = settings.API_TOKEN
    original_maas = settings.MAAS_ENDPOINT
    original_judge = settings.JUDGE_MODEL_NAME
    _decomposition_cache.clear()
    yield
    settings.API_TOKEN = original_token
    settings.MAAS_ENDPOINT = original_maas
    settings.JUDGE_MODEL_NAME = original_judge
    _decomposition_cache.clear()


def test_returns_original_when_no_model():
    """Should return original question when no model is configured."""
    from src.core.config import settings

    settings.JUDGE_MODEL_NAME = ""
    settings.API_TOKEN = ""

    result = asyncio.run(decompose_query(BROAD_QUESTION))
    assert result == [BROAD_QUESTION]


def test_returns_original_when_no_token():
    """Should return original question when no API token is set."""
    from src.core.config import settings

    settings.JUDGE_MODEL_NAME = "test-model"
    settings.API_TOKEN = ""

    result = asyncio.run(decompose_query(BROAD_QUESTION))
    assert result == [BROAD_QUESTION]


def test_returns_sub_queries_on_success():
    """Should return parsed sub-queries from LLM response."""
    from src.core.config import settings

    settings.API_TOKEN = "test-token"
    settings.MAAS_ENDPOINT = "https://example.com"
    settings.JUDGE_MODEL_NAME = "test-model"

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": '["What is Rule 6c-11?", "What are SAI requirements?", "What are N-PORT filings?"]'
                }
            }
        ]
    }

    with patch("src.services.query_decomposition.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = asyncio.run(decompose_query(BROAD_QUESTION))

    assert len(result) == 3
    assert "Rule 6c-11" in result[0]
    assert "SAI" in result[1]


def test_returns_original_on_invalid_json():
    """Should fall back to original question when LLM returns invalid JSON."""
    from src.core.config import settings

    settings.API_TOKEN = "test-token"
    settings.MAAS_ENDPOINT = "https://example.com"
    settings.JUDGE_MODEL_NAME = "test-model"

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "This is not JSON"}}]
    }

    with patch("src.services.query_decomposition.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = asyncio.run(decompose_query(BROAD_QUESTION))

    assert result == [BROAD_QUESTION]


def test_returns_original_on_api_error():
    """Should fall back to original question when API call fails."""
    from src.core.config import settings

    settings.API_TOKEN = "test-token"
    settings.MAAS_ENDPOINT = "https://example.com"
    settings.JUDGE_MODEL_NAME = "test-model"

    import httpx

    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Server Error", request=MagicMock(), response=mock_response
    )

    with patch("src.services.query_decomposition.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = asyncio.run(decompose_query(BROAD_QUESTION))

    assert result == [BROAD_QUESTION]


def test_respects_max_sub_queries():
    """Should cap the number of sub-queries returned."""
    from src.core.config import settings

    settings.API_TOKEN = "test-token"
    settings.MAAS_ENDPOINT = "https://example.com"
    settings.JUDGE_MODEL_NAME = "test-model"

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": '["q1", "q2", "q3", "q4", "q5", "q6", "q7"]'
                }
            }
        ]
    }

    with patch("src.services.query_decomposition.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = asyncio.run(decompose_query(BROAD_QUESTION, max_sub_queries=3))

    assert len(result) <= 3


def test_returns_original_on_empty_array():
    """Should fall back when LLM returns empty array."""
    from src.core.config import settings

    settings.API_TOKEN = "test-token"
    settings.MAAS_ENDPOINT = "https://example.com"
    settings.JUDGE_MODEL_NAME = "test-model"

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "[]"}}]
    }

    with patch("src.services.query_decomposition.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = asyncio.run(decompose_query(BROAD_QUESTION))

    assert result == [BROAD_QUESTION]


# --- Decomposition gate tests ---


def test_skips_short_questions():
    """Should skip decomposition for questions with fewer than 8 words."""
    from src.core.config import settings

    settings.API_TOKEN = "test-token"
    settings.MAAS_ENDPOINT = "https://example.com"
    settings.JUDGE_MODEL_NAME = "test-model"

    result = asyncio.run(decompose_query(NARROW_QUESTION))
    assert result == [NARROW_QUESTION]


def test_skips_questions_without_broad_signals():
    """Should skip decomposition when no broad-signal keywords are detected."""
    from src.core.config import settings

    settings.API_TOKEN = "test-token"
    settings.MAAS_ENDPOINT = "https://example.com"
    settings.JUDGE_MODEL_NAME = "test-model"

    # 8+ words but no broad-signal keywords
    question = "Does the filing deadline apply to quarterly submissions this year?"
    result = asyncio.run(decompose_query(question))
    assert result == [question]


# --- Cache tests ---


def test_cache_returns_cached_result():
    """Should return cached result without calling LLM again."""
    from src.core.config import settings

    settings.API_TOKEN = "test-token"
    settings.MAAS_ENDPOINT = "https://example.com"
    settings.JUDGE_MODEL_NAME = "test-model"

    # Pre-populate cache
    cached_result = ["sub-q1", "sub-q2"]
    _decomposition_cache[BROAD_QUESTION] = cached_result

    # No mock needed -- should never call the LLM
    result = asyncio.run(decompose_query(BROAD_QUESTION))
    assert result == cached_result


def test_cache_populated_after_success():
    """Should cache results after successful decomposition."""
    from src.core.config import settings

    settings.API_TOKEN = "test-token"
    settings.MAAS_ENDPOINT = "https://example.com"
    settings.JUDGE_MODEL_NAME = "test-model"

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "choices": [
            {"message": {"content": '["sub-q1", "sub-q2"]'}}
        ]
    }

    with patch("src.services.query_decomposition.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        asyncio.run(decompose_query(BROAD_QUESTION))

    assert BROAD_QUESTION in _decomposition_cache
    assert _decomposition_cache[BROAD_QUESTION] == ["sub-q1", "sub-q2"]


# --- JSON repair fallback tests ---


def test_parse_strips_markdown_fencing():
    """Should parse JSON wrapped in markdown code fences."""
    from src.services.query_decomposition import _parse_decomposition_json

    result = _parse_decomposition_json('```json\n["q1", "q2"]\n```')
    assert result == ["q1", "q2"]


def test_parse_repairs_trailing_comma():
    """Should parse JSON with trailing commas."""
    from src.services.query_decomposition import _parse_decomposition_json

    result = _parse_decomposition_json('["q1", "q2",]')
    assert result == ["q1", "q2"]


def test_parse_repairs_missing_comma():
    """Should parse JSON with missing commas between lines."""
    from src.services.query_decomposition import _parse_decomposition_json

    raw = '[\n"q1"\n"q2"\n]'
    result = _parse_decomposition_json(raw)
    assert result is not None
    assert len(result) >= 2


def test_parse_regex_fallback():
    """Should extract quoted strings when JSON is completely malformed."""
    from src.services.query_decomposition import _parse_decomposition_json

    raw = 'Here are the sub-queries:\n1. "What is Rule 6c-11?"\n2. "What are SAI requirements?"'
    result = _parse_decomposition_json(raw)
    assert result is not None
    assert len(result) == 2
    assert "Rule 6c-11" in result[0]


def test_parse_returns_none_for_unparseable():
    """Should return None when no strings can be extracted."""
    from src.services.query_decomposition import _parse_decomposition_json

    result = _parse_decomposition_json("no json here at all")
    assert result is None
