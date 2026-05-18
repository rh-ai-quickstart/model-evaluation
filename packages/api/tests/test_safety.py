# This project was developed with assistance from AI tools.
"""Tests for the safety service (Llama Guard integration)."""

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.services.safety import (
    check_input_safety,
    check_output_safety,
)


@pytest.fixture
def _safety_configured():
    """Configure a safety model for tests."""
    with patch("src.services.safety.settings") as mock_settings:
        mock_settings.SAFETY_MODEL_NAME = "llama-guard-3"
        mock_settings.SAFETY_ENABLED = True
        mock_settings.get_model_config.return_value = {
            "endpoint": "https://maas.example.com",
            "token": "test-token",
        }
        yield mock_settings


@pytest.fixture
def _safety_disabled():
    """Safety model configured but disabled."""
    with patch("src.services.safety.settings") as mock_settings:
        mock_settings.SAFETY_MODEL_NAME = "llama-guard-3"
        mock_settings.SAFETY_ENABLED = False
        yield mock_settings


@pytest.fixture
def _no_safety_model():
    """No safety model configured."""
    with patch("src.services.safety.settings") as mock_settings:
        mock_settings.SAFETY_MODEL_NAME = ""
        mock_settings.SAFETY_ENABLED = True
        yield mock_settings


def _mock_guard_response(content: str):
    """Create a mock httpx.Response with Llama Guard output."""
    mock_response = AsyncMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": content}}],
    }
    mock_response.raise_for_status = lambda: None
    return mock_response


# --- Core behavior tests ---


@pytest.mark.asyncio
async def test_safe_input_returns_is_safe(_safety_configured):
    """Should return is_safe=True when guard model responds 'safe'."""
    mock_client = AsyncMock()
    mock_client.post.return_value = _mock_guard_response("safe")
    mock_client.is_closed = False

    with patch("src.services.safety._get_client", return_value=mock_client):
        result = await check_input_safety("What are the capital requirements?")

    assert result.is_safe is True
    assert result.category is None


@pytest.mark.asyncio
async def test_unsafe_input_returns_category(_safety_configured):
    """Should return is_safe=False with category when guard flags content."""
    mock_client = AsyncMock()
    mock_client.post.return_value = _mock_guard_response("unsafe\nS1")
    mock_client.is_closed = False

    with patch("src.services.safety._get_client", return_value=mock_client):
        result = await check_input_safety("harmful content here")

    assert result.is_safe is False
    assert result.category == "s1"


@pytest.mark.asyncio
async def test_output_safety_checks_assistant_role(_safety_configured):
    """Should call guard with assistant role for output checks."""
    mock_client = AsyncMock()
    mock_client.post.return_value = _mock_guard_response("safe")
    mock_client.is_closed = False

    with patch("src.services.safety._get_client", return_value=mock_client):
        result = await check_output_safety("This is a safe response.")

    assert result.is_safe is True
    call_kwargs = mock_client.post.call_args
    payload = call_kwargs[1]["json"] if "json" in call_kwargs[1] else call_kwargs.kwargs["json"]
    assert payload["messages"][0]["role"] == "assistant"


# --- Graceful degradation tests ---


@pytest.mark.asyncio
async def test_skips_when_no_safety_model(_no_safety_model):
    """Should return safe when no safety model is configured."""
    result = await check_input_safety("anything")
    assert result.is_safe is True


@pytest.mark.asyncio
async def test_skips_when_safety_disabled(_safety_disabled):
    """Should return safe when safety is disabled."""
    result = await check_input_safety("anything")
    assert result.is_safe is True


@pytest.mark.asyncio
async def test_returns_safe_on_timeout(_safety_configured):
    """Should degrade gracefully on timeout."""
    mock_client = AsyncMock()
    mock_client.post.side_effect = httpx.TimeoutException("timed out")
    mock_client.is_closed = False

    with patch("src.services.safety._get_client", return_value=mock_client):
        result = await check_input_safety("test question")

    assert result.is_safe is True


@pytest.mark.asyncio
async def test_returns_safe_on_http_error(_safety_configured):
    """Should degrade gracefully on HTTP error."""
    mock_client = AsyncMock()
    mock_response = AsyncMock(spec=httpx.Response)
    mock_response.status_code = 500
    mock_client.post.side_effect = httpx.HTTPStatusError(
        "server error", request=AsyncMock(), response=mock_response
    )
    mock_client.is_closed = False

    with patch("src.services.safety._get_client", return_value=mock_client):
        result = await check_output_safety("test response")

    assert result.is_safe is True


@pytest.mark.asyncio
async def test_returns_safe_on_no_token(_safety_configured):
    """Should degrade gracefully when no API token is available."""
    _safety_configured.get_model_config.return_value = {
        "endpoint": "https://maas.example.com",
        "token": "",
    }
    result = await check_input_safety("test")
    assert result.is_safe is True
