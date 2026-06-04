"""Tests for generation service."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.generation import (
    _build_context_block,
    _build_generation_payload,
    _strip_reasoning_blocks,
    generate_answer,
)


@pytest.fixture(autouse=True)
def _reset_settings():
    """Ensure settings are restored after each test."""
    from src.core.config import settings

    original_token = settings.API_TOKEN
    original_debug = settings.DEBUG
    yield
    settings.API_TOKEN = original_token
    settings.DEBUG = original_debug


def _mock_chat_response(
    answer_text: str,
    total_tokens: int = 110,
    reasoning_content: str | None = None,
) -> MagicMock:
    """Build a successful MaaS completion response mock."""
    response = MagicMock()
    response.status_code = 200
    response.raise_for_status = MagicMock()
    message = {"content": answer_text}
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    response.json.return_value = {
        "choices": [{"message": message}],
        "usage": {
            "prompt_tokens": max(total_tokens - 10, 1),
            "completion_tokens": min(10, total_tokens),
            "total_tokens": total_tokens,
        },
    }
    return response


def test_build_generation_payload_respects_custom_max_tokens():
    """Profile-driven answers should pass through a higher completion budget."""
    payload, _, max_tokens = _build_generation_payload(
        question="What must be disclosed?",
        chunks=[{"source_document": "doc.pdf", "page_number": "1", "text": "context"}],
        model_name="granite-test",
        system_prompt=None,
        attempt=0,
        base_max_tokens=2048,
    )
    assert payload["max_tokens"] == max_tokens == 2048


def test_build_context_block_formats_chunks():
    """Should format chunks with source headers."""
    chunks = [
        {"source_document": "report.pdf", "page_number": "3", "text": "Revenue grew 10%."},
        {"source_document": "guide.pdf", "page_number": None, "text": "See section 4."},
    ]
    result = _build_context_block(chunks)
    assert "[Source 1: report.pdf, page 3]" in result
    assert "Revenue grew 10%." in result
    assert "[Source 2: guide.pdf]" in result
    assert "page" not in result.split("[Source 2:")[1].split("]")[0]


def test_strip_reasoning_blocks_removes_complete_think_block():
    """Should remove reasoning blocks before scoring model answers."""
    raw = "<think>work through sources</think>\n\nThe answer is 42."
    assert _strip_reasoning_blocks(raw) == "The answer is 42."


def test_strip_reasoning_blocks_removes_orphan_closing_think_block():
    """DeepSeek-style orphan closing tags should drop leading reasoning."""
    raw = "I will reason through this first.\n</think>\n\nThe requirement prevents dumping."
    assert _strip_reasoning_blocks(raw) == "The requirement prevents dumping."


def test_strip_reasoning_blocks_removes_redacted_thinking_pair():
    """Moderated/stack paths may wrap reasoning in redacted_thinking tags."""
    raw = (
        "<"
        + "redacted_thinking"
        + ">hidden</"
        + "redacted_thinking"
        + ">\n\nVisible answer."
    )
    assert _strip_reasoning_blocks(raw) == "Visible answer."


def test_build_generation_payload_uses_default_max_tokens():
    """Default max_tokens should be 2048 when no base_max_tokens is provided."""
    payload, _, max_tokens = _build_generation_payload(
        question="What?",
        chunks=[{"source_document": "doc.pdf", "page_number": "1", "text": "context"}],
        model_name="granite-test",
        system_prompt=None,
        attempt=0,
    )
    assert payload["max_tokens"] == max_tokens == 2048


def test_strip_reasoning_blocks_only_think_block():
    """Should return empty string when answer is only a reasoning block."""
    assert _strip_reasoning_blocks("<think>reasoning only</think>") == ""


def test_strip_reasoning_blocks_untagged_deepseek_distill():
    """Should strip untagged chain-of-thought from deepseek-r1-distill models."""
    raw = (
        "Okay, so I need to figure out the disclosure requirements for an ETF "
        "filing based on the provided context documents. Let me go through each "
        "source one by one to gather the necessary information.\n\n"
        "Starting with Source 1: It talks about Rule 6c-11 and how it provides "
        "exemptive relief to ETFs.\n\n"
        "Source 2 is from Form N-1A and outlines the requirements.\n\n"
        "Based on the provided context, the key disclosure requirements are:\n\n"
        "1. Prospectus disclosure of fund characteristics and risks in plain language\n"
        "2. Portfolio holdings transparency via daily website disclosure\n"
        "3. SAI structure and organization with table of contents"
    )
    result = _strip_reasoning_blocks(raw)
    assert result.startswith("1.")
    assert "Okay, so I need to" not in result
    assert "Source 1:" not in result
    assert "Based on the provided context" not in result


def test_strip_reasoning_blocks_untagged_with_summary_transition():
    """Should detect 'In summary' transition after untagged reasoning."""
    raw = (
        "Let me think through this step by step and analyze the sources.\n\n"
        "The first document discusses registration requirements.\n\n"
        "The second document covers compliance obligations.\n\n"
        "In summary, ETFs must comply with three main requirements:\n\n"
        "1. File registration statements with the SEC under the Securities Act of 1933\n"
        "2. Maintain daily transparency for portfolio holdings on a public website\n"
        "3. Provide standardized disclosures using Form N-1A including prospectus and SAI"
    )
    result = _strip_reasoning_blocks(raw)
    assert result.startswith("1.")
    assert "Let me think" not in result
    assert "In summary" not in result


def test_strip_reasoning_blocks_strips_trailing_reasoning():
    """Should remove trailing 'I should/I need to' fragments."""
    raw = (
        "Okay, so I need to figure out the ETF requirements by analyzing the sources.\n\n"
        "Putting it all together, here are the requirements:\n\n"
        "1. **Prospectus Requirements**: Clear disclosure of fund characteristics, "
        "risks, investment strategy, and balanced information.\n\n"
        "2. **Portfolio Transparency**: ETFs must disclose their complete daily "
        "portfolio holdings or index components on a website.\n\n"
        "I should make sure to structure the answer to cover each of these points clearly"
    )
    result = _strip_reasoning_blocks(raw)
    assert result.startswith("1.")
    assert "I should make sure" not in result
    assert "Putting it all together" not in result


def test_strip_reasoning_blocks_preserves_normal_answer():
    """Should not strip normal answers that don't start with reasoning markers."""
    answer = (
        "ETFs are required to file registration statements with the SEC. "
        "The key disclosure requirements include prospectus delivery, "
        "portfolio transparency, and standardized reporting formats."
    )
    assert _strip_reasoning_blocks(answer) == answer


def test_strip_reasoning_blocks_preserves_short_reasoning_like_text():
    """Should not strip when transition leads to a too-short answer fragment."""
    raw = (
        "Okay, let me figure out the answer from the sources.\n\n"
        "Based on the context, yes."
    )
    result = _strip_reasoning_blocks(raw)
    assert "Okay" in result


def test_returns_error_when_no_token():
    """Should return a helpful message when no API token is configured."""
    from src.core.config import settings

    settings.API_TOKEN = ""

    import asyncio

    result = asyncio.run(generate_answer("What is X?", [], "granite-3.1-8b-instruct"))
    assert "No API token configured" in result["answer"]
    assert result["model"] == "granite-3.1-8b-instruct"


def test_returns_answer_on_success():
    """Should return the generated answer from the LLM."""
    from src.core.config import settings

    settings.API_TOKEN = "test-token"

    mock_response = _mock_chat_response("The answer is 42.")

    import asyncio

    with patch("src.services.generation.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = asyncio.run(
            generate_answer(
                "What is the meaning?",
                [{"source_document": "doc.pdf", "page_number": "1", "text": "context"}],
                "granite-3.1-8b-instruct",
                max_tokens=2048,
            )
        )

        call_kw = mock_client.post.call_args
        assert call_kw[1]["json"]["max_tokens"] == 2048

    assert result["answer"] == "The answer is 42."
    assert result["model"] == "granite-3.1-8b-instruct"
    assert result["usage"]["total_tokens"] == 110


def test_returns_error_on_api_failure():
    """Should return error message when LLM API fails."""
    from src.core.config import settings

    settings.API_TOKEN = "test-token"

    import asyncio

    import httpx

    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Server Error", request=MagicMock(), response=mock_response
    )

    with patch("src.services.generation.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = asyncio.run(generate_answer("What?", [], "granite-3.1-8b-instruct"))

    assert "error" in result["answer"].lower() or "500" in result["answer"]


def test_skips_strip_when_reasoning_content_field_present():
    """Should not strip content when vLLM reasoning parser separated it."""
    from src.core.config import settings

    settings.API_TOKEN = "test-token"

    mock_response = _mock_chat_response(
        "The answer is 42.",
        reasoning_content="Let me figure this out step by step...",
    )

    import asyncio

    with patch("src.services.generation.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = asyncio.run(
            generate_answer(
                "What is the meaning?",
                [{"source_document": "doc.pdf", "page_number": "1", "text": "ctx"}],
                "deepseek-r1-distill-qwen-14b",
            )
        )

    assert result["answer"] == "The answer is 42."


def test_http_error_includes_upstream_message_when_debug():
    """When DEBUG is on, surface a short summary from the gateway error body."""
    from src.core.config import settings

    settings.API_TOKEN = "test-token"
    settings.DEBUG = True

    import asyncio

    import httpx

    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = '{"error":{"message":"model not found","type":"invalid_request"}}'
    mock_response.json.return_value = {
        "error": {"message": "model not found", "type": "invalid_request"}
    }
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Server Error", request=MagicMock(), response=mock_response
    )

    with patch("src.services.generation.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = asyncio.run(generate_answer("What?", [], "granite-3.1-8b-instruct"))

    assert "500" in result["answer"]
    assert "model not found" in result["answer"]
