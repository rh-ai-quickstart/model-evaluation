"""Tests for the RAG query endpoint (/query)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.core.config import settings
from src.services.safety import SafetyResult


@pytest.fixture(autouse=True)
def _ensure_token():
    """Ensure API_TOKEN is set for query tests."""
    original = settings.API_TOKEN
    if not settings.API_TOKEN:
        settings.API_TOKEN = "test-token"
    yield
    settings.API_TOKEN = original


# --- Tests ---


def test_query_returns_answer(client):
    """Should return an answer with sources from the RAG pipeline."""
    model_a = settings.MODEL_A_NAME
    mock_chunks = [
        {
            "id": 1,
            "text": "Revenue was $1B.",
            "source_document": "report.pdf",
            "page_number": "5",
            "score": 0.92,
        },
    ]
    mock_generation = {
        "answer": "Revenue was $1 billion.",
        "model": model_a,
        "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
    }

    with (
        patch("src.routes.query.retrieve_chunks", return_value=mock_chunks),
        patch("src.routes.query.generate_answer", return_value=mock_generation),
    ):
        response = client.post(
            "/query/",
            json={"question": "What was the revenue?"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Revenue was $1 billion."
    assert data["model"] == model_a
    assert len(data["sources"]) == 1
    assert data["sources"][0]["source_document"] == "report.pdf"
    assert data["sources"][0]["score"] == 0.92
    assert data["usage"]["total_tokens"] == 60


def test_query_with_custom_model(client):
    """Should use the specified model for generation."""
    model_b = settings.MODEL_B_NAME
    with (
        patch("src.routes.query.retrieve_chunks", return_value=[]),
        patch(
            "src.routes.query.generate_answer",
            return_value={
                "answer": "No context available.",
                "model": model_b,
                "usage": None,
            },
        ) as mock_gen,
    ):
        response = client.post(
            "/query/",
            json={"question": "What?", "model_name": model_b},
        )

    assert response.status_code == 200
    assert response.json()["model"] == model_b
    mock_gen.assert_called_once()
    assert mock_gen.call_args[1]["model_name"] == model_b


@pytest.mark.asyncio
async def test_generate_answer_uses_custom_system_prompt():
    """Should pass system_prompt override to the LLM payload."""
    from src.services.generation import SYSTEM_PROMPT, generate_answer

    custom_prompt = "You are an FSI compliance specialist."

    mock_client = AsyncMock()
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "test answer"}}],
        "usage": None,
    }
    mock_response.raise_for_status = lambda: None
    mock_client.post = AsyncMock(return_value=mock_response)

    with (
        patch("src.services.generation._get_client", return_value=mock_client),
        patch("src.services.generation.settings") as mock_settings,
    ):
        mock_settings.get_model_config.return_value = {
            "endpoint": "https://maas.example.com",
            "token": "test-token",
        }
        mock_settings.DEBUG = False

        await generate_answer("What?", [], "test-model", system_prompt=custom_prompt)

    call_kwargs = mock_client.post.call_args
    payload = call_kwargs[1]["json"]
    assert payload["messages"][0]["content"] == custom_prompt
    assert payload["messages"][0]["content"] != SYSTEM_PROMPT


def test_query_validates_empty_question(client):
    """Should reject empty questions."""
    response = client.post("/query/", json={"question": ""})
    assert response.status_code == 422


def test_query_with_no_sources(client):
    """Should return answer even when no chunks are retrieved."""
    model_a = settings.MODEL_A_NAME
    with (
        patch("src.routes.query.retrieve_chunks", return_value=[]),
        patch(
            "src.routes.query.generate_answer",
            return_value={
                "answer": "I don't have enough context to answer.",
                "model": model_a,
                "usage": None,
            },
        ),
    ):
        response = client.post(
            "/query/",
            json={"question": "Unknown topic?"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["sources"] == []
    assert data["usage"] is None


# --- Safety filtering tests ---


def test_query_filters_unsafe_input(client):
    """Should return safety-filtered response when input is flagged unsafe."""
    unsafe_result = SafetyResult(is_safe=False, category="S1")

    with patch(
        "src.routes.query.check_input_safety",
        new_callable=AsyncMock,
        return_value=unsafe_result,
    ):
        response = client.post(
            "/query/",
            json={"question": "Something harmful"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["safety_filtered"] is True
    assert data["sources"] == []
    assert "unable to process" in data["answer"].lower()


def test_query_filters_unsafe_output(client):
    """Should filter unsafe model output and return safety-filtered response."""
    safe_input = SafetyResult(is_safe=True)
    unsafe_output = SafetyResult(is_safe=False, category="S3")

    mock_chunks = [
        {
            "id": 1,
            "text": "context",
            "source_document": "doc.pdf",
            "page_number": "1",
            "score": 0.9,
        },
    ]
    mock_generation = {
        "answer": "Harmful output here",
        "model": settings.MODEL_A_NAME,
        "usage": None,
    }

    with (
        patch(
            "src.routes.query.check_input_safety", new_callable=AsyncMock, return_value=safe_input
        ),
        patch(
            "src.routes.query.check_output_safety",
            new_callable=AsyncMock,
            return_value=unsafe_output,
        ),
        patch("src.routes.query.retrieve_chunks", return_value=mock_chunks),
        patch("src.routes.query.generate_answer", return_value=mock_generation),
    ):
        response = client.post(
            "/query/",
            json={"question": "What are the compliance rules?"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["safety_filtered"] is True
    assert "filtered for safety" in data["answer"].lower()


def test_query_passes_through_when_safe(client):
    """Should proceed normally when both input and output pass safety checks."""
    safe_result = SafetyResult(is_safe=True)
    model_a = settings.MODEL_A_NAME

    mock_chunks = [
        {
            "id": 1,
            "text": "Revenue data",
            "source_document": "report.pdf",
            "page_number": "5",
            "score": 0.92,
        },
    ]
    mock_generation = {
        "answer": "Revenue was $1 billion.",
        "model": model_a,
        "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
    }

    with (
        patch(
            "src.routes.query.check_input_safety", new_callable=AsyncMock, return_value=safe_result
        ),
        patch(
            "src.routes.query.check_output_safety", new_callable=AsyncMock, return_value=safe_result
        ),
        patch("src.routes.query.retrieve_chunks", return_value=mock_chunks),
        patch("src.routes.query.generate_answer", return_value=mock_generation),
    ):
        response = client.post(
            "/query/",
            json={"question": "What was the revenue?"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["safety_filtered"] is False
    assert data["answer"] == "Revenue was $1 billion."
    assert len(data["sources"]) == 1


# --- Profile-based retrieval config tests ---


def test_query_loads_profile_retrieval_config(client):
    """Should pass profile retrieval kwargs (top_k, diversity, etc.) to retrieve_chunks."""
    model_a = settings.MODEL_A_NAME
    with (
        patch("src.routes.query.retrieve_chunks", return_value=[]) as mock_retrieve,
        patch(
            "src.routes.query.generate_answer",
            return_value={"answer": "test", "model": model_a, "usage": None},
        ),
    ):
        response = client.post(
            "/query/",
            json={"question": "What are the disclosure requirements?"},
        )

    assert response.status_code == 200
    call_kwargs = mock_retrieve.call_args[1]
    assert call_kwargs["top_k"] == 8
    assert call_kwargs["keyword_enabled"] is True
    assert call_kwargs["rerank_depth"] == 45
    assert call_kwargs["diversity_min"] == 3


def test_query_passes_profile_system_prompt_to_generation(client):
    """Should pass FSI profile system prompt to generate_answer."""
    model_a = settings.MODEL_A_NAME
    with (
        patch("src.routes.query.retrieve_chunks", return_value=[]),
        patch(
            "src.routes.query.generate_answer",
            return_value={"answer": "test", "model": model_a, "usage": None},
        ) as mock_gen,
    ):
        client.post(
            "/query/",
            json={"question": "What are the rules?"},
        )

    call_kwargs = mock_gen.call_args[1]
    assert call_kwargs["system_prompt"] is not None
    assert "compliance" in call_kwargs["system_prompt"].lower()


def test_query_explicit_top_k_overrides_profile(client):
    """Should use request top_k when explicitly provided, not the profile default."""
    model_a = settings.MODEL_A_NAME
    with (
        patch("src.routes.query.retrieve_chunks", return_value=[]) as mock_retrieve,
        patch(
            "src.routes.query.generate_answer",
            return_value={"answer": "test", "model": model_a, "usage": None},
        ),
    ):
        response = client.post(
            "/query/",
            json={"question": "Simple question?", "top_k": 3},
        )

    assert response.status_code == 200
    assert mock_retrieve.call_args[1]["top_k"] == 3


def test_query_degrades_gracefully_when_profile_missing(client):
    """Should still work when the profile file is missing."""
    model_a = settings.MODEL_A_NAME
    with (
        patch(
            "src.routes.query.load_profile",
            side_effect=FileNotFoundError("not found"),
        ),
        patch("src.routes.query.retrieve_chunks", return_value=[]) as mock_retrieve,
        patch(
            "src.routes.query.generate_answer",
            return_value={"answer": "test", "model": model_a, "usage": None},
        ) as mock_gen,
    ):
        response = client.post(
            "/query/",
            json={"question": "What is this?"},
        )

    assert response.status_code == 200
    assert mock_gen.call_args[1]["system_prompt"] is None
    assert "top_k" not in mock_retrieve.call_args[1]


# --- Debug retrieval endpoint tests ---


def test_debug_retrieval_returns_diagnostics(client):
    """Should return sub-queries, document scores, and final chunks."""
    mock_chunks = [
        {
            "id": 1,
            "text": "ETF Rule 6c-11 content",
            "source_document": "etf-rule.pdf",
            "page_number": "1",
            "score": 0.9,
        },
        {
            "id": 2,
            "text": "Form N-PORT filing requirements",
            "source_document": "form-n-port.pdf",
            "page_number": "3",
            "score": 0.5,
        },
    ]

    with (
        patch(
            "src.routes.query.decompose_query",
            new_callable=AsyncMock,
            return_value=["What is Rule 6c-11?", "What are N-PORT requirements?"],
        ),
        patch(
            "src.routes.query.retrieve_chunks",
            new_callable=AsyncMock,
            return_value=mock_chunks,
        ),
    ):
        response = client.post(
            "/query/debug",
            json={"question": "What are ETF regulatory requirements?"},
        )

    assert response.status_code == 200
    data = response.json()
    assert len(data["sub_queries"]) == 2
    assert data["total_candidates"] >= 2
    assert len(data["documents"]) >= 1
    assert len(data["final_chunks"]) >= 1
    assert "diversity_threshold" in data
    assert "top_k" in data


def test_debug_retrieval_single_query_fallback(client):
    """Should work when decomposition returns the original query."""
    mock_chunks = [
        {
            "id": 1,
            "text": "Some content",
            "source_document": "doc.pdf",
            "page_number": "1",
            "score": 0.8,
        },
    ]

    with (
        patch(
            "src.routes.query.decompose_query",
            new_callable=AsyncMock,
            return_value=["Simple question"],
        ),
        patch(
            "src.routes.query.retrieve_chunks",
            new_callable=AsyncMock,
            return_value=mock_chunks,
        ),
    ):
        response = client.post(
            "/query/debug",
            json={"question": "Simple question"},
        )

    assert response.status_code == 200
    data = response.json()
    assert len(data["sub_queries"]) == 1
    assert len(data["final_chunks"]) == 1
