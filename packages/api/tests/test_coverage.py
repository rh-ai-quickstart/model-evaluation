"""Tests for coverage gap detection service."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.coverage import (
    _check_concept_in_contexts,
    _concept_cache,
    _concept_cache_key,
    detect_coverage_gaps,
)


@pytest.fixture(autouse=True)
def _reset_settings_and_cache():
    """Ensure settings and cache are restored after each test."""
    from src.core.config import settings

    original_token = settings.API_TOKEN
    original_maas = settings.MAAS_ENDPOINT
    original_judge = settings.JUDGE_MODEL_NAME
    _concept_cache.clear()
    yield
    settings.API_TOKEN = original_token
    settings.MAAS_ENDPOINT = original_maas
    settings.JUDGE_MODEL_NAME = original_judge
    _concept_cache.clear()


def _make_mock_response(content: str) -> MagicMock:
    """Create a mock HTTP response with the given content."""
    mock = MagicMock()
    mock.status_code = 200
    mock.raise_for_status = MagicMock()
    mock.json.return_value = {
        "choices": [{"message": {"content": content}}]
    }
    return mock


def _setup_settings():
    """Configure settings for tests that need a model."""
    from src.core.config import settings

    settings.API_TOKEN = "test-token"
    settings.MAAS_ENDPOINT = "https://example.com"
    settings.JUDGE_MODEL_NAME = "test-model"


def _patch_two_phase(extract_content: str, check_content: str):
    """Patch httpx to return extract_content on first call, check_content on second."""
    extract_resp = _make_mock_response(extract_content)
    check_resp = _make_mock_response(check_content)

    mock_client = AsyncMock()
    mock_client.post.side_effect = [extract_resp, check_resp]
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    return patch(
        "src.services.coverage.httpx.AsyncClient",
        return_value=mock_client,
    )


def test_returns_none_when_no_model():
    """Should return None when no model is configured."""
    from src.core.config import settings

    settings.JUDGE_MODEL_NAME = ""
    settings.API_TOKEN = ""

    result = asyncio.run(
        detect_coverage_gaps("expected answer", "actual answer")
    )
    assert result is None


def test_returns_none_when_no_token():
    """Should return None when no API token is set."""
    from src.core.config import settings

    settings.JUDGE_MODEL_NAME = "test-model"
    settings.API_TOKEN = ""

    result = asyncio.run(
        detect_coverage_gaps("expected answer", "actual answer")
    )
    assert result is None


def test_returns_coverage_report_on_success():
    """Should return structured coverage report with concepts, covered, and missing."""
    _setup_settings()

    with _patch_two_phase(
        '["Rule 6c-11 provisions", "N-PORT filing deadlines", "SAI disclosure requirements"]',
        '["covered", "missing", "covered"]',
    ):
        result = asyncio.run(
            detect_coverage_gaps(
                "Rule 6c-11 provisions, N-PORT filing deadlines, SAI disclosures",
                "The answer covers Rule 6c-11 and SAI requirements.",
            )
        )

    assert result is not None
    assert len(result["concepts"]) == 3
    assert len(result["covered"]) == 2
    assert len(result["missing"]) == 1
    assert "N-PORT filing deadlines" in result["missing"]
    assert result["coverage_ratio"] == pytest.approx(2 / 3)


def test_returns_none_on_invalid_json_extraction():
    """Should return None when LLM returns invalid JSON for concept extraction."""
    _setup_settings()

    mock_client = AsyncMock()
    mock_client.post.return_value = _make_mock_response("This is not JSON")
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("src.services.coverage.httpx.AsyncClient", return_value=mock_client):
        result = asyncio.run(
            detect_coverage_gaps("expected", "actual")
        )

    assert result is None


def test_returns_none_on_api_error():
    """Should return None when API call fails."""
    _setup_settings()

    import httpx

    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Server Error", request=MagicMock(), response=mock_response
    )

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("src.services.coverage.httpx.AsyncClient", return_value=mock_client):
        result = asyncio.run(
            detect_coverage_gaps("expected", "actual")
        )

    assert result is None


def test_handles_markdown_fenced_json():
    """Should strip markdown fencing and parse the JSON."""
    _setup_settings()

    fenced_concepts = '```json\n["concept A", "concept B"]\n```'

    with _patch_two_phase(fenced_concepts, '["covered", "covered"]'):
        result = asyncio.run(
            detect_coverage_gaps("expected", "actual")
        )

    assert result is not None
    assert len(result["concepts"]) == 2
    assert result["coverage_ratio"] == 1.0


def test_returns_none_on_empty_concepts():
    """Should return None when LLM returns empty concepts list."""
    _setup_settings()

    mock_client = AsyncMock()
    mock_client.post.return_value = _make_mock_response("[]")
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("src.services.coverage.httpx.AsyncClient", return_value=mock_client):
        result = asyncio.run(
            detect_coverage_gaps("expected", "actual")
        )

    assert result is None


# --- Concept caching tests ---


def test_concept_cache_reused_across_calls():
    """Should reuse cached concepts for the same expected answer."""
    _setup_settings()

    expected = "Rule 6c-11 provisions and N-PORT deadlines"
    concepts = ["Rule 6c-11 provisions", "N-PORT filing deadlines"]

    # First call: extract + check (2 LLM calls)
    with _patch_two_phase(
        json.dumps(concepts),
        '["covered", "missing"]',
    ):
        result1 = asyncio.run(
            detect_coverage_gaps(expected, "answer A")
        )

    assert result1 is not None
    assert _concept_cache_key(None, expected) in _concept_cache

    # Second call: only check (1 LLM call), concepts from cache
    check_resp = _make_mock_response('["missing", "covered"]')
    mock_client2 = AsyncMock()
    mock_client2.post.return_value = check_resp
    mock_client2.__aenter__ = AsyncMock(return_value=mock_client2)
    mock_client2.__aexit__ = AsyncMock(return_value=False)

    with patch("src.services.coverage.httpx.AsyncClient", return_value=mock_client2):
        result2 = asyncio.run(
            detect_coverage_gaps(expected, "answer B")
        )

    assert result2 is not None
    # Same concepts, different coverage
    assert result2["concepts"] == result1["concepts"]
    assert result2["covered"] == ["N-PORT filing deadlines"]
    assert result2["missing"] == ["Rule 6c-11 provisions"]
    # Only 1 LLM call (check only, no extraction)
    assert mock_client2.post.call_count == 1


def test_concept_cache_separates_by_question():
    """Same expected answer under different questions should not share extraction cache."""
    _setup_settings()

    expected = "Same gold text"
    concepts_q1 = ["concept for Q1"]
    concepts_q2 = ["concept for Q2"]

    with _patch_two_phase(json.dumps(concepts_q1), '["covered"]'):
        r1 = asyncio.run(
            detect_coverage_gaps(expected, "answer", question="Question one?")
        )

    assert r1 is not None
    assert _concept_cache_key("Question one?", expected) in _concept_cache

    with _patch_two_phase(json.dumps(concepts_q2), '["missing"]'):
        r2 = asyncio.run(
            detect_coverage_gaps(expected, "answer", question="Question two?")
        )

    assert r2 is not None
    assert r2["concepts"] == concepts_q2
    assert _concept_cache_key("Question two?", expected) in _concept_cache
    assert len(_concept_cache) == 2


def test_concept_found_in_contexts():
    """Should detect concept when significant words appear in context."""
    contexts = ["Form N-PORT requires quarterly filing within 60 days."]
    assert _check_concept_in_contexts("N-PORT quarterly filing deadline", contexts) is True


def test_concept_not_found_in_contexts():
    """Should return False when concept words are absent from context."""
    contexts = ["Rule 6c-11 governs ETF operations and structure."]
    assert _check_concept_in_contexts("N-PORT quarterly filing deadline", contexts) is False


def test_concept_empty_contexts():
    """Should return False when contexts list is empty."""
    assert _check_concept_in_contexts("any concept", []) is False


# --- Retrieval vs generation failure classification tests ---


def test_classifies_retrieval_and_generation_failures():
    """Should classify missing concepts as retrieval or generation failures."""
    _setup_settings()

    # N-PORT appears in context (generation failure), SAI does not (retrieval failure)
    contexts = [
        "Form N-PORT requires quarterly filing of portfolio holdings within 60 days.",
        "Rule 6c-11 governs ETF operations.",
    ]

    with _patch_two_phase(
        '["Rule 6c-11 provisions", "N-PORT filing deadlines", "SAI disclosure requirements"]',
        '["covered", "missing", "missing"]',
    ):
        result = asyncio.run(
            detect_coverage_gaps(
                "Rule 6c-11, N-PORT deadlines, SAI disclosures",
                "The answer only covers Rule 6c-11.",
                contexts=contexts,
            )
        )

    assert result is not None
    assert len(result["missing"]) == 2
    assert "N-PORT filing deadlines" in result["generation_failures"]
    assert "SAI disclosure requirements" in result["retrieval_failures"]


def test_no_failure_classification_without_contexts():
    """Should not include failure classification when contexts is None."""
    _setup_settings()

    with _patch_two_phase(
        '["concept A"]',
        '["missing"]',
    ):
        result = asyncio.run(
            detect_coverage_gaps("expected", "actual")
        )

    assert result is not None
    assert "retrieval_failures" not in result
    assert "generation_failures" not in result


# --- Pre-extracted concepts tests ---


def test_pre_extracted_concepts_skips_llm_extraction():
    """Should skip LLM extraction and use pre-extracted concepts directly."""
    _setup_settings()

    pre_concepts = ["concept from truth A", "concept from truth B"]

    # Only one LLM call (coverage check), not two (no extraction)
    check_resp = _make_mock_response('["covered", "missing"]')
    mock_client = AsyncMock()
    mock_client.post.return_value = check_resp
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("src.services.coverage.httpx.AsyncClient", return_value=mock_client):
        result = asyncio.run(
            detect_coverage_gaps(
                "expected answer text",
                "actual answer text",
                pre_extracted_concepts=pre_concepts,
            )
        )

    assert result is not None
    assert result["concepts"] == pre_concepts
    assert result["covered"] == ["concept from truth A"]
    assert result["missing"] == ["concept from truth B"]
    assert result["coverage_ratio"] == 0.5
    # Only 1 LLM call (check), not 2 (extract + check)
    assert mock_client.post.call_count == 1


def test_pre_extracted_concepts_with_failure_classification():
    """Should classify failures correctly when using pre-extracted concepts."""
    _setup_settings()

    pre_concepts = ["quarterly filing deadline", "blockchain custody requirements"]
    contexts = ["Form N-PORT requires quarterly filing of portfolio holdings within 60 days."]

    check_resp = _make_mock_response('["missing", "missing"]')
    mock_client = AsyncMock()
    mock_client.post.return_value = check_resp
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("src.services.coverage.httpx.AsyncClient", return_value=mock_client):
        result = asyncio.run(
            detect_coverage_gaps(
                "expected",
                "actual",
                contexts=contexts,
                pre_extracted_concepts=pre_concepts,
            )
        )

    assert result is not None
    # "quarterly filing deadline" words appear in context -> generation failure
    assert "quarterly filing deadline" in result["generation_failures"]
    # "blockchain custody requirements" words absent from context -> retrieval failure
    assert "blockchain custody requirements" in result["retrieval_failures"]
