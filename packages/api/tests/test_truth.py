# This project was developed with assistance from AI tools.
"""Tests for structured truth schema and truth generation service."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.schemas.truth import AnswerTruth, RetrievalTruth, TruthMetadata, TruthPayload

# --- Schema validation tests ---


def test_truth_payload_valid():
    """Should accept a valid truth payload with all fields."""
    payload = TruthPayload(
        answer_truth=AnswerTruth(
            required_concepts=["ETFs must file Form N-1A", "Prospectus disclosures required"],
        ),
        retrieval_truth=RetrievalTruth(
            required_documents=["sec-etf-guide.pdf"],
            expected_chunk_refs=["chunk:42", "chunk:43"],
            evidence_mode="traced_from_synthesis",
        ),
        metadata=TruthMetadata(
            generated_by_model="mistral-small-24b",
            generated_at=datetime(2026, 4, 23, 12, 0, 0),
            source_chunk_ids=[42, 43],
        ),
    )
    assert len(payload.answer_truth.required_concepts) == 2
    assert payload.retrieval_truth.evidence_mode == "traced_from_synthesis"
    assert payload.metadata.truth_schema_version == "1.1"


def test_truth_payload_grounded_mode():
    """Should accept grounded_from_manual_answer evidence mode."""
    payload = TruthPayload(
        answer_truth=AnswerTruth(required_concepts=["concept"]),
        retrieval_truth=RetrievalTruth(evidence_mode="grounded_from_manual_answer"),
        metadata=TruthMetadata(
            generated_by_model="test-model",
            generated_at=datetime(2026, 4, 23),
        ),
    )
    assert payload.retrieval_truth.evidence_mode == "grounded_from_manual_answer"


def test_truth_payload_grounded_synthesis_mode():
    """Should accept generated truth grounded through retrieval."""
    payload = TruthPayload(
        answer_truth=AnswerTruth(required_concepts=["concept"]),
        retrieval_truth=RetrievalTruth(evidence_mode="grounded_from_synthesis"),
        metadata=TruthMetadata(
            generated_by_model="test-model",
            generated_at=datetime(2026, 4, 23),
        ),
    )
    assert payload.retrieval_truth.evidence_mode == "grounded_from_synthesis"


def test_truth_payload_invalid_evidence_mode():
    """Should reject invalid evidence mode."""
    with pytest.raises(Exception):
        TruthPayload(
            answer_truth=AnswerTruth(required_concepts=["concept"]),
            retrieval_truth=RetrievalTruth(evidence_mode="invalid_mode"),
            metadata=TruthMetadata(
                generated_by_model="test-model",
                generated_at=datetime(2026, 4, 23),
            ),
        )


def test_truth_payload_defaults():
    """Should use correct defaults for optional fields."""
    truth = RetrievalTruth(evidence_mode="traced_from_synthesis")
    assert truth.required_documents == []
    assert truth.expected_chunk_refs == []

    meta = TruthMetadata(
        generated_by_model="test-model",
        generated_at=datetime(2026, 4, 23),
    )
    assert meta.truth_schema_version == "1.1"
    assert meta.source_chunk_ids == []


def test_truth_payload_abstention_expected():
    """Should support abstention_expected flag."""
    truth = AnswerTruth(required_concepts=[], abstention_expected=True)
    assert truth.abstention_expected is True


def test_truth_payload_serializes_to_json():
    """Should serialize cleanly to/from JSON for storage in JSON column."""
    payload = TruthPayload(
        answer_truth=AnswerTruth(required_concepts=["concept A"]),
        retrieval_truth=RetrievalTruth(
            required_documents=["doc.pdf"],
            expected_chunk_refs=["chunk:1"],
            evidence_mode="traced_from_synthesis",
        ),
        metadata=TruthMetadata(
            generated_by_model="test-model",
            generated_at=datetime(2026, 4, 23, 12, 0, 0),
            source_chunk_ids=[1],
        ),
    )
    json_str = payload.model_dump_json()
    restored = TruthPayload.model_validate_json(json_str)
    assert restored.answer_truth.required_concepts == ["concept A"]
    assert restored.retrieval_truth.expected_chunk_refs == ["chunk:1"]


# --- Truth generation service tests ---


def _make_mock_response(content: str) -> MagicMock:
    """Create a mock HTTP response with the given content."""
    mock = MagicMock()
    mock.status_code = 200
    mock.raise_for_status = MagicMock()
    mock.json.return_value = {"choices": [{"message": {"content": content}}]}
    return mock


def _setup_settings():
    """Configure settings for tests that need a model."""
    from src.core.config import settings

    settings.API_TOKEN = "test-token"
    settings.MAAS_ENDPOINT = "https://example.com"
    settings.JUDGE_MODEL_NAME = "test-judge"


@pytest.fixture(autouse=True)
def _reset_settings():
    """Restore settings after each test."""
    from src.core.config import settings

    original_token = settings.API_TOKEN
    original_maas = settings.MAAS_ENDPOINT
    original_judge = settings.JUDGE_MODEL_NAME
    yield
    settings.API_TOKEN = original_token
    settings.MAAS_ENDPOINT = original_maas
    settings.JUDGE_MODEL_NAME = original_judge


def test_extract_answer_truth_returns_concepts():
    """Should extract concepts from expected answer via LLM."""
    from src.services.truth_generation import extract_answer_truth

    _setup_settings()

    concepts_json = '["ETFs must file Form N-1A", "Prospectus disclosures required"]'
    mock_resp = _make_mock_response(concepts_json)
    mock_client = AsyncMock()
    mock_client.post.return_value = mock_resp
    mock_client.is_closed = False

    with patch("src.services.truth_generation._get_client", return_value=mock_client):
        result = asyncio.run(extract_answer_truth("ETFs must file Form N-1A...", "test-judge"))

    assert isinstance(result, AnswerTruth)
    assert len(result.required_concepts) == 2
    assert "ETFs must file Form N-1A" in result.required_concepts
    assert result.abstention_expected is False


def test_extract_answer_truth_raises_on_empty_concepts():
    """Should raise RuntimeError when LLM returns empty list."""
    from src.services.truth_generation import extract_answer_truth

    _setup_settings()

    mock_resp = _make_mock_response("[]")
    mock_client = AsyncMock()
    mock_client.post.return_value = mock_resp
    mock_client.is_closed = False

    with (
        patch("src.services.truth_generation._get_client", return_value=mock_client),
        pytest.raises(RuntimeError, match="empty or non-list"),
    ):
        asyncio.run(extract_answer_truth("Some answer", "test-judge"))


def test_extract_answer_truth_raises_on_api_error():
    """Should raise RuntimeError on HTTP error."""
    from src.services.truth_generation import extract_answer_truth

    _setup_settings()

    import httpx

    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Server Error", request=MagicMock(), response=mock_response
    )

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response
    mock_client.is_closed = False

    with (
        patch("src.services.truth_generation._get_client", return_value=mock_client),
        pytest.raises(RuntimeError, match="HTTP 500"),
    ):
        asyncio.run(extract_answer_truth("Some answer", "test-judge"))


def test_extract_answer_truth_raises_on_no_token():
    """Should raise RuntimeError when no API token is configured."""
    from src.core.config import settings
    from src.services.truth_generation import extract_answer_truth

    settings.API_TOKEN = ""

    with pytest.raises(RuntimeError, match="No API token"):
        asyncio.run(extract_answer_truth("Some answer", "test-judge"))


def test_build_retrieval_truth_from_synthesis():
    """Should build retrieval truth with document classification."""
    from src.services.truth_generation import build_retrieval_truth_from_synthesis

    _setup_settings()

    source_chunks = [
        {"id": 42, "source_document": "sec-etf-guide.pdf", "text": "..."},
        {"id": 43, "source_document": "sec-etf-guide.pdf", "text": "..."},
        {"id": 67, "source_document": "form-n1a-instructions.pdf", "text": "..."},
    ]

    classification = {
        "required": ["sec-etf-guide.pdf"],
        "supporting": ["form-n1a-instructions.pdf"],
    }

    with patch(
        "src.services.truth_generation.classify_documents",
        new_callable=AsyncMock,
        return_value=classification,
    ):
        result = asyncio.run(
            build_retrieval_truth_from_synthesis(
                "What are ETF requirements?", "ETF answer...", source_chunks, "test-judge"
            )
        )

    assert isinstance(result, RetrievalTruth)
    assert result.evidence_mode == "traced_from_synthesis"
    assert result.required_documents == ["sec-etf-guide.pdf"]
    assert result.supporting_documents == ["form-n1a-instructions.pdf"]
    assert result.expected_chunk_refs == ["chunk:42", "chunk:43"]
    assert result.supporting_chunk_refs == ["chunk:67"]


def test_build_retrieval_truth_from_synthesis_empty_chunks():
    """Should handle empty chunk list."""
    from src.services.truth_generation import build_retrieval_truth_from_synthesis

    _setup_settings()

    with patch(
        "src.services.truth_generation.classify_documents",
        new_callable=AsyncMock,
        return_value={"required": [], "supporting": []},
    ):
        result = asyncio.run(
            build_retrieval_truth_from_synthesis("question?", "answer", [], "test-judge")
        )

    assert result.required_documents == []
    assert result.expected_chunk_refs == []
    assert result.evidence_mode == "traced_from_synthesis"


def test_ground_answer_to_corpus():
    """Should ground expected answer against corpus with classification."""
    from src.services.truth_generation import ground_answer_to_corpus

    _setup_settings()

    mock_chunks = [
        {"id": 12, "source_document": "sec-etf-guide.pdf", "text": "...", "score": 0.9},
        {"id": 19, "source_document": "sec-etf-guide.pdf", "text": "...", "score": 0.8},
    ]

    classification = {"required": ["sec-etf-guide.pdf"], "supporting": []}
    mock_session = AsyncMock()

    with (
        patch(
            "src.services.truth_generation.retrieve_chunks",
            new_callable=AsyncMock,
            return_value=mock_chunks,
        ) as mock_retrieve,
        patch(
            "src.services.truth_generation.classify_documents",
            new_callable=AsyncMock,
            return_value=classification,
        ),
    ):
        result, source_ids = asyncio.run(
            ground_answer_to_corpus(
                "What are ETF filing requirements?",
                "ETFs register under the Investment Company Act...",
                mock_session,
                "test-judge",
            )
        )

    assert isinstance(result, RetrievalTruth)
    assert result.evidence_mode == "grounded_from_manual_answer"
    assert result.required_documents == ["sec-etf-guide.pdf"]
    assert result.expected_chunk_refs == ["chunk:12", "chunk:19"]
    assert source_ids == [12, 19]
    mock_retrieve.assert_called_once()


def test_ground_answer_to_corpus_passes_retrieval_kwargs():
    """Should forward profile-driven retrieval parameters."""
    from src.services.truth_generation import ground_answer_to_corpus

    _setup_settings()

    mock_session = AsyncMock()
    kwargs = {"top_k": 10, "keyword_enabled": False}

    with (
        patch(
            "src.services.truth_generation.retrieve_chunks",
            new_callable=AsyncMock,
            return_value=[],
        ) as mock_retrieve,
        patch(
            "src.services.truth_generation.classify_documents",
            new_callable=AsyncMock,
            return_value={"required": [], "supporting": []},
        ),
    ):
        asyncio.run(
            ground_answer_to_corpus(
                "Some question?", "Some answer", mock_session, "test-judge",
                retrieval_kwargs=kwargs,
            )
        )

    call_kwargs = mock_retrieve.call_args
    assert call_kwargs.kwargs["top_k"] == 10
    assert call_kwargs.kwargs["keyword_enabled"] is False


def test_generate_truth_from_synthesis():
    """Should compose full truth payload for synthesized questions."""
    from src.services.truth_generation import generate_truth_from_synthesis

    _setup_settings()

    source_chunks = [
        {"id": 42, "source_document": "guide.pdf", "text": "..."},
        {"id": 43, "source_document": "guide.pdf", "text": "..."},
    ]

    concepts_json = '["ETFs must file Form N-1A", "Prospectus disclosures required"]'
    mock_resp = _make_mock_response(concepts_json)
    mock_client = AsyncMock()
    mock_client.post.return_value = mock_resp
    mock_client.is_closed = False

    classification = {"required": ["guide.pdf"], "supporting": []}

    with (
        patch("src.services.truth_generation._get_client", return_value=mock_client),
        patch(
            "src.services.truth_generation.classify_documents",
            new_callable=AsyncMock,
            return_value=classification,
        ),
    ):
        result = asyncio.run(
            generate_truth_from_synthesis(
                "What are ETF filing requirements?",
                "ETFs must file Form N-1A...", source_chunks, "test-judge",
            )
        )

    assert isinstance(result, TruthPayload)
    assert len(result.answer_truth.required_concepts) == 2
    assert result.retrieval_truth.evidence_mode == "traced_from_synthesis"
    assert result.retrieval_truth.expected_chunk_refs == ["chunk:42", "chunk:43"]
    assert result.metadata.generated_by_model == "test-judge"
    assert result.metadata.source_chunk_ids == [42, 43]


def test_generate_truth_from_synthesis_can_ground_via_retrieval():
    """Synthesized truth should use retrievable evidence when a session is provided."""
    from src.services.truth_generation import generate_truth_from_synthesis

    _setup_settings()

    concepts_json = '["ETF basket anti-dumping safeguard"]'
    mock_resp = _make_mock_response(concepts_json)
    mock_client = AsyncMock()
    mock_client.post.return_value = mock_resp
    mock_client.is_closed = False

    retrieved_chunks = [
        {
            "id": 12,
            "source_document": "33-10695.pdf",
            "text": "The requirement addresses dumping less liquid securities.",
            "score": 0.9,
        }
    ]
    classification = {"required": ["33-10695.pdf"], "supporting": []}
    mock_session = AsyncMock()

    with (
        patch("src.services.truth_generation._get_client", return_value=mock_client),
        patch(
            "src.services.truth_generation.retrieve_chunks",
            new_callable=AsyncMock,
            return_value=retrieved_chunks,
        ),
        patch(
            "src.services.truth_generation.classify_documents",
            new_callable=AsyncMock,
            return_value=classification,
        ),
    ):
        result = asyncio.run(
            generate_truth_from_synthesis(
                "What is the purpose of the pro rata basket requirement?",
                "The requirement addresses dumping less liquid securities.",
                [{"id": 42, "source_document": "33-10695.pdf", "text": "source prompt"}],
                "test-judge",
                session=mock_session,
            )
        )

    assert result.retrieval_truth.evidence_mode == "grounded_from_synthesis"
    assert result.retrieval_truth.expected_chunk_refs == ["chunk:12"]
    assert result.metadata.source_chunk_ids == [12]


def test_generate_truth_from_manual_answer():
    """Should compose full truth payload for manual questions with corpus grounding."""
    from src.services.truth_generation import generate_truth_from_manual_answer

    _setup_settings()

    mock_chunks = [
        {"id": 12, "source_document": "guide.pdf", "text": "...", "score": 0.9},
    ]

    concepts_json = '["Registration form is N-1A"]'
    mock_resp = _make_mock_response(concepts_json)
    mock_client = AsyncMock()
    mock_client.post.return_value = mock_resp
    mock_client.is_closed = False

    classification = {"required": ["guide.pdf"], "supporting": []}
    mock_session = AsyncMock()

    with (
        patch("src.services.truth_generation._get_client", return_value=mock_client),
        patch(
            "src.services.truth_generation.retrieve_chunks",
            new_callable=AsyncMock,
            return_value=mock_chunks,
        ),
        patch(
            "src.services.truth_generation.classify_documents",
            new_callable=AsyncMock,
            return_value=classification,
        ),
    ):
        result = asyncio.run(
            generate_truth_from_manual_answer(
                "What are ETF filing requirements?",
                "ETFs register using Form N-1A...", mock_session, "test-judge",
            )
        )

    assert isinstance(result, TruthPayload)
    assert result.answer_truth.required_concepts == ["Registration form is N-1A"]
    assert result.retrieval_truth.evidence_mode == "grounded_from_manual_answer"
    assert result.retrieval_truth.expected_chunk_refs == ["chunk:12"]
    assert result.metadata.source_chunk_ids == [12]


def test_build_truth_metadata():
    """Should build metadata with correct version fields and timestamp."""
    from src.services.truth_generation import build_truth_metadata

    meta = build_truth_metadata("test-model", [1, 2, 3])

    assert isinstance(meta, TruthMetadata)
    assert meta.generated_by_model == "test-model"
    assert meta.source_chunk_ids == [1, 2, 3]
    assert meta.truth_schema_version == "1.1"
    assert meta.concept_extraction_version == "v1"
    assert isinstance(meta.generated_at, datetime)
