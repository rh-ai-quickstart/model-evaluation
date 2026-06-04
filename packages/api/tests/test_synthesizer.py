"""Tests for question synthesizer endpoint and service."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from db import Chunk, Document

from src.core.config import settings


def _seed_documents_and_chunks(async_session):
    """Seed a document with chunks for synthesizer tests."""
    import asyncio

    async def _seed():
        async with async_session() as session:
            doc = Document(id=1, filename="test.pdf", status="ready", chunk_count=3)
            session.add(doc)
            await session.flush()

            for i in range(3):
                session.add(
                    Chunk(
                        document_id=1,
                        text=f"This is chunk {i} about artificial intelligence and machine learning.",
                        source_document="test.pdf",
                        element_type="paragraph",
                        token_count=10,
                    )
                )
            await session.commit()

    asyncio.run(_seed())


@pytest.fixture(autouse=True)
def _synthesis_env(monkeypatch):
    """Synthesis route requires a model name and MaaS settings."""
    monkeypatch.setattr(settings, "MODEL_A_NAME", "test-synth-model", raising=False)
    monkeypatch.setattr(settings, "API_TOKEN", "test-token", raising=False)
    monkeypatch.setattr(settings, "MAAS_ENDPOINT", "https://maas.test", raising=False)


def _patch_httpx_synthesize(questions_data):
    """Patch synthesizer._get_client so POST returns JSON questions from the model."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": json.dumps({"questions": questions_data})}}]
    }

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.is_closed = False

    return patch("src.services.synthesizer._get_client", return_value=mock_client)


# --- Tests ---


def test_synthesize_returns_questions(client, _setup_db):
    """Should return generated questions from document chunks."""
    _, async_session = _setup_db
    _seed_documents_and_chunks(async_session)

    with _patch_httpx_synthesize(
        [
            {
                "question": "What is artificial intelligence?",
                "expected_answer": "AI is the simulation of human intelligence by machines.",
            }
        ]
    ):
        response = client.post(
            "/evaluations/synthesize",
            json={"max_questions": 5},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 1
    assert data["questions"][0]["question"] == "What is artificial intelligence?"
    assert data["questions"][0]["expected_answer"] is not None


def test_synthesize_empty_when_no_documents(client):
    """Should return empty list when no documents exist."""
    response = client.post(
        "/evaluations/synthesize",
        json={"max_questions": 5},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 0
    assert data["questions"] == []


def test_synthesize_filters_by_document_ids(client, _setup_db):
    """Should only use chunks from specified document IDs."""
    _, async_session = _setup_db
    _seed_documents_and_chunks(async_session)

    with _patch_httpx_synthesize(
        [{"question": "What is ML?", "expected_answer": "Machine learning is a subset of AI."}]
    ):
        response = client.post(
            "/evaluations/synthesize",
            json={"document_ids": [1], "max_questions": 5},
        )

    assert response.status_code == 200
    assert response.json()["count"] == 1


def test_synthesize_filters_nonexistent_document(client):
    """Should return empty when filtering by document ID with no chunks."""
    response = client.post(
        "/evaluations/synthesize",
        json={"document_ids": [999], "max_questions": 5},
    )

    assert response.status_code == 200
    assert response.json()["count"] == 0


def test_synthesize_validates_max_questions(client):
    """Should reject max_questions outside 1-50 range."""
    response = client.post(
        "/evaluations/synthesize",
        json={"max_questions": 0},
    )
    assert response.status_code == 422

    response = client.post(
        "/evaluations/synthesize",
        json={"max_questions": 51},
    )
    assert response.status_code == 422


def test_parse_questions_json_raw():
    """Should parse raw JSON without fences."""
    from src.services.synthesizer import _parse_questions_json

    result = _parse_questions_json('{"questions": [{"question": "Q1"}]}')
    assert result["questions"][0]["question"] == "Q1"


def test_parse_questions_json_fenced():
    """Should parse JSON wrapped in markdown code fences."""
    from src.services.synthesizer import _parse_questions_json

    raw = '```json\n{"questions": [{"question": "Q2"}]}\n```'
    result = _parse_questions_json(raw)
    assert result["questions"][0]["question"] == "Q2"


def test_parse_questions_json_malformed():
    """Should raise on completely unparseable input."""
    from src.services.synthesizer import _parse_questions_json

    with pytest.raises(ValueError):
        _parse_questions_json("not json at all")


def test_synthesize_rejects_when_no_model_configured(client, monkeypatch):
    """Should 400 when no synthesis model is configured."""
    monkeypatch.setattr(settings, "MODEL_A_NAME", "", raising=False)
    monkeypatch.setattr(settings, "JUDGE_MODEL_NAME", "", raising=False)
    monkeypatch.setattr(settings, "QUESTION_SYNTHESIS_MODEL_NAME", "", raising=False)

    response = client.post(
        "/evaluations/synthesize",
        json={"max_questions": 5},
    )
    assert response.status_code == 400
    assert "question synthesis" in response.json()["detail"].lower()


def test_synthesize_with_fsi_profile(client, _setup_db):
    """Should use FSI domain rules when profile_id is provided."""
    _, async_session = _setup_db
    _seed_documents_and_chunks(async_session)

    with _patch_httpx_synthesize(
        [{"question": "What are the SEC reporting requirements?", "expected_answer": "Quarterly."}]
    ) as mock_httpx:
        response = client.post(
            "/evaluations/synthesize",
            json={"max_questions": 3, "profile_id": "fsi_compliance_v1"},
        )

    assert response.status_code == 200
    assert response.json()["count"] == 1
    # Verify the prompt sent to the model includes FSI-specific rules
    # Use call_args_list[0] to get the synthesis call (truth generation adds subsequent calls)
    call_kwargs = mock_httpx.return_value.post.call_args_list[0]
    payload = call_kwargs[1]["json"] if "json" in call_kwargs[1] else call_kwargs.kwargs["json"]
    prompt_content = payload["messages"][0]["content"]
    assert "SEC/FINRA" in prompt_content


def test_synthesize_without_profile_uses_default_rules(client, _setup_db):
    """Should use generic rules when no profile_id is provided."""
    _, async_session = _setup_db
    _seed_documents_and_chunks(async_session)

    with _patch_httpx_synthesize(
        [{"question": "What is AI?", "expected_answer": "Artificial intelligence."}]
    ) as mock_httpx:
        response = client.post(
            "/evaluations/synthesize",
            json={"max_questions": 3},
        )

    assert response.status_code == 200
    # Use call_args_list[0] to get the synthesis call (truth generation adds subsequent calls)
    call_kwargs = mock_httpx.return_value.post.call_args_list[0]
    payload = call_kwargs[1]["json"] if "json" in call_kwargs[1] else call_kwargs.kwargs["json"]
    prompt_content = payload["messages"][0]["content"]
    assert "SEC/FINRA" not in prompt_content
    assert "requirements, obligations, thresholds" in prompt_content


def test_synthesize_with_invalid_profile_falls_back_to_default(client, _setup_db):
    """Should use default rules when profile_id is invalid (graceful degradation)."""
    _, async_session = _setup_db
    _seed_documents_and_chunks(async_session)

    with _patch_httpx_synthesize(
        [{"question": "What is AI?", "expected_answer": "Artificial intelligence."}]
    ) as mock_httpx:
        response = client.post(
            "/evaluations/synthesize",
            json={"max_questions": 3, "profile_id": "nonexistent_profile"},
        )

    assert response.status_code == 200
    assert response.json()["count"] == 1
    # Use call_args_list[0] to get the synthesis call (truth generation adds subsequent calls)
    call_kwargs = mock_httpx.return_value.post.call_args_list[0]
    payload = call_kwargs[1]["json"] if "json" in call_kwargs[1] else call_kwargs.kwargs["json"]
    prompt_content = payload["messages"][0]["content"]
    assert "SEC/FINRA" not in prompt_content
    assert "requirements, obligations, thresholds" in prompt_content


def test_domain_rules_mapping():
    """Should have FSI-specific rules and a default fallback."""
    from src.services.synthesizer import _DEFAULT_DOMAIN_RULES, _DOMAIN_RULES

    assert "fsi" in _DOMAIN_RULES
    assert "SEC/FINRA" in _DOMAIN_RULES["fsi"]
    assert _DEFAULT_DOMAIN_RULES
    assert "SEC/FINRA" not in _DEFAULT_DOMAIN_RULES


# --- Truth generation tests ---


def test_synthesize_returns_truth_when_judge_configured(client, _setup_db):
    """Should include truth payload in synthesized questions when judge model is available."""
    _, async_session = _setup_db
    _seed_documents_and_chunks(async_session)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(settings, "JUDGE_MODEL_NAME", "test-judge", raising=False)

    concepts_json = '["AI simulates human intelligence"]'

    # Two HTTP calls per question: 1 synthesis + 1 concept extraction.
    # Retrieval grounding and document classification are patched below.
    synth_response = MagicMock()
    synth_response.raise_for_status = MagicMock()
    synth_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "questions": [
                                {
                                    "question": "What is AI?",
                                    "expected_answer": "AI is the simulation of human intelligence.",
                                }
                            ]
                        }
                    )
                }
            }
        ]
    }

    concept_response = MagicMock()
    concept_response.raise_for_status = MagicMock()
    concept_response.json.return_value = {"choices": [{"message": {"content": concepts_json}}]}

    synth_client = MagicMock()
    synth_client.post = AsyncMock(return_value=synth_response)
    synth_client.is_closed = False

    truth_client = MagicMock()
    truth_client.post = AsyncMock(return_value=concept_response)
    truth_client.is_closed = False

    with (
        patch("src.services.synthesizer._get_client", return_value=synth_client),
        patch("src.services.truth_generation._get_client", return_value=truth_client),
        patch(
            "src.services.truth_generation.retrieve_chunks",
            new_callable=AsyncMock,
            return_value=[
                {
                    "id": 1,
                    "source_document": "test.pdf",
                    "text": "AI is the simulation of human intelligence.",
                    "score": 0.9,
                }
            ],
        ),
        patch(
            "src.services.truth_generation.classify_documents",
            new_callable=AsyncMock,
            return_value={"required": ["test.pdf"], "supporting": []},
        ),
    ):
        response = client.post(
            "/evaluations/synthesize",
            json={"max_questions": 1},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 1
    q = data["questions"][0]
    assert q["truth"] is not None
    assert "answer_truth" in q["truth"]
    assert "retrieval_truth" in q["truth"]
    assert "metadata" in q["truth"]
    assert q["truth"]["retrieval_truth"]["evidence_mode"] == "grounded_from_synthesis"
    assert len(q["truth"]["answer_truth"]["required_concepts"]) > 0

    monkeypatch.undo()


def test_synthesize_works_without_judge_model(client, _setup_db):
    """Should return questions without truth when no judge model is configured."""
    _, async_session = _setup_db
    _seed_documents_and_chunks(async_session)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(settings, "JUDGE_MODEL_NAME", "", raising=False)

    with _patch_httpx_synthesize(
        [{"question": "What is AI?", "expected_answer": "AI is artificial intelligence."}]
    ):
        response = client.post(
            "/evaluations/synthesize",
            json={"max_questions": 1},
        )

    assert response.status_code == 200
    q = response.json()["questions"][0]
    assert q.get("truth") is None

    monkeypatch.undo()


def test_synthesize_graceful_on_truth_failure(client, _setup_db):
    """Should return questions without truth when truth generation fails."""
    _, async_session = _setup_db
    _seed_documents_and_chunks(async_session)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(settings, "JUDGE_MODEL_NAME", "test-judge", raising=False)

    with (
        _patch_httpx_synthesize(
            [{"question": "What is AI?", "expected_answer": "AI is artificial intelligence."}]
        ),
        patch(
            "src.services.synthesizer.generate_truth_from_synthesis",
            new_callable=AsyncMock,
            side_effect=RuntimeError("LLM unavailable"),
        ),
    ):
        response = client.post(
            "/evaluations/synthesize",
            json={"max_questions": 1},
        )

    assert response.status_code == 200
    q = response.json()["questions"][0]
    assert q.get("truth") is None
    assert q["expected_answer"] is not None

    monkeypatch.undo()


# --- Balanced sampling tests ---


def test_balanced_sample_distributes_across_documents():
    """Should give each document at least MIN_CHUNKS_PER_DOC chunks."""
    from src.services.synthesizer import _balanced_sample

    chunks_by_doc = {
        1: [{"id": i, "text": f"chunk {i}", "source_document": "a.pdf"} for i in range(20)],
        2: [
            {"id": 20 + i, "text": f"chunk {20 + i}", "source_document": "b.pdf"} for i in range(5)
        ],
        3: [
            {"id": 25 + i, "text": f"chunk {25 + i}", "source_document": "c.pdf"} for i in range(3)
        ],
    }

    result = _balanced_sample(chunks_by_doc, budget=10)
    assert len(result) == 10

    # Each document should have at least some representation
    docs_in_result = {c["source_document"] for c in result}
    assert docs_in_result == {"a.pdf", "b.pdf", "c.pdf"}


def test_balanced_sample_respects_budget():
    """Should not exceed the budget."""
    from src.services.synthesizer import _balanced_sample

    chunks_by_doc = {
        1: [{"id": i, "text": f"chunk {i}", "source_document": "a.pdf"} for i in range(100)],
    }
    result = _balanced_sample(chunks_by_doc, budget=10)
    assert len(result) <= 10


def test_balanced_sample_handles_small_documents():
    """Should take all chunks from documents smaller than MIN_CHUNKS_PER_DOC."""
    from src.services.synthesizer import _balanced_sample

    chunks_by_doc = {
        1: [{"id": 1, "text": "only chunk", "source_document": "tiny.pdf"}],
        2: [{"id": 2 + i, "text": f"chunk {i}", "source_document": "big.pdf"} for i in range(20)],
    }
    result = _balanced_sample(chunks_by_doc, budget=10)
    # tiny.pdf has only 1 chunk, should get it
    tiny_chunks = [c for c in result if c["source_document"] == "tiny.pdf"]
    assert len(tiny_chunks) == 1


def test_balanced_sample_empty_input():
    """Should return empty list for empty input."""
    from src.services.synthesizer import _balanced_sample

    assert _balanced_sample({}, budget=10) == []


def test_balanced_sample_sorted_by_id():
    """Should return chunks sorted by ID for stable prompt ordering."""
    from src.services.synthesizer import _balanced_sample

    chunks_by_doc = {
        1: [{"id": 10, "text": "a", "source_document": "a.pdf"}],
        2: [{"id": 5, "text": "b", "source_document": "b.pdf"}],
        3: [{"id": 15, "text": "c", "source_document": "c.pdf"}],
    }
    result = _balanced_sample(chunks_by_doc, budget=10)
    ids = [c["id"] for c in result]
    assert ids == sorted(ids)


# --- Per-question chunk alignment tests ---


def test_align_chunks_finds_matching_content():
    """Should align chunks whose content appears in the expected answer."""
    from src.services.truth_generation import _align_chunks_to_answer

    chunks = [
        {"id": 1, "text": "The SEC requires quarterly filings for all registered advisers."},
        {"id": 2, "text": "Python is a popular programming language."},
        {"id": 3, "text": "FINRA mandates supervisory procedures for broker-dealers."},
    ]
    answer = "The SEC requires quarterly filings for all registered advisers."
    aligned = _align_chunks_to_answer(answer, chunks)
    aligned_ids = {c["id"] for c in aligned}
    assert 1 in aligned_ids
    assert 2 not in aligned_ids


def test_align_chunks_falls_back_when_no_overlap():
    """Should return all chunks when no content overlap is found."""
    from src.services.truth_generation import _align_chunks_to_answer

    chunks = [
        {"id": 1, "text": "Completely unrelated content about weather."},
        {"id": 2, "text": "Another unrelated topic about cooking."},
    ]
    answer = "The SEC requires quarterly filings."
    aligned = _align_chunks_to_answer(answer, chunks)
    assert len(aligned) == 2  # falls back to all


def test_align_chunks_handles_empty_inputs():
    """Should handle empty answer or empty chunks gracefully."""
    from src.services.truth_generation import _align_chunks_to_answer

    chunks = [{"id": 1, "text": "Some content."}]
    assert _align_chunks_to_answer("", chunks) == chunks
    assert _align_chunks_to_answer("answer", []) == []
