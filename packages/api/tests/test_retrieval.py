"""Tests for retrieval service."""

from collections import namedtuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.embedding import EmbeddingsResult
from src.services.retrieval import (
    _apply_diversity,
    _deduplicate_chunks,
    _fallback_search,
    _reciprocal_rank_fusion,
    compute_search_depth,
    retrieve_chunks,
)


@pytest.fixture
def mock_session():
    """Create a mock async session with sync .all() on results."""
    session = AsyncMock()
    return session


def _make_mock_result(rows):
    """Create a mock query result where .all() is sync (like SQLAlchemy)."""
    mock_result = MagicMock()
    mock_result.all.return_value = rows
    return mock_result


Row = namedtuple("Row", ["id", "text", "source_document", "page_number", "section_path"])


def test_fallback_search_returns_chunks(mock_session):
    """Should return recent chunks when vector search is unavailable."""
    mock_rows = [
        Row(id=1, text="chunk one", source_document="doc.pdf", page_number="1", section_path=None),
        Row(id=2, text="chunk two", source_document="doc.pdf", page_number="2", section_path=None),
    ]
    mock_session.execute.return_value = _make_mock_result(mock_rows)

    import asyncio

    result = asyncio.run(_fallback_search(mock_session, top_k=5))

    assert len(result) == 2
    assert result[0]["text"] == "chunk one"
    assert result[0]["score"] == 0.0
    assert result[1]["source_document"] == "doc.pdf"


def test_retrieve_chunks_uses_fallback_when_no_embeddings(mock_session):
    """Should fall back to recent chunks when embedding generation fails."""
    mock_rows = [
        Row(
            id=1,
            text="fallback chunk",
            source_document="doc.pdf",
            page_number="1",
            section_path=None,
        ),
    ]
    mock_session.execute.return_value = _make_mock_result(mock_rows)

    import asyncio

    with (
        patch(
            "src.services.retrieval.generate_embeddings",
            new_callable=AsyncMock,
            return_value=EmbeddingsResult(vectors=None, error=None),
        ),
        patch(
            "src.services.retrieval._keyword_search",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
        # diversity_min=1 skips doc-count query so execute maps to fallback only
        result = asyncio.run(
            retrieve_chunks("test query", mock_session, diversity_min=1),
        )

    assert len(result) == 1
    assert result[0]["text"] == "fallback chunk"
    assert result[0]["score"] == 0.0


def test_retrieve_chunks_keyword_only_when_no_embeddings(mock_session):
    """Without embeddings, keyword hits should be used (query-specific), not recent-chunks fallback."""
    import asyncio

    kw_hit = {
        "id": 42,
        "text": "ETF basket rule text",
        "source_document": "rule.pdf",
        "page_number": "3",
        "section_path": None,
        "score": 0.88,
    }

    with (
        patch(
            "src.services.retrieval.generate_embeddings",
            new_callable=AsyncMock,
            return_value=EmbeddingsResult(vectors=None, error="token missing"),
        ),
        patch(
            "src.services.retrieval._keyword_search",
            new_callable=AsyncMock,
            return_value=[kw_hit],
        ),
    ):
        result = asyncio.run(
            retrieve_chunks("baskets pro rata ETF", mock_session, diversity_min=1, top_k=5),
        )

    assert len(result) >= 1
    assert result[0]["text"] == "ETF basket rule text"
    assert result[0]["id"] == 42


# --- RRF and diversity tests ---


def _chunk(chunk_id, doc="doc.pdf", score=0.5):
    return {
        "id": chunk_id,
        "text": f"chunk {chunk_id}",
        "source_document": doc,
        "page_number": "1",
        "section_path": None,
        "score": score,
    }


def test_rrf_merges_two_lists():
    """Should merge and re-rank results from two lists using RRF."""
    vector = [_chunk(1, score=0.9), _chunk(2, score=0.8), _chunk(3, score=0.7)]
    keyword = [_chunk(2, score=0.6), _chunk(4, score=0.5), _chunk(1, score=0.4)]

    merged = _reciprocal_rank_fusion(vector, keyword)

    # Chunks 1 and 2 appear in both lists, so they should rank higher
    ids = [c["id"] for c in merged]
    assert ids[0] in (1, 2)  # top should be one shared between lists
    assert 4 in ids  # keyword-only result should still appear


def test_rrf_single_list():
    """Should preserve order when only one list is provided."""
    results = [_chunk(1, score=0.9), _chunk(2, score=0.8)]
    merged = _reciprocal_rank_fusion(results)
    assert [c["id"] for c in merged] == [1, 2]


def test_diversity_caps_per_doc():
    """Should cap chunks per document."""
    chunks = [
        _chunk(1, doc="a.pdf"),
        _chunk(2, doc="a.pdf"),
        _chunk(3, doc="a.pdf"),
        _chunk(4, doc="b.pdf"),
    ]
    result = _apply_diversity(chunks, top_k=3, max_per_doc=2, diversity_min=1)
    a_count = sum(1 for c in result if c["source_document"] == "a.pdf")
    assert a_count <= 2


def test_diversity_promotes_underrepresented_docs():
    """Should promote chunks from underrepresented documents."""
    chunks = [
        _chunk(1, doc="a.pdf"),
        _chunk(2, doc="a.pdf"),
        _chunk(3, doc="a.pdf"),
        _chunk(4, doc="a.pdf"),
        _chunk(5, doc="b.pdf"),
        _chunk(6, doc="c.pdf"),
    ]
    result = _apply_diversity(chunks, top_k=4, max_per_doc=2, diversity_min=3)
    docs = {c["source_document"] for c in result}
    assert len(docs) >= 3  # should include a, b, c


def test_diversity_returns_empty_for_empty_input():
    """Should return empty list for no chunks."""
    assert _apply_diversity([], top_k=5, max_per_doc=2, diversity_min=3) == []


def test_compute_search_depth_no_diversity():
    """Should use rerank_depth only when diversity is disabled."""
    assert compute_search_depth(50, 1, 100, max_search_depth=400) == 50
    assert compute_search_depth(35, 1, 20, max_search_depth=400) == 35


def test_compute_search_depth_uncapped_below_limit():
    """Should use doc_count * rerank_depth when under cap."""
    assert compute_search_depth(50, 3, 5, max_search_depth=400) == 250


def test_compute_search_depth_applies_cap():
    """Should cap huge doc_count * rerank_depth products."""
    assert compute_search_depth(50, 3, 30, max_search_depth=400) == 400


def test_retrieve_chunks_passes_capped_limit_to_vector_search(mock_session):
    """Should cap DB LIMIT when diversity widens the candidate pool."""
    import asyncio

    captured: dict[str, int] = {}

    async def capture_vector_search(_emb, _session, limit):
        captured["limit"] = limit
        return [_chunk(1, doc="a.pdf")]

    with (
        patch(
            "src.services.retrieval.generate_embeddings",
            new_callable=AsyncMock,
            return_value=EmbeddingsResult(vectors=[[0.1, 0.2]], error=None),
        ),
        patch(
            "src.services.retrieval._vector_search",
            new_callable=AsyncMock,
            side_effect=capture_vector_search,
        ),
        patch(
            "src.services.retrieval._keyword_search",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "src.services.retrieval._count_ready_documents",
            new_callable=AsyncMock,
            return_value=30,
        ),
    ):
        asyncio.run(
            retrieve_chunks(
                "test query",
                mock_session,
                top_k=5,
                rerank_depth=50,
                diversity_min=3,
            )
        )

    # uncapped would be max(50, 30*50)=1500; default cap 400
    assert captured.get("limit") == 400


def test_retrieve_chunks_hybrid_path(mock_session):
    """Should merge vector and keyword results via RRF when embeddings succeed."""
    import asyncio

    vector_results = [_chunk(1, doc="a.pdf"), _chunk(2, doc="b.pdf")]
    keyword_results = [_chunk(2, doc="b.pdf"), _chunk(3, doc="c.pdf")]

    with (
        patch(
            "src.services.retrieval.generate_embeddings",
            new_callable=AsyncMock,
            return_value=EmbeddingsResult(vectors=[[0.1, 0.2]], error=None),
        ),
        patch(
            "src.services.retrieval._vector_search",
            new_callable=AsyncMock,
            return_value=vector_results,
        ),
        patch(
            "src.services.retrieval._keyword_search",
            new_callable=AsyncMock,
            return_value=keyword_results,
        ),
        patch(
            "src.services.retrieval._count_ready_documents",
            new_callable=AsyncMock,
            return_value=3,
        ),
    ):
        result = asyncio.run(retrieve_chunks("test query", mock_session, top_k=5))

    ids = [c["id"] for c in result]
    # Chunk 2 appears in both lists so should rank highest after RRF
    assert ids[0] == 2
    # All 3 chunks should appear
    assert set(ids) == {1, 2, 3}


def test_keyword_search_returns_empty_on_exception(mock_session):
    """Should gracefully return empty list when keyword search fails."""
    import asyncio

    from src.services.retrieval import _keyword_search

    mock_session.execute.side_effect = Exception("full-text search not supported")

    result = asyncio.run(_keyword_search("test query", mock_session, limit=10))
    assert result == []


# --- Deduplication tests ---


def test_deduplicate_removes_identical_chunks():
    """Should remove chunks with identical text."""
    chunks = [
        _chunk(1, doc="a.pdf") | {"text": "the quick brown fox jumps over the lazy dog"},
        _chunk(2, doc="b.pdf") | {"text": "the quick brown fox jumps over the lazy dog"},
        _chunk(3, doc="c.pdf") | {"text": "completely different text about regulations"},
    ]
    result = _deduplicate_chunks(chunks, threshold=0.85)
    assert len(result) == 2
    assert result[0]["id"] == 1
    assert result[1]["id"] == 3


def test_deduplicate_removes_near_duplicates():
    """Should remove chunks that are near-duplicates above threshold."""
    base = "the quick brown fox jumps over the lazy dog near the river bank"
    # Change one word -- Jaccard should be high
    variant = "the quick brown fox jumps over the lazy dog near the river shore"
    chunks = [
        _chunk(1) | {"text": base},
        _chunk(2) | {"text": variant},
        _chunk(3) | {"text": "SEC regulation filing compliance requirements for banking"},
    ]
    result = _deduplicate_chunks(chunks, threshold=0.7)
    assert len(result) == 2
    assert result[0]["id"] == 1
    assert result[1]["id"] == 3


def test_deduplicate_keeps_distinct_chunks():
    """Should keep chunks below the similarity threshold."""
    chunks = [
        _chunk(1) | {"text": "federal banking regulations require annual compliance audits"},
        _chunk(2) | {"text": "SEC filing deadlines are quarterly for public companies"},
        _chunk(3) | {"text": "risk management frameworks include operational and market risk"},
    ]
    result = _deduplicate_chunks(chunks, threshold=0.85)
    assert len(result) == 3


def test_deduplicate_preserves_rank_order():
    """Should keep higher-ranked chunks and drop lower-ranked duplicates."""
    chunks = [
        _chunk(10) | {"text": "important regulatory requirement for compliance"},
        _chunk(20) | {"text": "different topic about financial markets"},
        _chunk(30) | {"text": "important regulatory requirement for compliance"},
    ]
    result = _deduplicate_chunks(chunks, threshold=0.85)
    assert [c["id"] for c in result] == [10, 20]


def test_deduplicate_empty_input():
    """Should return empty list for empty input."""
    assert _deduplicate_chunks([], threshold=0.85) == []


def test_deduplicate_single_chunk():
    """Should return single chunk unchanged."""
    chunks = [_chunk(1)]
    result = _deduplicate_chunks(chunks, threshold=0.85)
    assert len(result) == 1


def test_deduplicate_disabled_at_threshold_one():
    """Should skip dedup entirely when threshold is 1.0."""
    chunks = [
        _chunk(1) | {"text": "identical text"},
        _chunk(2) | {"text": "identical text"},
    ]
    result = _deduplicate_chunks(chunks, threshold=1.0)
    assert len(result) == 2


def test_retrieve_chunks_hybrid_with_dedup(mock_session):
    """Should deduplicate near-duplicate chunks in the hybrid retrieval path."""
    import asyncio

    shared_text = "the federal reserve requires banks to maintain adequate capital reserves"
    vector_results = [
        _chunk(1, doc="a.pdf") | {"text": shared_text},
        _chunk(2, doc="b.pdf") | {"text": "SEC quarterly filing requirements for public companies"},
    ]
    keyword_results = [
        # Same text as chunk 1, different ID (keyword search found it separately)
        _chunk(3, doc="a.pdf") | {"text": shared_text},
        _chunk(4, doc="c.pdf") | {"text": "FINRA rules for broker-dealer registration"},
    ]

    with (
        patch(
            "src.services.retrieval.generate_embeddings",
            new_callable=AsyncMock,
            return_value=EmbeddingsResult(vectors=[[0.1, 0.2]], error=None),
        ),
        patch(
            "src.services.retrieval._vector_search",
            new_callable=AsyncMock,
            return_value=vector_results,
        ),
        patch(
            "src.services.retrieval._keyword_search",
            new_callable=AsyncMock,
            return_value=keyword_results,
        ),
        patch(
            "src.services.retrieval._count_ready_documents",
            new_callable=AsyncMock,
            return_value=3,
        ),
    ):
        result = asyncio.run(
            retrieve_chunks("test query", mock_session, top_k=5, dedup_threshold=0.85)
        )

    texts = [c["text"] for c in result]
    # The duplicate text should appear only once
    assert texts.count(shared_text) == 1
    # All unique chunks should be present
    assert len(result) == 3


# --- Threshold-based diversity tests ---


def test_diversity_skips_low_relevance_docs():
    """Should not promote documents whose best chunk is below the relevance threshold."""
    chunks = [
        _chunk(1, doc="strong.pdf", score=0.8),
        _chunk(2, doc="strong.pdf", score=0.7),
        _chunk(3, doc="strong.pdf", score=0.6),
        _chunk(4, doc="weak.pdf", score=0.1),  # below threshold
        _chunk(5, doc="medium.pdf", score=0.4),
    ]
    result = _apply_diversity(
        chunks, top_k=3, max_per_doc=None, diversity_min=3, relevance_threshold=0.3
    )
    docs = {c["source_document"] for c in result}
    # weak.pdf should NOT get a guaranteed slot (score 0.1 < threshold 0.3)
    # strong.pdf and medium.pdf should be present
    assert "strong.pdf" in docs
    assert "medium.pdf" in docs


def test_diversity_guarantees_slots_for_relevant_docs():
    """Should guarantee slots for all documents above the relevance threshold."""
    chunks = [
        _chunk(1, doc="a.pdf", score=0.9),
        _chunk(2, doc="a.pdf", score=0.85),
        _chunk(3, doc="a.pdf", score=0.8),
        _chunk(4, doc="a.pdf", score=0.75),
        _chunk(5, doc="b.pdf", score=0.5),
        _chunk(6, doc="c.pdf", score=0.4),
        _chunk(7, doc="d.pdf", score=0.35),
    ]
    result = _apply_diversity(
        chunks, top_k=5, max_per_doc=None, diversity_min=5, relevance_threshold=0.3
    )
    docs = {c["source_document"] for c in result}
    # All 4 docs are above threshold 0.3, so all should be represented
    assert docs == {"a.pdf", "b.pdf", "c.pdf", "d.pdf"}


def test_diversity_threshold_zero_promotes_all():
    """Should promote all documents when threshold is 0.0 (effectively disabled)."""
    chunks = [
        _chunk(1, doc="a.pdf", score=0.9),
        _chunk(2, doc="a.pdf", score=0.8),
        _chunk(3, doc="b.pdf", score=0.01),
    ]
    result = _apply_diversity(
        chunks, top_k=3, max_per_doc=None, diversity_min=3, relevance_threshold=0.0
    )
    docs = {c["source_document"] for c in result}
    assert "b.pdf" in docs


# --- Pre-computed embedding tests ---


def test_retrieve_chunks_uses_precomputed_embedding(mock_session):
    """Should skip generate_embeddings when query_embedding is provided."""
    import asyncio

    pre_embedding = [0.1, 0.2, 0.3]
    vector_results = [_chunk(1, doc="a.pdf")]

    with (
        patch(
            "src.services.retrieval.generate_embeddings",
            new_callable=AsyncMock,
        ) as mock_embed,
        patch(
            "src.services.retrieval._vector_search",
            new_callable=AsyncMock,
            return_value=vector_results,
        ) as mock_vs,
        patch(
            "src.services.retrieval._keyword_search",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
        result = asyncio.run(
            retrieve_chunks(
                "test query", mock_session, diversity_min=1,
                query_embedding=pre_embedding,
            ),
        )

    mock_embed.assert_not_called()
    mock_vs.assert_called_once()
    assert mock_vs.call_args[0][0] == pre_embedding
    assert len(result) == 1


def test_retrieve_chunks_falls_back_when_no_precomputed_embedding(mock_session):
    """Should call generate_embeddings when query_embedding is None."""
    import asyncio

    with (
        patch(
            "src.services.retrieval.generate_embeddings",
            new_callable=AsyncMock,
            return_value=EmbeddingsResult(vectors=[[0.1, 0.2]], error=None),
        ) as mock_embed,
        patch(
            "src.services.retrieval._vector_search",
            new_callable=AsyncMock,
            return_value=[_chunk(1)],
        ),
        patch(
            "src.services.retrieval._keyword_search",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
        result = asyncio.run(
            retrieve_chunks("test query", mock_session, diversity_min=1),
        )

    mock_embed.assert_called_once()
    assert len(result) == 1
