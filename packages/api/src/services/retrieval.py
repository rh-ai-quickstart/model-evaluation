# This project was developed with assistance from AI tools.
"""Retrieval service -- hybrid search with vector similarity and keyword matching.

Supports profile-driven configuration: top_k, max_chunks_per_document,
rerank_depth, document_diversity_min, and keyword_search_enabled.
Merges vector and keyword results via Reciprocal Rank Fusion (RRF).
"""

import logging
from collections import defaultdict

from db import Chunk, Document
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import settings
from .embedding import QUERY_PREFIX, generate_embeddings

logger = logging.getLogger(__name__)

TOP_K = 5  # default number of chunks to retrieve


def compute_search_depth(
    rerank_depth: int,
    diversity_min: int,
    doc_count: int,
    max_search_depth: int | None = None,
) -> int:
    """Size of the vector/keyword candidate pool before RRF and diversity filtering.

    When ``diversity_min > 1``, the pool widens to ``max(rerank_depth,
    doc_count * rerank_depth)`` so smaller documents are not starved by one
    large file. That product can grow without bound as the corpus grows; the
    cap keeps DB and RRF work predictable (latency).

    Args:
        rerank_depth: Profile baseline depth (RRF candidate budget).
        diversity_min: When <= 1, only ``rerank_depth`` is used.
        doc_count: Number of ready, non-deleted documents in the corpus.
        max_search_depth: Upper bound (defaults to ``settings.RETRIEVAL_MAX_SEARCH_DEPTH``).

    Returns:
        Effective LIMIT for vector and keyword retrieval passes.
    """
    cap = max_search_depth if max_search_depth is not None else settings.RETRIEVAL_MAX_SEARCH_DEPTH
    if diversity_min <= 1:
        return rerank_depth

    raw = max(rerank_depth, doc_count * rerank_depth)
    if raw > cap:
        logger.info(
            "Retrieval search_depth capped: uncapped=%d (docs=%d, rerank_depth=%d) -> %d",
            raw,
            doc_count,
            rerank_depth,
            cap,
        )
        return cap
    return raw


async def retrieve_chunks(
    query: str,
    session: AsyncSession,
    top_k: int = TOP_K,
    max_per_doc: int | None = None,
    rerank_depth: int = 20,
    diversity_min: int = 3,
    keyword_enabled: bool = True,
    dedup_threshold: float = 0.85,
    diversity_relevance_threshold: float = 0.3,
) -> list[dict]:
    """Retrieve the most relevant chunks for a query.

    Uses hybrid retrieval when keyword search is enabled: vector similarity
    and PostgreSQL full-text search merged via Reciprocal Rank Fusion (RRF).
    If embeddings are unavailable, runs keyword search when enabled; only if that is
    empty does it fall back to recent chunks (which are **not** query-specific—every
    question would otherwise see the same context).

    Args:
        query: The search query.
        session: Database session.
        top_k: Number of chunks to return.
        max_per_doc: Max chunks per source document (None = unlimited).
        rerank_depth: Number of RRF candidates to consider before selecting top_k.
        diversity_min: Soft target for minimum number of distinct source documents.
        keyword_enabled: Whether to include keyword search in hybrid retrieval.
        dedup_threshold: Jaccard similarity threshold for dropping near-duplicate
            chunks. Set to 1.0 to disable deduplication.
        diversity_relevance_threshold: Minimum score a document's best chunk must
            have to qualify for guaranteed diversity slots. Documents below this
            threshold get zero guaranteed representation.

    Returns:
        List of dicts with 'id', 'text', 'source_document', 'page_number',
        'section_path', 'score' keys, ordered by relevance.
    """
    doc_count = 1
    if diversity_min > 1:
        doc_count = await _count_ready_documents(session)
    search_depth = compute_search_depth(rerank_depth, diversity_min, doc_count)

    result = await generate_embeddings([query], prefix=QUERY_PREFIX)

    if result.vectors:
        vector_results = await _vector_search(result.vectors[0], session, search_depth)
    else:
        # Without vectors, never jump straight to "recent chunks" — that path ignores the
        # query and makes every eval question receive identical context (same answers at temp 0).
        logger.warning(
            "No query embeddings (%s); trying keyword-only retrieval before non-query fallback",
            result.error or "vectors absent",
        )
        keyword_only: list[dict] = []
        if keyword_enabled:
            keyword_only = await _keyword_search(query, session, search_depth)
        if keyword_only:
            deduped = _deduplicate_chunks(keyword_only, dedup_threshold)
            final = _apply_diversity(
                deduped,
                top_k,
                max_per_doc,
                diversity_min,
                diversity_relevance_threshold,
            )
            kw_doc_counts: dict[str, int] = defaultdict(int)
            for chunk in final:
                kw_doc_counts[chunk["source_document"]] += 1
            logger.info(
                "Embedding-less retrieval: keyword-only -> %d chunks, doc distribution: %s",
                len(final),
                dict(kw_doc_counts),
            )
            return final

        logger.error(
            "Embeddings unavailable and keyword search returned no hits — "
            "using recent chunks (identical for all queries). Fix embedding service or corpus."
        )
        return await _fallback_search(session, top_k)

    # Keyword search (graceful degradation: returns [] if not supported)
    keyword_results = []
    if keyword_enabled:
        keyword_results = await _keyword_search(query, session, search_depth)

    # Merge via RRF
    if keyword_results:
        merged = _reciprocal_rank_fusion(vector_results, keyword_results)
    else:
        merged = vector_results

    # Log per-document score breakdown before diversity enforcement
    doc_scores: dict[str, list[float]] = defaultdict(list)
    for chunk in merged:
        doc_scores[chunk["source_document"]].append(chunk.get("score", 0.0))
    logger.info(
        "Retrieval candidates: %d chunks from %d docs. Per-doc best scores: %s",
        len(merged),
        len(doc_scores),
        {doc: round(max(scores), 4) for doc, scores in doc_scores.items()},
    )

    # Remove near-duplicate chunks before diversity enforcement
    deduped = _deduplicate_chunks(merged, dedup_threshold)

    # Apply document diversity and per-doc caps
    final = _apply_diversity(
        deduped, top_k, max_per_doc, diversity_min, diversity_relevance_threshold
    )

    # Log final selection
    final_docs: dict[str, int] = defaultdict(int)
    for chunk in final:
        final_docs[chunk["source_document"]] += 1
    logger.info(
        "Final retrieval: %d chunks, doc distribution: %s",
        len(final),
        dict(final_docs),
    )

    return final


async def _count_ready_documents(session: AsyncSession) -> int:
    """Count documents with status 'ready' that haven't been deleted."""
    stmt = select(func.count(Document.id)).where(
        Document.status == "ready", Document.deleted_at.is_(None)
    )
    result = await session.execute(stmt)
    return result.scalar_one()


async def _vector_search(
    query_embedding: list[float],
    session: AsyncSession,
    limit: int,
) -> list[dict]:
    """Search chunks by cosine similarity to the query embedding."""
    dist = Chunk.embedding.cosine_distance(query_embedding)

    stmt = (
        select(
            Chunk.id,
            Chunk.text,
            Chunk.source_document,
            Chunk.page_number,
            Chunk.section_path,
            (1 - dist).label("score"),
        )
        .join(Document, Chunk.document_id == Document.id)
        .where(Chunk.embedding.isnot(None), Document.deleted_at.is_(None))
        .order_by(dist)
        .limit(limit)
    )

    result = await session.execute(stmt)
    rows = result.all()

    return [
        {
            "id": row.id,
            "text": row.text,
            "source_document": row.source_document,
            "page_number": row.page_number,
            "section_path": row.section_path,
            "score": round(float(row.score), 4),
        }
        for row in rows
    ]


async def _keyword_search(
    query: str,
    session: AsyncSession,
    limit: int,
) -> list[dict]:
    """Search chunks using PostgreSQL full-text search.

    Returns empty list gracefully if full-text search is not available
    (e.g., SQLite in tests).
    """
    try:
        ts_query = func.plainto_tsquery("english", query)
        ts_vector = func.to_tsvector("english", Chunk.text)
        rank = func.ts_rank(ts_vector, ts_query)

        stmt = (
            select(
                Chunk.id,
                Chunk.text,
                Chunk.source_document,
                Chunk.page_number,
                Chunk.section_path,
                rank.label("score"),
            )
            .join(Document, Chunk.document_id == Document.id)
            .where(ts_vector.bool_op("@@")(ts_query), Document.deleted_at.is_(None))
            .order_by(rank.desc())
            .limit(limit)
        )

        result = await session.execute(stmt)
        rows = result.all()

        return [
            {
                "id": row.id,
                "text": row.text,
                "source_document": row.source_document,
                "page_number": row.page_number,
                "section_path": row.section_path,
                "score": round(float(row.score), 4),
            }
            for row in rows
        ]
    except Exception:
        # Graceful fallback for SQLite or unsupported databases
        logger.debug("Keyword search not available, skipping")
        return []


def _reciprocal_rank_fusion(
    *result_lists: list[dict],
    k: int = 60,
) -> list[dict]:
    """Merge multiple ranked result lists using Reciprocal Rank Fusion.

    RRF score = sum(1 / (k + rank)) across all lists.

    Args:
        result_lists: Multiple ranked lists of chunk dicts.
        k: RRF constant (default 60, standard value from the RRF paper).

    Returns:
        Merged list sorted by combined RRF score.
    """
    rrf_scores: dict[int, float] = defaultdict(float)
    chunk_map: dict[int, dict] = {}

    for results in result_lists:
        for rank, chunk in enumerate(results):
            chunk_id = chunk["id"]
            rrf_scores[chunk_id] += 1.0 / (k + rank + 1)
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = chunk

    sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

    return [{**chunk_map[cid], "score": round(rrf_scores[cid], 4)} for cid in sorted_ids]


def _deduplicate_chunks(
    chunks: list[dict],
    threshold: float = 0.85,
) -> list[dict]:
    """Remove near-duplicate chunks using word-level Jaccard similarity.

    Iterates chunks in rank order (highest-ranked first). For each chunk,
    computes Jaccard similarity against all previously kept chunks. If any
    similarity exceeds the threshold, the chunk is dropped.

    Args:
        chunks: Ranked list of chunks (already sorted by relevance/RRF).
        threshold: Jaccard similarity above which a chunk is considered
            a duplicate. Set to 1.0 to disable deduplication.

    Returns:
        Filtered list with near-duplicates removed, preserving rank order.
    """
    if threshold >= 1.0 or len(chunks) <= 1:
        return chunks

    kept: list[dict] = []
    kept_word_sets: list[set[str]] = []

    for chunk in chunks:
        words = set(chunk["text"].lower().split())
        is_dup = False
        for kept_words in kept_word_sets:
            union = words | kept_words
            if not union:
                continue
            jaccard = len(words & kept_words) / len(union)
            if jaccard > threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(chunk)
            kept_word_sets.append(words)

    if len(kept) < len(chunks):
        logger.debug(
            "Dedup removed %d/%d chunks (threshold=%.2f)",
            len(chunks) - len(kept),
            len(chunks),
            threshold,
        )

    return kept


def _apply_diversity(
    chunks: list[dict],
    top_k: int,
    max_per_doc: int | None,
    diversity_min: int,
    relevance_threshold: float = 0.3,
) -> list[dict]:
    """Apply threshold-based document diversity and per-document caps.

    Strategy:
    1. If max_per_doc is set, cap chunks per source document.
    2. Identify documents that have at least one chunk above the relevance
       threshold -- only these qualify for guaranteed diversity slots.
    3. Guarantee one chunk per qualifying document (up to diversity_min).
    4. Fill remaining slots by rank order.

    Documents below the relevance threshold get zero guaranteed representation,
    preventing low-quality chunks from diluting strong signals.

    Args:
        chunks: Ranked list of chunks (already sorted by relevance/RRF).
        top_k: Number of chunks to return.
        max_per_doc: Max chunks per document (None = unlimited).
        diversity_min: Soft target for document diversity.
        relevance_threshold: Minimum score for a document's best chunk to
            qualify for a guaranteed diversity slot.

    Returns:
        Filtered list of up to top_k chunks.
    """
    if not chunks:
        return []

    # Apply per-document cap
    if max_per_doc:
        doc_counts: dict[str, int] = defaultdict(int)
        capped: list[dict] = []
        deferred: list[dict] = []
        for chunk in chunks:
            doc = chunk["source_document"]
            if doc_counts[doc] < max_per_doc:
                capped.append(chunk)
                doc_counts[doc] += 1
            else:
                deferred.append(chunk)
        chunks = capped + deferred

    # Threshold-based diversity: only promote documents with relevant chunks
    if diversity_min > 1 and len(chunks) > top_k:
        # Find the best score per document
        best_score_per_doc: dict[str, float] = {}
        for chunk in chunks:
            doc = chunk["source_document"]
            score = chunk.get("score", 0.0)
            if doc not in best_score_per_doc or score > best_score_per_doc[doc]:
                best_score_per_doc[doc] = score

        # Only documents above the relevance threshold qualify for guaranteed slots
        qualifying_docs = {
            doc for doc, score in best_score_per_doc.items() if score >= relevance_threshold
        }

        selected: list[dict] = []
        remaining: list[dict] = list(chunks)
        seen_docs: set[str] = set()

        # First pass: pick one from each qualifying unseen document
        for chunk in list(remaining):
            if len(seen_docs) >= diversity_min:
                break
            doc = chunk["source_document"]
            if doc not in seen_docs and doc in qualifying_docs:
                selected.append(chunk)
                remaining.remove(chunk)
                seen_docs.add(doc)

        # Fill rest by rank order
        for chunk in remaining:
            if len(selected) >= top_k:
                break
            selected.append(chunk)

        non_qualifying = set(best_score_per_doc.keys()) - qualifying_docs
        logger.info(
            "Diversity enforcement: %d/%d docs qualify (threshold=%.2f), "
            "%d guaranteed slots used. Qualifying: %s",
            len(qualifying_docs),
            len(best_score_per_doc),
            relevance_threshold,
            len(seen_docs),
            {doc: round(best_score_per_doc[doc], 4) for doc in qualifying_docs},
        )
        if non_qualifying:
            logger.info(
                "Non-qualifying docs (below threshold %.2f): %s",
                relevance_threshold,
                {doc: round(best_score_per_doc[doc], 4) for doc in non_qualifying},
            )

        return selected[:top_k]

    return chunks[:top_k]


async def _fallback_search(
    session: AsyncSession,
    top_k: int,
) -> list[dict]:
    """Return the most recent chunks when vector search is unavailable."""
    stmt = (
        select(
            Chunk.id,
            Chunk.text,
            Chunk.source_document,
            Chunk.page_number,
            Chunk.section_path,
        )
        .join(Document, Chunk.document_id == Document.id)
        .where(Document.deleted_at.is_(None))
        .order_by(Chunk.created_at.desc())
        .limit(top_k)
    )

    result = await session.execute(stmt)
    rows = result.all()

    return [
        {
            "id": row.id,
            "text": row.text,
            "source_document": row.source_document,
            "page_number": row.page_number,
            "section_path": getattr(row, "section_path", None),
            "score": 0.0,
        }
        for row in rows
    ]
