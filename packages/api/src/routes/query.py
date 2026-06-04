"""RAG query endpoint -- retrieve context and generate an answer."""

import logging
from collections import defaultdict

from db import get_db
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import settings
from ..schemas.query import (
    DebugRetrievalRequest,
    DebugRetrievalResponse,
    DocumentScore,
    QueryRequest,
    QueryResponse,
    SourceChunk,
    UsageInfo,
)
from ..services.generation import generate_answer
from ..services.profiles import load_profile
from ..services.query_decomposition import decompose_query
from ..services.retrieval import _apply_diversity, _deduplicate_chunks, retrieve_chunks
from ..services.safety import check_input_safety, check_output_safety

logger = logging.getLogger(__name__)
router = APIRouter()

CONFIDENCE_THRESHOLD = 0.5


@router.post("/", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    session: AsyncSession = Depends(get_db),
) -> QueryResponse:
    """Ask a question against the RAG knowledge base.

    1. Retrieves the top-k most relevant chunks via vector similarity.
    2. Sends the chunks as context to the specified LLM.
    3. Returns the generated answer with source citations.
    """
    if not settings.any_token_configured:
        raise HTTPException(
            status_code=400,
            detail="No API token configured. Set API_TOKEN in your environment.",
        )

    # Validate model_name
    valid_models = [settings.MODEL_A_NAME, settings.MODEL_B_NAME]
    if request.model_name not in valid_models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model name. Available: {settings.MODEL_A_NAME}, {settings.MODEL_B_NAME}",
        )

    # Pre-retrieval safety check on user input
    input_safety = await check_input_safety(request.question)
    if not input_safety.is_safe:
        logger.info("Query blocked by input safety filter (category=%s)", input_safety.category)
        return QueryResponse(
            answer="I'm unable to process this request.",
            model=request.model_name,
            sources=[],
            safety_filtered=True,
        )

    # Load profile retrieval config so chat uses the same pipeline as evaluation
    retrieval_kwargs: dict = {}
    system_prompt: str | None = None
    gen_max_tokens: int | None = None
    try:
        profile = load_profile("fsi_compliance_v1")
        r = profile.retrieval
        retrieval_kwargs = {
            "top_k": r.top_k,
            "max_per_doc": r.max_chunks_per_document,
            "rerank_depth": r.rerank_depth,
            "diversity_min": r.document_diversity_min,
            "keyword_enabled": r.keyword_search_enabled,
            "dedup_threshold": r.dedup_threshold,
            "diversity_relevance_threshold": r.diversity_relevance_threshold,
        }
        if profile.system_prompt:
            system_prompt = profile.system_prompt
        if profile.generation and profile.generation.max_tokens:
            gen_max_tokens = profile.generation.max_tokens
    except (FileNotFoundError, ValueError) as e:
        logger.warning("Could not load chat profile: %s", e)

    if request.top_k is not None:
        retrieval_kwargs["top_k"] = request.top_k

    chunks = await retrieve_chunks(
        query=request.question,
        session=session,
        **retrieval_kwargs,
    )

    result = await generate_answer(
        question=request.question,
        chunks=chunks,
        model_name=request.model_name,
        system_prompt=system_prompt,
        max_tokens=gen_max_tokens,
    )

    # Post-generation safety check on model output
    output_safety = await check_output_safety(result["answer"])
    if not output_safety.is_safe:
        logger.info(
            "Response blocked by output safety filter (category=%s)", output_safety.category
        )
        return QueryResponse(
            answer="The generated response was filtered for safety.",
            model=result["model"],
            sources=[],
            safety_filtered=True,
        )

    sources = [
        SourceChunk(
            id=c["id"],
            text=c["text"],
            source_document=c["source_document"],
            page_number=c.get("page_number"),
            score=c["score"],
        )
        for c in chunks
    ]

    usage = None
    if result.get("usage"):
        usage = UsageInfo(**result["usage"])

    low_confidence = len(sources) > 0 and all(s.score < CONFIDENCE_THRESHOLD for s in sources)

    return QueryResponse(
        answer=result["answer"],
        model=result["model"],
        sources=sources,
        usage=usage,
        low_confidence=low_confidence,
    )


@router.post("/debug", response_model=DebugRetrievalResponse)
async def debug_retrieval(
    request: DebugRetrievalRequest,
    session: AsyncSession = Depends(get_db),
) -> DebugRetrievalResponse:
    """Debug retrieval pipeline for a question.

    Shows query decomposition, per-document scores, diversity decisions,
    and final chunk selection without running generation or scoring.
    """
    # Load retrieval config from profile
    retrieval_kwargs: dict = {}
    diversity_threshold = 0.3
    top_k = 10
    if request.profile_id:
        try:
            profile = load_profile(request.profile_id)
            r = profile.retrieval
            retrieval_kwargs = {
                "top_k": r.top_k,
                "max_per_doc": r.max_chunks_per_document,
                "rerank_depth": r.rerank_depth,
                "diversity_min": r.document_diversity_min,
                "keyword_enabled": r.keyword_search_enabled,
                "dedup_threshold": r.dedup_threshold,
                "diversity_relevance_threshold": r.diversity_relevance_threshold,
            }
            diversity_threshold = r.diversity_relevance_threshold
            top_k = r.top_k
        except (FileNotFoundError, ValueError) as e:
            logger.warning("Could not load profile '%s': %s", request.profile_id, e)

    top_k = retrieval_kwargs.get("top_k", top_k)

    # Decompose query
    sub_queries = await decompose_query(request.question)

    # Retrieve chunks (same logic as evaluation)
    if len(sub_queries) <= 1:
        chunks = await retrieve_chunks(
            query=request.question, session=session, **retrieval_kwargs
        )
        all_candidates = chunks
    else:
        all_candidates = []
        seen_ids: set[int] = set()
        for sq in sub_queries:
            sq_chunks = await retrieve_chunks(
                query=sq, session=session, **retrieval_kwargs
            )
            for chunk in sq_chunks:
                if chunk["id"] not in seen_ids:
                    all_candidates.append(chunk)
                    seen_ids.add(chunk["id"])

        all_candidates.sort(key=lambda c: c.get("score", 0.0), reverse=True)
        dedup_threshold = retrieval_kwargs.get("dedup_threshold", 0.85)
        all_candidates = _deduplicate_chunks(all_candidates, dedup_threshold)

        diversity_min = retrieval_kwargs.get("diversity_min", 3)
        max_per_doc = retrieval_kwargs.get("max_per_doc")
        threshold = retrieval_kwargs.get("diversity_relevance_threshold", 0.3)
        chunks = _apply_diversity(
            all_candidates,
            top_k=top_k,
            max_per_doc=max_per_doc,
            diversity_min=diversity_min,
            relevance_threshold=threshold,
        )

    # Build per-document summary from all candidates
    doc_best: dict[str, float] = defaultdict(float)
    doc_count: dict[str, int] = defaultdict(int)
    for c in all_candidates:
        doc = c["source_document"]
        doc_count[doc] += 1
        score = c.get("score", 0.0)
        if score > doc_best[doc]:
            doc_best[doc] = score

    documents = [
        DocumentScore(
            document=doc,
            chunk_count=doc_count[doc],
            best_score=round(doc_best[doc], 4),
            qualifies_for_diversity=doc_best[doc] >= diversity_threshold,
        )
        for doc in sorted(doc_best, key=lambda d: doc_best[d], reverse=True)
    ]

    final_chunks = [
        SourceChunk(
            id=c["id"],
            text=c["text"][:200] + "..." if len(c["text"]) > 200 else c["text"],
            source_document=c["source_document"],
            page_number=c.get("page_number"),
            score=c.get("score", 0.0),
        )
        for c in chunks
    ]

    return DebugRetrievalResponse(
        question=request.question,
        sub_queries=sub_queries,
        total_candidates=len(all_candidates),
        documents=documents,
        final_chunks=final_chunks,
        diversity_threshold=diversity_threshold,
        top_k=top_k,
    )
