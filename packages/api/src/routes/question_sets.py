"""Question set endpoints -- create, list, and delete reusable question sets."""

import logging
from datetime import datetime
from typing import cast

from db import QuestionSet, get_db
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..core.config import settings
from ..schemas.question_set import (
    QuestionSetCreate,
    QuestionSetItem,
    QuestionSetResponse,
    QuestionSetUpdate,
)
from ..services.profiles import RetrievalConfig, load_profile
from ..services.truth_generation import generate_truth_from_manual_answer

logger = logging.getLogger(__name__)
router = APIRouter()


def _build_response(qs: QuestionSet) -> QuestionSetResponse:
    """Build response, normalizing old plain-string questions to objects."""
    items = []
    for q in qs.questions:
        if isinstance(q, str):
            items.append(QuestionSetItem(question=q))
        elif isinstance(q, dict):
            items.append(QuestionSetItem(**q))
        else:
            items.append(q)
    return QuestionSetResponse(
        id=qs.id,
        name=qs.name,
        questions=items,
        created_at=cast(datetime, qs.created_at),
        updated_at=cast(datetime | None, getattr(qs, "updated_at", None)),
    )


async def _normalize_and_enrich(
    questions: list[str | QuestionSetItem],
    profile_id: str | None,
    session: AsyncSession,
) -> list[dict]:
    """Normalize questions and generate truth payloads where needed."""
    normalized: list[dict] = []
    for q in questions:
        if isinstance(q, str):
            normalized.append({"question": q})
        else:
            normalized.append(q.model_dump(mode="json", exclude_none=True))

    judge_model = settings.resolved_judge_model_name
    if judge_model:
        retrieval_cfg = RetrievalConfig()
        if profile_id:
            try:
                prof = load_profile(profile_id)
                retrieval_cfg = prof.retrieval
            except (FileNotFoundError, ValueError):
                pass
        truth_retrieval_kwargs = {
            "top_k": retrieval_cfg.top_k,
            "max_per_doc": retrieval_cfg.max_chunks_per_document,
            "rerank_depth": retrieval_cfg.rerank_depth,
            "diversity_min": retrieval_cfg.document_diversity_min,
            "keyword_enabled": retrieval_cfg.keyword_search_enabled,
            "dedup_threshold": retrieval_cfg.dedup_threshold,
            "diversity_relevance_threshold": retrieval_cfg.diversity_relevance_threshold,
        }
        for q in normalized:
            if q.get("expected_answer") and not q.get("truth"):
                try:
                    truth = await generate_truth_from_manual_answer(
                        q["question"],
                        q["expected_answer"],
                        session,
                        judge_model,
                        retrieval_kwargs=truth_retrieval_kwargs,
                    )
                    q["truth"] = truth.model_dump(mode="json")
                except Exception as e:
                    logger.warning("Truth generation failed for manual question: %s", e)

    return normalized


@router.post("/", response_model=QuestionSetResponse, status_code=201)
async def create_question_set(
    request: QuestionSetCreate,
    session: AsyncSession = Depends(get_db),
) -> QuestionSetResponse:
    """Save a reusable set of evaluation questions."""
    normalized = await _normalize_and_enrich(request.questions, request.profile_id, session)

    qs = QuestionSet(name=request.name, questions=normalized)
    session.add(qs)
    await session.flush()
    response = _build_response(qs)
    await session.commit()
    return response


@router.get("/", response_model=list[QuestionSetResponse])
async def list_question_sets(
    session: AsyncSession = Depends(get_db),
) -> list[QuestionSetResponse]:
    """List all saved question sets, most recent first."""
    result = await session.execute(select(QuestionSet).order_by(QuestionSet.created_at.desc()))
    return [_build_response(qs) for qs in result.scalars().all()]


@router.get("/{question_set_id}", response_model=QuestionSetResponse)
async def get_question_set(
    question_set_id: int,
    session: AsyncSession = Depends(get_db),
) -> QuestionSetResponse:
    """Get a single question set by ID."""
    qs = await session.get(QuestionSet, question_set_id)
    if not qs:
        raise HTTPException(status_code=404, detail="Question set not found")
    return _build_response(qs)


@router.patch("/{question_set_id}", response_model=QuestionSetResponse)
async def update_question_set(
    question_set_id: int,
    request: QuestionSetUpdate,
    session: AsyncSession = Depends(get_db),
) -> QuestionSetResponse:
    """Partially update a question set (name, questions, or both)."""
    qs = await session.get(QuestionSet, question_set_id)
    if not qs:
        raise HTTPException(status_code=404, detail="Question set not found")

    if request.name is not None:
        qs.name = request.name

    if request.questions is not None:
        qs.questions = await _normalize_and_enrich(request.questions, request.profile_id, session)

    await session.flush()
    await session.refresh(qs)
    response = _build_response(qs)
    await session.commit()
    return response


@router.delete("/{question_set_id}", status_code=204)
async def delete_question_set(
    question_set_id: int,
    session: AsyncSession = Depends(get_db),
) -> None:
    """Delete a question set and all associated eval runs/results."""
    result = await session.execute(
        select(QuestionSet)
        .options(selectinload(QuestionSet.eval_runs))
        .where(QuestionSet.id == question_set_id)
    )
    qs = result.scalars().first()
    if not qs:
        raise HTTPException(status_code=404, detail="Question set not found")
    await session.delete(qs)
    await session.commit()
