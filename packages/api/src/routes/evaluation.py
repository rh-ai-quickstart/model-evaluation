# This project was developed with assistance from AI tools.
"""Evaluation endpoints -- create, list, and inspect evaluation runs."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

from db import Document, EvalResult, EvalRun, get_db
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import settings
from ..schemas.evaluation import (
    ComparisonDecision,
    ComparisonMetric,
    ComparisonResponse,
    ComparisonWarning,
    EvalQuestion,
    EvalResultResponse,
    EvalRunCreate,
    EvalRunCreateResponse,
    EvalRunDetailResponse,
    EvalRunRerun,
    EvalRunResponse,
    QuestionComparison,
    SynthesizedQuestion,
    SynthesizeRequest,
    SynthesizeResponse,
)
from ..services.coverage import detect_coverage_gaps
from ..services.deterministic_checks import run_deterministic_checks
from ..services.generation import generate_answer
from ..services.profiles import EvalProfile, list_profiles, load_profile
from ..services.query_decomposition import decompose_query
from ..services.retrieval import _apply_diversity, _deduplicate_chunks, retrieve_chunks
from ..services.scoring import compute_chunk_alignment, score_result
from ..services.synthesizer import generate_questions
from ..services.truth_generation import generate_truth_from_manual_answer
from ..services.verdicts import (
    QuestionVerdict,
    compute_comparison_decision,
    compute_question_verdict,
    compute_run_verdict,
)

logger = logging.getLogger(__name__)
router = APIRouter()

COMPARISON_TIE_THRESHOLD = 0.05

# In-memory set of run IDs that have been requested to cancel.
# The background task checks this between questions and stops early.
_cancelled_runs: set[int] = set()


@dataclass
class _QuestionResult:
    """Result of processing a single evaluation question."""

    question: str
    expected_answer: str | None = None
    answer: str | None = None
    contexts_str: str | None = None
    latency_ms: float = 0.0
    tokens: int | None = None
    scores: dict = field(default_factory=dict)
    chunk_alignment_score: float | None = None
    coverage_gaps: dict | None = None
    deterministic_checks: list[dict] | None = None
    truth_payload: dict | None = None
    error: str | None = None
    verdict: QuestionVerdict | None = None


_MAX_CONCURRENT_QUESTIONS = 6

# Keys accepted by retrieve_chunks(); profile-only merge options must be excluded.
_RETRIEVE_CHUNKS_KWARGS = frozenset(
    {
        "top_k",
        "max_per_doc",
        "rerank_depth",
        "diversity_min",
        "keyword_enabled",
        "dedup_threshold",
        "diversity_relevance_threshold",
    }
)


def _build_retrieval_kwargs(profile) -> dict:
    """Map profile retrieval config to retrieve_chunks parameter names."""
    r = profile.retrieval
    return {
        "top_k": r.top_k,
        "max_per_doc": r.max_chunks_per_document,
        "rerank_depth": r.rerank_depth,
        "diversity_min": r.document_diversity_min,
        "keyword_enabled": r.keyword_search_enabled,
        "dedup_threshold": r.dedup_threshold,
        "diversity_relevance_threshold": r.diversity_relevance_threshold,
        "ensure_sub_query_representation": r.ensure_sub_query_representation,
        "sub_query_swap_score_multiplier": r.sub_query_swap_score_multiplier,
    }


def _retrieve_chunks_kwargs(retrieval_kwargs: dict) -> dict:
    """Strip evaluation-only keys before calling retrieve_chunks."""
    return {k: v for k, v in retrieval_kwargs.items() if k in _RETRIEVE_CHUNKS_KWARGS}


async def _build_corpus_snapshot(session: AsyncSession) -> dict:
    """Build a lightweight corpus snapshot for reproducibility.

    Returns a dict with document metadata and totals. Does not store
    chunk text -- only IDs, filenames, and counts.
    """
    result = await session.execute(
        select(Document)
        .where(Document.status == "ready", Document.deleted_at.is_(None))
        .order_by(Document.id)
    )
    docs = result.scalars().all()
    total_chunks = sum(d.chunk_count for d in docs)
    return {
        "documents": [
            {
                "id": d.id,
                "filename": d.filename,
                "chunk_count": d.chunk_count,
                "status": d.status,
            }
            for d in docs
        ],
        "total_documents": len(docs),
        "total_chunks": total_chunks,
    }


async def _process_question(
    q_item: EvalQuestion,
    model_name: str,
    semaphore: asyncio.Semaphore,
    eval_run_id: int,
    profile: object | None,
) -> _QuestionResult:
    """Process a single question: retrieve, generate, score.

    Runs under a semaphore to limit concurrency. Uses its own DB session
    for retrieval (read-only) to avoid sharing sessions across coroutines.
    """
    from db.database import SessionLocal

    question = q_item.question
    expected_answer = q_item.expected_answer
    result = _QuestionResult(
        question=question,
        expected_answer=expected_answer,
        truth_payload=q_item.truth.model_dump(mode="json") if q_item.truth else None,
    )

    async with semaphore:
        if eval_run_id in _cancelled_runs:
            return result

        start = time.time()
        try:
            # Build retrieval kwargs from profile settings
            retrieval_kwargs: dict = {}
            if profile and hasattr(profile, "retrieval"):
                retrieval_kwargs = _build_retrieval_kwargs(profile)

            # Decompose broad questions into targeted sub-queries for
            # multi-document retrieval, then merge results across sub-queries
            sub_queries = await decompose_query(question)

            rk = _retrieve_chunks_kwargs(retrieval_kwargs)
            if len(sub_queries) <= 1:
                async with SessionLocal() as retrieval_session:
                    chunks = await retrieve_chunks(
                        query=question,
                        session=retrieval_session,
                        **rk,
                    )
            else:
                # Run retrieval for original question + each sub-query in parallel (each
                # uses its own DB session) then merge. Sequential retrieval was a major
                # latency source when decomposition produces several sub-queries.
                all_queries = [question] + sub_queries
                logger.info(
                    "Query decomposed into %d sub-queries (+ original): %s",
                    len(sub_queries),
                    sub_queries,
                )

                async def _retrieve_for_sq(sq_idx: int, sq: str) -> tuple[int, str, list[dict]]:
                    async with SessionLocal() as retrieval_session:
                        sq_chunks = await retrieve_chunks(
                            query=sq,
                            session=retrieval_session,
                            **rk,
                        )
                    return sq_idx, sq, sq_chunks

                retrieved_lists = await asyncio.gather(
                    *[_retrieve_for_sq(i, sq) for i, sq in enumerate(all_queries)]
                )

                all_chunks: list[dict] = []
                chunk_by_id: dict[int, int] = {}
                # Track which sub-query produced each chunk for concept diversity
                chunk_sub_queries: dict[int, int] = {}
                for sq_idx, sq, sq_chunks in retrieved_lists:
                    new_count = 0
                    for chunk in sq_chunks:
                        cid = chunk["id"]
                        if cid not in chunk_by_id:
                            chunk_by_id[cid] = len(all_chunks)
                            all_chunks.append(chunk)
                            chunk_sub_queries[cid] = sq_idx
                            new_count += 1
                        else:
                            existing = all_chunks[chunk_by_id[cid]]
                            if chunk.get("score", 0.0) > existing.get("score", 0.0):
                                all_chunks[chunk_by_id[cid]] = chunk
                                chunk_sub_queries[cid] = sq_idx
                    logger.info(
                        "Sub-query '%s': %d chunks retrieved, %d new (not seen before)",
                        sq[:80],
                        len(sq_chunks),
                        new_count,
                    )

                # Sort merged results by score descending and dedup
                all_chunks.sort(key=lambda c: c.get("score", 0.0), reverse=True)
                dedup_threshold = retrieval_kwargs.get("dedup_threshold", 0.85)
                all_chunks = _deduplicate_chunks(all_chunks, dedup_threshold)

                # Apply diversity enforcement on merged results (not just
                # per-sub-query). Without this, sorting by score and taking
                # top_k re-concentrates on the highest-scoring document.
                top_k = retrieval_kwargs.get("top_k", 10)
                diversity_min = retrieval_kwargs.get("diversity_min", 3)
                max_per_doc = retrieval_kwargs.get("max_per_doc")
                threshold = retrieval_kwargs.get("diversity_relevance_threshold", 0.3)
                chunks = _apply_diversity(
                    all_chunks,
                    top_k=top_k,
                    max_per_doc=max_per_doc,
                    diversity_min=diversity_min,
                    relevance_threshold=threshold,
                )

                # Sub-query representation: ensure each sub-query contributes
                # at least one chunk when enabled on the profile.
                if retrieval_kwargs.get("ensure_sub_query_representation", True):
                    swap_mult = retrieval_kwargs.get("sub_query_swap_score_multiplier", 1.1)
                    min_swap_score = threshold * swap_mult
                    represented_sqs = {
                        chunk_sub_queries[c["id"]]
                        for c in chunks
                        if c["id"] in chunk_sub_queries
                    }
                    missing_sqs = set(range(len(all_queries))) - represented_sqs
                    if missing_sqs:
                        chunk_ids = {c["id"] for c in chunks}
                        swapped = 0
                        for sq_idx in missing_sqs:
                            best = None
                            for c in all_chunks:
                                if chunk_sub_queries.get(c["id"]) == sq_idx:
                                    best = c
                                    break  # all_chunks is score-sorted
                            if (
                                not best
                                or best["id"] in chunk_ids
                                or best.get("score", 0.0) < min_swap_score
                            ):
                                continue
                            if len(chunks) >= top_k:
                                worst_idx = min(
                                    range(len(chunks)),
                                    key=lambda i: chunks[i].get("score", 0.0),
                                )
                                worst_score = chunks[worst_idx].get("score", 0.0)
                                if best.get("score", 0.0) <= worst_score:
                                    continue
                                chunks[worst_idx] = best
                            else:
                                chunks.append(best)
                            chunk_ids.add(best["id"])
                            swapped += 1
                        if swapped:
                            logger.info(
                                "Sub-query representation: swapped in %d chunks "
                                "for under-represented sub-queries",
                                swapped,
                            )

                # Log final document distribution
                doc_counts: dict[str, int] = {}
                for c in chunks:
                    doc = c.get("source_document", "unknown")
                    doc_counts[doc] = doc_counts.get(doc, 0) + 1
                logger.info(
                    "Post-merge retrieval: %d chunks from %d documents: %s",
                    len(chunks),
                    len(doc_counts),
                    doc_counts,
                )

            if eval_run_id in _cancelled_runs:
                return result

            profile_prompt = (
                profile.system_prompt
                if profile and hasattr(profile, "system_prompt") and profile.system_prompt
                else None
            )
            gen_max_tokens = None
            if profile and getattr(profile, "generation", None) is not None:
                gen_max_tokens = profile.generation.max_tokens
            gen_result = await generate_answer(
                question=question,
                chunks=chunks,
                model_name=model_name,
                system_prompt=profile_prompt,
                max_tokens=gen_max_tokens,
            )
            if gen_result.get("error"):
                result.latency_ms = (time.time() - start) * 1000
                result.error = gen_result["error"]
                return result

            result.latency_ms = (time.time() - start) * 1000
            result.answer = gen_result["answer"]
            result.tokens = (
                gen_result.get("usage", {}).get("total_tokens") if gen_result.get("usage") else None
            )

            context_texts = [c["text"] for c in chunks] if chunks else []
            # Format context with source metadata for UI display
            if chunks:
                context_parts = []
                for c in chunks:
                    header_parts = [c.get("source_document", "")]
                    if c.get("page_number"):
                        header_parts.append(f"p.{c['page_number']}")
                    if c.get("section_path"):
                        header_parts.append(c["section_path"])
                    header = " | ".join(p for p in header_parts if p)
                    context_parts.append(f"[{header}]\n{c['text']}")
                result.contexts_str = "\n---\n".join(context_parts)

            # Compute chunk alignment: prefer truth refs, fall back to expected_chunks
            chunk_refs = None
            chunk_texts = None
            if q_item.truth and q_item.truth.retrieval_truth.expected_chunk_refs:
                chunk_refs = q_item.truth.retrieval_truth.expected_chunk_refs
                chunk_texts = q_item.truth.retrieval_truth.expected_chunk_texts or None
            elif q_item.expected_chunks:
                chunk_refs = q_item.expected_chunks
            if chunk_refs and chunks:
                result.chunk_alignment_score = compute_chunk_alignment(
                    chunks, chunk_refs, chunk_texts
                )

            # Run deterministic checks (fast, no LLM cost)
            if chunks:
                result.deterministic_checks = run_deterministic_checks(
                    truth=q_item.truth,
                    retrieved_chunks=chunks,
                    answer=gen_result.get("answer"),
                )

            if eval_run_id in _cancelled_runs:
                return result

            if gen_result["answer"]:
                scoring_coro = score_result(
                    question=question,
                    answer=gen_result["answer"],
                    contexts=context_texts,
                    expected_answer=expected_answer,
                    evaluated_model_name=model_name,
                )

                coverage_coro = None
                if expected_answer:
                    pre_concepts = None
                    if q_item.truth and q_item.truth.answer_truth.required_concepts:
                        pre_concepts = q_item.truth.answer_truth.required_concepts
                    coverage_coro = detect_coverage_gaps(
                        expected_answer=expected_answer,
                        actual_answer=gen_result["answer"],
                        contexts=context_texts or None,
                        question=question,
                        pre_extracted_concepts=pre_concepts,
                    )

                if coverage_coro:
                    result.scores, result.coverage_gaps = await asyncio.gather(
                        scoring_coro, coverage_coro
                    )
                else:
                    result.scores = await scoring_coro

            # Compute per-question verdict if profile is loaded
            if profile and result.scores:
                ev_mode = None
                if q_item.truth:
                    ev_mode = q_item.truth.retrieval_truth.evidence_mode
                result.verdict = compute_question_verdict(
                    result.scores,
                    profile,
                    deterministic_checks=result.deterministic_checks,
                    coverage_gaps=result.coverage_gaps,
                    evidence_mode=ev_mode,
                )

        except Exception as e:
            logger.error("Eval question failed: %s", e)
            result.error = str(e)

    return result


async def _run_evaluation(
    eval_run_id: int,
    model_name: str,
    questions: list[EvalQuestion],
    profile_id: str | None = None,
) -> None:
    """Execute an evaluation run in the background.

    Processes questions concurrently (up to _MAX_CONCURRENT_QUESTIONS at a time)
    for retrieval, generation, and scoring. DB writes are batched sequentially
    as results complete.
    """
    from db.database import SessionLocal

    async with SessionLocal() as session:
        run = await session.get(EvalRun, eval_run_id)
        if not run:
            return

        run.status = "running"
        if profile_id:
            run.profile_id = profile_id
        await session.commit()

        # Load profile for verdict computation (if specified)
        profile = None
        if profile_id:
            try:
                profile = load_profile(profile_id)
                run.profile_version = profile.version
            except (FileNotFoundError, ValueError) as e:
                logger.warning("Could not load profile '%s': %s", profile_id, e)

        try:
            # Process all questions concurrently with a semaphore
            semaphore = asyncio.Semaphore(_MAX_CONCURRENT_QUESTIONS)
            tasks = [
                _process_question(q_item, model_name, semaphore, eval_run_id, profile)
                for q_item in questions
            ]
            question_results = await asyncio.gather(*tasks)

            # Aggregate results sequentially (DB writes are not concurrent)
            total_latency = 0.0
            total_tokens_sum = 0
            completed = 0
            all_relevancy = []
            all_groundedness = []
            all_context_precision = []
            all_context_relevancy = []
            all_completeness = []
            all_correctness = []
            all_compliance_accuracy = []
            all_abstention = []
            all_chunk_alignment = []
            hallucination_count = 0
            hallucination_scored_count = 0
            question_verdicts: list[QuestionVerdict] = []

            for qr in question_results:
                if not qr.answer and not qr.error:
                    # Skipped (e.g., cancelled before processing)
                    continue

                if qr.error:
                    eval_result = EvalResult(
                        eval_run_id=eval_run_id,
                        question=qr.question,
                        error_message=qr.error,
                    )
                else:
                    eval_result = EvalResult(
                        eval_run_id=eval_run_id,
                        question=qr.question,
                        expected_answer=qr.expected_answer,
                        answer=qr.answer,
                        contexts=qr.contexts_str,
                        latency_ms=qr.latency_ms,
                        relevancy_score=qr.scores.get("relevancy_score"),
                        groundedness_score=qr.scores.get("groundedness_score"),
                        context_precision_score=qr.scores.get("context_precision_score"),
                        context_relevancy_score=qr.scores.get("context_relevancy_score"),
                        completeness_score=qr.scores.get("completeness_score"),
                        correctness_score=qr.scores.get("correctness_score"),
                        compliance_accuracy_score=qr.scores.get("compliance_accuracy_score"),
                        abstention_score=qr.scores.get("abstention_score"),
                        is_hallucination=qr.scores.get("is_hallucination"),
                        chunk_alignment_score=qr.chunk_alignment_score,
                        coverage_gaps=qr.coverage_gaps,
                        deterministic_checks=qr.deterministic_checks,
                        truth_payload=qr.truth_payload,
                        total_tokens=qr.tokens,
                    )

                    if qr.verdict:
                        eval_result.verdict = qr.verdict.verdict
                        eval_result.fail_reasons = qr.verdict.fail_reasons
                        question_verdicts.append(qr.verdict)

                    total_latency += qr.latency_ms
                    if qr.tokens:
                        total_tokens_sum += qr.tokens
                    if qr.scores.get("relevancy_score") is not None:
                        all_relevancy.append(qr.scores["relevancy_score"])
                    if qr.scores.get("groundedness_score") is not None:
                        all_groundedness.append(qr.scores["groundedness_score"])
                    if qr.scores.get("context_precision_score") is not None:
                        all_context_precision.append(qr.scores["context_precision_score"])
                    if qr.scores.get("context_relevancy_score") is not None:
                        all_context_relevancy.append(qr.scores["context_relevancy_score"])
                    if qr.scores.get("completeness_score") is not None:
                        all_completeness.append(qr.scores["completeness_score"])
                    if qr.scores.get("correctness_score") is not None:
                        all_correctness.append(qr.scores["correctness_score"])
                    if qr.scores.get("compliance_accuracy_score") is not None:
                        all_compliance_accuracy.append(qr.scores["compliance_accuracy_score"])
                    if qr.scores.get("abstention_score") is not None:
                        all_abstention.append(qr.scores["abstention_score"])
                    if qr.chunk_alignment_score is not None:
                        all_chunk_alignment.append(qr.chunk_alignment_score)
                    if qr.scores.get("is_hallucination") is not None:
                        hallucination_scored_count += 1
                        if qr.scores["is_hallucination"]:
                            hallucination_count += 1

                session.add(eval_result)
                completed += 1
                run.completed_questions = completed
                await session.commit()

            # Set final status
            if eval_run_id in _cancelled_runs:
                _cancelled_runs.discard(eval_run_id)
                run.status = "cancelled"
                run.completed_at = datetime.now(UTC).replace(tzinfo=None)
            else:
                run.status = "completed"
                run.completed_at = datetime.now(UTC).replace(tzinfo=None)

            # Compute aggregates from whatever questions were completed
            run.avg_latency_ms = total_latency / completed if completed else None
            run.total_tokens = total_tokens_sum or None
            run.avg_relevancy = sum(all_relevancy) / len(all_relevancy) if all_relevancy else None
            run.avg_groundedness = (
                sum(all_groundedness) / len(all_groundedness) if all_groundedness else None
            )
            run.avg_context_precision = (
                sum(all_context_precision) / len(all_context_precision)
                if all_context_precision
                else None
            )
            run.avg_context_relevancy = (
                sum(all_context_relevancy) / len(all_context_relevancy)
                if all_context_relevancy
                else None
            )
            run.avg_completeness = (
                sum(all_completeness) / len(all_completeness) if all_completeness else None
            )
            run.avg_correctness = (
                sum(all_correctness) / len(all_correctness) if all_correctness else None
            )
            run.avg_compliance_accuracy = (
                sum(all_compliance_accuracy) / len(all_compliance_accuracy)
                if all_compliance_accuracy
                else None
            )
            run.avg_abstention = (
                sum(all_abstention) / len(all_abstention) if all_abstention else None
            )
            run.avg_chunk_alignment = (
                sum(all_chunk_alignment) / len(all_chunk_alignment) if all_chunk_alignment else None
            )
            run.hallucination_rate = (
                hallucination_count / hallucination_scored_count
                if hallucination_scored_count > 0
                else None
            )

            # Compute run-level verdict
            if profile and question_verdicts:
                run_verdict = compute_run_verdict(question_verdicts)
                run.overall_verdict = run_verdict.overall
                run.pass_count = run_verdict.pass_count
                run.fail_count = run_verdict.fail_count
                run.review_count = run_verdict.review_count

            await session.commit()

        except Exception as e:
            logger.exception("Evaluation run failed: %s", e)
            if run.status != "cancelled":
                run.status = "failed"
                run.error_message = str(e)
            await session.commit()


@router.post("/", response_model=EvalRunCreateResponse, status_code=201)
async def create_eval_run(
    request: EvalRunCreate,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_db),
) -> EvalRunCreateResponse:
    """Create a new evaluation run.

    Queues the evaluation to run in the background. Each question is
    sent through the RAG pipeline using the specified model.
    """
    # Normalize questions: accept both plain strings and {question, expected_answer} objects
    normalized: list[EvalQuestion] = []
    for q in request.questions:
        if isinstance(q, str):
            normalized.append(EvalQuestion(question=q))
        else:
            normalized.append(q)

    # Load profile BEFORE truth generation so retrieval kwargs match evaluation
    retrieval_cfg = None
    profile_version = None
    truth_retrieval_kwargs = None
    if request.profile_id:
        try:
            prof = load_profile(request.profile_id)
            retrieval_cfg = prof.retrieval.model_dump()
            profile_version = prof.version
            truth_retrieval_kwargs = _build_retrieval_kwargs(prof)
        except (FileNotFoundError, ValueError):
            pass

    # Generate truth for questions with expected answers but no truth
    judge_model = settings.resolved_judge_model_name
    if judge_model:
        for q in normalized:
            if q.expected_answer and not q.truth:
                try:
                    q.truth = await generate_truth_from_manual_answer(
                        q.question,
                        q.expected_answer,
                        session,
                        judge_model,
                        retrieval_kwargs=truth_retrieval_kwargs,
                    )
                except Exception as e:
                    logger.warning("Truth generation failed for inline question: %s", e)

    corpus_snapshot = await _build_corpus_snapshot(session)

    run = EvalRun(
        model_name=request.model_name,
        question_set_id=request.question_set_id,
        profile_id=request.profile_id,
        profile_version=profile_version,
        judge_model_name=judge_model or None,
        retrieval_config=retrieval_cfg,
        corpus_snapshot=corpus_snapshot,
        status="pending",
        total_questions=len(normalized),
    )
    session.add(run)
    await session.flush()

    # Capture values before commit (commit expires ORM attributes in async context)
    run_id = run.id
    run_model = run.model_name
    run_status = run.status
    run_total = run.total_questions

    background_tasks.add_task(
        _run_evaluation,
        eval_run_id=run_id,
        model_name=request.model_name,
        questions=normalized,
        profile_id=request.profile_id,
    )

    await session.commit()

    message = f"Evaluation started with {len(request.questions)} questions"
    model_cfg = settings.get_model_config(request.model_name)
    if not model_cfg["token"]:
        message += f". Warning: No API token configured for {request.model_name}."

    return EvalRunCreateResponse(
        eval_run_id=run_id,
        model_name=run_model,
        status=run_status,
        total_questions=run_total,
        message=message,
    )


@router.get("/", response_model=list[EvalRunResponse])
async def list_eval_runs(
    session: AsyncSession = Depends(get_db),
) -> list[EvalRunResponse]:
    """List all evaluation runs, most recent first."""
    result = await session.execute(select(EvalRun).order_by(EvalRun.created_at.desc()))
    runs = result.scalars().all()
    return [_build_run_response(r) for r in runs]


@router.delete("/{eval_run_id}", status_code=204)
async def delete_eval_run(
    eval_run_id: int,
    session: AsyncSession = Depends(get_db),
) -> None:
    """Delete an evaluation run and its results."""
    run = await session.get(EvalRun, eval_run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    await session.delete(run)
    await session.commit()


@router.post("/{eval_run_id}/cancel")
async def cancel_eval_run(
    eval_run_id: int,
    session: AsyncSession = Depends(get_db),
) -> dict:
    """Cancel a running or pending evaluation.

    Signals the background task to stop after the current question.
    Partial results are retained with aggregated scores.
    """
    run = await session.get(EvalRun, eval_run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    if run.status not in ("pending", "running"):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel evaluation with status '{run.status}'",
        )

    _cancelled_runs.add(eval_run_id)

    # If still pending (background task hasn't started yet), mark directly
    if run.status == "pending":
        run.status = "cancelled"
        run.completed_at = datetime.now(UTC).replace(tzinfo=None)
        await session.commit()
        _cancelled_runs.discard(eval_run_id)

    return {"message": f"Cancellation requested for evaluation run #{eval_run_id}"}


@router.get("/profiles")
async def get_profiles() -> list[dict]:
    """List available evaluation profiles."""
    profile_ids = list_profiles()
    result = []
    for pid in profile_ids:
        try:
            p = load_profile(pid)
            result.append(
                {
                    "id": p.id,
                    "version": p.version,
                    "domain": p.domain,
                    "description": p.description,
                    "has_system_prompt": bool(p.system_prompt),
                }
            )
        except Exception:
            result.append({"id": pid, "version": "", "domain": "", "description": ""})
    return result


@router.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize_questions(
    request: SynthesizeRequest,
    session: AsyncSession = Depends(get_db),
) -> SynthesizeResponse:
    """Auto-generate evaluation questions from ingested documents.

    Calls the shared MaaS chat endpoint (same path as RAG) with
    ``question_synthesis_model_name``. Optionally filter by document IDs.
    """
    if not settings.API_TOKEN:
        raise HTTPException(
            status_code=400,
            detail="No API token configured. Set API_TOKEN to enable question generation.",
        )
    if not settings.question_synthesis_model_name:
        raise HTTPException(
            status_code=400,
            detail=(
                "No model configured for question synthesis. Set MODEL_A_NAME, "
                "JUDGE_MODEL_NAME, or QUESTION_SYNTHESIS_MODEL_NAME."
            ),
        )

    # Load profile domain for domain-specific question generation
    domain = ""
    truth_retrieval_kwargs = None
    if request.profile_id:
        try:
            profile = load_profile(request.profile_id)
            domain = profile.domain
            truth_retrieval_kwargs = _build_retrieval_kwargs(profile)
        except (FileNotFoundError, ValueError) as e:
            logger.warning("Could not load profile '%s' for synthesis: %s", request.profile_id, e)

    try:
        questions = await generate_questions(
            session=session,
            document_ids=request.document_ids,
            max_questions=request.max_questions,
            domain=domain,
            retrieval_kwargs=truth_retrieval_kwargs,
        )
    except Exception as e:
        logger.exception("Question synthesis failed")
        raise HTTPException(status_code=500, detail=f"Question synthesis failed: {str(e)[:200]}")
    return SynthesizeResponse(
        questions=[
            SynthesizedQuestion(
                question=q["question"],
                expected_answer=q.get("expected_answer"),
                truth=q.get("truth"),
            )
            for q in questions
        ],
        count=len(questions),
    )


def _build_run_response(run: EvalRun) -> EvalRunResponse:
    """Build an EvalRunResponse from a model instance."""
    return EvalRunResponse(
        id=run.id,
        model_name=run.model_name,
        question_set_name=run.question_set.name if run.question_set else None,
        status=run.status,
        total_questions=run.total_questions,
        completed_questions=run.completed_questions,
        avg_latency_ms=run.avg_latency_ms,
        avg_relevancy=run.avg_relevancy,
        avg_groundedness=run.avg_groundedness,
        avg_context_precision=run.avg_context_precision,
        avg_context_relevancy=run.avg_context_relevancy,
        avg_completeness=run.avg_completeness,
        avg_correctness=run.avg_correctness,
        avg_compliance_accuracy=run.avg_compliance_accuracy,
        avg_abstention=run.avg_abstention,
        hallucination_rate=run.hallucination_rate,
        avg_chunk_alignment=run.avg_chunk_alignment,
        profile_id=run.profile_id,
        profile_version=run.profile_version,
        judge_model_name=run.judge_model_name,
        synthesis_model_name=run.synthesis_model_name,
        retrieval_config=run.retrieval_config,
        corpus_snapshot=run.corpus_snapshot,
        overall_verdict=run.overall_verdict,
        pass_count=run.pass_count,
        fail_count=run.fail_count,
        review_count=run.review_count,
        total_tokens=run.total_tokens,
        error_message=run.error_message,
        created_at=run.created_at,
        completed_at=run.completed_at,
    )


def _build_result_response(r: EvalResult) -> EvalResultResponse:
    """Build an EvalResultResponse from a model instance."""
    return EvalResultResponse(
        id=r.id,
        question=r.question,
        expected_answer=r.expected_answer,
        answer=r.answer,
        contexts=r.contexts,
        latency_ms=r.latency_ms,
        relevancy_score=r.relevancy_score,
        groundedness_score=r.groundedness_score,
        context_precision_score=r.context_precision_score,
        context_relevancy_score=r.context_relevancy_score,
        completeness_score=r.completeness_score,
        correctness_score=r.correctness_score,
        compliance_accuracy_score=r.compliance_accuracy_score,
        abstention_score=r.abstention_score,
        is_hallucination=r.is_hallucination,
        chunk_alignment_score=r.chunk_alignment_score,
        coverage_gaps=r.coverage_gaps,
        deterministic_checks=r.deterministic_checks,
        truth=r.truth_payload,
        verdict=r.verdict,
        fail_reasons=r.fail_reasons,
        total_tokens=r.total_tokens,
        error_message=r.error_message,
    )


def _compare_metric(
    name: str, val_a: float | None, val_b: float | None, lower_is_better: bool = False
) -> ComparisonMetric:
    """Compare a single metric between two runs.

    Args:
        name: Metric name.
        val_a: Value from run A.
        val_b: Value from run B.
        lower_is_better: If True, lower values win (e.g., latency, hallucination rate).
    """
    winner = None
    if val_a is not None and val_b is not None:
        if abs(val_a - val_b) < COMPARISON_TIE_THRESHOLD:
            winner = "tie"
        elif lower_is_better:
            winner = "run_a" if val_a < val_b else "run_b"
        else:
            winner = "run_a" if val_a > val_b else "run_b"
    return ComparisonMetric(metric=name, run_a=val_a, run_b=val_b, winner=winner)


def _deterministic_pass_rate(results: dict[str, EvalResult], check_name: str) -> float | None:
    """Compute the pass rate for a deterministic check across question results.

    Args:
        results: Dict of question text -> EvalResult.
        check_name: Name of the deterministic check to aggregate.

    Returns:
        Pass rate as a float (0.0-1.0), or None if no results have the check.
    """
    total = 0
    passed = 0
    for r in results.values():
        if not r.deterministic_checks:
            continue
        for check in r.deterministic_checks:
            if check.get("check_name") == check_name:
                total += 1
                if check.get("passed"):
                    passed += 1
                break
    return passed / total if total > 0 else None


def _load_comparison_profile(run_a: EvalRun, run_b: EvalRun) -> EvalProfile | None:
    """Try to load a profile for comparison gates.

    Prefers run_a's profile; falls back to run_b's.
    Returns None if neither run has a profile or loading fails.
    """
    profile_id = run_a.profile_id or run_b.profile_id
    if not profile_id:
        return None
    try:
        return load_profile(profile_id)
    except (FileNotFoundError, ValueError) as exc:
        logger.warning("Could not load profile '%s' for comparison: %s", profile_id, exc)
        return None


@router.get("/compare", response_model=ComparisonResponse)
async def compare_eval_runs(
    run_a_id: int,
    run_b_id: int,
    session: AsyncSession = Depends(get_db),
) -> ComparisonResponse:
    """Compare two evaluation runs side-by-side.

    Returns aggregate metric comparison and per-question breakdown.
    Questions are matched by text across both runs.
    """
    run_a = await session.get(EvalRun, run_a_id)
    run_b = await session.get(EvalRun, run_b_id)
    if not run_a:
        raise HTTPException(status_code=404, detail=f"Evaluation run {run_a_id} not found")
    if not run_b:
        raise HTTPException(status_code=404, detail=f"Evaluation run {run_b_id} not found")

    metrics = [
        _compare_metric("groundedness", run_a.avg_groundedness, run_b.avg_groundedness),
        _compare_metric("relevancy", run_a.avg_relevancy, run_b.avg_relevancy),
        _compare_metric(
            "context_precision", run_a.avg_context_precision, run_b.avg_context_precision
        ),
        _compare_metric(
            "context_relevancy", run_a.avg_context_relevancy, run_b.avg_context_relevancy
        ),
        _compare_metric("completeness", run_a.avg_completeness, run_b.avg_completeness),
        _compare_metric("correctness", run_a.avg_correctness, run_b.avg_correctness),
        _compare_metric(
            "compliance_accuracy", run_a.avg_compliance_accuracy, run_b.avg_compliance_accuracy
        ),
        _compare_metric("abstention", run_a.avg_abstention, run_b.avg_abstention),
        _compare_metric("chunk_alignment", run_a.avg_chunk_alignment, run_b.avg_chunk_alignment),
        _compare_metric(
            "hallucination_rate",
            run_a.hallucination_rate,
            run_b.hallucination_rate,
            lower_is_better=True,
        ),
        _compare_metric(
            "latency_ms", run_a.avg_latency_ms, run_b.avg_latency_ms, lower_is_better=True
        ),
    ]

    results_a = await session.execute(
        select(EvalResult).where(EvalResult.eval_run_id == run_a_id).order_by(EvalResult.id)
    )
    results_b = await session.execute(
        select(EvalResult).where(EvalResult.eval_run_id == run_b_id).order_by(EvalResult.id)
    )

    a_by_question = {r.question: r for r in results_a.scalars().all()}
    b_by_question = {r.question: r for r in results_b.scalars().all()}

    # Add deterministic check pass rates as comparison metrics
    for check_name in ("document_presence", "chunk_alignment"):
        rate_a = _deterministic_pass_rate(a_by_question, check_name)
        rate_b = _deterministic_pass_rate(b_by_question, check_name)
        metrics.append(_compare_metric(f"{check_name}_pass_rate", rate_a, rate_b))

    # Build expected_answer lookup from question sets (fallback for old results)
    expected_by_question: dict[str, str] = {}
    for run in (run_a, run_b):
        if run.question_set and run.question_set.questions:
            for q_item in run.question_set.questions:
                if isinstance(q_item, dict) and q_item.get("expected_answer"):
                    expected_by_question.setdefault(q_item["question"], q_item["expected_answer"])

    all_questions = list(dict.fromkeys(list(a_by_question.keys()) + list(b_by_question.keys())))

    questions = []
    for q in all_questions:
        r_a = a_by_question.get(q)
        r_b = b_by_question.get(q)
        # Prefer expected_answer from eval result, fall back to question set
        expected = (
            (r_a.expected_answer if r_a else None)
            or (r_b.expected_answer if r_b else None)
            or expected_by_question.get(q)
        )
        questions.append(
            QuestionComparison(
                question=q,
                expected_answer=expected,
                run_a=_build_result_response(r_a) if r_a else None,
                run_b=_build_result_response(r_b) if r_b else None,
            )
        )

    run_a_resp = _build_run_response(run_a)
    run_b_resp = _build_run_response(run_b)

    # --- Precondition warnings (Step 7) ---
    warnings: list[ComparisonWarning] = []
    if run_a.profile_id != run_b.profile_id:
        warnings.append(
            ComparisonWarning(
                code="PROFILE_MISMATCH",
                message=(
                    f"Runs use different profiles: "
                    f"{run_a.profile_id or 'none'} vs {run_b.profile_id or 'none'}"
                ),
            )
        )
    if run_a.question_set_id != run_b.question_set_id:
        warnings.append(
            ComparisonWarning(
                code="QUESTION_SET_MISMATCH",
                message="Runs use different question sets",
            )
        )
    if (
        run_a.profile_id
        and run_b.profile_id
        and run_a.profile_id == run_b.profile_id
        and run_a.profile_version != run_b.profile_version
    ):
        warnings.append(
            ComparisonWarning(
                code="PROFILE_VERSION_MISMATCH",
                message=(
                    f"Same profile but different versions: "
                    f"{run_a.profile_version or 'unknown'} vs "
                    f"{run_b.profile_version or 'unknown'}"
                ),
            )
        )
    if run_a.judge_model_name and run_b.judge_model_name:
        if run_a.judge_model_name != run_b.judge_model_name:
            warnings.append(
                ComparisonWarning(
                    code="JUDGE_MODEL_MISMATCH",
                    message=(
                        f"Runs used different judge models: "
                        f"{run_a.judge_model_name} vs {run_b.judge_model_name}"
                    ),
                )
            )
    elif not run_a.judge_model_name or not run_b.judge_model_name:
        missing = "run A" if not run_a.judge_model_name else "run B"
        warnings.append(
            ComparisonWarning(
                code="JUDGE_MODEL_UNKNOWN",
                message=f"Judge model not recorded for {missing}; scoring conditions may differ",
            )
        )
    if run_a.corpus_snapshot and run_b.corpus_snapshot:
        if run_a.corpus_snapshot != run_b.corpus_snapshot:
            warnings.append(
                ComparisonWarning(
                    code="CORPUS_MISMATCH",
                    message="Corpus changed between runs; retrieval results may not be comparable",
                )
            )
    elif not run_a.corpus_snapshot or not run_b.corpus_snapshot:
        missing = "run A" if not run_a.corpus_snapshot else "run B"
        warnings.append(
            ComparisonWarning(
                code="CORPUS_UNKNOWN",
                message=f"Corpus snapshot not recorded for {missing}; cannot verify same corpus",
            )
        )
    if not run_a.profile_id and not run_b.profile_id:
        warnings.append(
            ComparisonWarning(
                code="NO_PROFILE",
                message="Neither run has a profile; verdict and gate checks are skipped",
            )
        )

    # --- Comparison decision (Steps 1-3) ---
    profile = _load_comparison_profile(run_a, run_b)
    metric_winners = [(m.metric, m.winner) for m in metrics]

    decision_result = compute_comparison_decision(
        run_a_data={
            "model_name": run_a.model_name,
            "overall_verdict": run_a.overall_verdict,
            "fail_count": run_a.fail_count or 0,
            "review_count": run_a.review_count or 0,
            "avg_completeness": run_a.avg_completeness,
            "avg_correctness": run_a.avg_correctness,
            "avg_compliance_accuracy": run_a.avg_compliance_accuracy,
        },
        run_b_data={
            "model_name": run_b.model_name,
            "overall_verdict": run_b.overall_verdict,
            "fail_count": run_b.fail_count or 0,
            "review_count": run_b.review_count or 0,
            "avg_completeness": run_b.avg_completeness,
            "avg_correctness": run_b.avg_correctness,
            "avg_compliance_accuracy": run_b.avg_compliance_accuracy,
        },
        metric_winners=metric_winners,
        profile=profile,
    )

    decision = ComparisonDecision(
        winner=decision_result.winner,
        winner_name=decision_result.winner_name,
        decision_status=decision_result.decision_status,
        reason_codes=decision_result.reason_codes,
        summary=decision_result.summary,
        risk_flags=decision_result.risk_flags,
        disqualified=decision_result.disqualified,
    )

    return ComparisonResponse(
        run_a=run_a_resp,
        run_b=run_b_resp,
        metrics=metrics,
        questions=questions,
        decision=decision,
        warnings=warnings,
    )


@router.get("/{eval_run_id}", response_model=EvalRunDetailResponse)
async def get_eval_run(
    eval_run_id: int,
    session: AsyncSession = Depends(get_db),
) -> EvalRunDetailResponse:
    """Get an evaluation run with its individual results."""
    run = await session.get(EvalRun, eval_run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Evaluation run not found")

    results = await session.execute(
        select(EvalResult).where(EvalResult.eval_run_id == eval_run_id).order_by(EvalResult.id)
    )
    result_rows = results.scalars().all()

    # Backfill expected_answer from question set for old results
    expected_by_question: dict[str, str] = {}
    if run.question_set and run.question_set.questions:
        for q_item in run.question_set.questions:
            if isinstance(q_item, dict) and q_item.get("expected_answer"):
                expected_by_question.setdefault(q_item["question"], q_item["expected_answer"])

    result_responses = []
    for r in result_rows:
        resp = _build_result_response(r)
        if not resp.expected_answer and r.question in expected_by_question:
            resp.expected_answer = expected_by_question[r.question]
        result_responses.append(resp)

    base_response = _build_run_response(run)
    return EvalRunDetailResponse(
        **base_response.model_dump(),
        results=result_responses,
    )


@router.post("/{eval_run_id}/rerun", response_model=EvalRunCreateResponse, status_code=201)
async def rerun_eval(
    eval_run_id: int,
    request: EvalRunRerun,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_db),
) -> EvalRunCreateResponse:
    """Re-run an evaluation with a different model.

    Copies the questions from an existing run and creates a new run
    using the specified model. Useful for comparing a new model against
    previously evaluated results without re-entering questions.
    """
    original_run = await session.get(EvalRun, eval_run_id)
    if not original_run:
        raise HTTPException(status_code=404, detail="Evaluation run not found")

    results = await session.execute(
        select(EvalResult.question, EvalResult.expected_answer, EvalResult.truth_payload)
        .where(EvalResult.eval_run_id == eval_run_id)
        .order_by(EvalResult.id)
    )
    from ..schemas.truth import TruthPayload

    questions = []
    truth_parse_failures = 0
    for row in results.all():
        truth = None
        if row[2]:
            try:
                truth = TruthPayload.model_validate(row[2])
            except Exception as e:
                truth_parse_failures += 1
                logger.warning(
                    "Could not parse truth_payload for rerun of run %d, question '%.80s': %s",
                    eval_run_id,
                    row[0],
                    e,
                )
        questions.append(EvalQuestion(question=row[0], expected_answer=row[1], truth=truth))
    if not questions:
        raise HTTPException(status_code=400, detail="Original run has no questions to re-run")

    # Snapshot run metadata for reproducibility (same as create_eval_run)
    retrieval_cfg = None
    profile_version = None
    profile_id = original_run.profile_id
    if profile_id:
        try:
            prof = load_profile(profile_id)
            retrieval_cfg = prof.retrieval.model_dump()
            profile_version = prof.version
        except (FileNotFoundError, ValueError):
            pass

    judge_model = settings.resolved_judge_model_name
    corpus_snapshot = await _build_corpus_snapshot(session)

    run = EvalRun(
        model_name=request.model_name,
        question_set_id=original_run.question_set_id,
        profile_id=profile_id,
        profile_version=profile_version,
        judge_model_name=judge_model or None,
        retrieval_config=retrieval_cfg,
        corpus_snapshot=corpus_snapshot,
        status="pending",
        total_questions=len(questions),
    )
    session.add(run)
    await session.flush()

    run_id = run.id
    run_model = run.model_name
    run_status = run.status
    run_total = run.total_questions

    background_tasks.add_task(
        _run_evaluation,
        eval_run_id=run_id,
        model_name=request.model_name,
        questions=questions,
        profile_id=original_run.profile_id,
    )

    await session.commit()

    message = f"Re-run started with {len(questions)} questions from run #{eval_run_id}"
    if truth_parse_failures:
        message += (
            f". Warning: {truth_parse_failures} question(s) had unparseable truth data"
            " -- deterministic checks will be skipped for those questions."
        )

    return EvalRunCreateResponse(
        eval_run_id=run_id,
        model_name=run_model,
        status=run_status,
        total_questions=run_total,
        message=message,
    )
