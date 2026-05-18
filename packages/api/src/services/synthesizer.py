# This project was developed with assistance from AI tools.
"""Question synthesizer -- generates evaluation questions from document chunks."""

import json
import logging
import random
import re
from collections import defaultdict

import httpx
from db import Chunk, Document
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import settings
from .generation import GENERATION_TIMEOUT, _summarize_upstream_error
from .truth_generation import generate_truth_from_synthesis

logger = logging.getLogger(__name__)

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=GENERATION_TIMEOUT)
    return _client


MAX_CONTEXTS = 50
_MIN_CHUNKS_PER_DOC = 2


_SYNTHESIZE_PROMPT = """\
You are given excerpts from documents. \
Generate exactly {count} question-and-answer pairs that can ONLY be answered \
using the information in the excerpts below. Do NOT invent facts or use outside knowledge.

Rules:
- Each question must be answerable from the provided text.
- Each answer must quote or closely paraphrase the source text.
{domain_rules}
Respond in JSON: {{"questions": [{{"question": "...", "expected_answer": "..."}}]}}

--- DOCUMENT EXCERPTS ---
{context}
--- END EXCERPTS ---"""

_DEFAULT_DOMAIN_RULES = (
    "- Focus on specific requirements, obligations, thresholds, and definitions."
)

# Domain-specific synthesis rules keyed by profile domain.
# When a profile is active, these replace the default rules to generate
# questions that match the domain's evaluation criteria.
_DOMAIN_RULES: dict[str, str] = {
    "fsi": (
        "- Focus on specific regulatory requirements, obligations, thresholds, and definitions.\n"
        "- Prioritize questions about SEC/FINRA rules, compliance procedures, "
        "supervisory obligations, and risk controls.\n"
        "- Include questions that test whether the source correctly identifies "
        "regulatory deadlines, reporting requirements, and escalation procedures."
    ),
}


def _balanced_sample(
    chunks_by_doc: dict[int, list[dict]],
    budget: int,
) -> list[dict]:
    """Select chunks with equal-first-with-caps distribution across documents.

    Each document gets at least ``_MIN_CHUNKS_PER_DOC`` chunks (or all
    chunks if the document has fewer). Remaining budget is distributed
    proportionally by chunk count. Within each document, chunks are
    selected randomly to avoid insertion-order bias.

    Args:
        chunks_by_doc: Mapping of document_id -> list of chunk dicts.
        budget: Maximum total chunks to return.

    Returns:
        List of chunk dicts, balanced across documents.
    """
    if not chunks_by_doc:
        return []

    doc_ids = sorted(chunks_by_doc.keys())
    selected: list[dict] = []

    # Phase 1: guarantee minimum allocation per document
    remaining_budget = budget
    per_doc_selected: dict[int, list[dict]] = {}
    for doc_id in doc_ids:
        pool = list(chunks_by_doc[doc_id])
        random.shuffle(pool)
        take = min(len(pool), _MIN_CHUNKS_PER_DOC, remaining_budget)
        per_doc_selected[doc_id] = pool[:take]
        remaining_budget -= take
        if remaining_budget <= 0:
            break

    # Phase 2: distribute remaining budget proportionally by chunk count
    if remaining_budget > 0:
        eligible = {
            doc_id: len(chunks_by_doc[doc_id]) - len(per_doc_selected.get(doc_id, []))
            for doc_id in doc_ids
            if len(chunks_by_doc[doc_id]) > len(per_doc_selected.get(doc_id, []))
        }
        total_eligible = sum(eligible.values())
        if total_eligible > 0:
            for doc_id in doc_ids:
                extra_pool_size = eligible.get(doc_id, 0)
                if extra_pool_size <= 0:
                    continue
                share = max(1, round(remaining_budget * extra_pool_size / total_eligible))
                share = min(share, extra_pool_size, remaining_budget)
                already_selected_ids = {c["id"] for c in per_doc_selected.get(doc_id, [])}
                extra_pool = [
                    c for c in chunks_by_doc[doc_id] if c["id"] not in already_selected_ids
                ]
                random.shuffle(extra_pool)
                per_doc_selected.setdefault(doc_id, []).extend(extra_pool[:share])
                remaining_budget -= share
                if remaining_budget <= 0:
                    break

    # Flatten and sort by chunk ID for stable prompt ordering
    for doc_id in doc_ids:
        selected.extend(per_doc_selected.get(doc_id, []))
    selected.sort(key=lambda c: c["id"])

    return selected[:budget]


def _parse_questions_json(raw: str) -> dict:
    """Parse model JSON; tolerate fences, trailing commas, and missing commas."""
    text = (raw or "").strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Repair: remove trailing commas before } or ]
    repaired = re.sub(r",\s*([}\]])", r"\1", text)
    # Repair: insert missing commas between "value"\n"key" patterns
    repaired = re.sub(r'"\s*\n(\s*")', r'",\n\1', repaired)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Last resort: regex-extract question/answer pairs
    pairs = re.findall(
        r'"question"\s*:\s*"((?:[^"\\]|\\.)*)"\s*[,}]\s*"expected_answer"\s*:\s*"((?:[^"\\]|\\.)*)"',
        text,
    )
    if pairs:
        logger.warning("JSON parse failed, extracted %d questions via regex fallback", len(pairs))
        return {"questions": [{"question": q, "expected_answer": a} for q, a in pairs]}

    raise ValueError(f"Could not parse question synthesis response: {text[:200]}")


async def generate_questions(
    session: AsyncSession,
    document_ids: list[int] | None = None,
    max_questions: int = 10,
    domain: str = "",
    retrieval_kwargs: dict | None = None,
) -> list[dict]:
    """Generate evaluation questions from ingested document chunks.

    Calls the same OpenAI-compatible chat endpoint as RAG generation (httpx),
    using ``question_synthesis_model_name`` (see Settings).

    Args:
        session: Database session.
        document_ids: Optional list of document IDs to filter chunks.
            If None, uses chunks from all ready documents.
        max_questions: Maximum number of questions to generate.
        domain: Optional domain key (e.g. 'fsi') to generate
            domain-specific questions matching the evaluation profile.
        retrieval_kwargs: Optional profile-driven retrieval parameters used
            when grounding generated truth.

    Returns:
        List of dicts with 'question' and 'expected_answer' keys.
    """
    query = (
        select(Chunk.id, Chunk.text, Chunk.source_document, Chunk.document_id)
        .join(Document, Chunk.document_id == Document.id)
        .where(Document.status == "ready", Document.deleted_at.is_(None))
    )
    if document_ids:
        query = query.where(Chunk.document_id.in_(document_ids))

    result = await session.execute(query)
    rows = result.all()

    # Group chunks by document for balanced sampling
    chunks_by_doc: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        chunks_by_doc[row[3]].append({"id": row[0], "text": row[1], "source_document": row[2]})

    chunk_data = _balanced_sample(chunks_by_doc, MAX_CONTEXTS)
    chunk_texts = [c["text"] for c in chunk_data]

    if not chunk_texts:
        return []

    model_name = settings.question_synthesis_model_name
    if not model_name:
        raise RuntimeError(
            "No model configured for question synthesis. Set MODEL_A_NAME, JUDGE_MODEL_NAME, "
            "or QUESTION_SYNTHESIS_MODEL_NAME."
        )

    model_cfg = settings.get_model_config(model_name)
    if not model_cfg["token"]:
        raise RuntimeError("No API token configured for question synthesis.")

    context = "\n\n".join(chunk_texts[:MAX_CONTEXTS])
    domain_rules = (
        _DOMAIN_RULES.get(domain, _DEFAULT_DOMAIN_RULES) if domain else _DEFAULT_DOMAIN_RULES
    )
    prompt = _SYNTHESIZE_PROMPT.format(
        count=max_questions, context=context, domain_rules=domain_rules
    )

    url = f"{model_cfg['endpoint']}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {model_cfg['token']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 4096,
    }

    try:
        client = _get_client()
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()

        data = response.json()
        text = data["choices"][0]["message"]["content"]
        parsed = _parse_questions_json(text)

        raw_questions = parsed.get("questions", [])
        questions = []
        for item in raw_questions[:max_questions]:
            if isinstance(item, dict) and item.get("question"):
                questions.append(
                    {
                        "question": item["question"],
                        "expected_answer": item.get("expected_answer"),
                    }
                )

        # Generate structured truth for each question with an expected answer (parallel)
        judge_model = settings.resolved_judge_model_name
        if judge_model and chunk_data:
            truth_tasks = []
            truth_indices = []
            for i, q in enumerate(questions):
                if q.get("expected_answer"):
                    truth_tasks.append(
                        generate_truth_from_synthesis(
                            q["question"],
                            q["expected_answer"],
                            chunk_data,
                            judge_model,
                            session=session,
                            retrieval_kwargs=retrieval_kwargs,
                        )
                    )
                    truth_indices.append(i)

            if truth_tasks:
                import asyncio

                results = await asyncio.gather(*truth_tasks, return_exceptions=True)
                for idx, result in zip(truth_indices, results):
                    if isinstance(result, Exception):
                        logger.warning(
                            "Truth generation failed for synthesized question: %s", result
                        )
                    else:
                        questions[idx]["truth"] = result

        return questions

    except httpx.HTTPStatusError as e:
        detail = _summarize_upstream_error(e.response)
        logger.error(
            "Question synthesis HTTP %s for model %r: %s",
            e.response.status_code,
            model_name,
            detail or "(empty body)",
        )
        raise RuntimeError(detail or f"Model returned HTTP {e.response.status_code}") from e
    except Exception as e:
        logger.error("Question synthesis failed: %s", e, exc_info=True)
        raise
