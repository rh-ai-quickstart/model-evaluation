# This project was developed with assistance from AI tools.
"""Truth generation service -- creates structured truth payloads for questions.

Generates truth for two entry paths:
- Synthesis: concepts extracted from expected answer, retrieval truth traced from
  the chunks used during question generation.
- Manual: concepts extracted from expected answer, retrieval truth grounded by
  running the expected answer through the same retrieval pipeline as evaluation.
"""

import hashlib
import json
import logging
from datetime import UTC, datetime

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import settings
from ..schemas.truth import AnswerTruth, EvidenceMode, RetrievalTruth, TruthMetadata, TruthPayload
from .coverage import COVERAGE_TIMEOUT, EXTRACT_CONCEPTS_PROMPT, _strip_markdown_fencing
from .retrieval import retrieve_chunks

logger = logging.getLogger(__name__)

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=COVERAGE_TIMEOUT)
    return _client

_classification_cache: dict[str, dict[str, list[str]]] = {}
_CLASSIFICATION_CACHE_MAX = 200

CLASSIFY_DOCUMENTS_PROMPT = """\
Given this user question, expected answer, and the following documents with \
excerpts, classify each document's role in answering THIS SPECIFIC question.

User question:
{question}

Expected answer:
{expected_answer}

A document is "required" if the expected answer depends on it for a core part \
of the answer to the user's question:
- the answer explains, analyzes, or describes the document's content substantively
- removing the document would leave a gap in the answer's main argument

A document is "supporting" if:
- the answer only mentions it briefly or in passing (e.g. "may include X")
- it provides examples, illustrations, or elaboration on points covered by other documents
- it covers secondary, optional, or background disclosures
- removing it would not change the core substance of the answer

Do NOT classify a document as required only because its excerpt contains \
mandatory words like "must", "shall", "required", or "filed with the SEC". \
A regulatory document can contain mandatory language and still be supporting \
for this specific question.

Classify based on how the expected answer uses the document to answer the \
user's question, not on how important the document's content is in general.

If a document supports only an ongoing-reporting example, secondary filing, \
parenthetical note, or "may include" clause, classify it as supporting.

Documents:
{documents_block}

Return only valid JSON: {{"required": ["doc1.pdf", ...], "supporting": ["doc2.pdf", ...]}}
Every document must appear in exactly one list.\
"""


def _fallback_classify(doc_to_chunks: dict[str, list[dict]]) -> dict[str, list[str]]:
    """Chunk-count heuristic fallback when LLM classification fails.

    Ranks documents by chunk count and marks only the top document as
    required. This is conservative — better to under-classify than to
    mark all documents as required and trigger false document_presence
    failures.
    """
    ranked = sorted(doc_to_chunks.items(), key=lambda x: len(x[1]), reverse=True)
    if not ranked:
        return {"required": [], "supporting": []}
    required = [ranked[0][0]]
    supporting = sorted(d for d, _ in ranked[1:])
    return {"required": required, "supporting": supporting}


async def classify_documents(
    question: str,
    expected_answer: str,
    doc_to_chunks: dict[str, list[dict]],
    model_name: str,
) -> dict[str, list[str]]:
    """Classify documents as required vs supporting using the judge model.

    Uses LLM-based classification with explicit criteria. Falls back to
    chunk-count heuristic on failure. Results are cached by content hash.

    Args:
        question: The user question text (provides classification context).
        expected_answer: The expected answer text.
        doc_to_chunks: Mapping of document filename to list of chunk dicts
            (each with at least 'text' key).
        model_name: Judge model for classification.

    Returns:
        Dict with 'required' and 'supporting' lists of document filenames.
    """
    if not doc_to_chunks:
        return {"required": [], "supporting": []}

    cache_input = question + expected_answer + json.dumps(
        sorted((k, [c.get("text", "")[:200] for c in v]) for k, v in doc_to_chunks.items())
    )
    cache_key = hashlib.sha256(cache_input.encode()).hexdigest()

    if cache_key in _classification_cache:
        return _classification_cache[cache_key]

    model_cfg = settings.get_model_config(model_name)
    if not model_cfg["token"]:
        logger.warning("No API token for classification, using fallback heuristic")
        result = _fallback_classify(doc_to_chunks)
        _classification_cache[cache_key] = result
        return result

    docs_block_parts = []
    for i, (doc_name, chunks) in enumerate(sorted(doc_to_chunks.items()), 1):
        excerpts = " ".join(c.get("text", "")[:200] for c in chunks)
        docs_block_parts.append(f"{i}. {doc_name}: {excerpts}")
    documents_block = "\n".join(docs_block_parts)

    prompt_text = CLASSIFY_DOCUMENTS_PROMPT.format(
        question=question,
        expected_answer=expected_answer,
        documents_block=documents_block,
    )

    url = f"{model_cfg['endpoint']}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {model_cfg['token']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": 0.0,
        "max_tokens": 512,
    }

    try:
        client = _get_client()
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()

        data = response.json()
        message = data["choices"][0]["message"]
        content = (message.get("content") or "").strip()
        if not content and message.get("reasoning_content"):
            content = message["reasoning_content"].strip()
            logger.info("Classification used reasoning_content field (content was empty)")
        content = _strip_markdown_fencing(content)
        if not content:
            raise ValueError(
                f"Model returned empty classification (raw keys: {list(message.keys())})"
            )
        classification = json.loads(content)

        if not isinstance(classification, dict):
            raise ValueError("Classification response is not a dict")

        all_docs = set(doc_to_chunks.keys())
        required = [d for d in classification.get("required", []) if d in all_docs]
        supporting = [d for d in classification.get("supporting", []) if d in all_docs]

        classified = set(required) | set(supporting)
        for doc in all_docs - classified:
            supporting.append(doc)
        required = sorted(required)
        supporting = sorted(supporting)

        if not required and supporting:
            required = supporting[:1]
            supporting = supporting[1:]

        logger.info(
            "Document classification: %d required, %d supporting",
            len(required),
            len(supporting),
        )

    except Exception as e:
        logger.warning("LLM document classification failed (%s), using fallback", e)
        result = _fallback_classify(doc_to_chunks)
        _classification_cache[cache_key] = result
        return result

    result = {"required": required, "supporting": supporting}
    if len(_classification_cache) >= _CLASSIFICATION_CACHE_MAX:
        _classification_cache.clear()
    _classification_cache[cache_key] = result
    return result


async def extract_answer_truth(expected_answer: str, model_name: str) -> AnswerTruth:
    """Extract required concepts from an expected answer using the judge model.

    Reuses the same concept extraction prompt as coverage.py to ensure
    consistent concept definitions.

    Args:
        expected_answer: The ground truth answer text.
        model_name: Model to use for concept extraction.

    Returns:
        AnswerTruth with extracted concepts.

    Raises:
        RuntimeError: If concept extraction fails.
    """
    model_cfg = settings.get_model_config(model_name)
    if not model_cfg["token"]:
        raise RuntimeError("No API token configured for truth generation.")

    url = f"{model_cfg['endpoint']}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {model_cfg['token']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": EXTRACT_CONCEPTS_PROMPT.format(expected_answer=expected_answer),
            },
        ],
        "temperature": 0.0,
        "max_tokens": 512,
    }

    try:
        client = _get_client()
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()

        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()
        content = _strip_markdown_fencing(content)

        concepts = json.loads(content)
        if not isinstance(concepts, list) or not concepts:
            raise RuntimeError("Concept extraction returned empty or non-list result")

        required_concepts = [c.strip() for c in concepts if isinstance(c, str) and c.strip()]
        if not required_concepts:
            raise RuntimeError("No valid concepts extracted from expected answer")

        logger.info("Truth generation: extracted %d concepts", len(required_concepts))
        return AnswerTruth(required_concepts=required_concepts)

    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse concept extraction response: {e}") from e
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"Concept extraction HTTP {e.response.status_code}") from e


async def build_retrieval_truth_from_synthesis(
    question: str,
    expected_answer: str,
    source_chunks: list[dict],
    model_name: str,
) -> RetrievalTruth:
    """Build retrieval truth from chunks used during question synthesis.

    Classifies documents as required vs supporting using the judge model,
    falling back to chunk-count heuristic on failure.

    Args:
        question: The user question text.
        expected_answer: The expected answer text for classification context.
        source_chunks: Chunk dicts with 'id', 'text', and 'source_document' keys.
        model_name: Judge model for document classification.

    Returns:
        RetrievalTruth with traced evidence and document classification.
    """
    doc_to_chunks: dict[str, list[dict]] = {}
    for chunk in source_chunks:
        doc = chunk.get("source_document", "")
        if doc:
            doc_to_chunks.setdefault(doc, []).append(chunk)

    classification = await classify_documents(question, expected_answer, doc_to_chunks, model_name)
    required_docs = set(classification["required"])
    supporting_docs = set(classification["supporting"])

    required_chunks = [
        c for c in source_chunks
        if c.get("id") and c.get("source_document", "") in required_docs
    ]
    required_refs = [f"chunk:{c['id']}" for c in required_chunks]
    required_texts = [c.get("text", "") for c in required_chunks]
    supporting_refs = [
        f"chunk:{c['id']}" for c in source_chunks
        if c.get("id") and c.get("source_document", "") in supporting_docs
    ]

    return RetrievalTruth(
        required_documents=sorted(required_docs),
        expected_chunk_refs=required_refs,
        expected_chunk_texts=required_texts,
        supporting_documents=sorted(supporting_docs),
        supporting_chunk_refs=supporting_refs,
        evidence_mode="traced_from_synthesis",
    )


async def ground_answer_to_corpus(
    question: str,
    expected_answer: str,
    session: AsyncSession,
    model_name: str,
    retrieval_kwargs: dict | None = None,
    evidence_mode: EvidenceMode = "grounded_from_manual_answer",
) -> tuple[RetrievalTruth, list[int]]:
    """Ground an expected answer against the uploaded corpus.

    Uses the same retrieval pipeline as evaluation to find chunks that
    best match the expected answer text, then classifies documents as
    required vs supporting.

    Args:
        question: The user question text (provides classification context).
        expected_answer: The expected answer text to ground.
        session: Database session for retrieval queries.
        model_name: Judge model for document classification.
        retrieval_kwargs: Profile-driven retrieval parameters (top_k, etc.).
        evidence_mode: Provenance label to store on retrieval truth.

    Returns:
        Tuple of (RetrievalTruth with classification, source_chunk_ids).
    """
    _accepted = {
        "top_k",
        "max_per_doc",
        "rerank_depth",
        "diversity_min",
        "keyword_enabled",
        "dedup_threshold",
        "diversity_relevance_threshold",
    }
    kwargs = {k: v for k, v in (retrieval_kwargs or {}).items() if k in _accepted}
    chunks = await retrieve_chunks(
        query=question,
        session=session,
        **kwargs,
    )

    source_chunk_ids = [c["id"] for c in chunks if c.get("id")]

    doc_to_chunks: dict[str, list[dict]] = {}
    for chunk in chunks:
        doc = chunk.get("source_document", "")
        if doc:
            doc_to_chunks.setdefault(doc, []).append(chunk)

    classification = await classify_documents(
        question, expected_answer, doc_to_chunks, model_name
    )
    required_docs = set(classification["required"])
    supporting_docs = set(classification["supporting"])

    required_chunks = [
        c for c in chunks
        if c.get("id") and c.get("source_document", "") in required_docs
    ]
    required_refs = [f"chunk:{c['id']}" for c in required_chunks]
    required_texts = [c.get("text", "") for c in required_chunks]
    supporting_refs = [
        f"chunk:{c['id']}" for c in chunks
        if c.get("id") and c.get("source_document", "") in supporting_docs
    ]

    all_docs = sorted(required_docs | supporting_docs)
    logger.info(
        "Corpus grounding: %d retrieved, %d documents (%d required, %d supporting)",
        len(chunks),
        len(all_docs),
        len(required_docs),
        len(supporting_docs),
    )

    return RetrievalTruth(
        required_documents=sorted(required_docs),
        expected_chunk_refs=required_refs,
        expected_chunk_texts=required_texts,
        supporting_documents=sorted(supporting_docs),
        supporting_chunk_refs=supporting_refs,
        evidence_mode=evidence_mode,
    ), source_chunk_ids


def build_truth_metadata(model_name: str, source_chunk_ids: list[int]) -> TruthMetadata:
    """Build truth metadata with version fields and provenance.

    Args:
        model_name: Model used for concept extraction.
        source_chunk_ids: IDs of chunks that informed this truth.

    Returns:
        TruthMetadata with current timestamp and version fields.
    """
    return TruthMetadata(
        generated_by_model=model_name,
        generated_at=datetime.now(UTC).replace(tzinfo=None),
        source_chunk_ids=source_chunk_ids,
    )


def _align_chunks_to_answer(
    expected_answer: str,
    source_chunks: list[dict],
    min_overlap_words: int = 3,
) -> list[dict]:
    """Find which source chunks are actually referenced by the expected answer.

    Uses word n-gram overlap to identify chunks whose content appears in the
    answer. This avoids making every synthesized question depend on all
    synthesis chunks, which would make deterministic checks too strict.

    Args:
        expected_answer: The expected answer text.
        source_chunks: All chunks from the synthesis context.
        min_overlap_words: Minimum word n-gram length for a match.

    Returns:
        Subset of source_chunks that have content overlap with the answer.
        Falls back to all chunks if no overlap is found (conservative).
    """
    if not expected_answer or not source_chunks:
        return source_chunks

    answer_lower = expected_answer.lower()
    answer_words = answer_lower.split()
    if len(answer_words) < min_overlap_words:
        return source_chunks

    # Build answer n-grams for matching
    answer_ngrams: set[str] = set()
    for i in range(len(answer_words) - min_overlap_words + 1):
        answer_ngrams.add(" ".join(answer_words[i : i + min_overlap_words]))

    aligned = []
    for chunk in source_chunks:
        chunk_text = (chunk.get("text") or "").lower()
        chunk_words = chunk_text.split()
        for i in range(len(chunk_words) - min_overlap_words + 1):
            ngram = " ".join(chunk_words[i : i + min_overlap_words])
            if ngram in answer_ngrams:
                aligned.append(chunk)
                break

    # Fall back to all chunks if alignment finds nothing (conservative)
    return aligned if aligned else source_chunks


async def generate_truth_from_synthesis(
    question: str,
    expected_answer: str,
    source_chunks: list[dict],
    model_name: str,
    session: AsyncSession | None = None,
    retrieval_kwargs: dict | None = None,
) -> TruthPayload:
    """Generate a complete truth payload for a synthesized question.

    When a DB session is provided, grounds generated truth through the
    same retrieval pipeline used by evaluation. This avoids freezing exact
    chunk requirements from the broad synthesis prompt that may not be
    reachable from the final generated question. Without a session, falls
    back to tracing source chunks from the synthesis context.

    Args:
        question: The synthesized question text.
        expected_answer: LLM-generated expected answer.
        source_chunks: Chunks used during synthesis, each with 'id',
            'text', and 'source_document' keys.
        model_name: Judge model for concept extraction.
        session: Optional DB session used to ground generated truth via retrieval.
        retrieval_kwargs: Profile-driven retrieval parameters.

    Returns:
        Complete TruthPayload with generated-answer retrieval truth.
    """
    answer_truth = await extract_answer_truth(expected_answer, model_name)
    if session is not None:
        retrieval_truth, source_chunk_ids = await ground_answer_to_corpus(
            question,
            expected_answer,
            session,
            model_name,
            retrieval_kwargs=retrieval_kwargs,
            evidence_mode="grounded_from_synthesis",
        )
        metadata = build_truth_metadata(model_name, source_chunk_ids)
        return TruthPayload(
            answer_truth=answer_truth,
            retrieval_truth=retrieval_truth,
            metadata=metadata,
        )

    aligned_chunks = _align_chunks_to_answer(expected_answer, source_chunks)
    retrieval_truth = await build_retrieval_truth_from_synthesis(
        question, expected_answer, aligned_chunks, model_name
    )
    source_chunk_ids = [c["id"] for c in aligned_chunks if c.get("id")]
    metadata = build_truth_metadata(model_name, source_chunk_ids)

    return TruthPayload(
        answer_truth=answer_truth,
        retrieval_truth=retrieval_truth,
        metadata=metadata,
    )


async def generate_truth_from_manual_answer(
    question: str,
    expected_answer: str,
    session: AsyncSession,
    model_name: str,
    retrieval_kwargs: dict | None = None,
) -> TruthPayload:
    """Generate a complete truth payload for a manual question with expected answer.

    Grounds the expected answer against the uploaded corpus using the same
    retrieval pipeline as evaluation.

    Args:
        question: The user question text.
        expected_answer: User-provided expected answer.
        session: Database session for corpus retrieval.
        model_name: Judge model for concept extraction.
        retrieval_kwargs: Profile-driven retrieval parameters.

    Returns:
        Complete TruthPayload with corpus-grounded retrieval truth.
    """
    answer_truth = await extract_answer_truth(expected_answer, model_name)
    retrieval_truth, source_chunk_ids = await ground_answer_to_corpus(
        question, expected_answer, session, model_name, retrieval_kwargs
    )
    metadata = build_truth_metadata(model_name, source_chunk_ids)

    return TruthPayload(
        answer_truth=answer_truth,
        retrieval_truth=retrieval_truth,
        metadata=metadata,
    )
