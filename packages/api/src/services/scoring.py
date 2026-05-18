# This project was developed with assistance from AI tools.
"""Consolidated LLM-as-judge scoring for evaluation results.

Scores RAG responses on faithfulness (groundedness), answer relevancy,
contextual precision, contextual relevancy, completeness, correctness,
compliance accuracy, and abstention quality using two consolidated judge
prompts instead of 8 separate calls.

Prompt A (always runs): faithfulness, relevancy, context_relevancy, abstention.
Prompt B (when expected_answer provided): completeness, correctness,
compliance_accuracy, context_precision.

All scores are 0.0-1.0 raw values. Pass/fail semantics live exclusively
in the verdict layer (verdicts.py) using profile thresholds.
"""

import asyncio
import json
import logging
import re

from openai import AsyncOpenAI, OpenAI

from ..core.config import settings

logger = logging.getLogger(__name__)

HALLUCINATION_THRESHOLD = 0.7

_SCORE_PATTERN = re.compile(r'"(\w+)":\s*([\d.]+|null)', re.IGNORECASE)


def _judge_model_name_for_run(evaluated_model_name: str) -> str:
    """Which model to use as LLM-as-judge.

    Resolution order: JUDGE_MODEL_NAME, MODEL_A_NAME, MODEL_B_NAME,
    then falls back to evaluated_model_name. Returns empty string if
    nothing is configured (caller should skip scoring).
    """
    name = settings.resolved_judge_model_name
    if not name:
        name = evaluated_model_name or ""
    if not name:
        return ""
    if name == evaluated_model_name:
        logger.warning(
            "Judge model (%s) is the same as the evaluated model -- "
            "results may be biased by self-evaluation",
            name,
        )
    return name


class MaaSJudgeModel:
    """OpenAI-compatible judge model backed by MaaS endpoint."""

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
    ):
        self._model_name = model_name
        self._base_url = (base_url or "").rstrip("/")
        t = (api_key or "").strip()
        if t.lower().startswith("bearer "):
            t = t[7:].strip()
        self._api_key = t
        self._sync_client = OpenAI(
            base_url=self._base_url + "/v1",
            api_key=self._api_key,
            timeout=120.0,
            max_retries=2,
        )
        self._async_client = AsyncOpenAI(
            base_url=self._base_url + "/v1",
            api_key=self._api_key,
            timeout=120.0,
            max_retries=2,
        )

    def generate(self, prompt: str) -> str:
        response = self._sync_client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        response = await self._async_client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content

    def get_model_name(self) -> str:
        return self._model_name


def _get_judge_model(model_name: str) -> MaaSJudgeModel:
    """Create judge model for the given resolved judge / evaluated model name."""
    logger.info("Creating judge model: %s at %s", model_name, settings.MAAS_ENDPOINT)
    return MaaSJudgeModel(
        model_name=model_name,
        base_url=settings.MAAS_ENDPOINT,
        api_key=settings.api_token_bare,
    )


# ---------------------------------------------------------------------------
# Consolidated judge prompts
# ---------------------------------------------------------------------------

_PROMPT_A = """\
You are an expert evaluation judge for RAG (Retrieval-Augmented Generation) systems. \
Evaluate the following response across four criteria. Score each from 0.0 to 1.0.

QUESTION:
{question}

ANSWER:
{answer}

RETRIEVAL CONTEXT:
{contexts}

Evaluate these four metrics:

1. **faithfulness** (groundedness): Is the answer factually supported by the \
retrieval context? Penalize claims not grounded in the provided context. \
1.0 = fully grounded; 0.0 = entirely fabricated.

2. **relevancy** (answer relevancy): Does the answer directly address the \
question asked? Penalize irrelevant information and tangents. \
1.0 = perfectly relevant; 0.0 = completely off-topic.

3. **context_relevancy**: Are the retrieved context chunks relevant to the \
question? Penalize irrelevant or noisy context that does not help answer \
the question. 1.0 = all context is relevant; 0.0 = none is relevant.

4. **abstention_quality**: Does the answer appropriately handle uncertainty? \
When the retrieval context does not contain sufficient information, the answer \
should acknowledge the limitation rather than fabricating or guessing. \
When context is sufficient, the answer should use it appropriately. \
Penalize confident assertions not supported by context. \
1.0 = appropriate handling; 0.0 = confidently wrong or inappropriately uncertain.

Respond with ONLY a JSON object containing the four scores. No other text.
Example: {{"faithfulness": 0.85, "relevancy": 0.90, "context_relevancy": 0.75, \
"abstention_quality": 0.95}}"""

_PROMPT_B = """\
You are an expert evaluation judge for RAG (Retrieval-Augmented Generation) systems. \
Evaluate the following response against the expected answer across four criteria. \
Score each from 0.0 to 1.0.

QUESTION:
{question}

ANSWER:
{answer}

EXPECTED ANSWER:
{expected_answer}

RETRIEVAL CONTEXT:
{contexts}

Evaluate these four metrics:

1. **completeness**: Does the answer cover all key points, requirements, and \
information present in the expected answer? Penalize omissions of important \
details, conditions, or qualifications. \
1.0 = all key points covered; 0.0 = none covered.

2. **correctness**: Are the claims and facts in the answer consistent with \
the expected answer? Penalize statements that contradict, misstate, or distort \
information from the expected answer. \
1.0 = fully correct; 0.0 = entirely incorrect.

3. **compliance_accuracy**: Does the answer correctly handle domain-specific \
compliance items (obligations, restrictions, approvals, disclosures, thresholds, \
escalation conditions, evidence requirements, cited authorities)? Check that \
claims are supported by the retrieval context and that critical requirements \
from the expected answer are not omitted. \
1.0 = full compliance accuracy; 0.0 = none.

4. **context_precision**: Are the most relevant context chunks ranked higher? \
Given the expected answer, check whether the context chunks that actually \
contain the needed information appear before irrelevant ones. \
1.0 = perfectly ranked; 0.0 = relevant chunks buried or absent.

Respond with ONLY a JSON object containing the four scores. No other text.
Example: {{"completeness": 0.80, "correctness": 0.90, "compliance_accuracy": 0.85, \
"context_precision": 0.75}}"""


def _parse_scores(raw: str, expected_keys: list[str]) -> dict[str, float | None]:
    """Parse JSON scores from judge response with fallback regex extraction."""
    text = raw.strip()

    # Strip markdown fencing
    if text.startswith("```"):
        lines = text.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return {
                k: _clamp_score(parsed.get(k))
                for k in expected_keys
            }
    except json.JSONDecodeError:
        pass

    # Fallback: regex extraction
    scores: dict[str, float | None] = {}
    for match in _SCORE_PATTERN.finditer(text):
        key = match.group(1).lower()
        val = match.group(2)
        if val.lower() == "null":
            scores[key] = None
        else:
            try:
                scores[key] = _clamp_score(float(val))
            except ValueError:
                scores[key] = None

    return {k: scores.get(k) for k in expected_keys}


def _clamp_score(value) -> float | None:
    """Clamp a score to [0.0, 1.0] or return None for invalid values."""
    if value is None:
        return None
    try:
        f = float(value)
        return max(0.0, min(1.0, f))
    except (TypeError, ValueError):
        return None


async def _run_prompt_a(
    judge: MaaSJudgeModel,
    question: str,
    answer: str,
    contexts: list[str],
) -> dict[str, float | None]:
    """Run consolidated Prompt A: faithfulness, relevancy, context_relevancy, abstention."""
    contexts_str = "\n---\n".join(contexts) if contexts else "(no context provided)"
    prompt = _PROMPT_A.format(
        question=question,
        answer=answer,
        contexts=contexts_str,
    )
    try:
        raw = await judge.a_generate(prompt)
        scores = _parse_scores(
            raw, ["faithfulness", "relevancy", "context_relevancy", "abstention_quality"]
        )
        return scores
    except Exception as e:
        logger.error("Prompt A scoring failed: %s", e, exc_info=True)
        return {
            "faithfulness": None,
            "relevancy": None,
            "context_relevancy": None,
            "abstention_quality": None,
        }


async def _run_prompt_b(
    judge: MaaSJudgeModel,
    question: str,
    answer: str,
    expected_answer: str,
    contexts: list[str],
) -> dict[str, float | None]:
    """Run consolidated Prompt B: completeness, correctness, compliance_accuracy, context_precision."""
    contexts_str = "\n---\n".join(contexts) if contexts else "(no context provided)"
    prompt = _PROMPT_B.format(
        question=question,
        answer=answer,
        expected_answer=expected_answer,
        contexts=contexts_str,
    )
    try:
        raw = await judge.a_generate(prompt)
        scores = _parse_scores(
            raw,
            ["completeness", "correctness", "compliance_accuracy", "context_precision"],
        )
        return scores
    except Exception as e:
        logger.error("Prompt B scoring failed: %s", e, exc_info=True)
        return {
            "completeness": None,
            "correctness": None,
            "compliance_accuracy": None,
            "context_precision": None,
        }


# ---------------------------------------------------------------------------
# Chunk alignment (deterministic, no LLM)
# ---------------------------------------------------------------------------

_CHUNK_MATCH_NGRAM_SIZE = 4
_CHUNK_MATCH_MIN_NGRAMS = 3


def _text_overlap_match(
    expected_text: str,
    retrieved_texts: list[str],
    ngram_size: int = _CHUNK_MATCH_NGRAM_SIZE,
    min_ngrams: int = _CHUNK_MATCH_MIN_NGRAMS,
) -> bool:
    """Check if an expected chunk's text overlaps substantially with any retrieved chunk."""
    expected_words = expected_text.lower().split()
    if len(expected_words) < ngram_size:
        return any(expected_text.lower() in rt.lower() for rt in retrieved_texts)

    expected_ngrams: set[str] = set()
    for i in range(len(expected_words) - ngram_size + 1):
        expected_ngrams.add(" ".join(expected_words[i : i + ngram_size]))

    for rt in retrieved_texts:
        rt_words = rt.lower().split()
        hits = 0
        for i in range(len(rt_words) - ngram_size + 1):
            if " ".join(rt_words[i : i + ngram_size]) in expected_ngrams:
                hits += 1
                if hits >= min_ngrams:
                    return True
    return False


def compute_chunk_alignment(
    retrieved_chunks: list[dict],
    expected_chunks: list[str],
    expected_chunk_texts: list[str] | None = None,
) -> float:
    """Score how well retrieved chunks match expected source chunks.

    Deterministic recall metric -- no LLM call required. Each expected chunk
    is a reference string in one of three formats:

    - ``"chunk:{id}"`` -- canonical format, matches on chunk database ID
    - ``"filename.pdf:3"`` -- legacy format, matches on document + page
    - ``"filename.pdf"`` -- legacy format, matches on document name only

    When ID-based matching yields zero hits and ``expected_chunk_texts``
    is provided, falls back to n-gram text overlap matching. This handles
    the case where documents were re-uploaded (new chunk IDs).

    Args:
        retrieved_chunks: Chunks from retrieval, each with ``id``,
            ``source_document``, and optionally ``page_number``.
        expected_chunks: Expected chunk references.
        expected_chunk_texts: Optional text content of expected chunks
            for text-based fallback matching.

    Returns:
        Recall score between 0.0 and 1.0 (matched / expected).
    """
    if not expected_chunks:
        return 1.0

    # Build lookup structures from retrieved chunks
    retrieved_ids: set[int] = set()
    retrieved_set: set[tuple[str, str | None]] = set()
    retrieved_docs: set[str] = set()
    for chunk in retrieved_chunks:
        if chunk.get("id") is not None:
            retrieved_ids.add(int(chunk["id"]))
        doc = chunk.get("source_document", "")
        page = str(chunk["page_number"]) if chunk.get("page_number") else None
        retrieved_set.add((doc, page))
        retrieved_docs.add(doc)

    matched = 0
    for ref in expected_chunks:
        if ref.startswith("chunk:"):
            try:
                chunk_id = int(ref[6:])
                if chunk_id in retrieved_ids:
                    matched += 1
            except ValueError:
                pass
        elif ":" in ref:
            doc, page = ref.rsplit(":", 1)
            if (doc, page) in retrieved_set:
                matched += 1
        else:
            if ref in retrieved_docs:
                matched += 1

    # Text-based fallback when ID matching finds nothing
    if matched == 0 and expected_chunk_texts:
        retrieved_texts = [c.get("text", "") for c in retrieved_chunks if c.get("text")]
        for exp_text in expected_chunk_texts:
            if exp_text and _text_overlap_match(exp_text, retrieved_texts):
                matched += 1

    return matched / len(expected_chunks)


# ---------------------------------------------------------------------------
# Main scoring entry point
# ---------------------------------------------------------------------------


async def score_result(
    question: str,
    answer: str,
    contexts: list[str],
    expected_answer: str | None = None,
    *,
    evaluated_model_name: str = "",
) -> dict:
    """Score a single RAG response using consolidated judge prompts.

    Runs two prompts concurrently:
    - Prompt A (always): faithfulness, relevancy, context_relevancy, abstention
    - Prompt B (when expected_answer provided): completeness, correctness,
      compliance_accuracy, context_precision

    Reduces judge calls from 8 to at most 2 per question.

    Args:
        question: The input question.
        answer: The model's generated answer.
        contexts: Retrieved context chunks used for generation (may be empty).
        expected_answer: Optional ground truth answer.
        evaluated_model_name: Model under test; used as the judge LLM when no
            JUDGE_MODEL_NAME / MODEL_A_NAME / MODEL_B_NAME is configured.

    Returns:
        Dict with metric scores and is_hallucination flag.
    """
    if not settings.API_TOKEN:
        logger.warning("No API token for judge model, skipping scoring")
        return {}

    judge_model_name = _judge_model_name_for_run(evaluated_model_name)
    if not judge_model_name:
        logger.warning(
            "No judge model name configured (set JUDGE_MODEL_NAME or MODEL_A_NAME, "
            "or pass a non-empty evaluated model for the run), skipping scoring"
        )
        return {}

    judge = _get_judge_model(judge_model_name)

    scores: dict = {}

    if expected_answer:
        prompt_a_result, prompt_b_result = await asyncio.gather(
            _run_prompt_a(judge, question, answer, contexts),
            _run_prompt_b(judge, question, answer, expected_answer, contexts),
        )
    else:
        prompt_a_result = await _run_prompt_a(judge, question, answer, contexts)
        prompt_b_result = None

    # Map prompt A results to canonical score keys
    scores["groundedness_score"] = prompt_a_result.get("faithfulness")
    scores["relevancy_score"] = prompt_a_result.get("relevancy")
    scores["context_relevancy_score"] = prompt_a_result.get("context_relevancy")
    scores["abstention_score"] = prompt_a_result.get("abstention_quality")

    # Map prompt B results when available
    if prompt_b_result:
        scores["completeness_score"] = prompt_b_result.get("completeness")
        scores["correctness_score"] = prompt_b_result.get("correctness")
        scores["compliance_accuracy_score"] = prompt_b_result.get("compliance_accuracy")
        scores["context_precision_score"] = prompt_b_result.get("context_precision")

    groundedness = scores.get("groundedness_score")
    if groundedness is not None:
        scores["is_hallucination"] = groundedness < HALLUCINATION_THRESHOLD
    else:
        scores["is_hallucination"] = None

    return scores
