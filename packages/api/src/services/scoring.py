# This project was developed with assistance from AI tools.
"""DeepEval-based scoring for evaluation results.

Uses LLM-as-judge via DeepEval metrics to score RAG responses on
faithfulness (groundedness), answer relevancy, contextual precision,
contextual relevancy, completeness, correctness, compliance accuracy,
and abstention quality. The judge model is configurable — defaults
to MaaS endpoint.

All metrics use threshold=0.0 (raw scorer only). Pass/fail semantics
live exclusively in the verdict layer (verdicts.py) using profile
thresholds.
"""

import asyncio
import logging

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from openai import AsyncOpenAI, OpenAI

from ..core.config import settings

logger = logging.getLogger(__name__)

HALLUCINATION_THRESHOLD = 0.7


def _judge_model_name_for_run(evaluated_model_name: str) -> str:
    """Which model DeepEval calls for LLM-as-judge.

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


class MaaSJudgeModel(DeepEvalBaseLLM):
    """OpenAI-compatible judge model for DeepEval, backed by MaaS endpoint."""

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
        super().__init__(model=model_name)

    def load_model(self):
        return self._sync_client

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


def _get_judge_model(model_name: str) -> DeepEvalBaseLLM:
    """Create judge model for the given resolved judge / evaluated model name."""
    logger.info("Creating judge model: %s at %s", model_name, settings.MAAS_ENDPOINT)
    return MaaSJudgeModel(
        model_name=model_name,
        base_url=settings.MAAS_ENDPOINT,
        api_key=settings.api_token_bare,
    )


def _completeness_metric(judge: DeepEvalBaseLLM) -> GEval:
    """GEval metric: did the answer cover all key points from the expected answer?"""
    return GEval(
        name="Completeness",
        criteria=(
            "Evaluate whether the actual output covers all the key points, requirements, "
            "and information present in the expected output. Penalize omissions of important "
            "details, conditions, or qualifications."
        ),
        evaluation_steps=[
            "Identify all key points and requirements in the expected output.",
            "Check which of those key points appear in the actual output.",
            "Penalize for each missing key point proportional to its importance.",
            "A score of 1.0 means all key points are covered; 0.0 means none are.",
        ],
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        model=judge,
        threshold=0.0,
        async_mode=True,
    )


def _correctness_metric(judge: DeepEvalBaseLLM) -> GEval:
    """GEval metric: is what the answer said factually consistent with the expected answer?"""
    return GEval(
        name="Correctness",
        criteria=(
            "Evaluate whether the claims and facts in the actual output are consistent with "
            "the expected output. Penalize any statements that contradict, misstate, or "
            "distort information from the expected output."
        ),
        evaluation_steps=[
            "Identify factual claims in the actual output.",
            "Compare each claim against the expected output for consistency.",
            "Penalize contradictions and misstatements.",
            "A score of 1.0 means fully correct; 0.0 means entirely incorrect.",
        ],
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        model=judge,
        threshold=0.0,
        async_mode=True,
    )


def _compliance_accuracy_metric(judge: DeepEvalBaseLLM) -> GEval:
    """GEval metric: are domain-specific compliance items correctly handled?"""
    return GEval(
        name="Compliance Accuracy",
        criteria=(
            "Evaluate whether the answer correctly handles domain-specific compliance items: "
            "obligations, restrictions, approvals, disclosures, thresholds, escalation "
            "conditions, evidence requirements, and cited authorities. Check that claims are "
            "supported by the retrieval context and that critical requirements from the "
            "expected output are not omitted."
        ),
        evaluation_steps=[
            "Identify compliance-relevant items in the expected output (obligations, "
            "thresholds, restrictions, disclosures, authorities).",
            "Check whether each item is present and correctly stated in the actual output.",
            "Verify that compliance claims are supported by the retrieval context.",
            "Penalize omitted critical requirements and unsupported claims.",
            "A score of 1.0 means full compliance accuracy; 0.0 means none.",
        ],
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        model=judge,
        threshold=0.0,
        async_mode=True,
    )


def _abstention_metric(judge: DeepEvalBaseLLM) -> GEval:
    """GEval metric: does the answer appropriately handle uncertainty?"""
    return GEval(
        name="Abstention Quality",
        criteria=(
            "Evaluate whether the answer appropriately handles uncertainty. When the "
            "retrieval context does not contain sufficient information to fully answer "
            "the question, the answer should explicitly acknowledge the limitation rather "
            "than fabricating or guessing. Penalize confident-sounding answers that go "
            "beyond what the context supports."
        ),
        evaluation_steps=[
            "Assess whether the retrieval context contains enough information to answer "
            "the question.",
            "If context is sufficient, check that the answer uses it appropriately "
            "(score 1.0 for correct use).",
            "If context is insufficient, check that the answer acknowledges the gap.",
            "Penalize confident assertions not supported by the retrieval context.",
            "A score of 1.0 means appropriate handling; 0.0 means confidently wrong.",
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        model=judge,
        threshold=0.0,
        async_mode=True,
    )


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


async def score_result(
    question: str,
    answer: str,
    contexts: list[str],
    expected_answer: str | None = None,
    *,
    evaluated_model_name: str = "",
) -> dict:
    """Score a single RAG response using DeepEval metrics.

    Always runs: faithfulness, answer relevancy, context relevancy, abstention.
    When expected_answer is provided: also runs completeness, correctness,
    compliance accuracy, and context precision.

    All metrics use threshold=0.0 (raw scores only). Pass/fail logic lives
    in the verdict layer using profile thresholds.

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

    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=contexts,
        expected_output=expected_answer,
    )

    scores: dict = {}

    # Always-on metrics (no expected_answer needed)
    metrics: list[tuple[str, object]] = [
        ("groundedness_score", FaithfulnessMetric(model=judge, threshold=0.0, async_mode=True)),
        ("relevancy_score", AnswerRelevancyMetric(model=judge, threshold=0.0, async_mode=True)),
        (
            "context_relevancy_score",
            ContextualRelevancyMetric(model=judge, threshold=0.0, async_mode=True),
        ),
        ("abstention_score", _abstention_metric(judge)),
    ]

    # Metrics that require expected_answer (ground truth)
    if expected_answer:
        metrics.extend(
            [
                (
                    "context_precision_score",
                    ContextualPrecisionMetric(model=judge, threshold=0.0, async_mode=True),
                ),
                ("completeness_score", _completeness_metric(judge)),
                ("correctness_score", _correctness_metric(judge)),
                ("compliance_accuracy_score", _compliance_accuracy_metric(judge)),
            ]
        )

    # Run all metrics concurrently -- each is an independent LLM judge call
    async def _measure(name: str, metric):
        try:
            await metric.a_measure(test_case)
            return name, metric.score
        except Exception as e:
            logger.error("Scoring failed for %s: %s", name, e, exc_info=True)
            return name, None

    results = await asyncio.gather(*[_measure(name, metric) for name, metric in metrics])
    for name, score in results:
        scores[name] = score

    groundedness = scores.get("groundedness_score")
    if groundedness is not None:
        scores["is_hallucination"] = groundedness < HALLUCINATION_THRESHOLD
    else:
        scores["is_hallucination"] = None

    return scores
