"""Coverage gap detection -- extracts key concepts from expected answers
and checks which are present in the model's actual answer, distinguishing
between retrieval failures (concept not in context) and generation failures
(concept in context but omitted by model).

Two-phase approach for consistency:
1. Extract concepts from the expected answer (cached so the same expected
   answer always produces the same concept list across runs)
2. Check which concepts the actual answer covers (per-run)
3. For missing concepts, check if they appear in the retrieval context
"""

import json
import logging

import httpx

from ..core.config import settings

logger = logging.getLogger(__name__)

COVERAGE_TIMEOUT = 60.0  # seconds

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=COVERAGE_TIMEOUT)
    return _client

# Cache extracted concepts so the same (question, expected answer) pair
# always produces the same concept list across model runs.
_concept_cache: dict[str, list[str]] = {}
_CONCEPT_CACHE_MAX = 200


def _concept_cache_key(question: str | None, expected_answer: str) -> str:
    """Isolate cache entries when the same gold text appears under different questions."""
    q = (question or "").strip()
    return f"{q}\x00{expected_answer}"


EXTRACT_CONCEPTS_PROMPT = """\
You are an evaluation analyst. Break the EXPECTED ANSWER into its key \
high-level concepts for coverage checking.

RULES:
- Extract BROAD, HIGH-LEVEL concepts, not fine-grained sub-facts. Each \
concept should capture a major theme, requirement, or conclusion -- not \
individual details, examples, or qualifications within that theme.
- Each concept MUST be a short phrase (at most 15 words).
- Return between 3 and 10 concepts. Short answers yield 3-5; long answers \
yield 6-10. Never exceed 10.
- Merge related facts into a single broader concept. For example, merge \
"must file Form X" and "must file Form Y" into "must file required \
registration forms".
- Do NOT extract individual examples, specific thresholds, or parenthetical \
details as separate concepts -- fold them into the broader concept they \
support.
- Do NOT pad the list or invent concepts not grounded in the expected answer.
- Order concepts by importance: core requirements first, minor details last.

Respond with a JSON array of short strings. No other text.

Example:
EXPECTED ANSWER: "ETFs must file Form N-1A and publish daily NAV. The SAI \
provides details on portfolio holdings and tax information."
Output: [
  "ETFs must file registration forms and publish daily NAV",
  "SAI provides portfolio holdings and tax information"
]

EXPECTED ANSWER:
{expected_answer}
"""

CHECK_COVERAGE_PROMPT = """\
You are an evaluation analyst. Given a list of REQUIRED CONCEPTS and an \
ACTUAL ANSWER, determine which concepts the answer covers.

For each concept, respond with:
- "covered" if the concept is present or adequately addressed
- "missing" if the concept is absent or not addressed

Respond with a JSON array matching the order of the input concepts. \
Each element should be "covered" or "missing". No other text.

REQUIRED CONCEPTS:
{concepts_json}

ACTUAL ANSWER:
{actual_answer}
"""


def _check_concept_in_contexts(concept: str, contexts: list[str]) -> bool:
    """Check if a concept appears in retrieval contexts using keyword overlap.

    Uses word-boundary matching -- if >40% of the concept's significant
    words appear as whole tokens in the combined context, the concept
    is considered present in retrieval.
    """
    if not contexts:
        return False

    concept_words = {w.lower() for w in concept.split() if len(w) >= 3}
    if not concept_words:
        return False

    combined = " ".join(contexts).lower()
    context_tokens = {w.rstrip(".,;:!?'\")]}") for w in combined.split()}
    matched = sum(1 for w in concept_words if w in context_tokens)
    return matched / len(concept_words) >= 0.4


def _strip_markdown_fencing(content: str) -> str:
    """Remove markdown code fencing if present."""
    if content.startswith("```"):
        lines = content.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        return "\n".join(lines).strip()
    return content


async def _extract_concepts(
    expected_answer: str,
    model_name: str,
    endpoint: str,
    token: str,
    *,
    cache_key: str,
) -> list[str] | None:
    """Extract key concepts from an expected answer (Phase 1).

    Results are cached so the same cache key always produces the same concept
    list, regardless of which model's actual answer is being evaluated.
    """
    cached = _concept_cache.get(cache_key)
    if cached is not None:
        logger.info("Using cached concept extraction (%d concepts)", len(cached))
        return cached

    url = f"{endpoint}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": EXTRACT_CONCEPTS_PROMPT.format(
                    expected_answer=expected_answer,
                ),
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
            logger.warning("Concept extraction returned non-list or empty")
            return None

        result = [c.strip() for c in concepts if isinstance(c, str) and c.strip()]
        if not result:
            return None

        if len(result) > 15:
            logger.info(
                "Concept extraction returned %d concepts (prompt asks for ≤15), "
                "keeping all to avoid dropping relevant ones",
                len(result),
            )

        # Cache the result
        if len(_concept_cache) >= _CONCEPT_CACHE_MAX:
            oldest = next(iter(_concept_cache))
            del _concept_cache[oldest]
        _concept_cache[cache_key] = result

        logger.info("Extracted %d concepts from expected answer", len(result))
        return result

    except json.JSONDecodeError:
        logger.warning("Failed to parse concept extraction response as JSON")
        return None
    except Exception as e:
        logger.warning("Concept extraction failed (%s)", e)
        return None


async def _check_coverage(
    concepts: list[str],
    actual_answer: str,
    model_name: str,
    endpoint: str,
    token: str,
) -> list[str] | None:
    """Check which concepts the actual answer covers (Phase 2).

    Returns a list of statuses ("covered"/"missing") matching the
    concept list order, or None on failure.
    """
    url = f"{endpoint}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": CHECK_COVERAGE_PROMPT.format(
                    concepts_json=json.dumps(concepts),
                    actual_answer=actual_answer,
                ),
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

        statuses = json.loads(content)
        if not isinstance(statuses, list):
            logger.warning("Coverage check returned non-list")
            return None

        # Validate and normalize
        valid = []
        for s in statuses:
            if isinstance(s, str) and s.strip().lower() in ("covered", "missing"):
                valid.append(s.strip().lower())
            else:
                valid.append("missing")  # default to missing if unclear

        # Pad or truncate to match concept count
        while len(valid) < len(concepts):
            valid.append("missing")
        valid = valid[: len(concepts)]

        return valid

    except json.JSONDecodeError:
        logger.warning("Failed to parse coverage check response as JSON")
        return None
    except Exception as e:
        logger.warning("Coverage check failed (%s)", e)
        return None


async def detect_coverage_gaps(
    expected_answer: str,
    actual_answer: str,
    contexts: list[str] | None = None,
    model_name: str | None = None,
    question: str | None = None,
    pre_extracted_concepts: list[str] | None = None,
) -> dict | None:
    """Extract key concepts from expected answer and check coverage.

    Uses a two-phase approach: concept extraction is cached so the same
    (question, expected answer) pair always produces the same concept list.
    Coverage checking runs per actual answer.

    When contexts are provided, classifies each missing concept as either
    a retrieval failure (not in context) or a generation failure (in context
    but omitted by the model).

    Args:
        expected_answer: The ground truth answer.
        actual_answer: The model's generated answer.
        contexts: Retrieved context chunks used for generation.
        model_name: Model to use for analysis. Defaults to the judge model.
        question: Optional eval question; included in the concept cache key when
            the same expected answer appears under different questions.
        pre_extracted_concepts: When provided, skip LLM concept extraction
            and use these concepts directly. Used when structured truth is
            available from truth generation.

    Returns:
        Dict with 'concepts', 'covered', 'missing', 'coverage_ratio',
        and optionally 'retrieval_failures' and 'generation_failures' keys,
        or None if analysis fails.
    """
    resolved_model = model_name or settings.resolved_judge_model_name
    if not resolved_model:
        logger.info("No model configured for coverage analysis, skipping")
        return None

    model_cfg = settings.get_model_config(resolved_model)
    if not model_cfg["token"]:
        logger.info("No API token for coverage model, skipping")
        return None

    endpoint = model_cfg["endpoint"]
    token = model_cfg["token"]

    # Phase 1: use pre-extracted concepts or extract via LLM (cached)
    if pre_extracted_concepts:
        concepts = pre_extracted_concepts
        logger.info("Using %d pre-extracted concepts from structured truth", len(concepts))
    else:
        cache_key = _concept_cache_key(question, expected_answer)
        concepts = await _extract_concepts(
            expected_answer, resolved_model, endpoint, token, cache_key=cache_key
        )
    if not concepts:
        return None

    # Phase 2: check coverage
    statuses = await _check_coverage(concepts, actual_answer, resolved_model, endpoint, token)
    if not statuses:
        return None

    covered = [c for c, s in zip(concepts, statuses) if s == "covered"]
    missing = [c for c, s in zip(concepts, statuses) if s == "missing"]

    result: dict = {
        "concepts": concepts,
        "covered": covered,
        "missing": missing,
        "coverage_ratio": len(covered) / len(concepts) if concepts else 1.0,
    }

    # Classify missing concepts as retrieval or generation failures
    if contexts is not None and missing:
        retrieval_failures = []
        generation_failures = []
        for concept in missing:
            if _check_concept_in_contexts(concept, contexts):
                generation_failures.append(concept)
            else:
                retrieval_failures.append(concept)
        result["retrieval_failures"] = retrieval_failures
        result["generation_failures"] = generation_failures

    logger.info(
        "Coverage analysis: %d/%d concepts covered (%.0f%%), missing: %s",
        len(covered),
        len(concepts),
        result["coverage_ratio"] * 100,
        missing[:5] if missing else "none",
    )
    if contexts is not None and missing:
        logger.info(
            "Failure classification: %d retrieval, %d generation",
            len(result.get("retrieval_failures", [])),
            len(result.get("generation_failures", [])),
        )

    return result
