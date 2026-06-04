"""Tests for consolidated judge scoring service."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def _no_api_token():
    """Ensure no API token is set."""
    with patch("src.services.scoring.settings") as mock_settings:
        mock_settings.API_TOKEN = ""
        yield mock_settings


@pytest.fixture
def _mock_settings():
    """Provide valid settings for scoring."""
    with patch("src.services.scoring.settings") as mock_settings:
        mock_settings.API_TOKEN = "test-token"
        mock_settings.MAAS_ENDPOINT = "https://maas.example.com"
        mock_settings.JUDGE_MODEL_NAME = "granite-3.1-8b-instruct"
        mock_settings.MODEL_A_NAME = ""
        mock_settings.MODEL_B_NAME = ""
        mock_settings.resolved_judge_model_name = "granite-3.1-8b-instruct"
        mock_settings.api_token_bare = "test-token"
        yield mock_settings


def _make_prompt_a_response(
    faithfulness=0.85, relevancy=0.90, context_relevancy=0.75, abstention_quality=0.95
):
    """Build a valid Prompt A JSON response."""
    return json.dumps({
        "faithfulness": faithfulness,
        "relevancy": relevancy,
        "context_relevancy": context_relevancy,
        "abstention_quality": abstention_quality,
    })


def _make_prompt_b_response(
    completeness=0.80,
    correctness=0.90,
    compliance_accuracy=0.85,
    context_precision=0.75,
    concept_coverage=None,
):
    """Build a valid Prompt B JSON response."""
    data = {
        "completeness": completeness,
        "correctness": correctness,
        "compliance_accuracy": compliance_accuracy,
        "context_precision": context_precision,
    }
    if concept_coverage is not None:
        data["concept_coverage"] = concept_coverage
    return json.dumps(data)


@pytest.mark.asyncio
async def test_score_result_skips_when_no_token(_no_api_token):
    """Should return empty dict when no API token is configured."""
    from src.services.scoring import score_result

    result = await score_result(
        question="What is AI?",
        answer="AI is artificial intelligence.",
        contexts=["AI stands for artificial intelligence."],
    )
    assert result == {}


@pytest.mark.asyncio
async def test_score_result_skips_when_no_judge_model_name():
    """Should return empty dict when token is set but no judge/chat model name."""
    from src.services.scoring import score_result

    with patch("src.services.scoring.settings") as mock_settings:
        mock_settings.API_TOKEN = "test-token"
        mock_settings.resolved_judge_model_name = ""

        result = await score_result(
            question="What is AI?",
            answer="AI is artificial intelligence.",
            contexts=["AI stands for artificial intelligence."],
            evaluated_model_name="",
        )

    assert result == {}


@pytest.mark.asyncio
async def test_score_result_uses_evaluated_model_when_no_env_judge():
    """Should use evaluated_model_name as judge when env judge chain is empty."""
    from src.services.scoring import score_result

    with patch("src.services.scoring.settings") as mock_settings:
        mock_settings.API_TOKEN = "test-token"
        mock_settings.MAAS_ENDPOINT = "https://maas.example.com"
        mock_settings.JUDGE_MODEL_NAME = ""
        mock_settings.MODEL_A_NAME = ""
        mock_settings.MODEL_B_NAME = ""
        mock_settings.resolved_judge_model_name = ""
        mock_settings.api_token_bare = "test-token"

        with patch("src.services.scoring.MaaSJudgeModel") as mock_judge_cls:
            mock_judge = MagicMock()
            mock_judge.a_generate = AsyncMock(return_value=_make_prompt_a_response())
            mock_judge_cls.return_value = mock_judge

            result = await score_result(
                question="What is AI?",
                answer="AI is artificial intelligence.",
                contexts=["AI stands for artificial intelligence."],
                evaluated_model_name="qwen3-14b",
            )

    mock_judge_cls.assert_called_once()
    assert mock_judge_cls.call_args.kwargs["model_name"] == "qwen3-14b"
    assert result.get("relevancy_score") == 0.9


@pytest.mark.asyncio
async def test_score_result_returns_all_metrics(_mock_settings):
    """Should return all 8 metric scores when expected_answer is provided."""
    from src.services.scoring import score_result

    prompt_a_resp = _make_prompt_a_response(
        faithfulness=0.85, relevancy=0.90, context_relevancy=0.75, abstention_quality=0.95
    )
    prompt_b_resp = _make_prompt_b_response(
        completeness=0.80, correctness=0.88, compliance_accuracy=0.82, context_precision=0.70
    )

    with patch("src.services.scoring.MaaSJudgeModel") as mock_judge_cls:
        mock_judge = MagicMock()
        mock_judge.a_generate = AsyncMock(side_effect=[prompt_a_resp, prompt_b_resp])
        mock_judge_cls.return_value = mock_judge

        result = await score_result(
            question="What is AI?",
            answer="AI is artificial intelligence.",
            contexts=["AI stands for artificial intelligence."],
            expected_answer="AI is artificial intelligence.",
        )

    assert result["groundedness_score"] == 0.85
    assert result["relevancy_score"] == 0.90
    assert result["context_relevancy_score"] == 0.75
    assert result["abstention_score"] == 0.95
    assert result["completeness_score"] == 0.80
    assert result["correctness_score"] == 0.88
    assert result["compliance_accuracy_score"] == 0.82
    assert result["context_precision_score"] == 0.70
    assert result["is_hallucination"] is False


@pytest.mark.asyncio
async def test_score_result_detects_hallucination(_mock_settings):
    """Should flag hallucination when groundedness score is below threshold."""
    from src.services.scoring import score_result

    prompt_a_resp = _make_prompt_a_response(
        faithfulness=0.4, relevancy=0.90, context_relevancy=0.75, abstention_quality=0.95
    )

    with patch("src.services.scoring.MaaSJudgeModel") as mock_judge_cls:
        mock_judge = MagicMock()
        mock_judge.a_generate = AsyncMock(return_value=prompt_a_resp)
        mock_judge_cls.return_value = mock_judge

        result = await score_result(
            question="What is the capital requirement?",
            answer="Banks need 50% capital reserves.",
            contexts=["Basel III requires 8% minimum capital."],
        )

    assert result["groundedness_score"] == 0.4
    assert result["is_hallucination"] is True


@pytest.mark.asyncio
async def test_score_result_handles_prompt_a_failure(_mock_settings):
    """Should return None scores when Prompt A judge call fails."""
    from src.services.scoring import score_result

    with patch("src.services.scoring.MaaSJudgeModel") as mock_judge_cls:
        mock_judge = MagicMock()
        mock_judge.a_generate = AsyncMock(
            side_effect=RuntimeError("Judge model unavailable")
        )
        mock_judge_cls.return_value = mock_judge

        result = await score_result(
            question="What is AI?",
            answer="AI is artificial intelligence.",
            contexts=["AI stands for artificial intelligence."],
        )

    assert result["groundedness_score"] is None
    assert result["relevancy_score"] is None
    assert result["context_relevancy_score"] is None
    assert result["abstention_score"] is None
    assert result["is_hallucination"] is None


@pytest.mark.asyncio
async def test_score_result_handles_prompt_b_failure_keeps_prompt_a(_mock_settings):
    """Should keep Prompt A scores even when Prompt B fails."""
    from src.services.scoring import score_result

    prompt_a_resp = _make_prompt_a_response()

    with patch("src.services.scoring.MaaSJudgeModel") as mock_judge_cls:
        mock_judge = MagicMock()
        call_count = 0

        async def side_effect(prompt):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return prompt_a_resp
            raise RuntimeError("Prompt B failed")

        mock_judge.a_generate = AsyncMock(side_effect=side_effect)
        mock_judge_cls.return_value = mock_judge

        result = await score_result(
            question="What is AI?",
            answer="AI is artificial intelligence.",
            contexts=["AI stands for artificial intelligence."],
            expected_answer="AI is artificial intelligence.",
        )

    assert result["groundedness_score"] == 0.85
    assert result["relevancy_score"] == 0.90
    assert result["completeness_score"] is None
    assert result["correctness_score"] is None
    assert result["is_hallucination"] is False


@pytest.mark.asyncio
async def test_score_result_without_expected_answer(_mock_settings):
    """Should omit Prompt B metrics when no expected_answer is provided."""
    from src.services.scoring import score_result

    prompt_a_resp = _make_prompt_a_response()

    with patch("src.services.scoring.MaaSJudgeModel") as mock_judge_cls:
        mock_judge = MagicMock()
        mock_judge.a_generate = AsyncMock(return_value=prompt_a_resp)
        mock_judge_cls.return_value = mock_judge

        result = await score_result(
            question="What is AI?",
            answer="AI is artificial intelligence.",
            contexts=["AI stands for artificial intelligence."],
        )

    assert "groundedness_score" in result
    assert "relevancy_score" in result
    assert "context_relevancy_score" in result
    assert "abstention_score" in result
    assert "context_precision_score" not in result
    assert "completeness_score" not in result
    assert "correctness_score" not in result
    assert "compliance_accuracy_score" not in result
    assert result["is_hallucination"] is False
    # Only 1 judge call (Prompt A), not 2
    assert mock_judge.a_generate.call_count == 1


@pytest.mark.asyncio
async def test_score_result_makes_two_concurrent_calls_with_expected_answer(_mock_settings):
    """Should make exactly 2 judge calls (Prompt A + B) when expected_answer is provided."""
    from src.services.scoring import score_result

    prompt_a_resp = _make_prompt_a_response()
    prompt_b_resp = _make_prompt_b_response()

    with patch("src.services.scoring.MaaSJudgeModel") as mock_judge_cls:
        mock_judge = MagicMock()
        mock_judge.a_generate = AsyncMock(side_effect=[prompt_a_resp, prompt_b_resp])
        mock_judge_cls.return_value = mock_judge

        await score_result(
            question="What is AI?",
            answer="AI is artificial intelligence.",
            contexts=["AI stands for artificial intelligence."],
            expected_answer="AI is artificial intelligence.",
        )

    assert mock_judge.a_generate.call_count == 2


def test_maas_judge_model_get_model_name():
    """MaaSJudgeModel should return the configured model name."""
    from src.services.scoring import MaaSJudgeModel

    judge = MaaSJudgeModel(
        model_name="granite-3.1-8b-instruct",
        base_url="https://maas.example.com",
        api_key="test-token",
    )
    assert judge.get_model_name() == "granite-3.1-8b-instruct"


# --- Parse scores tests ---


def test_parse_scores_valid_json():
    """Should parse well-formed JSON scores correctly."""
    from src.services.scoring import _parse_scores

    raw = '{"faithfulness": 0.85, "relevancy": 0.90, "context_relevancy": 0.75}'
    result = _parse_scores(raw, ["faithfulness", "relevancy", "context_relevancy"])
    assert result == {"faithfulness": 0.85, "relevancy": 0.90, "context_relevancy": 0.75}


def test_parse_scores_with_markdown_fencing():
    """Should strip markdown code fences before parsing."""
    from src.services.scoring import _parse_scores

    raw = '```json\n{"faithfulness": 0.85, "relevancy": 0.90}\n```'
    result = _parse_scores(raw, ["faithfulness", "relevancy"])
    assert result == {"faithfulness": 0.85, "relevancy": 0.90}


def test_parse_scores_regex_fallback():
    """Should fall back to regex extraction when JSON parse fails."""
    from src.services.scoring import _parse_scores

    raw = 'Here are the scores: "faithfulness": 0.85, "relevancy": 0.90, extra text'
    result = _parse_scores(raw, ["faithfulness", "relevancy"])
    assert result["faithfulness"] == 0.85
    assert result["relevancy"] == 0.90


def test_parse_scores_missing_key_returns_none():
    """Should return None for keys not present in the response."""
    from src.services.scoring import _parse_scores

    raw = '{"faithfulness": 0.85}'
    result = _parse_scores(raw, ["faithfulness", "relevancy"])
    assert result["faithfulness"] == 0.85
    assert result["relevancy"] is None


def test_parse_scores_clamps_out_of_range():
    """Should clamp scores to [0.0, 1.0]."""
    from src.services.scoring import _parse_scores

    raw = '{"faithfulness": 1.5, "relevancy": -0.3}'
    result = _parse_scores(raw, ["faithfulness", "relevancy"])
    assert result["faithfulness"] == 1.0
    assert result["relevancy"] == 0.0


def test_parse_scores_handles_null():
    """Should return None for null values in JSON."""
    from src.services.scoring import _parse_scores

    raw = '{"faithfulness": null, "relevancy": 0.85}'
    result = _parse_scores(raw, ["faithfulness", "relevancy"])
    assert result["faithfulness"] is None
    assert result["relevancy"] == 0.85


# --- Chunk alignment tests ---


def test_chunk_alignment_perfect_match():
    """Should return 1.0 when all expected chunks are retrieved."""
    from src.services.scoring import compute_chunk_alignment

    retrieved = [
        {"source_document": "report.pdf", "page_number": "3"},
        {"source_document": "guide.pdf", "page_number": "1"},
    ]
    expected = ["report.pdf:3", "guide.pdf:1"]
    assert compute_chunk_alignment(retrieved, expected) == 1.0


def test_chunk_alignment_partial_match():
    """Should return fraction of matched expected chunks."""
    from src.services.scoring import compute_chunk_alignment

    retrieved = [
        {"source_document": "report.pdf", "page_number": "3"},
        {"source_document": "other.pdf", "page_number": "5"},
    ]
    expected = ["report.pdf:3", "guide.pdf:1"]
    assert compute_chunk_alignment(retrieved, expected) == 0.5


def test_chunk_alignment_no_match():
    """Should return 0.0 when no expected chunks are retrieved."""
    from src.services.scoring import compute_chunk_alignment

    retrieved = [
        {"source_document": "other.pdf", "page_number": "1"},
    ]
    expected = ["report.pdf:3", "guide.pdf:1"]
    assert compute_chunk_alignment(retrieved, expected) == 0.0


def test_chunk_alignment_doc_only_match():
    """Should match on document name when no page is specified in expected."""
    from src.services.scoring import compute_chunk_alignment

    retrieved = [
        {"source_document": "report.pdf", "page_number": "7"},
    ]
    expected = ["report.pdf"]
    assert compute_chunk_alignment(retrieved, expected) == 1.0


def test_chunk_alignment_empty_expected():
    """Should return 1.0 when no expected chunks are specified."""
    from src.services.scoring import compute_chunk_alignment

    retrieved = [{"source_document": "report.pdf", "page_number": "1"}]
    assert compute_chunk_alignment(retrieved, []) == 1.0


def test_chunk_alignment_empty_retrieved():
    """Should return 0.0 when nothing was retrieved but chunks were expected."""
    from src.services.scoring import compute_chunk_alignment

    assert compute_chunk_alignment([], ["report.pdf:3"]) == 0.0


def test_chunk_alignment_mixed_format():
    """Should handle mix of doc-only and doc:page expected chunks."""
    from src.services.scoring import compute_chunk_alignment

    retrieved = [
        {"source_document": "report.pdf", "page_number": "3"},
        {"source_document": "guide.pdf", "page_number": None},
    ]
    expected = ["report.pdf:3", "guide.pdf"]
    assert compute_chunk_alignment(retrieved, expected) == 1.0


def test_chunk_alignment_chunk_id_format():
    """Should match chunk:{id} refs against retrieved chunk IDs."""
    from src.services.scoring import compute_chunk_alignment

    retrieved = [
        {"id": 42, "source_document": "guide.pdf", "page_number": "3"},
        {"id": 43, "source_document": "guide.pdf", "page_number": "4"},
        {"id": 67, "source_document": "report.pdf", "page_number": "1"},
    ]
    expected = ["chunk:42", "chunk:67"]
    assert compute_chunk_alignment(retrieved, expected) == 1.0


def test_chunk_alignment_chunk_id_partial():
    """Should return fraction when some chunk:{id} refs are not retrieved."""
    from src.services.scoring import compute_chunk_alignment

    retrieved = [
        {"id": 42, "source_document": "guide.pdf", "page_number": "3"},
    ]
    expected = ["chunk:42", "chunk:99"]
    assert compute_chunk_alignment(retrieved, expected) == 0.5


def test_chunk_alignment_mixed_chunk_id_and_legacy():
    """Should handle mix of chunk:{id} and legacy filename refs."""
    from src.services.scoring import compute_chunk_alignment

    retrieved = [
        {"id": 42, "source_document": "guide.pdf", "page_number": "3"},
        {"id": 43, "source_document": "report.pdf", "page_number": "1"},
    ]
    expected = ["chunk:42", "report.pdf:1"]
    assert compute_chunk_alignment(retrieved, expected) == 1.0


# --- Concept coverage tests ---


def test_parse_concept_coverage_from_json():
    """Should extract concept_coverage array from JSON response."""
    from src.services.scoring import _parse_concept_coverage

    raw = json.dumps({
        "completeness": 0.80,
        "concept_coverage": ["covered", "missing", "covered"],
    })
    result = _parse_concept_coverage(raw, 3)
    assert result == ["covered", "missing", "covered"]


def test_parse_concept_coverage_pads_short_array():
    """Should pad with 'missing' when array is shorter than expected."""
    from src.services.scoring import _parse_concept_coverage

    raw = json.dumps({"concept_coverage": ["covered"]})
    result = _parse_concept_coverage(raw, 3)
    assert result == ["covered", "missing", "missing"]


def test_parse_concept_coverage_truncates_long_array():
    """Should truncate when array is longer than expected."""
    from src.services.scoring import _parse_concept_coverage

    raw = json.dumps({"concept_coverage": ["covered", "missing", "covered", "covered"]})
    result = _parse_concept_coverage(raw, 2)
    assert result == ["covered", "missing"]


def test_parse_concept_coverage_regex_fallback():
    """Should extract concept_coverage via regex when JSON parsing fails."""
    from src.services.scoring import _parse_concept_coverage

    raw = 'some text "concept_coverage": ["covered", "missing"] more text'
    result = _parse_concept_coverage(raw, 2)
    assert result == ["covered", "missing"]


def test_parse_concept_coverage_returns_none_when_missing():
    """Should return None when concept_coverage is not in the response."""
    from src.services.scoring import _parse_concept_coverage

    raw = json.dumps({"completeness": 0.80})
    result = _parse_concept_coverage(raw, 3)
    assert result is None


@pytest.mark.asyncio
async def test_score_result_with_concepts_returns_coverage_gaps(_mock_settings):
    """Should return coverage_gaps when required_concepts are provided."""
    from src.services.scoring import score_result

    prompt_a_resp = _make_prompt_a_response()
    prompt_b_resp = _make_prompt_b_response(
        concept_coverage=["covered", "missing", "covered"],
    )

    with patch("src.services.scoring.MaaSJudgeModel") as mock_judge_cls:
        mock_judge = MagicMock()
        mock_judge.a_generate = AsyncMock(side_effect=[prompt_a_resp, prompt_b_resp])
        mock_judge_cls.return_value = mock_judge

        result = await score_result(
            question="What are ETF requirements?",
            answer="ETFs must file registration forms and provide transparency.",
            contexts=["Form N-1A requires registration."],
            expected_answer="ETFs must file forms, provide transparency, and report quarterly.",
            required_concepts=[
                "ETF registration forms",
                "quarterly reporting",
                "portfolio transparency",
            ],
        )

    gaps = result["coverage_gaps"]
    assert gaps["concepts"] == [
        "ETF registration forms",
        "quarterly reporting",
        "portfolio transparency",
    ]
    assert gaps["covered"] == ["ETF registration forms", "portfolio transparency"]
    assert gaps["missing"] == ["quarterly reporting"]
    assert gaps["coverage_ratio"] == pytest.approx(2 / 3)


@pytest.mark.asyncio
async def test_score_result_concepts_classifies_failures(_mock_settings):
    """Should classify missing concepts as retrieval or generation failures."""
    from src.services.scoring import score_result

    prompt_a_resp = _make_prompt_a_response()
    prompt_b_resp = _make_prompt_b_response(
        concept_coverage=["covered", "missing", "missing"],
    )

    with patch("src.services.scoring.MaaSJudgeModel") as mock_judge_cls:
        mock_judge = MagicMock()
        mock_judge.a_generate = AsyncMock(side_effect=[prompt_a_resp, prompt_b_resp])
        mock_judge_cls.return_value = mock_judge

        result = await score_result(
            question="What are the requirements?",
            answer="ETFs must file registration forms.",
            contexts=["Form N-PORT requires quarterly filing of portfolio holdings."],
            expected_answer="Registration, quarterly filing, blockchain custody.",
            required_concepts=[
                "registration forms",
                "quarterly filing deadline",
                "blockchain custody requirements",
            ],
        )

    gaps = result["coverage_gaps"]
    assert "quarterly filing deadline" in gaps["generation_failures"]
    assert "blockchain custody requirements" in gaps["retrieval_failures"]


@pytest.mark.asyncio
async def test_score_result_no_coverage_gaps_without_concepts(_mock_settings):
    """Should not include coverage_gaps when no concepts are provided."""
    from src.services.scoring import score_result

    prompt_a_resp = _make_prompt_a_response()
    prompt_b_resp = _make_prompt_b_response()

    with patch("src.services.scoring.MaaSJudgeModel") as mock_judge_cls:
        mock_judge = MagicMock()
        mock_judge.a_generate = AsyncMock(side_effect=[prompt_a_resp, prompt_b_resp])
        mock_judge_cls.return_value = mock_judge

        result = await score_result(
            question="What is AI?",
            answer="AI is artificial intelligence.",
            contexts=["AI stands for artificial intelligence."],
            expected_answer="AI is artificial intelligence.",
        )

    assert "coverage_gaps" not in result


def test_resolved_judge_model_name_order():
    """Judge model should prefer JUDGE_MODEL_NAME, then MODEL_A_NAME, then MODEL_B_NAME."""
    from src.core.config import Settings

    s = Settings(
        JUDGE_MODEL_NAME="judge-m",
        MODEL_A_NAME="model-a",
        MODEL_B_NAME="model-b",
        MAAS_ENDPOINT="https://x",
        API_TOKEN="t",
    )
    assert s.resolved_judge_model_name == "judge-m"

    s2 = Settings(
        JUDGE_MODEL_NAME="",
        MODEL_A_NAME="model-a",
        MODEL_B_NAME="model-b",
        MAAS_ENDPOINT="https://x",
        API_TOKEN="t",
    )
    assert s2.resolved_judge_model_name == "model-a"

    s3 = Settings(
        JUDGE_MODEL_NAME="",
        MODEL_A_NAME="",
        MODEL_B_NAME="model-b",
        MAAS_ENDPOINT="https://x",
        API_TOKEN="t",
    )
    assert s3.resolved_judge_model_name == "model-b"
