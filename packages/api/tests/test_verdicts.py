"""Tests for the verdict service and profile loader."""

import pytest

from src.services.profiles import (
    EvalProfile,
    GenerationConfig,
    RetrievalConfig,
    list_profiles,
    load_profile,
)
from src.services.verdicts import compute_question_verdict, compute_run_verdict


@pytest.fixture
def fsi_profile() -> EvalProfile:
    """A test profile matching the FSI compliance profile structure."""
    return EvalProfile(
        id="test_profile",
        version="1.0",
        domain="fsi",
        thresholds={
            "groundedness_score": 0.7,
            "relevancy_score": 0.5,
            "completeness_score": 0.5,
            "correctness_score": 0.5,
            "compliance_accuracy_score": 0.5,
            "abstention_score": 0.5,
        },
        critical_thresholds={
            "groundedness_score": 0.5,
            "compliance_accuracy_score": 0.3,
        },
    )


# --- Profile loader tests ---


def test_load_fsi_compliance_profile():
    """Should load the built-in FSI compliance profile."""
    profile = load_profile("fsi_compliance_v1")
    assert profile.id == "fsi_compliance_v1"
    assert profile.domain == "fsi"
    assert "groundedness_score" in profile.thresholds
    assert "groundedness_score" in profile.critical_thresholds
    assert isinstance(profile.retrieval, RetrievalConfig)
    assert profile.retrieval.top_k == 8
    assert isinstance(profile.generation, GenerationConfig)
    assert profile.generation.max_tokens == 768


def test_fsi_profile_has_system_prompt():
    """Should include a domain-specific system prompt in the FSI profile."""
    profile = load_profile("fsi_compliance_v1")
    assert profile.system_prompt
    assert "compliance" in profile.system_prompt.lower()
    assert "financial" in profile.system_prompt.lower() or "SEC" in profile.system_prompt


def test_profile_system_prompt_defaults_empty():
    """Should default system_prompt to empty string when not specified."""
    profile = EvalProfile(id="minimal")
    assert profile.system_prompt == ""


def test_load_profile_not_found():
    """Should raise FileNotFoundError for unknown profile."""
    with pytest.raises(FileNotFoundError, match="nonexistent"):
        load_profile("nonexistent")


def test_list_profiles_includes_fsi():
    """Should list at least the FSI compliance profile."""
    profiles = list_profiles()
    assert "fsi_compliance_v1" in profiles


# --- Question verdict tests ---


def test_verdict_pass_when_all_above_threshold(fsi_profile):
    """Should return PASS when all scores exceed thresholds."""
    scores = {
        "groundedness_score": 0.9,
        "relevancy_score": 0.8,
        "completeness_score": 0.7,
        "correctness_score": 0.6,
        "compliance_accuracy_score": 0.8,
        "abstention_score": 0.9,
    }
    verdict = compute_question_verdict(scores, fsi_profile)
    assert verdict.verdict == "PASS"
    assert len(verdict.fail_reasons) == 0
    assert len(verdict.failed_metrics) == 0


def test_verdict_review_when_below_regular_threshold(fsi_profile):
    """Should return REVIEW_REQUIRED when a score is below threshold but above critical."""
    scores = {
        "groundedness_score": 0.6,  # below 0.7 threshold, above 0.5 critical
        "relevancy_score": 0.8,
        "completeness_score": 0.7,
        "correctness_score": 0.6,
        "compliance_accuracy_score": 0.8,
        "abstention_score": 0.9,
    }
    verdict = compute_question_verdict(scores, fsi_profile)
    assert verdict.verdict == "REVIEW_REQUIRED"
    assert "FAIL_LOW_GROUNDEDNESS" in verdict.fail_reasons
    assert "groundedness_score" in verdict.failed_metrics


def test_verdict_fail_when_below_critical_threshold(fsi_profile):
    """Should return FAIL when a score is below critical threshold."""
    scores = {
        "groundedness_score": 0.3,  # below 0.5 critical
        "relevancy_score": 0.8,
        "completeness_score": 0.7,
        "correctness_score": 0.6,
        "compliance_accuracy_score": 0.8,
        "abstention_score": 0.9,
    }
    verdict = compute_question_verdict(scores, fsi_profile)
    assert verdict.verdict == "FAIL"
    assert "FAIL_LOW_GROUNDEDNESS" in verdict.fail_reasons


def test_verdict_compliance_critical_fail(fsi_profile):
    """Should FAIL when compliance accuracy is below its critical threshold."""
    scores = {
        "groundedness_score": 0.9,
        "relevancy_score": 0.8,
        "compliance_accuracy_score": 0.2,  # below 0.3 critical
        "abstention_score": 0.9,
    }
    verdict = compute_question_verdict(scores, fsi_profile)
    assert verdict.verdict == "FAIL"
    assert "FAIL_COMPLIANCE_VIOLATION" in verdict.fail_reasons


def test_verdict_skips_none_scores(fsi_profile):
    """Should skip metrics with None scores (not penalized)."""
    scores = {
        "groundedness_score": 0.9,
        "relevancy_score": 0.8,
        "completeness_score": None,  # no expected_answer provided
        "correctness_score": None,
        "compliance_accuracy_score": None,
        "abstention_score": 0.9,
    }
    verdict = compute_question_verdict(scores, fsi_profile)
    assert verdict.verdict == "PASS"
    assert len(verdict.fail_reasons) == 0


def test_verdict_multiple_failures(fsi_profile):
    """Should collect all fail reasons when multiple metrics fail."""
    scores = {
        "groundedness_score": 0.3,  # critical fail
        "relevancy_score": 0.3,  # below threshold
        "completeness_score": 0.2,  # below threshold
        "abstention_score": 0.9,
    }
    verdict = compute_question_verdict(scores, fsi_profile)
    assert verdict.verdict == "FAIL"
    assert len(verdict.fail_reasons) >= 2


# --- Run verdict tests ---


def test_run_verdict_all_pass(fsi_profile):
    """Should return PASS when all questions pass."""
    scores = {
        "groundedness_score": 0.9,
        "relevancy_score": 0.8,
        "abstention_score": 0.9,
    }
    q_verdicts = [compute_question_verdict(scores, fsi_profile) for _ in range(5)]
    run_verdict = compute_run_verdict(q_verdicts)
    assert run_verdict.overall == "PASS"
    assert run_verdict.pass_count == 5
    assert run_verdict.fail_count == 0
    assert "5/5" in run_verdict.summary


def test_run_verdict_with_failures(fsi_profile):
    """Should return FAIL when any question fails."""
    pass_scores = {"groundedness_score": 0.9, "abstention_score": 0.9}
    fail_scores = {"groundedness_score": 0.3, "abstention_score": 0.9}  # critical fail

    verdicts = [
        compute_question_verdict(pass_scores, fsi_profile),
        compute_question_verdict(pass_scores, fsi_profile),
        compute_question_verdict(fail_scores, fsi_profile),
    ]
    run_verdict = compute_run_verdict(verdicts)
    assert run_verdict.overall == "FAIL"
    assert run_verdict.pass_count == 2
    assert run_verdict.fail_count == 1
    assert "2/3" in run_verdict.summary


def test_run_verdict_review_required(fsi_profile):
    """Should return REVIEW_REQUIRED when no fails but some reviews."""
    pass_scores = {"groundedness_score": 0.9, "abstention_score": 0.9}
    review_scores = {"groundedness_score": 0.6, "abstention_score": 0.9}  # below threshold

    verdicts = [
        compute_question_verdict(pass_scores, fsi_profile),
        compute_question_verdict(review_scores, fsi_profile),
    ]
    run_verdict = compute_run_verdict(verdicts)
    assert run_verdict.overall == "REVIEW_REQUIRED"
    assert run_verdict.review_count == 1


# --- Deterministic check integration ---


def test_verdict_fail_on_retrieval_check_failure(fsi_profile):
    """Should return FAIL when a retrieval deterministic check fails."""
    scores = {
        "groundedness_score": 0.9,
        "relevancy_score": 0.8,
        "abstention_score": 0.9,
    }
    checks = [
        {
            "check_name": "document_presence",
            "passed": False,
            "detail": "Missing",
            "category": "retrieval",
        },
    ]
    verdict = compute_question_verdict(scores, fsi_profile, deterministic_checks=checks)
    assert verdict.verdict == "FAIL"
    assert "FAIL_RETRIEVAL_INCOMPLETE" in verdict.fail_reasons
    assert "document_presence" in verdict.failed_metrics


def test_verdict_fail_on_chunk_alignment_failure(fsi_profile):
    """Should return FAIL when chunk_alignment check fails."""
    scores = {
        "groundedness_score": 0.9,
        "relevancy_score": 0.8,
        "abstention_score": 0.9,
    }
    checks = [
        {
            "check_name": "chunk_alignment",
            "passed": False,
            "detail": "0/3",
            "category": "retrieval",
        },
    ]
    verdict = compute_question_verdict(scores, fsi_profile, deterministic_checks=checks)
    assert verdict.verdict == "FAIL"
    assert "FAIL_RETRIEVAL_INCOMPLETE" in verdict.fail_reasons


def test_verdict_ignores_chunk_alignment_for_generation_only_gaps(fsi_profile):
    """Should not force FAIL on chunk_alignment when misses are generation-only."""
    scores = {
        "groundedness_score": 0.9,
        "relevancy_score": 0.8,
        "abstention_score": 0.9,
    }
    checks = [
        {
            "check_name": "document_presence",
            "passed": True,
            "detail": "All required documents present",
            "category": "retrieval",
        },
        {
            "check_name": "chunk_alignment",
            "passed": False,
            "detail": "4/13",
            "category": "retrieval",
        },
    ]
    coverage_gaps = {
        "missing": ["Rule 6c-11 requires daily website transparency"],
        "retrieval_failures": [],
        "generation_failures": ["Rule 6c-11 requires daily website transparency"],
    }

    verdict = compute_question_verdict(
        scores,
        fsi_profile,
        deterministic_checks=checks,
        coverage_gaps=coverage_gaps,
    )
    assert verdict.verdict == "PASS"
    assert "FAIL_RETRIEVAL_INCOMPLETE" not in verdict.fail_reasons


def test_verdict_keeps_chunk_alignment_fail_for_retrieval_gaps(fsi_profile):
    """Should still FAIL when chunk_alignment misses are retrieval-side."""
    scores = {
        "groundedness_score": 0.9,
        "relevancy_score": 0.8,
        "abstention_score": 0.9,
    }
    checks = [
        {
            "check_name": "document_presence",
            "passed": True,
            "detail": "All required documents present",
            "category": "retrieval",
        },
        {
            "check_name": "chunk_alignment",
            "passed": False,
            "detail": "1/10",
            "category": "retrieval",
        },
    ]
    coverage_gaps = {
        "missing": ["N-PORT filing deadline"],
        "retrieval_failures": ["N-PORT filing deadline"],
        "generation_failures": [],
    }

    verdict = compute_question_verdict(
        scores,
        fsi_profile,
        deterministic_checks=checks,
        coverage_gaps=coverage_gaps,
    )
    assert verdict.verdict == "FAIL"
    assert "FAIL_RETRIEVAL_INCOMPLETE" in verdict.fail_reasons


def test_verdict_pass_with_passing_retrieval_checks(fsi_profile):
    """Should not affect verdict when retrieval checks pass."""
    scores = {
        "groundedness_score": 0.9,
        "relevancy_score": 0.8,
        "abstention_score": 0.9,
    }
    checks = [
        {
            "check_name": "document_presence",
            "passed": True,
            "detail": "OK",
            "category": "retrieval",
        },
        {"check_name": "chunk_alignment", "passed": True, "detail": "2/2", "category": "retrieval"},
    ]
    verdict = compute_question_verdict(scores, fsi_profile, deterministic_checks=checks)
    assert verdict.verdict == "PASS"
    assert len(verdict.fail_reasons) == 0


def test_verdict_ignores_generation_check_failures(fsi_profile):
    """Should not fail verdict based on generation check failures (warnings only)."""
    scores = {
        "groundedness_score": 0.9,
        "relevancy_score": 0.8,
        "abstention_score": 0.9,
    }
    checks = [
        {
            "check_name": "abstention_validation",
            "passed": False,
            "detail": "Mismatch",
            "category": "generation",
        },
        {
            "check_name": "source_reference",
            "passed": False,
            "detail": "Unsupported",
            "category": "generation",
        },
    ]
    verdict = compute_question_verdict(scores, fsi_profile, deterministic_checks=checks)
    assert verdict.verdict == "PASS"
    assert len(verdict.fail_reasons) == 0


def test_verdict_no_duplicate_retrieval_reasons(fsi_profile):
    """Should not duplicate FAIL_RETRIEVAL_INCOMPLETE when both retrieval checks fail."""
    scores = {
        "groundedness_score": 0.9,
        "abstention_score": 0.9,
    }
    checks = [
        {
            "check_name": "document_presence",
            "passed": False,
            "detail": "Missing",
            "category": "retrieval",
        },
        {
            "check_name": "chunk_alignment",
            "passed": False,
            "detail": "0/2",
            "category": "retrieval",
        },
    ]
    verdict = compute_question_verdict(scores, fsi_profile, deterministic_checks=checks)
    assert verdict.verdict == "FAIL"
    assert verdict.fail_reasons.count("FAIL_RETRIEVAL_INCOMPLETE") == 1


def test_verdict_none_deterministic_checks(fsi_profile):
    """Should behave normally when deterministic_checks is None."""
    scores = {
        "groundedness_score": 0.9,
        "abstention_score": 0.9,
    }
    verdict = compute_question_verdict(scores, fsi_profile, deterministic_checks=None)
    assert verdict.verdict == "PASS"


def test_verdict_ignores_chunk_alignment_for_manual_truth(fsi_profile):
    """Should not force FAIL on chunk_alignment when evidence_mode is manual."""
    scores = {
        "groundedness_score": 0.9,
        "relevancy_score": 0.8,
        "abstention_score": 0.9,
    }
    checks = [
        {
            "check_name": "document_presence",
            "passed": True,
            "detail": "All required documents present",
            "category": "retrieval",
        },
        {
            "check_name": "chunk_alignment",
            "passed": False,
            "detail": "0/3",
            "category": "retrieval",
        },
    ]

    verdict = compute_question_verdict(
        scores,
        fsi_profile,
        deterministic_checks=checks,
        evidence_mode="grounded_from_manual_answer",
    )
    assert verdict.verdict == "PASS"
    assert "FAIL_RETRIEVAL_INCOMPLETE" not in verdict.fail_reasons


def test_verdict_ignores_chunk_alignment_for_grounded_synthesis(fsi_profile):
    """Should not force FAIL when synthesized truth was grounded via retrieval."""
    scores = {
        "groundedness_score": 0.9,
        "relevancy_score": 0.8,
        "abstention_score": 0.9,
    }
    checks = [
        {
            "check_name": "document_presence",
            "passed": True,
            "detail": "All required documents present",
            "category": "retrieval",
        },
        {
            "check_name": "chunk_alignment",
            "passed": False,
            "detail": "1/3",
            "category": "retrieval",
        },
    ]

    verdict = compute_question_verdict(
        scores,
        fsi_profile,
        deterministic_checks=checks,
        evidence_mode="grounded_from_synthesis",
    )
    assert verdict.verdict == "PASS"
    assert "FAIL_RETRIEVAL_INCOMPLETE" not in verdict.fail_reasons


def test_verdict_keeps_chunk_alignment_fail_for_synthesis_truth(fsi_profile):
    """Should still FAIL on chunk_alignment when evidence_mode is synthesis."""
    scores = {
        "groundedness_score": 0.9,
        "relevancy_score": 0.8,
        "abstention_score": 0.9,
    }
    checks = [
        {
            "check_name": "document_presence",
            "passed": True,
            "detail": "All required documents present",
            "category": "retrieval",
        },
        {
            "check_name": "chunk_alignment",
            "passed": False,
            "detail": "1/5",
            "category": "retrieval",
        },
    ]

    verdict = compute_question_verdict(
        scores,
        fsi_profile,
        deterministic_checks=checks,
        evidence_mode="traced_from_synthesis",
    )
    assert verdict.verdict == "FAIL"
    assert "FAIL_RETRIEVAL_INCOMPLETE" in verdict.fail_reasons
