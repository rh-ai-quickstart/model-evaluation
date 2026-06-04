"""Tests for comparison decision logic.

Covers disqualification gates, verdict hierarchy, failure counts,
and precondition validation.
"""

import pytest

from src.services.profiles import EvalProfile
from src.services.verdicts import compute_comparison_decision


@pytest.fixture
def fsi_profile() -> EvalProfile:
    """Profile with thresholds matching the FSI compliance profile."""
    return EvalProfile(
        id="test_fsi",
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


def _make_run(
    model_name: str = "model-a",
    overall_verdict: str | None = "PASS",
    fail_count: int = 0,
    review_count: int = 0,
    avg_completeness: float | None = 0.8,
    avg_correctness: float | None = 0.8,
    avg_compliance_accuracy: float | None = 0.8,
) -> dict:
    return {
        "model_name": model_name,
        "overall_verdict": overall_verdict,
        "fail_count": fail_count,
        "review_count": review_count,
        "avg_completeness": avg_completeness,
        "avg_correctness": avg_correctness,
        "avg_compliance_accuracy": avg_compliance_accuracy,
    }


# --- Disqualification gate tests ---


def test_completeness_gate_disqualifies_winner(fsi_profile):
    """Model A wins more metrics but fails completeness gate -- should lose."""
    run_a = _make_run(
        model_name="model-a",
        overall_verdict="PASS",
        avg_completeness=0.3,  # below 0.5 threshold
    )
    run_b = _make_run(
        model_name="model-b",
        overall_verdict="PASS",
        avg_completeness=0.8,
    )
    # A wins 5 metrics, B wins 3 -- but A is disqualified
    metric_winners = [
        ("groundedness", "run_a"),
        ("relevancy", "run_a"),
        ("context_precision", "run_a"),
        ("context_relevancy", "run_a"),
        ("abstention", "run_a"),
        ("completeness", "run_b"),
        ("correctness", "run_b"),
        ("compliance_accuracy", "run_b"),
    ]

    result = compute_comparison_decision(run_a, run_b, metric_winners, fsi_profile)

    assert result.winner == "run_b"
    assert "OPPONENT_DISQUALIFIED" in result.reason_codes
    assert result.decision_status == "decisive"
    assert len(result.disqualified["run_a"]) > 0
    assert any("completeness" in r for r in result.disqualified["run_a"])


def test_correctness_gate_disqualifies(fsi_profile):
    """Model below correctness threshold cannot be overall winner."""
    run_a = _make_run(model_name="model-a", avg_correctness=0.3)
    run_b = _make_run(model_name="model-b", avg_correctness=0.8)
    metric_winners = [("groundedness", "run_a"), ("relevancy", "run_a")]

    result = compute_comparison_decision(run_a, run_b, metric_winners, fsi_profile)

    assert result.winner == "run_b"
    assert "OPPONENT_DISQUALIFIED" in result.reason_codes
    assert any("correctness" in r for r in result.disqualified["run_a"])


def test_compliance_accuracy_gate_disqualifies(fsi_profile):
    """Model below compliance_accuracy threshold cannot be overall winner."""
    run_a = _make_run(model_name="model-a", avg_compliance_accuracy=0.3)
    run_b = _make_run(model_name="model-b", avg_compliance_accuracy=0.8)
    metric_winners = [("groundedness", "run_a")]

    result = compute_comparison_decision(run_a, run_b, metric_winners, fsi_profile)

    assert result.winner == "run_b"
    assert "OPPONENT_DISQUALIFIED" in result.reason_codes


def test_both_disqualified_produces_no_winner(fsi_profile):
    """When both models are disqualified, no winner is declared."""
    run_a = _make_run(
        model_name="model-a",
        overall_verdict="REVIEW_REQUIRED",
        avg_completeness=0.3,
    )
    run_b = _make_run(
        model_name="model-b",
        overall_verdict="FAIL",
        avg_completeness=0.2,
    )
    metric_winners = [("groundedness", "tie")]

    result = compute_comparison_decision(run_a, run_b, metric_winners, fsi_profile)

    assert result.winner is None
    assert result.decision_status == "inconclusive"
    assert "BOTH_DISQUALIFIED" in result.reason_codes
    assert "Both models fail critical quality gates" in result.risk_flags


# --- Verdict hierarchy tests ---


def test_verdict_pass_beats_fail(fsi_profile):
    """PASS verdict beats FAIL regardless of metric wins."""
    run_a = _make_run(model_name="model-a", overall_verdict="PASS")
    run_b = _make_run(model_name="model-b", overall_verdict="FAIL", fail_count=2)
    # B wins more metrics, but A has better verdict
    metric_winners = [
        ("groundedness", "run_b"),
        ("relevancy", "run_b"),
        ("completeness", "run_b"),
    ]

    result = compute_comparison_decision(run_a, run_b, metric_winners, fsi_profile)

    assert result.winner == "run_a"
    assert "BETTER_VERDICT" in result.reason_codes
    assert result.decision_status == "decisive"


def test_verdict_pass_beats_review_required(fsi_profile):
    """PASS verdict beats REVIEW_REQUIRED."""
    run_a = _make_run(model_name="model-a", overall_verdict="PASS")
    run_b = _make_run(model_name="model-b", overall_verdict="REVIEW_REQUIRED", review_count=3)
    metric_winners = []

    result = compute_comparison_decision(run_a, run_b, metric_winners, fsi_profile)

    assert result.winner == "run_a"
    assert "BETTER_VERDICT" in result.reason_codes


def test_verdict_review_required_beats_fail(fsi_profile):
    """REVIEW_REQUIRED verdict beats FAIL."""
    run_a = _make_run(
        model_name="model-a",
        overall_verdict="REVIEW_REQUIRED",
        review_count=1,
    )
    run_b = _make_run(model_name="model-b", overall_verdict="FAIL", fail_count=1)
    metric_winners = []

    result = compute_comparison_decision(run_a, run_b, metric_winners, fsi_profile)

    assert result.winner == "run_a"
    assert "BETTER_VERDICT" in result.reason_codes


# --- Fail/review count tests ---


def test_fewer_failures_wins_when_same_verdict(fsi_profile):
    """With same verdict, fewer failures wins."""
    run_a = _make_run(
        model_name="model-a",
        overall_verdict="FAIL",
        fail_count=1,
    )
    run_b = _make_run(
        model_name="model-b",
        overall_verdict="FAIL",
        fail_count=3,
    )
    metric_winners = [("groundedness", "run_b")]

    result = compute_comparison_decision(run_a, run_b, metric_winners, fsi_profile)

    assert result.winner == "run_a"
    assert "FEWER_FAILURES" in result.reason_codes


def test_fewer_reviews_wins_when_counts_equal(fsi_profile):
    """With same verdict and fail counts, fewer reviews wins."""
    run_a = _make_run(
        model_name="model-a",
        overall_verdict="REVIEW_REQUIRED",
        fail_count=0,
        review_count=1,
    )
    run_b = _make_run(
        model_name="model-b",
        overall_verdict="REVIEW_REQUIRED",
        fail_count=0,
        review_count=3,
    )
    metric_winners = []

    result = compute_comparison_decision(run_a, run_b, metric_winners, fsi_profile)

    assert result.winner == "run_a"
    assert "FEWER_REVIEWS" in result.reason_codes


# --- Metric tie-breaker tests ---


def test_metric_advantage_as_final_tiebreaker(fsi_profile):
    """When verdicts and counts are equal, metric wins decide."""
    run_a = _make_run(model_name="model-a", overall_verdict="PASS")
    run_b = _make_run(model_name="model-b", overall_verdict="PASS")
    metric_winners = [
        ("groundedness", "run_a"),
        ("completeness", "run_a"),
        ("relevancy", "run_b"),
    ]

    result = compute_comparison_decision(run_a, run_b, metric_winners, fsi_profile)

    assert result.winner == "run_a"
    assert "METRIC_ADVANTAGE" in result.reason_codes
    assert result.decision_status == "marginal"


def test_tie_when_all_equal(fsi_profile):
    """Should return tie when verdicts, counts, and metrics are all equal."""
    run_a = _make_run(model_name="model-a", overall_verdict="PASS")
    run_b = _make_run(model_name="model-b", overall_verdict="PASS")
    metric_winners = [
        ("groundedness", "run_a"),
        ("completeness", "run_b"),
    ]

    result = compute_comparison_decision(run_a, run_b, metric_winners, fsi_profile)

    assert result.winner == "tie"
    assert result.decision_status == "inconclusive"


# --- No-profile tests ---


def test_no_profile_skips_gates_and_verdicts():
    """Without a profile, gates are skipped and decision falls to metrics."""
    run_a = _make_run(
        model_name="model-a",
        overall_verdict=None,
        avg_completeness=0.1,  # would be disqualified with a profile
    )
    run_b = _make_run(model_name="model-b", overall_verdict=None)
    metric_winners = [
        ("groundedness", "run_a"),
        ("relevancy", "run_a"),
        ("completeness", "run_b"),
    ]

    result = compute_comparison_decision(run_a, run_b, metric_winners, profile=None)

    # Without profile, no disqualification -- raw metric count decides
    assert result.winner == "run_a"
    assert "METRIC_ADVANTAGE" in result.reason_codes
    assert any("No evaluation profile" in f for f in result.risk_flags)


# --- Summary and risk flag tests ---


def test_disqualification_produces_risk_flags(fsi_profile):
    """Disqualification should produce descriptive risk flags."""
    run_a = _make_run(
        model_name="qwen3-14b",
        avg_completeness=0.3,
        avg_correctness=0.2,
    )
    run_b = _make_run(model_name="granite-3-2-8b")
    metric_winners = []

    result = compute_comparison_decision(run_a, run_b, metric_winners, fsi_profile)

    assert result.winner == "run_b"
    assert any("completeness" in f and "qwen3-14b" in f for f in result.risk_flags)
    assert any("correctness" in f and "qwen3-14b" in f for f in result.risk_flags)


def test_summary_contains_winner_name(fsi_profile):
    """Summary should name the winning model."""
    run_a = _make_run(model_name="alpha-model", overall_verdict="PASS")
    run_b = _make_run(model_name="beta-model", overall_verdict="FAIL", fail_count=1)
    metric_winners = []

    result = compute_comparison_decision(run_a, run_b, metric_winners, fsi_profile)

    assert "alpha-model" in result.summary


# --- The original bug scenario ---


def test_original_bug_granite_should_not_win(fsi_profile):
    """Reproduce the original bug: granite had 30% completeness and should not win.

    Model A (qwen) is run_a, Model B (granite) is run_b.
    Granite wins more metric rows but fails the completeness gate.
    """
    run_a = _make_run(
        model_name="qwen3-14b",
        overall_verdict="REVIEW_REQUIRED",
        review_count=1,
        avg_completeness=0.25,
        avg_correctness=0.6,
        avg_compliance_accuracy=0.6,
    )
    run_b = _make_run(
        model_name="granite-3-2-8b-instruct",
        overall_verdict="FAIL",
        fail_count=1,
        avg_completeness=0.3,  # 30% -- below 0.5 threshold
        avg_correctness=0.6,
        avg_compliance_accuracy=0.6,
    )
    # Granite wins more individual metric comparisons
    metric_winners = [
        ("groundedness", "run_b"),
        ("relevancy", "run_b"),
        ("context_relevancy", "run_b"),
        ("completeness", "run_b"),
        ("correctness", "tie"),
        ("compliance_accuracy", "tie"),
        ("abstention", "run_a"),
        ("hallucination_rate", "run_a"),
    ]

    result = compute_comparison_decision(run_a, run_b, metric_winners, fsi_profile)

    # Both are disqualified (both below completeness threshold) -- no winner
    assert len(result.disqualified["run_a"]) > 0
    assert len(result.disqualified["run_b"]) > 0
    assert result.winner is None
    assert result.decision_status == "inconclusive"
    assert "BOTH_DISQUALIFIED" in result.reason_codes
