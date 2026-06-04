"""Business verdict layer -- applies profile thresholds to raw metric scores.

All pass/fail logic lives here. The scoring layer (scoring.py) produces raw
scores only (threshold=0.0). This module applies profile-defined thresholds
to produce PASS/FAIL/REVIEW_REQUIRED verdicts with specific fail reasons.

Also computes comparison decisions: which model wins a head-to-head comparison,
using verdict hierarchy, failure counts, and disqualification gates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .profiles import EvalProfile


@dataclass
class QuestionVerdict:
    """Verdict for a single evaluation question."""

    verdict: str  # PASS | FAIL | REVIEW_REQUIRED
    fail_reasons: list[str] = field(default_factory=list)
    passed_metrics: list[str] = field(default_factory=list)
    failed_metrics: list[str] = field(default_factory=list)


@dataclass
class RunVerdict:
    """Aggregate verdict for an evaluation run."""

    overall: str  # PASS | FAIL | REVIEW_REQUIRED
    pass_count: int = 0
    fail_count: int = 0
    review_count: int = 0
    total: int = 0
    summary: str = ""


_FAIL_REASON_MAP = {
    "groundedness_score": "FAIL_LOW_GROUNDEDNESS",
    "relevancy_score": "FAIL_LOW_RELEVANCY",
    "context_precision_score": "FAIL_LOW_CONTEXT_PRECISION",
    "context_relevancy_score": "FAIL_LOW_CONTEXT_RELEVANCY",
    "completeness_score": "FAIL_INSUFFICIENT_COVERAGE",
    "correctness_score": "FAIL_UNSUPPORTED_CLAIM",
    "compliance_accuracy_score": "FAIL_COMPLIANCE_VIOLATION",
    "abstention_score": "FAIL_CONFIDENT_WITHOUT_CONTEXT",
}


# Deterministic check names that can force a FAIL verdict.
# Retrieval checks are objective (tied to structured truth refs) and are
# strong gate candidates. Generation checks (abstention, source_reference)
# are heuristic and stored as warnings only.
_DETERMINISTIC_FAIL_CHECKS = frozenset({"document_presence", "chunk_alignment"})

_DETERMINISTIC_FAIL_REASON_MAP = {
    "document_presence": "FAIL_RETRIEVAL_INCOMPLETE",
    "chunk_alignment": "FAIL_RETRIEVAL_INCOMPLETE",
}

_GROUNDED_EVIDENCE_MODES = frozenset(
    {"grounded_from_manual_answer", "grounded_from_synthesis"}
)


def _is_generation_only_gap(coverage_gaps: dict | None) -> bool:
    """Return True when missing concepts are classified as generation-only.

    This requires coverage data to include:
    - at least one missing concept
    - zero retrieval_failures
    - one or more generation_failures
    """
    if not coverage_gaps:
        return False

    missing = coverage_gaps.get("missing")
    retrieval_failures = coverage_gaps.get("retrieval_failures")
    generation_failures = coverage_gaps.get("generation_failures")

    return (
        isinstance(missing, list)
        and len(missing) > 0
        and isinstance(retrieval_failures, list)
        and len(retrieval_failures) == 0
        and isinstance(generation_failures, list)
        and len(generation_failures) > 0
    )


def compute_question_verdict(
    scores: dict[str, float | None],
    profile: EvalProfile,
    deterministic_checks: list[dict] | None = None,
    coverage_gaps: dict | None = None,
    evidence_mode: str | None = None,
) -> QuestionVerdict:
    """Apply profile thresholds to metric scores and produce a verdict.

    Logic:
    - If any deterministic retrieval check fails -> FAIL
      - Exception: chunk_alignment does not force FAIL when:
        (a) document_presence passes, and
        (b) coverage gaps show generation-only misses.
      - Exception: chunk_alignment is non-gating for grounded truth
        because exact chunk recall is brittle when truth was produced through
        retrieval rather than direct source tracing.
    - If any metric is below its critical_threshold -> FAIL
    - If any metric is below its regular threshold -> REVIEW_REQUIRED
    - If all pass -> PASS
    - Metrics with None scores are skipped (not penalized).
    - Generation checks (abstention, source_reference) are recorded but
      do not affect the verdict.

    Args:
        scores: Dict of metric_name -> score (from score_result()).
        profile: The evaluation profile with thresholds.
        deterministic_checks: Optional list of deterministic check result dicts,
            each with check_name, passed, detail, category.
        coverage_gaps: Optional coverage gap result dict (from detect_coverage_gaps).
        evidence_mode: How retrieval truth was produced. chunk_alignment is
            non-gating for grounded evidence modes.

    Returns:
        QuestionVerdict with verdict, fail_reasons, and metric lists.
    """
    has_critical_fail = False
    has_threshold_fail = False
    fail_reasons: list[str] = []
    passed_metrics: list[str] = []
    failed_metrics: list[str] = []

    # Check deterministic retrieval gates first
    if deterministic_checks:
        document_presence_passed = False
        for check in deterministic_checks:
            if check.get("check_name", "") == "document_presence":
                document_presence_passed = bool(check.get("passed", True))
                break

        generation_only_coverage = (
            document_presence_passed and _is_generation_only_gap(coverage_gaps)
        )

        grounded_truth = evidence_mode in _GROUNDED_EVIDENCE_MODES

        for check in deterministic_checks:
            name = check.get("check_name", "")
            if name == "chunk_alignment" and not check.get("passed", True):
                if generation_only_coverage or grounded_truth:
                    continue
            if name in _DETERMINISTIC_FAIL_CHECKS and not check.get("passed", True):
                has_critical_fail = True
                reason = _DETERMINISTIC_FAIL_REASON_MAP.get(name, f"FAIL_{name.upper()}")
                if reason not in fail_reasons:
                    fail_reasons.append(reason)
                failed_metrics.append(name)

    # Check critical thresholds
    for metric_key, critical_threshold in profile.critical_thresholds.items():
        score = scores.get(metric_key)
        if score is None:
            continue
        if score < critical_threshold:
            has_critical_fail = True
            reason = _FAIL_REASON_MAP.get(metric_key, f"FAIL_{metric_key.upper()}")
            fail_reasons.append(reason)
            failed_metrics.append(metric_key)

    # Check regular thresholds
    for metric_key, threshold in profile.thresholds.items():
        score = scores.get(metric_key)
        if score is None:
            continue
        if metric_key in failed_metrics:
            continue  # Already counted as critical fail
        if score < threshold:
            has_threshold_fail = True
            reason = _FAIL_REASON_MAP.get(metric_key, f"FAIL_{metric_key.upper()}")
            fail_reasons.append(reason)
            failed_metrics.append(metric_key)
        else:
            passed_metrics.append(metric_key)

    if has_critical_fail:
        verdict = "FAIL"
    elif has_threshold_fail:
        verdict = "REVIEW_REQUIRED"
    else:
        verdict = "PASS"

    return QuestionVerdict(
        verdict=verdict,
        fail_reasons=fail_reasons,
        passed_metrics=passed_metrics,
        failed_metrics=failed_metrics,
    )


def compute_run_verdict(question_verdicts: list[QuestionVerdict]) -> RunVerdict:
    """Aggregate question verdicts into a run-level verdict.

    Args:
        question_verdicts: List of per-question verdicts.

    Returns:
        RunVerdict with counts and overall verdict.
    """
    total = len(question_verdicts)
    pass_count = sum(1 for v in question_verdicts if v.verdict == "PASS")
    fail_count = sum(1 for v in question_verdicts if v.verdict == "FAIL")
    review_count = sum(1 for v in question_verdicts if v.verdict == "REVIEW_REQUIRED")

    if fail_count > 0:
        overall = "FAIL"
    elif review_count > 0:
        overall = "REVIEW_REQUIRED"
    else:
        overall = "PASS"

    summary = f"{pass_count}/{total} questions passed all criteria"

    return RunVerdict(
        overall=overall,
        pass_count=pass_count,
        fail_count=fail_count,
        review_count=review_count,
        total=total,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Comparison decision
# ---------------------------------------------------------------------------

_VERDICT_RANK: dict[str, int] = {"PASS": 0, "REVIEW_REQUIRED": 1, "FAIL": 2}

# Metrics that act as hard gates: if a run's average is below the profile
# threshold for any of these, it cannot be declared the overall winner.
_GATE_METRIC_FIELDS: dict[str, str] = {
    "completeness_score": "avg_completeness",
    "correctness_score": "avg_correctness",
    "compliance_accuracy_score": "avg_compliance_accuracy",
}

# Quality metrics checked in priority order for the final tie-breaker.
_BUSINESS_PRIORITY_METRICS = [
    "groundedness",
    "completeness",
    "correctness",
    "compliance_accuracy",
    "relevancy",
    "context_precision",
    "context_relevancy",
    "abstention",
]


@dataclass
class ComparisonDecisionResult:
    """Result of comparing two evaluation runs."""

    winner: str | None = None  # "run_a" | "run_b" | "tie"
    winner_name: str | None = None
    decision_status: str = "inconclusive"
    reason_codes: list[str] = field(default_factory=list)
    summary: str = ""
    risk_flags: list[str] = field(default_factory=list)
    disqualified: dict[str, list[str]] = field(default_factory=dict)


def compute_comparison_decision(
    run_a_data: dict,
    run_b_data: dict,
    metric_winners: list[tuple[str, str | None]],
    profile: EvalProfile | None = None,
) -> ComparisonDecisionResult:
    """Decide which run wins a head-to-head comparison.

    Decision order:
    1. Hard disqualification gates (completeness, correctness, compliance_accuracy
       below profile threshold).
    2. Compare overall_verdict (PASS > REVIEW_REQUIRED > FAIL).
    3. Compare fail_count (fewer is better).
    4. Compare review_count (fewer is better).
    5. Business-priority metric wins (tie-breaker).

    Args:
        run_a_data: Dict with keys: model_name, overall_verdict, fail_count,
            review_count, avg_completeness, avg_correctness,
            avg_compliance_accuracy.
        run_b_data: Same structure as run_a_data.
        metric_winners: List of (metric_name, winner) tuples where winner is
            "run_a", "run_b", "tie", or None.
        profile: Evaluation profile for threshold gates.  None skips gates.

    Returns:
        ComparisonDecisionResult with winner, reason codes, risk flags.
    """
    risk_flags: list[str] = []
    disqualified: dict[str, list[str]] = {"run_a": [], "run_b": []}
    reason_codes: list[str] = []
    winner: str | None = None
    decision_status = "inconclusive"

    model_a = run_a_data["model_name"]
    model_b = run_b_data["model_name"]

    # --- Step 3: Hard disqualification gates ---
    if profile:
        for metric_key, avg_field in _GATE_METRIC_FIELDS.items():
            threshold = profile.thresholds.get(metric_key)
            if threshold is None:
                continue
            label = metric_key.replace("_score", "")
            for run_key, run_data, name in [
                ("run_a", run_a_data, model_a),
                ("run_b", run_b_data, model_b),
            ]:
                val = run_data.get(avg_field)
                if val is not None and val < threshold:
                    disqualified[run_key].append(f"{label}_below_threshold")
                    risk_flags.append(
                        f"{name}: {label} ({val:.0%}) below threshold ({threshold:.0%})"
                    )

    a_disq = len(disqualified["run_a"]) > 0
    b_disq = len(disqualified["run_b"]) > 0

    if a_disq and not b_disq:
        winner = "run_b"
        reason_codes.append("OPPONENT_DISQUALIFIED")
        decision_status = "decisive"
    elif b_disq and not a_disq:
        winner = "run_a"
        reason_codes.append("OPPONENT_DISQUALIFIED")
        decision_status = "decisive"
    elif a_disq and b_disq:
        risk_flags.append("Both models fail critical quality gates")
        reason_codes.append("BOTH_DISQUALIFIED")
        summary = _build_decision_summary(None, None, reason_codes, risk_flags)
        return ComparisonDecisionResult(
            winner=None,
            winner_name=None,
            decision_status="inconclusive",
            reason_codes=reason_codes,
            summary=summary,
            risk_flags=risk_flags,
            disqualified=disqualified,
        )

    # --- Step 2: Compare overall_verdict ---
    if winner is None:
        v_a = run_a_data.get("overall_verdict")
        v_b = run_b_data.get("overall_verdict")
        if v_a and v_b and v_a in _VERDICT_RANK and v_b in _VERDICT_RANK:
            rank_a = _VERDICT_RANK[v_a]
            rank_b = _VERDICT_RANK[v_b]
            if rank_a < rank_b:
                winner = "run_a"
                reason_codes.append("BETTER_VERDICT")
                decision_status = "decisive"
            elif rank_b < rank_a:
                winner = "run_b"
                reason_codes.append("BETTER_VERDICT")
                decision_status = "decisive"

    # --- Compare fail_count ---
    if winner is None:
        fc_a = run_a_data.get("fail_count") or 0
        fc_b = run_b_data.get("fail_count") or 0
        if fc_a < fc_b:
            winner = "run_a"
            reason_codes.append("FEWER_FAILURES")
            decision_status = "decisive" if abs(fc_a - fc_b) > 1 else "marginal"
        elif fc_b < fc_a:
            winner = "run_b"
            reason_codes.append("FEWER_FAILURES")
            decision_status = "decisive" if abs(fc_a - fc_b) > 1 else "marginal"

    # --- Compare review_count ---
    if winner is None:
        rc_a = run_a_data.get("review_count") or 0
        rc_b = run_b_data.get("review_count") or 0
        if rc_a < rc_b:
            winner = "run_a"
            reason_codes.append("FEWER_REVIEWS")
            decision_status = "marginal"
        elif rc_b < rc_a:
            winner = "run_b"
            reason_codes.append("FEWER_REVIEWS")
            decision_status = "marginal"

    # --- Business-priority metric tie-breaker ---
    if winner is None:
        winners_by_metric = {name: w for name, w in metric_winners}
        a_wins = 0
        b_wins = 0
        for metric_name in _BUSINESS_PRIORITY_METRICS:
            w = winners_by_metric.get(metric_name)
            if w == "run_a":
                a_wins += 1
            elif w == "run_b":
                b_wins += 1

        if a_wins > b_wins:
            winner = "run_a"
            reason_codes.append("METRIC_ADVANTAGE")
            decision_status = "marginal"
        elif b_wins > a_wins:
            winner = "run_b"
            reason_codes.append("METRIC_ADVANTAGE")
            decision_status = "marginal"
        else:
            winner = "tie"
            decision_status = "inconclusive"

    if not profile:
        risk_flags.append("No evaluation profile: verdict and gate checks were skipped")

    winner_name = model_a if winner == "run_a" else model_b if winner == "run_b" else None
    summary = _build_decision_summary(winner, winner_name, reason_codes, risk_flags)

    return ComparisonDecisionResult(
        winner=winner,
        winner_name=winner_name,
        decision_status=decision_status,
        reason_codes=reason_codes,
        summary=summary,
        risk_flags=risk_flags,
        disqualified=disqualified,
    )


def _build_decision_summary(
    winner: str | None,
    winner_name: str | None,
    reason_codes: list[str],
    risk_flags: list[str],
) -> str:
    """Build a human-readable summary of the comparison decision."""
    if winner == "tie":
        return "Both models perform comparably with no decisive advantage."

    if not winner_name:
        if "BOTH_DISQUALIFIED" in reason_codes:
            return "No winner: both models fail minimum quality thresholds."
        return "Comparison inconclusive."

    if "OPPONENT_DISQUALIFIED" in reason_codes:
        return f"{winner_name} wins because the other model fails critical quality gates."

    parts: list[str] = []
    if "BETTER_VERDICT" in reason_codes:
        parts.append("has a better overall verdict")
    if "FEWER_FAILURES" in reason_codes:
        parts.append("has fewer critical failures")
    if "FEWER_REVIEWS" in reason_codes:
        parts.append("requires fewer manual reviews")
    if "METRIC_ADVANTAGE" in reason_codes:
        parts.append("leads on more business-priority metrics")

    explanation = ", ".join(parts) if parts else "leads overall"
    suffix = ""
    flag_count = len(risk_flags)
    if flag_count > 0:
        suffix = f" ({flag_count} risk flag{'s' if flag_count != 1 else ''} noted)"

    return f"{winner_name} wins: {explanation}.{suffix}"
