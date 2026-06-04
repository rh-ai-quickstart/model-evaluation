"""Pydantic schemas for the evaluation endpoints."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from .truth import TruthPayload


class EvalQuestion(BaseModel):
    """A single evaluation question with optional expected answer."""

    question: str = Field(..., min_length=1)
    expected_answer: str | None = None
    expected_chunks: list[str] | None = Field(
        default=None, description='Expected source chunks, e.g. ["report.pdf:3", "guide.pdf"].'
    )
    truth: TruthPayload | None = None


class EvalRunCreate(BaseModel):
    """Request to create a new evaluation run."""

    model_name: str = Field(..., min_length=1)
    questions: list[str | EvalQuestion] = Field(..., min_length=1, max_length=100)
    question_set_id: int | None = None
    profile_id: str | None = None


class EvalResultResponse(BaseModel):
    """Response for a single evaluation result."""

    id: int
    question: str
    expected_answer: str | None = None
    answer: str | None = None
    contexts: str | None = None
    latency_ms: float | None = None
    relevancy_score: float | None = None
    groundedness_score: float | None = None
    context_precision_score: float | None = None
    context_relevancy_score: float | None = None
    completeness_score: float | None = None
    correctness_score: float | None = None
    compliance_accuracy_score: float | None = None
    abstention_score: float | None = None
    is_hallucination: bool | None = None
    chunk_alignment_score: float | None = None
    coverage_gaps: dict | None = None
    deterministic_checks: list[dict] | None = None
    truth: TruthPayload | None = None
    verdict: str | None = None
    fail_reasons: list[str] | None = None
    total_tokens: int | None = None
    error_message: str | None = None


class EvalRunResponse(BaseModel):
    """Response for an evaluation run."""

    id: int
    model_name: str
    question_set_name: str | None = None
    status: Literal["pending", "running", "completed", "failed", "cancelled"]
    total_questions: int
    completed_questions: int
    avg_latency_ms: float | None = None
    avg_relevancy: float | None = None
    avg_groundedness: float | None = None
    avg_context_precision: float | None = None
    avg_context_relevancy: float | None = None
    avg_completeness: float | None = None
    avg_correctness: float | None = None
    avg_compliance_accuracy: float | None = None
    avg_abstention: float | None = None
    hallucination_rate: float | None = None
    avg_chunk_alignment: float | None = None
    profile_id: str | None = None
    profile_version: str | None = None
    judge_model_name: str | None = None
    synthesis_model_name: str | None = None
    retrieval_config: dict | None = None
    corpus_snapshot: dict | None = None
    overall_verdict: str | None = None
    pass_count: int | None = None
    fail_count: int | None = None
    review_count: int | None = None
    total_tokens: int | None = None
    error_message: str | None = None
    created_at: datetime | None = None
    completed_at: datetime | None = None

    model_config = {"from_attributes": True}


class EvalRunDetailResponse(EvalRunResponse):
    """Evaluation run with individual results."""

    results: list[EvalResultResponse] = []


class EvalRunCreateResponse(BaseModel):
    """Response after creating an evaluation run."""

    eval_run_id: int
    model_name: str
    status: str
    total_questions: int
    message: str


class EvalRunRerun(BaseModel):
    """Request to re-run an evaluation with a different model."""

    model_name: str = Field(..., min_length=1)


class ComparisonMetric(BaseModel):
    """A single metric compared across two runs."""

    metric: str
    run_a: float | None = None
    run_b: float | None = None
    winner: str | None = None


class QuestionComparison(BaseModel):
    """Side-by-side comparison of a single question across two runs."""

    question: str
    expected_answer: str | None = None
    run_a: EvalResultResponse | None = None
    run_b: EvalResultResponse | None = None


class ComparisonDecision(BaseModel):
    """Backend-computed comparison verdict with disqualification gates."""

    winner: str | None = None  # "run_a" | "run_b" | "tie"
    winner_name: str | None = None
    decision_status: str = "inconclusive"  # "decisive" | "marginal" | "inconclusive"
    reason_codes: list[str] = Field(default_factory=list)
    summary: str = ""
    risk_flags: list[str] = Field(default_factory=list)
    disqualified: dict[str, list[str]] = Field(default_factory=dict)


class ComparisonWarning(BaseModel):
    """Warning about comparison precondition mismatch."""

    code: str
    message: str


class ComparisonResponse(BaseModel):
    """Side-by-side comparison of two evaluation runs."""

    run_a: EvalRunResponse
    run_b: EvalRunResponse
    metrics: list[ComparisonMetric] = []
    questions: list[QuestionComparison] = []
    decision: ComparisonDecision | None = None
    warnings: list[ComparisonWarning] = Field(default_factory=list)


class SynthesizeRequest(BaseModel):
    """Request to auto-generate evaluation questions from documents."""

    document_ids: list[int] | None = Field(
        default=None, description="Document IDs to generate from. None = all documents."
    )
    max_questions: int = Field(default=3, ge=1, le=50)
    profile_id: str | None = Field(
        default=None, description="Profile to use for domain-specific question generation."
    )


class SynthesizedQuestion(BaseModel):
    """A single auto-generated question with expected answer."""

    question: str
    expected_answer: str | None = None
    truth: TruthPayload | None = None


class SynthesizeResponse(BaseModel):
    """Response with auto-generated evaluation questions."""

    questions: list[SynthesizedQuestion] = []
    count: int = 0
