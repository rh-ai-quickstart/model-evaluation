
import { z } from 'zod';

export const EvalRunCreateResponseSchema = z.object({
    eval_run_id: z.number(),
    model_name: z.string(),
    status: z.string(),
    total_questions: z.number(),
    message: z.string(),
});

export const CoverageGapsSchema = z.object({
    concepts: z.array(z.string()),
    covered: z.array(z.string()),
    missing: z.array(z.string()),
    coverage_ratio: z.number(),
    retrieval_failures: z.array(z.string()).optional(),
    generation_failures: z.array(z.string()).optional(),
});

export const DeterministicCheckSchema = z.object({
    check_name: z.string(),
    passed: z.boolean(),
    detail: z.string().optional(),
});

export const AnswerTruthSchema = z.object({
    required_concepts: z.array(z.string()),
    abstention_expected: z.boolean().optional(),
});

export const RetrievalTruthSchema = z.object({
    required_documents: z.array(z.string()),
    expected_chunk_refs: z.array(z.string()),
    supporting_documents: z.array(z.string()).optional().default([]),
    supporting_chunk_refs: z.array(z.string()).optional().default([]),
    evidence_mode: z.string(),
});

export const TruthMetadataSchema = z.object({
    truth_schema_version: z.string().optional(),
    concept_extraction_version: z.string().optional(),
    evidence_alignment_version: z.string().optional(),
    generated_by_model: z.string(),
    generated_at: z.string(),
    source_chunk_ids: z.array(z.number()),
});

export const TruthPayloadSchema = z.object({
    answer_truth: AnswerTruthSchema,
    retrieval_truth: RetrievalTruthSchema,
    metadata: TruthMetadataSchema,
});

export const EvalResultSchema = z.object({
    id: z.number(),
    question: z.string(),
    expected_answer: z.string().nullable().optional(),
    answer: z.string().nullable().optional(),
    contexts: z.string().nullable().optional(),
    latency_ms: z.number().nullable().optional(),
    relevancy_score: z.number().nullable().optional(),
    groundedness_score: z.number().nullable().optional(),
    context_precision_score: z.number().nullable().optional(),
    context_relevancy_score: z.number().nullable().optional(),
    completeness_score: z.number().nullable().optional(),
    correctness_score: z.number().nullable().optional(),
    compliance_accuracy_score: z.number().nullable().optional(),
    abstention_score: z.number().nullable().optional(),
    is_hallucination: z.boolean().nullable().optional(),
    chunk_alignment_score: z.number().nullable().optional(),
    coverage_gaps: CoverageGapsSchema.nullable().optional(),
    deterministic_checks: z.array(DeterministicCheckSchema).nullable().optional(),
    truth: TruthPayloadSchema.nullable().optional(),
    verdict: z.string().nullable().optional(),
    fail_reasons: z.array(z.string()).nullable().optional(),
    total_tokens: z.number().nullable().optional(),
    error_message: z.string().nullable().optional(),
});

export const EvalRunSchema = z.object({
    id: z.number(),
    model_name: z.string(),
    question_set_name: z.string().nullable().optional(),
    status: z.string(),
    total_questions: z.number(),
    completed_questions: z.number(),
    avg_latency_ms: z.number().nullable().optional(),
    avg_relevancy: z.number().nullable().optional(),
    avg_groundedness: z.number().nullable().optional(),
    avg_context_precision: z.number().nullable().optional(),
    avg_context_relevancy: z.number().nullable().optional(),
    avg_completeness: z.number().nullable().optional(),
    avg_correctness: z.number().nullable().optional(),
    avg_compliance_accuracy: z.number().nullable().optional(),
    avg_abstention: z.number().nullable().optional(),
    hallucination_rate: z.number().nullable().optional(),
    avg_chunk_alignment: z.number().nullable().optional(),
    profile_id: z.string().nullable().optional(),
    profile_version: z.string().nullable().optional(),
    judge_model_name: z.string().nullable().optional(),
    synthesis_model_name: z.string().nullable().optional(),
    retrieval_config: z.record(z.unknown()).nullable().optional(),
    corpus_snapshot: z.record(z.unknown()).nullable().optional(),
    overall_verdict: z.string().nullable().optional(),
    pass_count: z.number().nullable().optional(),
    fail_count: z.number().nullable().optional(),
    review_count: z.number().nullable().optional(),
    total_tokens: z.number().nullable().optional(),
    error_message: z.string().nullable().optional(),
    created_at: z.string().nullable().optional(),
    completed_at: z.string().nullable().optional(),
});

export const EvalRunDetailSchema = EvalRunSchema.extend({
    results: z.array(EvalResultSchema),
});

export const ComparisonMetricSchema = z.object({
    metric: z.string(),
    run_a: z.number().nullable().optional(),
    run_b: z.number().nullable().optional(),
    winner: z.string().nullable().optional(),
});

export const QuestionComparisonSchema = z.object({
    question: z.string(),
    expected_answer: z.string().nullable().optional(),
    run_a: EvalResultSchema.nullable().optional(),
    run_b: EvalResultSchema.nullable().optional(),
});

export const ComparisonDecisionSchema = z.object({
    winner: z.string().nullable().optional(),
    winner_name: z.string().nullable().optional(),
    decision_status: z.string(),
    reason_codes: z.array(z.string()),
    summary: z.string(),
    risk_flags: z.array(z.string()),
    disqualified: z.record(z.array(z.string())),
});

export const ComparisonWarningSchema = z.object({
    code: z.string(),
    message: z.string(),
});

export const ComparisonResponseSchema = z.object({
    run_a: EvalRunSchema,
    run_b: EvalRunSchema,
    metrics: z.array(ComparisonMetricSchema),
    questions: z.array(QuestionComparisonSchema),
    decision: ComparisonDecisionSchema.nullable().optional(),
    warnings: z.array(ComparisonWarningSchema).optional(),
});

export const SynthesizedQuestionSchema = z.object({
    question: z.string(),
    expected_answer: z.string().nullable().optional(),
    truth: TruthPayloadSchema.nullable().optional(),
});

export const SynthesizeResponseSchema = z.object({
    questions: z.array(SynthesizedQuestionSchema),
    count: z.number(),
});

export type EvalRunCreateResponse = z.infer<typeof EvalRunCreateResponseSchema>;
export type CoverageGaps = z.infer<typeof CoverageGapsSchema>;
export type DeterministicCheck = z.infer<typeof DeterministicCheckSchema>;
export type AnswerTruth = z.infer<typeof AnswerTruthSchema>;
export type RetrievalTruth = z.infer<typeof RetrievalTruthSchema>;
export type TruthMetadata = z.infer<typeof TruthMetadataSchema>;
export type TruthPayload = z.infer<typeof TruthPayloadSchema>;
export type EvalResult = z.infer<typeof EvalResultSchema>;
export type EvalRun = z.infer<typeof EvalRunSchema>;
export type EvalRunDetail = z.infer<typeof EvalRunDetailSchema>;
export type ComparisonMetric = z.infer<typeof ComparisonMetricSchema>;
export type QuestionComparison = z.infer<typeof QuestionComparisonSchema>;
export type ComparisonDecision = z.infer<typeof ComparisonDecisionSchema>;
export type ComparisonWarning = z.infer<typeof ComparisonWarningSchema>;
export type ComparisonResponse = z.infer<typeof ComparisonResponseSchema>;
export type SynthesizedQuestion = z.infer<typeof SynthesizedQuestionSchema>;
export type SynthesizeResponse = z.infer<typeof SynthesizeResponseSchema>;
