// This project was developed with assistance from AI tools.

import { useState, useEffect } from 'react';
import { createFileRoute, useNavigate, Link } from '@tanstack/react-router';
import { useCompareEvalRuns, useEvalRuns } from '../../hooks/evaluation';
import {
    ArrowLeft,
    Loader2,
    Trophy,
    Minus,
    GitCompareArrows,
    BarChart3,
    FileText,
    Trash2,
    Clock,
    Info,
    AlertTriangle,
    CheckCircle2,
    XCircle,
    ChevronDown,
} from 'lucide-react';
import { useDocuments } from '../../hooks/documents';
import type { ComparisonMetric, ComparisonResponse, CoverageGaps, EvalResult, EvalRun } from '../../schemas/evaluation';
import { formatScore, formatLatency, formatMetricValue } from '../../lib/format';

interface CompareSearch {
    run_a: number;
    run_b: number;
}

export const Route = createFileRoute('/evaluations/compare')({
    validateSearch: (search: Record<string, unknown>): CompareSearch => ({
        run_a: Number(search.run_a) || 0,
        run_b: Number(search.run_b) || 0,
    }),
    component: ComparePage,
});

const METRIC_LABELS: Record<string, string> = {
    groundedness: 'Faithfulness',
    relevancy: 'Relevancy',
    context_precision: 'Context Precision',
    context_relevancy: 'Context Relevancy',
    completeness: 'Completeness',
    correctness: 'Correctness',
    compliance_accuracy: 'Compliance Accuracy',
    abstention: 'Abstention Quality',
    hallucination_rate: 'Hallucination Rate',
    latency_ms: 'Avg Latency',
};

const METRIC_TOOLTIPS: Record<string, string> = {
    groundedness: 'Is the answer grounded in the retrieved context? Higher means fewer unsupported claims.',
    relevancy: 'Is the answer relevant and useful for the question asked?',
    context_precision: 'Are the retrieved chunks relevant to the expected answer? Measures retrieval quality.',
    context_relevancy: 'Are the retrieved chunks relevant to the question? Measures retrieval targeting.',
    completeness: 'Does the answer cover all key points from the expected answer?',
    correctness: 'Is the answer factually consistent with the expected answer?',
    compliance_accuracy: 'Are domain-specific obligations, thresholds, and cited authorities correct and complete?',
    abstention: 'When context is insufficient, does the model say so instead of guessing?',
    hallucination_rate: 'Percentage of answers containing claims not supported by the retrieved context. Lower is better.',
    latency_ms: 'Average time to generate an answer. Lower is better.',
};

interface ComparisonHistoryEntry {
    run_a_id: number;
    run_b_id: number;
    model_a: string;
    model_b: string;
    timestamp: string;
}

const HISTORY_KEY = 'comparison-history';
const MAX_HISTORY = 20;

function loadHistory(): ComparisonHistoryEntry[] {
    try {
        const raw = localStorage.getItem(HISTORY_KEY);
        return raw ? JSON.parse(raw) : [];
    } catch {
        return [];
    }
}

function saveToHistory(entry: Omit<ComparisonHistoryEntry, 'timestamp'>): void {
    const history = loadHistory();
    const exists = history.some(
        (h) => h.run_a_id === entry.run_a_id && h.run_b_id === entry.run_b_id,
    );
    if (exists) return;
    const updated = [
        { ...entry, timestamp: new Date().toISOString() },
        ...history,
    ].slice(0, MAX_HISTORY);
    localStorage.setItem(HISTORY_KEY, JSON.stringify(updated));
}

function removeFromHistory(runAId: number, runBId: number): ComparisonHistoryEntry[] {
    const history = loadHistory().filter(
        (h) => !(h.run_a_id === runAId && h.run_b_id === runBId),
    );
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
    return history;
}

function WinnerIcon({ winner }: { winner: string | null | undefined }) {
    if (!winner || winner === 'tie') {
        return <Minus className="h-4 w-4 text-muted-foreground" />;
    }
    return <Trophy className="h-4 w-4 text-amber-500" />;
}

function formatEvidenceMode(mode: string): string {
    if (mode === 'traced_from_synthesis') return 'Traced from synthesis';
    if (mode === 'grounded_from_synthesis') return 'Grounded from synthesis';
    if (mode === 'grounded_from_manual_answer') return 'Grounded from manual answer';
    return mode.split('_').join(' ');
}

function isGroundedEvidenceMode(mode: string | undefined): boolean {
    return mode === 'grounded_from_manual_answer' || mode === 'grounded_from_synthesis';
}

function formatCheckName(name: string): string {
    return name
        .split('_')
        .map((part) => (part ? part[0].toUpperCase() + part.slice(1) : part))
        .join(' ');
}

function getRequiredDocsProgress(result: EvalResult): {
    found: number | null;
    total: number;
    supportingTotal: number;
} {
    const total = result.truth?.retrieval_truth.required_documents.length ?? 0;
    const supportingTotal = result.truth?.retrieval_truth.supporting_documents?.length ?? 0;
    if (total === 0) return { found: null, total: 0, supportingTotal };

    const docPresence = result.deterministic_checks?.find(
        (check) => check.check_name === 'document_presence',
    );
    if (!docPresence) return { found: null, total, supportingTotal };
    if (docPresence.passed) return { found: total, total, supportingTotal };

    const missingMatch = docPresence.detail?.match(/Missing\s+(\d+)\/(\d+)/i);
    if (missingMatch) {
        const missing = Number(missingMatch[1]);
        const parsedTotal = Number(missingMatch[2]) || total;
        return { found: Math.max(0, parsedTotal - missing), total: parsedTotal, supportingTotal };
    }

    return { found: null, total, supportingTotal };
}

function MetricRow({
    metric,
    modelA,
    modelB,
}: {
    metric: ComparisonMetric;
    modelA: string;
    modelB: string;
}) {
    const aWins = metric.winner === 'run_a';
    const bWins = metric.winner === 'run_b';

    const tooltip = METRIC_TOOLTIPS[metric.metric];

    return (
        <div className="flex items-center rounded-lg border p-3">
            <div className="flex w-40 items-center gap-1.5 text-sm font-medium">
                {METRIC_LABELS[metric.metric] ?? metric.metric}
                {tooltip && (
                    <span className="group relative">
                        <Info className="h-3.5 w-3.5 cursor-help text-muted-foreground/50" />
                        <span className="pointer-events-none absolute bottom-full left-1/2 z-50 mb-2 w-56 -translate-x-1/2 rounded-lg border bg-popover px-3 py-2 text-xs font-normal text-popover-foreground opacity-0 shadow-md transition-opacity group-hover:opacity-100">
                            {tooltip}
                        </span>
                    </span>
                )}
            </div>
            <div className="flex flex-1 items-center justify-center gap-8">
                <div
                    className={`text-center ${aWins ? 'font-bold text-emerald-600 dark:text-emerald-400' : ''}`}
                >
                    <div className="text-xs text-muted-foreground">{modelA}</div>
                    <div className="text-lg">
                        {formatMetricValue(metric.metric, metric.run_a)}
                    </div>
                    {aWins && <WinnerIcon winner={metric.winner} />}
                </div>
                <div className="text-sm text-muted-foreground">
                    {metric.winner === 'tie' ? 'Tie' : 'vs'}
                </div>
                <div
                    className={`text-center ${bWins ? 'font-bold text-emerald-600 dark:text-emerald-400' : ''}`}
                >
                    <div className="text-xs text-muted-foreground">{modelB}</div>
                    <div className="text-lg">
                        {formatMetricValue(metric.metric, metric.run_b)}
                    </div>
                    {bWins && <WinnerIcon winner={metric.winner} />}
                </div>
            </div>
        </div>
    );
}

function CoverageGapsSummary({ gaps }: { gaps: CoverageGaps | null | undefined }) {
    if (!gaps || gaps.missing.length === 0) return null;

    return (
        <div className="mt-2 rounded border border-amber-200 bg-amber-50/50 px-2.5 py-1.5 dark:border-amber-900 dark:bg-amber-950/20">
            <span className="text-[10px] font-semibold uppercase tracking-wide text-amber-800 dark:text-amber-300">
                Missing ({gaps.missing.length})
            </span>
            <p className="mt-0.5 text-xs text-amber-900 dark:text-amber-200">
                {gaps.missing.join(', ')}
            </p>
            {((gaps.retrieval_failures && gaps.retrieval_failures.length > 0) ||
                (gaps.generation_failures && gaps.generation_failures.length > 0)) && (
                <div className="mt-1 flex flex-wrap gap-2 text-[10px]">
                    {gaps.retrieval_failures && gaps.retrieval_failures.length > 0 && (
                        <span className="text-rose-700 dark:text-rose-400">
                            Retrieval gaps: {gaps.retrieval_failures.length}
                        </span>
                    )}
                    {gaps.generation_failures && gaps.generation_failures.length > 0 && (
                        <span className="text-orange-700 dark:text-orange-400">
                            Generation gaps: {gaps.generation_failures.length}
                        </span>
                    )}
                </div>
            )}
        </div>
    );
}

function InlineVerdictAndFailReasons({ result }: { result: EvalResult | null | undefined }) {
    if (!result?.verdict) return null;

    const verdictLabel = result.verdict === 'REVIEW_REQUIRED' ? 'Review' : result.verdict;
    const verdictClass =
        result.verdict === 'PASS'
            ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-300'
            : result.verdict === 'FAIL'
              ? 'bg-rose-100 text-rose-700 dark:bg-rose-950/40 dark:text-rose-300'
              : 'bg-amber-100 text-amber-700 dark:bg-amber-950/40 dark:text-amber-300';

    return (
        <div className="mt-2 flex flex-wrap items-center gap-1.5">
            <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-semibold ${verdictClass}`}>
                {verdictLabel}
            </span>
            {result.fail_reasons?.map((reason, i) => (
                <span
                    key={i}
                    className={`inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-medium ${
                        result.verdict === 'FAIL'
                            ? 'bg-rose-100 text-rose-700 dark:bg-rose-950/40 dark:text-rose-300'
                            : 'bg-amber-100 text-amber-700 dark:bg-amber-950/40 dark:text-amber-300'
                    }`}
                >
                    {reason}
                </span>
            ))}
        </div>
    );
}

function InlineDeterministicChecks({
    result,
    evidenceMode,
}: {
    result: EvalResult | null | undefined;
    evidenceMode?: string;
}) {
    if (!result?.deterministic_checks || result.deterministic_checks.length === 0) return null;

    return (
        <div className="mt-1.5 flex flex-wrap gap-2 text-xs">
            {result.deterministic_checks.map((check, i) => {
                const isGroundedTruth = isGroundedEvidenceMode(evidenceMode);
                const isChunkAlignmentInfo =
                    check.check_name === 'chunk_alignment' && !check.passed && isGroundedTruth;
                const hasSupportingWarning =
                    check.passed &&
                    check.detail?.toLowerCase().includes('missing') &&
                    check.detail?.toLowerCase().includes('supporting');
                const Icon = isChunkAlignmentInfo
                    ? Info
                    : check.passed
                      ? hasSupportingWarning
                          ? AlertTriangle
                          : CheckCircle2
                      : XCircle;
                const iconClass = isChunkAlignmentInfo
                    ? 'text-slate-400 dark:text-slate-500'
                    : check.passed
                      ? hasSupportingWarning
                          ? 'text-amber-600 dark:text-amber-400'
                          : 'text-emerald-600 dark:text-emerald-400'
                      : 'text-rose-600 dark:text-rose-400';
                return (
                    <span key={i} className="inline-flex items-center gap-1">
                        <Icon className={`h-3 w-3 ${iconClass}`} />
                        <span className="text-muted-foreground">
                            {formatCheckName(check.check_name)}
                        </span>
                    </span>
                );
            })}
        </div>
    );
}

function QuestionResultPanel({ result }: { result: EvalResult | null | undefined }) {
    const [truthExpanded, setTruthExpanded] = useState(false);
    const [showAllConcepts, setShowAllConcepts] = useState(false);

    if (!result) {
        return <span className="text-sm text-muted-foreground">Not evaluated</span>;
    }

    const truth = result.truth;
    const requiredConcepts = truth?.answer_truth.required_concepts ?? [];
    const missingConcepts = result.coverage_gaps?.missing ?? [];
    const defaultConcepts =
        missingConcepts.length > 0 ? missingConcepts : requiredConcepts.slice(0, 6);
    const visibleConcepts = showAllConcepts ? requiredConcepts : defaultConcepts;
    const docsProgress = getRequiredDocsProgress(result);

    return (
        <>
            <p className="mb-2 text-sm whitespace-pre-wrap break-words">{result.answer ?? 'No answer'}</p>
            <div className="flex flex-wrap gap-3 text-xs">
                <span>
                    Faith: <strong>{formatScore(result.groundedness_score)}</strong>
                </span>
                <span>
                    Relev: <strong>{formatScore(result.relevancy_score)}</strong>
                </span>
                <span>
                    Compl: <strong>{formatScore(result.completeness_score)}</strong>
                </span>
                <span>{formatLatency(result.latency_ms)}</span>
            </div>
            {result.is_hallucination && (
                <span className="mt-1 inline-block rounded-full bg-rose-100 px-2 py-0.5 text-[10px] font-medium text-rose-700 dark:bg-rose-950/40 dark:text-rose-300">
                    Hallucination
                </span>
            )}
            <InlineVerdictAndFailReasons result={result} />
            <InlineDeterministicChecks result={result} evidenceMode={truth?.retrieval_truth.evidence_mode} />
            <CoverageGapsSummary gaps={result.coverage_gaps} />

            {truth && (
                <div className="mt-2 rounded border bg-muted/30 px-2.5 py-1.5">
                    <div className="flex flex-wrap gap-1.5 text-[10px]">
                        <span className="rounded-full border bg-background px-2 py-0.5">
                            Missing concepts: {missingConcepts.length}/{requiredConcepts.length}
                        </span>
                        {docsProgress.total > 0 && (
                            <span className="rounded-full border bg-background px-2 py-0.5">
                                {docsProgress.found == null
                                    ? `Required docs: ${docsProgress.total}`
                                    : `Required docs found: ${docsProgress.found}/${docsProgress.total}`}
                            </span>
                        )}
                        {docsProgress.supportingTotal > 0 && (
                            <span className="rounded-full border bg-background px-2 py-0.5">
                                Supporting docs: {docsProgress.supportingTotal}
                            </span>
                        )}
                        <span className="rounded-full border bg-background px-2 py-0.5">
                            Evidence: {formatEvidenceMode(truth.retrieval_truth.evidence_mode)}
                        </span>
                    </div>
                </div>
            )}

            {truth && (
                <div className="mt-2 rounded border bg-muted/40">
                    <button
                        onClick={() => setTruthExpanded(!truthExpanded)}
                        className="flex w-full items-center justify-between px-3 py-2 text-left text-xs font-semibold"
                    >
                        Truth
                        <ChevronDown
                            className={`h-3.5 w-3.5 text-muted-foreground transition-transform ${truthExpanded ? 'rotate-180' : ''}`}
                        />
                    </button>
                    {truthExpanded && (
                        <div className="space-y-2 border-t px-3 pb-2.5 pt-2">
                            <div>
                                <span className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                                    {showAllConcepts
                                        ? `Required Concepts (${requiredConcepts.length})`
                                        : missingConcepts.length > 0
                                          ? `Missing Concepts (${missingConcepts.length})`
                                          : `Required Concepts (${visibleConcepts.length}/${requiredConcepts.length})`}
                                </span>
                                <div className="mt-1 flex flex-wrap gap-1">
                                    {visibleConcepts.map((concept, i) => (
                                        <span
                                            key={i}
                                            className="inline-flex items-center rounded-full border bg-background px-1.5 py-0.5 text-[10px]"
                                        >
                                            {concept}
                                        </span>
                                    ))}
                                </div>
                                {requiredConcepts.length > visibleConcepts.length && (
                                    <button
                                        onClick={() => setShowAllConcepts(!showAllConcepts)}
                                        className="mt-1 text-[10px] text-muted-foreground underline-offset-2 hover:underline"
                                    >
                                        {showAllConcepts
                                            ? 'Show fewer concepts'
                                            : missingConcepts.length > 0
                                              ? 'Show all required concepts'
                                              : 'Show all concepts'}
                                    </button>
                                )}
                            </div>
                            {truth.retrieval_truth.required_documents.length > 0 && (
                                <div>
                                    <span className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                                        Required Documents
                                    </span>
                                    <div className="mt-1 flex flex-wrap gap-1">
                                        {truth.retrieval_truth.required_documents.map((doc, i) => (
                                            <span
                                                key={i}
                                                className="inline-flex items-center rounded-full border bg-background px-1.5 py-0.5 text-[10px]"
                                            >
                                                {doc}
                                            </span>
                                        ))}
                                    </div>
                                </div>
                            )}
                            {truth.retrieval_truth.supporting_documents.length > 0 && (
                                <div>
                                    <span className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                                        Supporting Documents
                                    </span>
                                    <div className="mt-1 flex flex-wrap gap-1">
                                        {truth.retrieval_truth.supporting_documents.map((doc, i) => (
                                            <span
                                                key={i}
                                                className="inline-flex items-center rounded-full border border-dashed bg-muted/30 px-1.5 py-0.5 text-[10px] text-muted-foreground"
                                            >
                                                {doc}
                                            </span>
                                        ))}
                                    </div>
                                </div>
                            )}
                            <div className="flex flex-wrap gap-2 text-[10px] text-muted-foreground">
                                <span>Evidence: {formatEvidenceMode(truth.retrieval_truth.evidence_mode)}</span>
                                <span>Truth model: {truth.metadata.generated_by_model}</span>
                            </div>
                        </div>
                    )}
                </div>
            )}
        </>
    );
}

function QuestionRow({
    question,
    expectedAnswer,
    resultA,
    resultB,
    modelA,
    modelB,
}: {
    question: string;
    expectedAnswer: string | null | undefined;
    resultA: EvalResult | null | undefined;
    resultB: EvalResult | null | undefined;
    modelA: string;
    modelB: string;
}) {
    return (
        <div className="rounded-lg border bg-card">
            <div className="border-b px-4 py-3">
                <span className="text-sm font-medium">{question}</span>
            </div>
            {expectedAnswer && (
                <div className="border-b bg-muted/30 px-4 py-3">
                    <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                        Expected answer
                    </span>
                    <p className="mt-1 text-sm whitespace-pre-wrap break-words">{expectedAnswer}</p>
                </div>
            )}
            <div className="grid grid-cols-2 divide-x border-b bg-muted/20">
                <div className="px-4 py-2">
                    <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">{modelA}</span>
                </div>
                <div className="px-4 py-2">
                    <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">{modelB}</span>
                </div>
            </div>
            <div className="grid grid-cols-2 divide-x">
                <div className="p-4">
                    <QuestionResultPanel result={resultA} />
                </div>
                <div className="p-4">
                    <QuestionResultPanel result={resultB} />
                </div>
            </div>
        </div>
    );
}

function ComparisonHistory() {
    const [history, setHistory] = useState<ComparisonHistoryEntry[]>(loadHistory);

    if (history.length === 0) return null;

    return (
        <div className="mt-6 rounded-xl border bg-card p-4">
            <h3 className="mb-3 text-sm font-semibold">Recent Comparisons</h3>
            <div className="space-y-2">
                {history.map((entry) => (
                    <div
                        key={`${entry.run_a_id}-${entry.run_b_id}`}
                        className="flex items-center justify-between rounded-lg border bg-background px-3 py-2 transition-colors hover:bg-accent"
                    >
                        <Link
                            to="/evaluations/compare"
                            search={{ run_a: entry.run_a_id, run_b: entry.run_b_id }}
                            className="flex flex-1 flex-col"
                        >
                            <span className="text-sm font-medium">
                                #{entry.run_a_id} {entry.model_a} vs #{entry.run_b_id}{' '}
                                {entry.model_b}
                            </span>
                            <span className="flex items-center gap-1 text-xs text-muted-foreground">
                                <Clock className="h-3 w-3" />
                                {new Date(entry.timestamp).toLocaleString()}
                            </span>
                        </Link>
                        <button
                            onClick={(e) => {
                                e.preventDefault();
                                e.stopPropagation();
                                setHistory(removeFromHistory(entry.run_a_id, entry.run_b_id));
                            }}
                            className="rounded p-1.5 text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive"
                            title="Remove from history"
                        >
                            <Trash2 className="h-3.5 w-3.5" />
                        </button>
                    </div>
                ))}
            </div>
        </div>
    );
}

const REASON_LABELS: Record<string, string> = {
    OPPONENT_DISQUALIFIED: 'Opponent disqualified',
    BOTH_DISQUALIFIED: 'Both disqualified',
    BETTER_VERDICT: 'Better verdict',
    FEWER_FAILURES: 'Fewer critical failures',
    FEWER_REVIEWS: 'Fewer reviews needed',
    METRIC_ADVANTAGE: 'Metric advantage',
};

function ComparisonHeader({
    data,
    docCount,
}: {
    data: ComparisonResponse;
    docCount: number | undefined;
}) {
    const modelA = data.run_a.model_name;
    const modelB = data.run_b.model_name;
    const profileId = data.run_a.profile_id ?? data.run_b.profile_id;
    const totalQuestions = data.run_a.total_questions;

    const pills: string[] = [];
    if (profileId) pills.push(`profile: ${profileId}`);
    pills.push(`dataset: ${totalQuestions} question${totalQuestions !== 1 ? 's' : ''}`);
    if (docCount != null) pills.push(`docs: ${docCount} indexed source${docCount !== 1 ? 's' : ''}`);

    return (
        <div className="mb-6 rounded-xl border bg-muted/30 p-6">
            <div className="mb-3 inline-flex items-center gap-1.5 rounded-full border bg-background px-3 py-1 text-xs text-muted-foreground">
                <GitCompareArrows className="h-3.5 w-3.5" />
                Comparison screen
                {profileId && data.run_a.profile_id === data.run_b.profile_id && (
                    <>&middot; same profile</>
                )}
                {docCount != null && <>&middot; same docs</>}
            </div>
            <h1 className="mb-2 text-3xl font-bold tracking-tight">
                {modelA} vs {modelB}
            </h1>
            <p className="mb-4 text-sm text-muted-foreground">
                This is the decision screen: it shows which model is stronger, why it wins under the
                business rubric, and what risks remain before production use.
            </p>
            <div className="flex flex-wrap gap-2">
                {pills.map((pill) => (
                    <span
                        key={pill}
                        className="inline-flex items-center rounded-full border bg-background px-3 py-1 text-xs font-medium"
                    >
                        {pill}
                    </span>
                ))}
            </div>
        </div>
    );
}

function ExecutiveVerdictCard({ data }: { data: ComparisonResponse }) {
    const decision = data.decision;

    if (!decision) {
        return (
            <div className="mb-8 rounded-xl border bg-card p-6">
                <h2 className="mb-1 text-lg font-semibold">Executive verdict</h2>
                <p className="text-sm text-muted-foreground">
                    Verdict not available.
                </p>
            </div>
        );
    }

    const isTie = decision.winner === 'tie';
    const isBothDisqualified = decision.reason_codes.includes('BOTH_DISQUALIFIED');
    const isInconclusive = decision.decision_status === 'inconclusive' && !isTie && !isBothDisqualified;
    const hasRisks = decision.risk_flags.length > 0;
    // Choose card color based on decision quality
    const bgClass = isBothDisqualified
        ? 'bg-rose-50/80 dark:bg-rose-950/20'
        : isInconclusive
          ? 'bg-amber-50/80 dark:bg-amber-950/20'
          : isTie
            ? 'bg-muted/50'
            : 'bg-emerald-50/80 dark:bg-emerald-950/20';

    const badgeClass = isBothDisqualified
        ? 'bg-rose-600 dark:bg-rose-700'
        : isInconclusive
          ? 'bg-amber-600 dark:bg-amber-700'
          : isTie
            ? 'bg-muted-foreground'
            : 'bg-emerald-600 dark:bg-emerald-700';

    const textClass = isBothDisqualified
        ? 'text-rose-900 dark:text-rose-200'
        : isInconclusive
          ? 'text-amber-900 dark:text-amber-200'
          : isTie
            ? 'text-muted-foreground'
            : 'text-emerald-900 dark:text-emerald-200';

    return (
        <div className="mb-8 rounded-xl border bg-card p-6">
            <h2 className="mb-1 text-lg font-semibold">Executive verdict</h2>
            <p className="mb-4 text-sm text-muted-foreground">
                Start with the business answer, not the raw metrics.
            </p>
            <div className={`rounded-lg p-4 ${bgClass}`}>
                <div className="mb-2 flex flex-wrap items-center gap-2">
                    <span
                        className={`inline-flex items-center rounded-full px-3 py-1 text-sm font-medium text-white ${badgeClass}`}
                    >
                        {isBothDisqualified
                            ? 'No acceptable model'
                            : isTie
                              ? 'No clear winner'
                              : isInconclusive
                                ? 'Inconclusive'
                                : `Overall winner: ${decision.winner_name}`}
                    </span>
                    {decision.decision_status === 'marginal' && (
                        <span className="inline-flex items-center rounded-full border border-amber-300 bg-white px-3 py-1 text-xs font-medium text-amber-800 dark:border-amber-800 dark:bg-amber-950/40 dark:text-amber-300">
                            Marginal advantage
                        </span>
                    )}
                    {decision.reason_codes.map((code) => (
                        <span
                            key={code}
                            className={`inline-flex items-center rounded-full border bg-white px-3 py-1 text-xs font-medium ${
                                code === 'BOTH_DISQUALIFIED'
                                    ? 'border-rose-300 text-rose-800 dark:border-rose-800 dark:bg-rose-950/40 dark:text-rose-300'
                                    : 'border-emerald-300 text-emerald-800 dark:border-emerald-800 dark:bg-emerald-950/40 dark:text-emerald-300'
                            }`}
                        >
                            {REASON_LABELS[code] ?? code}
                        </span>
                    ))}
                </div>
                <p className={`text-sm ${textClass}`}>
                    {decision.summary}
                </p>
            </div>

            {/* Risk flags */}
            {hasRisks && (
                <div className="mt-4 rounded-lg border border-amber-200 bg-amber-50/50 p-3 dark:border-amber-900 dark:bg-amber-950/10">
                    <h3 className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-amber-800 dark:text-amber-300">
                        Risk flags
                    </h3>
                    <ul className="space-y-1">
                        {decision.risk_flags.map((flag, i) => (
                            <li key={i} className="text-sm text-amber-900 dark:text-amber-200">
                                {flag}
                            </li>
                        ))}
                    </ul>
                </div>
            )}


            {/* Comparison warnings */}
            {data.warnings && data.warnings.length > 0 && (
                <div className="mt-3 rounded-lg border border-muted bg-muted/30 p-3">
                    <h3 className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                        Comparison caveats
                    </h3>
                    <ul className="space-y-1">
                        {data.warnings.map((w, i) => (
                            <li key={i} className="text-sm text-muted-foreground">
                                {w.message}
                            </li>
                        ))}
                    </ul>
                </div>
            )}
        </div>
    );
}

function CompareSelector() {
    const navigate = useNavigate();
    const { data: runs } = useEvalRuns();
    const completedRuns =
        runs?.filter((r: EvalRun) => r.status === 'completed' || r.status === 'complete') ?? [];
    const [runA, setRunA] = useState<number>(0);
    const [runB, setRunB] = useState<number>(0);

    if (completedRuns.length < 2) {
        return (
            <div className="rounded-xl border bg-card p-8 text-center">
                <GitCompareArrows className="mx-auto mb-3 h-8 w-8 text-muted-foreground" />
                <h2 className="mb-1 text-base font-semibold">No comparable runs yet</h2>
                <p className="text-sm text-muted-foreground">
                    You need two completed evaluation runs to compare models side by side.
                </p>
                <div className="mt-4 flex flex-wrap justify-center gap-2">
                    <button
                        onClick={() => navigate({ to: '/evaluations' })}
                        className="inline-flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90"
                    >
                        <BarChart3 className="h-4 w-4" />
                        Run New Evaluation
                    </button>
                    <button
                        onClick={() => navigate({ to: '/documents' })}
                        className="inline-flex items-center gap-2 rounded-lg border px-4 py-2 text-sm font-medium transition-colors hover:bg-accent"
                    >
                        <FileText className="h-4 w-4" />
                        Upload Documents
                    </button>
                </div>
                <ComparisonHistory />
            </div>
        );
    }

    return (
        <>
            <div className="rounded-xl border bg-card p-4">
                <div className="flex items-end gap-3">
                    <div className="flex-1">
                        <label className="mb-1 block text-xs text-muted-foreground">
                            Baseline Run
                        </label>
                        <select
                            value={runA}
                            onChange={(e) => setRunA(Number(e.target.value))}
                            className="w-full rounded-lg border bg-background px-3 py-2 text-sm"
                        >
                            <option value={0}>Select run</option>
                            {completedRuns.map((r: EvalRun) => (
                                <option key={r.id} value={r.id}>
                                    #{r.id} - {r.model_name}
                                </option>
                            ))}
                        </select>
                    </div>
                    <span className="pb-2 text-sm text-muted-foreground">vs</span>
                    <div className="flex-1">
                        <label className="mb-1 block text-xs text-muted-foreground">
                            Candidate Run
                        </label>
                        <select
                            value={runB}
                            onChange={(e) => setRunB(Number(e.target.value))}
                            className="w-full rounded-lg border bg-background px-3 py-2 text-sm"
                        >
                            <option value={0}>Select run</option>
                            {completedRuns.map((r: EvalRun) => (
                                <option key={r.id} value={r.id}>
                                    #{r.id} - {r.model_name}
                                </option>
                            ))}
                        </select>
                    </div>
                    <button
                        onClick={() =>
                            navigate({
                                to: '/evaluations/compare',
                                search: { run_a: runA, run_b: runB },
                            })
                        }
                        disabled={!runA || !runB || runA === runB}
                        className="rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90 disabled:opacity-50"
                    >
                        Compare
                    </button>
                </div>
            </div>
            <ComparisonHistory />
        </>
    );
}

function ComparePage() {
    const navigate = useNavigate();
    const { run_a, run_b } = Route.useSearch();
    const { data, isLoading, error } = useCompareEvalRuns(run_a, run_b);
    const { data: documents } = useDocuments();

    useEffect(() => {
        if (data) {
            saveToHistory({
                run_a_id: data.run_a.id,
                run_b_id: data.run_b.id,
                model_a: data.run_a.model_name,
                model_b: data.run_b.model_name,
            });
        }
    }, [data]);

    if (!run_a || !run_b) {
        return (
            <div className="p-4 sm:p-6 lg:p-8">
                <div className="mx-auto max-w-5xl">
                    <h1 className="mb-1 text-2xl font-bold tracking-tight">Compare Evaluations</h1>
                    <p className="mb-6 text-sm text-muted-foreground">
                        Compare completed runs across models.
                    </p>
                    <CompareSelector />
                </div>
            </div>
        );
    }

    if (isLoading) {
        return (
            <div className="flex items-center justify-center p-12">
                <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            </div>
        );
    }

    if (error || !data) {
        return (
            <div className="p-8 text-center text-sm text-destructive">
                {error?.message ?? 'Failed to load comparison'}
            </div>
        );
    }

    const modelA = data.run_a.model_name;
    const modelB = data.run_b.model_name;
    const readyDocCount = documents?.filter(
        (d) => d.status === 'ready',
    ).length;

    return (
        <div className="p-4 sm:p-6 lg:p-8">
            <div className="mx-auto max-w-5xl">
                <button
                    onClick={() =>
                        navigate({
                            to: '/evaluations/compare',
                            search: { run_a: 0, run_b: 0 },
                        })
                    }
                    className="mb-4 flex items-center gap-1 text-sm text-muted-foreground transition-colors hover:text-foreground"
                >
                    <ArrowLeft className="h-4 w-4" />
                    New comparison
                </button>

                <ComparisonHeader data={data} docCount={readyDocCount} />

                <ExecutiveVerdictCard data={data} />

                {/* Aggregate metrics */}
                <div className="mb-8">
                    <h2 className="mb-3 text-lg font-semibold">Aggregate Metrics</h2>
                    <div className="space-y-2">
                        {data.metrics.map((metric) => (
                            <MetricRow
                                key={metric.metric}
                                metric={metric}
                                modelA={modelA}
                                modelB={modelB}
                            />
                        ))}
                    </div>
                </div>

                {/* Per-question breakdown */}
                <div>
                    <h2 className="mb-3 text-lg font-semibold">Per-Question Breakdown</h2>
                    <div className="space-y-3">
                        {data.questions.map((q, i) => (
                            <QuestionRow
                                key={i}
                                question={q.question}
                                expectedAnswer={q.expected_answer}
                                resultA={q.run_a}
                                resultB={q.run_b}
                                modelA={modelA}
                                modelB={modelB}
                            />
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
}
