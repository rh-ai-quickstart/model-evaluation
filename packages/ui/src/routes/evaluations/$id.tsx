// This project was developed with assistance from AI tools.

import { createFileRoute, useNavigate } from '@tanstack/react-router';
import { useState } from 'react';
import { useEvalRun, useRerunEval } from '../../hooks/evaluation';
import { useModels } from '../../hooks/models';
import {
    ArrowLeft,
    RefreshCw,
    Loader2,
    AlertTriangle,
    CheckCircle2,
    XCircle,
    ChevronDown,
    Clock,
    Info,
} from 'lucide-react';
import type { EvalResult, CoverageGaps } from '../../schemas/evaluation';
import { formatScore, formatLatency, formatUtcDate } from '../../lib/format';
import { EVAL_STATUS_COLORS } from '../../lib/status-colors';
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from '../../components/atoms/select/select';
import { Skeleton } from '../../components/atoms/skeleton/skeleton';
import { toast } from 'sonner';
import {
    Tooltip,
    TooltipContent,
    TooltipProvider,
    TooltipTrigger,
} from '../../components/atoms/tooltip/tooltip';

const METRIC_DESCRIPTIONS: Record<string, string> = {
    Faithfulness:
        'Are all claims in the answer grounded in the retrieved context? Below 70% is flagged as hallucination.',
    Relevancy: 'Does the answer directly address the question that was asked?',
    'Context Precision':
        'Do the retrieved chunks actually contain the information needed to produce the expected answer?',
    'Context Relevancy': 'Are the retrieved chunks relevant to the question?',
    Completeness:
        'Does the answer cover all key points from the expected answer? Requires expected answer.',
    Correctness:
        'Are the claims in the answer consistent with the expected answer, without contradictions? Requires expected answer.',
    'Compliance Accuracy':
        'Are regulatory obligations, thresholds, disclosures, and authorities correctly stated? Requires expected answer.',
    'Abstention Quality':
        'Does the answer appropriately acknowledge when the context is insufficient, rather than fabricating information?',
    'Hallucination Rate': 'Percentage of questions where faithfulness scored below 70%.',
    'Avg Latency': 'Average time to generate an answer across all questions in this run.',
};

export const Route = createFileRoute('/evaluations/$id')({
    component: EvalRunDetailPage,
});

function ScoreColor({ score }: { score: number | null | undefined }) {
    if (score == null) return <span className="text-muted-foreground">--</span>;
    const pct = score * 100;
    const color =
        pct >= 80
            ? 'text-emerald-600 dark:text-emerald-400'
            : pct >= 60
              ? 'text-amber-600 dark:text-amber-400'
              : 'text-rose-600 dark:text-rose-400';
    return <span className={`font-medium ${color}`}>{pct.toFixed(0)}%</span>;
}

function MetricLabel({ label }: { label: string }) {
    const description = METRIC_DESCRIPTIONS[label];
    if (!description) return <span>{label}</span>;
    return (
        <Tooltip>
            <TooltipTrigger asChild>
                <span className="cursor-help border-b border-dotted border-muted-foreground/40">
                    {label} *
                </span>
            </TooltipTrigger>
            <TooltipContent className="max-w-xs">
                <p>{description}</p>
            </TooltipContent>
        </Tooltip>
    );
}

function MetricCard({
    label,
    value,
    icon,
}: {
    label: string;
    value: string;
    icon: React.ReactNode;
}) {
    return (
        <div className="rounded-lg border bg-card p-4">
            <div className="mb-1 flex items-center gap-2 text-xs text-muted-foreground">
                {icon}
                <MetricLabel label={label} />
            </div>
            <div className="text-2xl font-bold">{value}</div>
        </div>
    );
}

const VERDICT_STYLES: Record<string, string> = {
    PASS: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-300',
    FAIL: 'bg-rose-100 text-rose-700 dark:bg-rose-950/40 dark:text-rose-300',
    REVIEW_REQUIRED:
        'bg-amber-100 text-amber-700 dark:bg-amber-950/40 dark:text-amber-300',
};

function VerdictBadge({ verdict }: { verdict: string }) {
    const label = verdict === 'REVIEW_REQUIRED' ? 'Review' : verdict;
    return (
        <span
            className={`inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-medium ${VERDICT_STYLES[verdict] ?? VERDICT_STYLES.REVIEW_REQUIRED}`}
        >
            {label}
        </span>
    );
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

interface ParsedChunk {
    document: string;
    page: string | null;
    section: string | null;
    text: string;
}

function parseContextChunks(contexts: string): ParsedChunk[] {
    const raw = contexts.split('\n---\n');
    return raw.map((block) => {
        const lines = block.trim().split('\n');
        let document = '';
        let page: string | null = null;
        let section: string | null = null;
        let textStart = 0;

        // Parse header line like [doc.pdf | p.5 | Section Name]
        if (lines[0]?.startsWith('[') && lines[0]?.includes(']')) {
            const headerEnd = lines[0].indexOf(']');
            const header = lines[0].slice(1, headerEnd);
            const parts = header.split('|').map((p) => p.trim());
            document = parts[0] ?? '';
            for (let i = 1; i < parts.length; i++) {
                if (parts[i].startsWith('p.')) {
                    page = parts[i];
                } else {
                    section = parts[i];
                }
            }
            textStart = 1;
        }

        const text = lines.slice(textStart).join('\n').trim();
        return { document, page, section, text };
    }).filter((c) => c.text.length > 0);
}

function ChunkCard({ chunk, index }: { chunk: ParsedChunk; index: number }) {
    return (
        <div className="rounded-lg border bg-card p-4">
            <div className="mb-2 flex items-start justify-between">
                <div>
                    <div className="flex items-center gap-2">
                        <span className="inline-flex h-6 min-w-6 items-center justify-center rounded-full bg-muted px-1.5 text-xs font-medium">
                            #{index + 1}
                        </span>
                        <span className="text-sm font-semibold">{chunk.document || 'Unknown document'}</span>
                    </div>
                    {(chunk.page || chunk.section) && (
                        <div className="mt-0.5 pl-8 text-xs text-muted-foreground">
                            {[chunk.page, chunk.section].filter(Boolean).join(' \u00b7 ')}
                        </div>
                    )}
                </div>
            </div>
            <p className="pl-8 text-sm">{chunk.text}</p>
        </div>
    );
}

function CoverageGapsSummary({ gaps }: { gaps: CoverageGaps | null | undefined }) {
    if (!gaps || gaps.missing.length === 0) return null;

    return (
        <div className="rounded border border-amber-200 bg-amber-50/50 px-2.5 py-1.5 dark:border-amber-900 dark:bg-amber-950/20">
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

function AnswerBlock({ label, text }: { label: string; text: string }) {
    return (
        <div>
            <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                {label}
            </span>
            <div className="mt-1 rounded-lg bg-muted/40 p-3">
                <p className="text-sm whitespace-pre-wrap break-words">{text}</p>
            </div>
        </div>
    );
}

function ResultRow({ result }: { result: EvalResult }) {
    const [expanded, setExpanded] = useState(false);
    const [truthExpanded, setTruthExpanded] = useState(false);
    const [showAllConcepts, setShowAllConcepts] = useState(false);
    const [chunksExpanded, setChunksExpanded] = useState(false);

    const parsedChunks = result.contexts ? parseContextChunks(result.contexts) : [];
    const visibleChunks = chunksExpanded ? parsedChunks : parsedChunks.slice(0, 2);

    const truth = result.truth;
    const requiredConcepts = truth?.answer_truth.required_concepts ?? [];
    const missingConcepts = result.coverage_gaps?.missing ?? [];
    const defaultConcepts =
        missingConcepts.length > 0 ? missingConcepts : requiredConcepts.slice(0, 8);
    const visibleConcepts = showAllConcepts ? requiredConcepts : defaultConcepts;

    const requiredDocs = truth?.retrieval_truth.required_documents ?? [];
    const supportingDocs = truth?.retrieval_truth.supporting_documents ?? [];
    const retrievedDocs = new Set(
        parsedChunks
            .map((chunk) => chunk.document.trim().toLowerCase())
            .filter((doc) => doc.length > 0),
    );
    const requiredDocsFound = requiredDocs.filter((doc) =>
        retrievedDocs.has(doc.trim().toLowerCase()),
    ).length;
    const supportingDocsFound = supportingDocs.filter((doc) =>
        retrievedDocs.has(doc.trim().toLowerCase()),
    ).length;

    return (
        <div className="rounded-lg border bg-card">
            <button
                onClick={() => setExpanded(!expanded)}
                className="flex w-full items-center justify-between p-4 text-left"
            >
                <div className="flex-1">
                    <div className="flex items-center gap-2">
                        <span className="text-sm font-medium">{result.question}</span>
                        {result.verdict && (
                            <VerdictBadge verdict={result.verdict} />
                        )}
                        {result.is_hallucination && (
                            <span className="inline-flex items-center gap-1 rounded-full bg-rose-100 px-2 py-0.5 text-[10px] font-medium text-rose-700 dark:bg-rose-950/40 dark:text-rose-300">
                                <AlertTriangle className="h-3 w-3" />
                                Hallucination
                            </span>
                        )}
                    </div>
                    {result.error_message && (
                        <p className="mt-1 text-xs text-destructive">{result.error_message}</p>
                    )}
                </div>
                <div className="flex items-center gap-4 text-sm">
                    <ScoreColor score={result.groundedness_score} />
                    <ScoreColor score={result.relevancy_score} />
                    <span className="text-xs text-muted-foreground">
                        {formatLatency(result.latency_ms)}
                    </span>
                </div>
            </button>

            {expanded && (
                <div className="border-t px-4 pb-4 pt-3 space-y-4">
                    {result.expected_answer && (
                        <AnswerBlock label="Expected answer" text={result.expected_answer} />
                    )}

                    {result.answer && <AnswerBlock label="Model answer" text={result.answer} />}

                    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                        <div>
                            <div className="text-xs text-muted-foreground">
                                <MetricLabel label="Faithfulness" />
                            </div>
                            <ScoreColor score={result.groundedness_score} />
                        </div>
                        <div>
                            <div className="text-xs text-muted-foreground">
                                <MetricLabel label="Relevancy" />
                            </div>
                            <ScoreColor score={result.relevancy_score} />
                        </div>
                        <div>
                            <div className="text-xs text-muted-foreground">
                                <MetricLabel label="Context Precision" />
                            </div>
                            <ScoreColor score={result.context_precision_score} />
                        </div>
                        <div>
                            <div className="text-xs text-muted-foreground">
                                <MetricLabel label="Context Relevancy" />
                            </div>
                            <ScoreColor score={result.context_relevancy_score} />
                        </div>
                        <div>
                            <div className="text-xs text-muted-foreground">
                                <MetricLabel label="Completeness" />
                            </div>
                            <ScoreColor score={result.completeness_score} />
                        </div>
                        <div>
                            <div className="text-xs text-muted-foreground">
                                <MetricLabel label="Correctness" />
                            </div>
                            <ScoreColor score={result.correctness_score} />
                        </div>
                        <div>
                            <div className="text-xs text-muted-foreground">
                                <MetricLabel label="Compliance Accuracy" />
                            </div>
                            <ScoreColor score={result.compliance_accuracy_score} />
                        </div>
                        <div>
                            <div className="text-xs text-muted-foreground">
                                <MetricLabel label="Abstention Quality" />
                            </div>
                            <ScoreColor score={result.abstention_score} />
                        </div>
                    </div>

                    {result.fail_reasons && result.fail_reasons.length > 0 && (
                        <div>
                            <h3 className="mb-1.5 text-sm font-semibold">Fail Reasons</h3>
                            <div className="flex flex-wrap gap-1.5">
                                {result.fail_reasons.map((reason, i) => (
                                    <span
                                        key={i}
                                        className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${
                                            result.verdict === 'FAIL'
                                                ? 'bg-rose-100 text-rose-700 dark:bg-rose-950/40 dark:text-rose-300'
                                                : 'bg-amber-100 text-amber-700 dark:bg-amber-950/40 dark:text-amber-300'
                                        }`}
                                    >
                                        {reason}
                                    </span>
                                ))}
                            </div>
                        </div>
                    )}

                    {result.deterministic_checks && result.deterministic_checks.length > 0 && (
                        <div>
                            <h3 className="mb-1.5 text-sm font-semibold">Deterministic Checks</h3>
                            <div className="space-y-1">
                                {result.deterministic_checks.map((check, i) => {
                                    const isGroundedTruth = isGroundedEvidenceMode(
                                        truth?.retrieval_truth.evidence_mode,
                                    );
                                    const isChunkAlignmentInfo =
                                        check.check_name === 'chunk_alignment' &&
                                        !check.passed &&
                                        isGroundedTruth;
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
                                        <div key={i} className="flex items-start gap-2 text-sm">
                                            <Icon
                                                className={`mt-0.5 h-4 w-4 shrink-0 ${iconClass}`}
                                            />
                                            <div>
                                                <span className="font-medium">
                                                    {check.check_name}
                                                </span>
                                                {check.detail && (
                                                    <span className="ml-1 text-muted-foreground">
                                                        &mdash; {check.detail}
                                                    </span>
                                                )}
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                    )}

                    <CoverageGapsSummary gaps={result.coverage_gaps} />

                    {truth && (
                        <div className="rounded-lg border bg-muted/30 p-3">
                            <div className="flex flex-wrap gap-2 text-xs">
                                <span className="rounded-full border bg-background px-2 py-0.5">
                                    Missing concepts: {missingConcepts.length}/{requiredConcepts.length}
                                </span>
                                {requiredDocs.length > 0 && (
                                    <span className="rounded-full border bg-background px-2 py-0.5">
                                        Required docs found: {requiredDocsFound}/{requiredDocs.length}
                                    </span>
                                )}
                                {supportingDocs.length > 0 && (
                                    <span className="rounded-full border bg-background px-2 py-0.5">
                                        Supporting docs found: {supportingDocsFound}/
                                        {supportingDocs.length}
                                    </span>
                                )}
                                <span className="rounded-full border bg-background px-2 py-0.5">
                                    Evidence: {formatEvidenceMode(truth.retrieval_truth.evidence_mode)}
                                </span>
                            </div>
                        </div>
                    )}

                    {truth && (
                        <div className="rounded-lg border bg-muted/40">
                            <button
                                onClick={() => setTruthExpanded(!truthExpanded)}
                                className="flex w-full items-center justify-between px-4 py-2.5 text-left text-sm font-semibold"
                            >
                                Truth
                                <ChevronDown
                                    className={`h-4 w-4 text-muted-foreground transition-transform ${truthExpanded ? 'rotate-180' : ''}`}
                                />
                            </button>
                            {truthExpanded && (
                                <div className="space-y-3 border-t px-4 pb-3 pt-2">
                                    <div>
                                        <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                                            {showAllConcepts
                                                ? `Required Concepts (${requiredConcepts.length})`
                                                : missingConcepts.length > 0
                                                  ? `Missing Concepts (${missingConcepts.length})`
                                                  : `Required Concepts (${visibleConcepts.length}/${requiredConcepts.length})`}
                                        </span>
                                        <div className="mt-1 flex flex-wrap gap-1.5">
                                            {visibleConcepts.map((concept, i) => (
                                                <span
                                                    key={i}
                                                    className="inline-flex items-center rounded-full border bg-background px-2 py-0.5 text-xs"
                                                >
                                                    {concept}
                                                </span>
                                            ))}
                                        </div>
                                        {requiredConcepts.length > visibleConcepts.length && (
                                            <button
                                                onClick={() => setShowAllConcepts(!showAllConcepts)}
                                                className="mt-2 text-xs text-muted-foreground underline-offset-2 hover:underline"
                                            >
                                                {showAllConcepts
                                                    ? 'Show fewer concepts'
                                                    : missingConcepts.length > 0
                                                      ? 'Show all required concepts'
                                                      : 'Show all concepts'}
                                            </button>
                                        )}
                                        {truth.answer_truth.abstention_expected && (
                                            <span className="mt-1 inline-flex items-center rounded-full bg-amber-100 px-2 py-0.5 text-[10px] font-medium text-amber-700 dark:bg-amber-950/40 dark:text-amber-300">
                                                Abstention expected
                                            </span>
                                        )}
                                    </div>
                                    {truth.retrieval_truth.required_documents.length > 0 && (
                                        <div>
                                            <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                                                Required Documents
                                            </span>
                                            <div className="mt-1 flex flex-wrap gap-1.5">
                                                {truth.retrieval_truth.required_documents.map((doc, i) => (
                                                    <span
                                                        key={i}
                                                        className="inline-flex items-center rounded-full border bg-background px-2 py-0.5 text-xs"
                                                    >
                                                        {doc}
                                                    </span>
                                                ))}
                                            </div>
                                        </div>
                                    )}
                                    {truth.retrieval_truth.supporting_documents.length > 0 && (
                                        <div>
                                            <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                                                Supporting Documents
                                            </span>
                                            <div className="mt-1 flex flex-wrap gap-1.5">
                                                {truth.retrieval_truth.supporting_documents.map(
                                                    (doc, i) => (
                                                        <span
                                                            key={i}
                                                            className="inline-flex items-center rounded-full border border-dashed bg-muted/30 px-2 py-0.5 text-xs text-muted-foreground"
                                                        >
                                                            {doc}
                                                        </span>
                                                    ),
                                                )}
                                            </div>
                                        </div>
                                    )}
                                    <div className="flex flex-wrap gap-3 text-xs text-muted-foreground">
                                        <span>
                                            Evidence:{' '}
                                            {formatEvidenceMode(
                                                truth.retrieval_truth.evidence_mode,
                                            )}
                                        </span>
                                        <span>
                                            Truth model: {truth.metadata.generated_by_model}
                                        </span>
                                    </div>
                                </div>
                            )}
                        </div>
                    )}

                    {parsedChunks.length > 0 && (
                        <div>
                            <h3 className="mb-1 text-sm font-semibold">Retrieved chunks</h3>
                            <p className="mb-3 text-xs text-muted-foreground">
                                Document, page, section, and snippet for each retrieved chunk.
                            </p>
                            <div className="space-y-3">
                                {visibleChunks.map((chunk, i) => (
                                    <ChunkCard key={i} chunk={chunk} index={i} />
                                ))}
                            </div>
                            {parsedChunks.length > 2 && (
                                <button
                                    onClick={() => setChunksExpanded(!chunksExpanded)}
                                    className="mt-2 text-xs text-muted-foreground underline-offset-2 hover:underline"
                                >
                                    {chunksExpanded
                                        ? 'Show fewer chunks'
                                        : `Show all ${parsedChunks.length} chunks`}
                                </button>
                            )}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

function EvalRunDetailPage() {
    const { id } = Route.useParams();
    const runId = Number(id);
    const navigate = useNavigate();
    const { data: run, isLoading, error } = useEvalRun(runId);
    const { data: models } = useModels();
    const rerunMutation = useRerunEval();
    const [rerunModel, setRerunModel] = useState('');

    if (isLoading) {
        return (
            <div className="p-4 sm:p-6 lg:p-8">
                <div className="mx-auto max-w-5xl">
                    <Skeleton className="mb-4 h-4 w-36" />
                    <div className="mb-6 flex items-start justify-between">
                        <div>
                            <Skeleton className="mb-2 h-7 w-64" />
                            <Skeleton className="h-4 w-40" />
                        </div>
                        <Skeleton className="h-6 w-20 rounded-full" />
                    </div>
                    <div className="mb-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
                        {Array.from({ length: 8 }).map((_, i) => (
                            <div key={i} className="rounded-lg border bg-card p-3">
                                <Skeleton className="mb-2 h-3 w-20" />
                                <Skeleton className="h-6 w-14" />
                            </div>
                        ))}
                    </div>
                    {Array.from({ length: 3 }).map((_, i) => (
                        <div key={i} className="mb-3 rounded-lg border p-4">
                            <Skeleton className="mb-2 h-4 w-3/4" />
                            <Skeleton className="h-3 w-1/2" />
                        </div>
                    ))}
                </div>
            </div>
        );
    }

    if (error || !run) {
        return (
            <div className="p-8 text-center text-sm text-destructive">
                {error?.message ?? 'Evaluation run not found'}
            </div>
        );
    }

    const handleRerun = () => {
        if (!rerunModel) return;
        rerunMutation.mutate(
            { evalRunId: runId, modelName: rerunModel },
            {
                onSuccess: (data) => {
                    toast.success('Evaluation started');
                    navigate({ to: '/evaluations/$id', params: { id: String(data.eval_run_id) } });
                },
                onError: (err) => {
                    toast.error(err.message);
                },
            },
        );
    };

    const isRunning = run.status === 'pending' || run.status === 'running';

    return (
        <TooltipProvider delayDuration={200}>
        <div className="p-4 sm:p-6 lg:p-8">
            <div className="mx-auto max-w-5xl">
                <button
                    onClick={() => navigate({ to: '/evaluations' })}
                    className="mb-4 flex items-center gap-1 text-sm text-muted-foreground transition-colors hover:text-foreground"
                >
                    <ArrowLeft className="h-4 w-4" />
                    Back to evaluations
                </button>

                <div className="mb-6 flex items-start justify-between">
                    <div>
                        <h1 className="text-2xl font-bold tracking-tight">
                            Run #{run.id} - {run.model_name}
                        </h1>
                        <p className="text-sm text-muted-foreground">
                            {run.created_at && formatUtcDate(run.created_at)}
                        </p>
                    </div>
                    <div className="flex items-center gap-2">
                        {isRunning && (
                            <span className="flex items-center gap-1 text-sm text-blue-600 dark:text-blue-400">
                                <Loader2 className="h-4 w-4 animate-spin" />
                                {run.completed_questions}/{run.total_questions}
                            </span>
                        )}
                        <span
                            className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${EVAL_STATUS_COLORS[run.status] ?? EVAL_STATUS_COLORS.pending}`}
                        >
                            {run.status}
                        </span>
                    </div>
                </div>

                {/* Verdict summary */}
                {run.overall_verdict && (
                    <div className="mb-4 flex items-center gap-3 rounded-lg border bg-card p-3">
                        <VerdictBadge verdict={run.overall_verdict} />
                        <span className="text-sm text-muted-foreground">
                            {run.pass_count}/{run.total_questions} questions passed
                            {run.fail_count ? ` | ${run.fail_count} failed` : ''}
                            {run.review_count ? ` | ${run.review_count} need review` : ''}
                        </span>
                        {run.profile_id && (
                            <span className="ml-auto text-xs text-muted-foreground">
                                Profile: {run.profile_id}
                            </span>
                        )}
                    </div>
                )}

                {/* Summary metrics */}
                {run.total_questions > 1 && (
                    <div className="mb-6">
                        <h2 className="mb-2 text-sm font-semibold text-muted-foreground">Run averages</h2>
                        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                            <MetricCard
                                label="Faithfulness"
                                value={formatScore(run.avg_groundedness)}
                                icon={<CheckCircle2 className="h-3.5 w-3.5" />}
                            />
                            <MetricCard
                                label="Relevancy"
                                value={formatScore(run.avg_relevancy)}
                                icon={<CheckCircle2 className="h-3.5 w-3.5" />}
                            />
                            <MetricCard
                                label="Hallucination Rate"
                                value={
                                    run.hallucination_rate != null
                                        ? (run.hallucination_rate * 100).toFixed(0) + '%'
                                        : '--'
                                }
                                icon={<AlertTriangle className="h-3.5 w-3.5" />}
                            />
                            <MetricCard
                                label="Avg Latency"
                                value={formatLatency(run.avg_latency_ms)}
                                icon={<Clock className="h-3.5 w-3.5" />}
                            />
                            <MetricCard
                                label="Completeness"
                                value={formatScore(run.avg_completeness)}
                                icon={<CheckCircle2 className="h-3.5 w-3.5" />}
                            />
                            <MetricCard
                                label="Correctness"
                                value={formatScore(run.avg_correctness)}
                                icon={<CheckCircle2 className="h-3.5 w-3.5" />}
                            />
                            <MetricCard
                                label="Compliance Accuracy"
                                value={formatScore(run.avg_compliance_accuracy)}
                                icon={<CheckCircle2 className="h-3.5 w-3.5" />}
                            />
                            <MetricCard
                                label="Abstention Quality"
                                value={formatScore(run.avg_abstention)}
                                icon={<CheckCircle2 className="h-3.5 w-3.5" />}
                            />
                        </div>
                    </div>
                )}

                {/* Rerun */}
                {run.status === 'completed' && (
                    <div className="mb-6 rounded-xl border bg-card p-4">
                        <h3 className="mb-3 text-sm font-semibold">
                            Re-run with a different model
                        </h3>
                        <div className="flex items-end gap-3">
                            <div className="flex-1">
                                <Select value={rerunModel || undefined} onValueChange={setRerunModel}>
                                    <SelectTrigger className="w-full">
                                        <SelectValue placeholder="Select a model" />
                                    </SelectTrigger>
                                    <SelectContent>
                                        {models?.map((m) => (
                                            <SelectItem key={m.id} value={m.name}>
                                                {m.name} ({m.deployment_mode})
                                            </SelectItem>
                                        ))}
                                    </SelectContent>
                                </Select>
                            </div>
                            <button
                                onClick={handleRerun}
                                disabled={!rerunModel || rerunMutation.isPending}
                                className="flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90 disabled:opacity-50"
                            >
                                {rerunMutation.isPending ? (
                                    <Loader2 className="h-4 w-4 animate-spin" />
                                ) : (
                                    <RefreshCw className="h-4 w-4" />
                                )}
                                Re-run
                            </button>
                        </div>
                    </div>
                )}

                {/* Results */}
                <div>
                    <div className="mb-3 flex items-center justify-between">
                        <h2 className="text-lg font-semibold">Results</h2>
                        <div className="flex items-center gap-4 text-xs text-muted-foreground">
                            <span>Faith.</span>
                            <span>Relev.</span>
                            <span>Latency</span>
                        </div>
                    </div>
                    <div className="space-y-2">
                        {run.results.map((result) => (
                            <ResultRow key={result.id} result={result} />
                        ))}
                    </div>
                    {run.results.length === 0 && !isRunning && (
                        <p className="text-sm text-muted-foreground">No results yet.</p>
                    )}
                </div>
            </div>
        </div>
        </TooltipProvider>
    );
}
