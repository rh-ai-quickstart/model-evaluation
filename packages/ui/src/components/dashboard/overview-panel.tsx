// This project was developed with assistance from AI tools.

import { Link } from '@tanstack/react-router';
import { useEvalRuns } from '../../hooks/evaluation';
import { useDocuments } from '../../hooks/documents';
import { useModels, useModelMetadata } from '../../hooks/models';
import {
    ArrowRight,
    BarChart3,
    GitCompareArrows,
    AlertTriangle,
    CheckCircle2,
    FileText,
    Activity,
} from 'lucide-react';
import { ModelSpecsCard, findModelMetadata } from './model-specs-card';

export function OverviewPanel({ selectedModelId }: { selectedModelId: number | null }) {
    const { data: runs } = useEvalRuns();
    const { data: documents } = useDocuments();
    const { data: models } = useModels();
    const { data: metadataResponse } = useModelMetadata();

    const completedRuns =
        runs?.filter((r) => r.status === 'completed' || r.status === 'complete') ?? [];
    const hasDocuments = (documents?.length ?? 0) > 0;
    const hasComparableRuns = completedRuns.length >= 2;
    const comparisonReadinessLabel = hasComparableRuns
        ? 'Ready to compare'
        : hasDocuments
          ? 'Documents ready'
          : 'Setup required';
    const comparisonReadinessIsWarning = !hasComparableRuns && !hasDocuments;

    return (
        <div className="flex h-full flex-col p-4">
            <div className="rounded-xl border bg-card p-4">
                <div className="flex items-center gap-2">
                    <BarChart3 className="h-5 w-5 text-blue-600 dark:text-blue-400" />
                    <h1 className="text-xl font-bold tracking-tight">Overview</h1>
                </div>
                <p className="mt-0.5 text-sm text-muted-foreground">
                    Evaluate and compare AI models on your compliance documents.
                </p>
                <div className="mt-2.5 flex gap-2">
                    <Link
                        to="/evaluations"
                        className="inline-flex items-center gap-1.5 rounded-lg bg-blue-600 px-3 py-1.5 text-sm font-medium text-white transition-colors hover:bg-blue-700 dark:bg-blue-600 dark:hover:bg-blue-700"
                    >
                        <BarChart3 className="h-4 w-4" />
                        Run New Evaluation
                    </Link>
                    <Link
                        to="/evaluations/compare"
                        search={{ run_a: 0, run_b: 0 }}
                        className="inline-flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-sm font-medium transition-colors hover:bg-accent"
                    >
                        <GitCompareArrows className="h-4 w-4" />
                        Go to Comparisons
                    </Link>
                </div>
            </div>

            <div className="mt-3 rounded-xl border bg-card p-4">
                <div className="mb-2.5 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <CheckCircle2 className="h-4 w-4 text-blue-600 dark:text-blue-400" />
                        <h2 className="text-sm font-semibold">Comparison Readiness</h2>
                    </div>
                    <span
                        className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium ${
                            comparisonReadinessIsWarning
                                ? 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400'
                                : 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400'
                        }`}
                    >
                        {comparisonReadinessIsWarning ? (
                            <AlertTriangle className="h-3 w-3" aria-hidden />
                        ) : (
                            <CheckCircle2 className="h-3 w-3" aria-hidden />
                        )}
                        {comparisonReadinessLabel}
                    </span>
                </div>
                <div className="grid gap-2.5 sm:grid-cols-2">
                    <div className="rounded-lg border p-3">
                        <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                            <FileText className="h-3.5 w-3.5 text-blue-500" />
                            Documents uploaded
                        </div>
                        {hasDocuments ? (
                            <div className="mt-0.5 flex items-baseline gap-1.5">
                                <span className="text-lg font-bold">{documents?.length ?? 0}</span>
                                <span className="text-sm text-muted-foreground">ready</span>
                            </div>
                        ) : (
                            <div className="mt-0.5">
                                <Link
                                    to="/documents"
                                    className="inline-flex items-center gap-1 text-sm text-primary underline underline-offset-4 hover:text-primary/80"
                                >
                                    Upload at least one
                                    <ArrowRight className="h-3.5 w-3.5" />
                                </Link>
                            </div>
                        )}
                    </div>
                    <div className="rounded-lg border p-3">
                        <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                            <Activity className="h-3.5 w-3.5 text-emerald-500" />
                            Completed evaluations
                        </div>
                        <div className="mt-0.5 flex items-baseline gap-1.5">
                            <span className="text-lg font-bold">
                                {completedRuns.length >= 2
                                    ? completedRuns.length
                                    : `${completedRuns.length}/2`}
                            </span>
                            <span className="text-sm text-muted-foreground">
                                {completedRuns.length >= 2 ? 'available' : 'complete'}
                            </span>
                        </div>
                    </div>
                </div>
            </div>

            {models && models.length >= 2 && metadataResponse?.available && (() => {
                const currentModel = models.find((m) => m.id === selectedModelId) ?? models[0];
                const otherModel = models.find((m) => m.id !== currentModel.id) ?? models[1];
                return (
                    <div className="mt-3 flex min-h-0 flex-1 flex-col">
                        <ModelSpecsCard
                            modelAName={currentModel.name}
                            modelBName={otherModel.name}
                            metaA={findModelMetadata(metadataResponse?.models, currentModel.name)}
                            metaB={findModelMetadata(metadataResponse?.models, otherModel.name)}
                        />
                    </div>
                );
            })()}

        </div>
    );
}
