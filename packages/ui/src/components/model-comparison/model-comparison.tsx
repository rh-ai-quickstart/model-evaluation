
import { Link } from '@tanstack/react-router';
import { GitCompareArrows, ArrowRight } from 'lucide-react';
import { useEvalRuns } from '../../hooks/evaluation';
import { formatScore, formatLatency } from '../../lib/format';

export function ModelComparison() {
    const { data: runs } = useEvalRuns();
    const completedRuns = runs?.filter((r) => r.status === 'completed') ?? [];

    return (
        <div className="rounded-xl border bg-card p-4">
            <div className="mb-4 flex items-center gap-2">
                <GitCompareArrows className="h-5 w-5" />
                <h2 className="text-lg font-semibold">Evaluation Results</h2>
            </div>

            {completedRuns.length === 0 ? (
                <div className="rounded-lg border border-dashed p-4 text-center">
                    <p className="text-sm text-muted-foreground">
                        No completed evaluations yet.
                    </p>
                    <Link
                        to="/evaluations"
                        className="mt-2 inline-flex items-center gap-1 text-sm font-medium text-primary hover:underline"
                    >
                        Run your first evaluation
                        <ArrowRight className="h-3.5 w-3.5" />
                    </Link>
                </div>
            ) : (
                <div className="space-y-3">
                    {completedRuns.slice(0, 3).map((run) => (
                        <Link
                            key={run.id}
                            to="/evaluations/$id"
                            params={{ id: String(run.id) }}
                            className="flex items-center justify-between rounded-lg border p-3 transition-colors hover:bg-accent"
                        >
                            <div>
                                <span className="text-sm font-medium">{run.model_name}</span>
                                <span className="ml-2 text-xs text-muted-foreground">
                                    Run #{run.id}
                                </span>
                            </div>
                            <div className="flex items-center gap-4 text-xs">
                                <div className="text-center">
                                    <div className="text-muted-foreground">Ground.</div>
                                    <div className="font-medium">{formatScore(run.avg_groundedness)}</div>
                                </div>
                                <div className="text-center">
                                    <div className="text-muted-foreground">Relev.</div>
                                    <div className="font-medium">{formatScore(run.avg_relevancy)}</div>
                                </div>
                                <div className="text-center">
                                    <div className="text-muted-foreground">Latency</div>
                                    <div className="font-medium">{formatLatency(run.avg_latency_ms)}</div>
                                </div>
                                <ArrowRight className="h-3.5 w-3.5 text-muted-foreground" />
                            </div>
                        </Link>
                    ))}

                    {completedRuns.length > 3 && (
                        <Link
                            to="/evaluations"
                            className="block text-center text-sm text-muted-foreground hover:text-foreground"
                        >
                            View all {completedRuns.length} completed evaluations
                        </Link>
                    )}

                    {completedRuns.length >= 2 && (
                        <Link
                            to="/evaluations"
                            className="mt-2 inline-flex items-center gap-1 text-sm font-medium text-primary hover:underline"
                        >
                            Compare evaluations
                            <ArrowRight className="h-3.5 w-3.5" />
                        </Link>
                    )}
                </div>
            )}
        </div>
    );
}
