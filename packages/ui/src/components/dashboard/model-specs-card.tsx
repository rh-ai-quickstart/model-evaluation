
import { Cpu, Sparkles } from 'lucide-react';
import type { ModelMetadata } from '../../schemas/models';
import { formatContextLength, formatTokenPrice } from '../../lib/format';

export function findModelMetadata(
    models: ModelMetadata[] | undefined,
    modelName: string,
): ModelMetadata | undefined {
    if (!models) return undefined;
    return (
        models.find((m) => m.id === modelName) ??
        models.find((m) => m.name === modelName) ??
        models.find((m) => m.id.toLowerCase() === modelName.toLowerCase()) ??
        models.find((m) => modelName.toLowerCase().includes(m.id.toLowerCase()))
    );
}

function computeInsights(
    metaA: ModelMetadata | undefined,
    metaB: ModelMetadata | undefined,
    nameA: string,
    nameB: string,
): string[] {
    const insights: string[] = [];

    if (metaA?.context_length && metaB?.context_length) {
        const ratio = metaA.context_length / metaB.context_length;
        if (ratio > 1.5) {
            insights.push(`${nameA} has ${Math.round(ratio)}x larger context window`);
        } else if (ratio < 1 / 1.5) {
            insights.push(`${nameB} has ${Math.round(1 / ratio)}x larger context window`);
        }
    }

    if (metaA?.pricing?.input != null && metaB?.pricing?.input != null) {
        if (metaA.pricing.input > 0 && metaB.pricing.input > 0) {
            const cheaper = metaA.pricing.input < metaB.pricing.input ? nameA : nameB;
            const ratio =
                Math.max(metaA.pricing.input, metaB.pricing.input) /
                Math.min(metaA.pricing.input, metaB.pricing.input);
            const pctCheaper = Math.round((1 - 1 / ratio) * 100);
            if (pctCheaper >= 10) {
                insights.push(`${cheaper} is ${pctCheaper}% cheaper per input token`);
            }
        }
    }

    if (metaA?.capabilities.length && metaB?.capabilities.length) {
        const aOnly = metaA.capabilities.filter((c) => !metaB!.capabilities.includes(c));
        const bOnly = metaB.capabilities.filter((c) => !metaA!.capabilities.includes(c));
        if (aOnly.length > 0) insights.push(`${nameA} uniquely supports: ${aOnly.join(', ')}`);
        if (bOnly.length > 0) insights.push(`${nameB} uniquely supports: ${bOnly.join(', ')}`);
    }

    if (metaA?.supports_vision !== metaB?.supports_vision) {
        if (metaA?.supports_vision) insights.push(`${nameA} supports vision`);
        if (metaB?.supports_vision) insights.push(`${nameB} supports vision`);
    }

    return insights;
}

function SpecRow({ label, valA, valB }: { label: string; valA: string; valB: string }) {
    return (
        <div className="grid grid-cols-3 items-center gap-2 rounded-lg border px-3 py-2 text-sm">
            <div className="text-muted-foreground">{label}</div>
            <div className="text-center font-medium">{valA}</div>
            <div className="text-center font-medium">{valB}</div>
        </div>
    );
}

interface ModelSpecsCardProps {
    modelAName: string;
    modelBName: string;
    metaA: ModelMetadata | undefined;
    metaB: ModelMetadata | undefined;
}

export function ModelSpecsCard({ modelAName, modelBName, metaA, metaB }: ModelSpecsCardProps) {
    if (!metaA && !metaB) return null;

    const insights = computeInsights(metaA, metaB, modelAName, modelBName);

    return (
        <div className="flex flex-1 flex-col rounded-xl border bg-card p-4">
            <div className="mb-2.5 flex items-center gap-2">
                <Cpu className="h-4 w-4 text-muted-foreground" />
                <h2 className="text-sm font-semibold">Your Models</h2>
            </div>

            <div className="mb-2 grid grid-cols-3 gap-2 text-xs text-muted-foreground">
                <div />
                <div className="flex flex-col items-center gap-1">
                    <span className="font-medium">{modelAName}</span>
                    <span className="rounded-full bg-blue-100 px-2 py-0.5 text-[10px] font-medium text-blue-700 dark:bg-blue-900/40 dark:text-blue-300">
                        Current
                    </span>
                </div>
                <div className="text-center font-medium">{modelBName}</div>
            </div>

            <div className="space-y-1">
                <SpecRow
                    label="Context Window"
                    valA={formatContextLength(metaA?.context_length)}
                    valB={formatContextLength(metaB?.context_length)}
                />
                <SpecRow
                    label="Input Price"
                    valA={formatTokenPrice(metaA?.pricing?.input)}
                    valB={formatTokenPrice(metaB?.pricing?.input)}
                />
                <SpecRow
                    label="Output Price"
                    valA={formatTokenPrice(metaA?.pricing?.output)}
                    valB={formatTokenPrice(metaB?.pricing?.output)}
                />
                <SpecRow
                    label="Rate Limits"
                    valA={
                        metaA?.tpm != null
                            ? `${(metaA.tpm / 1000).toFixed(0)}K TPM / ${metaA.rpm ?? '--'} RPM`
                            : '--'
                    }
                    valB={
                        metaB?.tpm != null
                            ? `${(metaB.tpm / 1000).toFixed(0)}K TPM / ${metaB.rpm ?? '--'} RPM`
                            : '--'
                    }
                />
                <div className="grid grid-cols-3 items-start gap-2 rounded-lg border px-3 py-2 text-sm">
                    <div className="text-muted-foreground">Capabilities</div>
                    <div className="flex flex-wrap justify-center gap-1">
                        {(metaA?.capabilities ?? []).map((cap) => (
                            <span
                                key={cap}
                                className="inline-flex items-center rounded-full border bg-background px-1.5 py-0.5 text-[10px]"
                            >
                                {cap}
                            </span>
                        ))}
                        {!metaA?.capabilities?.length && (
                            <span className="text-muted-foreground">--</span>
                        )}
                    </div>
                    <div className="flex flex-wrap justify-center gap-1">
                        {(metaB?.capabilities ?? []).map((cap) => (
                            <span
                                key={cap}
                                className="inline-flex items-center rounded-full border bg-background px-1.5 py-0.5 text-[10px]"
                            >
                                {cap}
                            </span>
                        ))}
                        {!metaB?.capabilities?.length && (
                            <span className="text-muted-foreground">--</span>
                        )}
                    </div>
                </div>
            </div>

            {insights.length > 0 && (
                <div className="mt-auto rounded-lg border border-blue-200 bg-blue-50/60 p-3 pt-2.5 dark:border-blue-800 dark:bg-blue-950/30">
                    <div className="mb-1.5 flex items-center gap-1.5">
                        <Sparkles className="h-3.5 w-3.5 text-blue-600 dark:text-blue-400" />
                        <h3 className="text-xs font-semibold uppercase tracking-wide text-blue-800 dark:text-blue-300">
                            Key Differences
                        </h3>
                    </div>
                    <ul className="space-y-1">
                        {insights.map((insight, i) => (
                            <li
                                key={i}
                                className="flex items-start gap-2 text-sm text-blue-900 dark:text-blue-200"
                            >
                                <span className="mt-1.5 h-1.5 w-1.5 flex-shrink-0 rounded-full bg-blue-400 dark:bg-blue-500" />
                                {insight}
                            </li>
                        ))}
                    </ul>
                </div>
            )}
        </div>
    );
}
