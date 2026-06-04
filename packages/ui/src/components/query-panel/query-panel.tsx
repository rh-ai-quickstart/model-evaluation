
import { useState } from 'react';
import { Send, FileText, AlertTriangle, Loader2 } from 'lucide-react';
import { useSubmitQuery } from '../../hooks/query';
import { ModelSelector } from '../model-selector/model-selector';
import type { Model } from '../../schemas/models';
import type { QueryResponse, SourceChunk } from '../../schemas/query';
import { cn } from '../../lib/utils';

function ConfidenceBadge({ score, isLow }: { score: number; isLow: boolean }) {
    return (
        <span
            className={cn(
                'inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-medium',
                isLow
                    ? 'bg-amber-100 text-amber-700 dark:bg-amber-950/40 dark:text-amber-300'
                    : 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-300'
            )}
        >
            {isLow && <AlertTriangle className="h-3 w-3" />}
            {(score * 100).toFixed(0)}% match
        </span>
    );
}

function SourceCard({ source, isLowConfidence }: { source: SourceChunk; isLowConfidence: boolean }) {
    const [isExpanded, setIsExpanded] = useState(false);

    return (
        <div className="rounded-lg border bg-muted/30 p-3">
            <div className="flex items-start justify-between gap-2">
                <div className="flex items-center gap-2 min-w-0">
                    <FileText className="h-4 w-4 shrink-0 text-muted-foreground" />
                    <span className="truncate text-xs font-medium">{source.source_document}</span>
                    {source.page_number && (
                        <span className="shrink-0 text-[10px] text-muted-foreground">p. {source.page_number}</span>
                    )}
                </div>
                <ConfidenceBadge score={source.score} isLow={isLowConfidence} />
            </div>
            <button
                onClick={() => setIsExpanded(!isExpanded)}
                className="mt-1.5 text-[10px] text-muted-foreground hover:text-foreground"
            >
                {isExpanded ? 'Hide context' : 'Show context'}
            </button>
            {isExpanded && (
                <p className="mt-1.5 text-xs text-muted-foreground leading-relaxed whitespace-pre-wrap">
                    {source.text.length > 500 ? source.text.slice(0, 500) + '...' : source.text}
                </p>
            )}
        </div>
    );
}

function AnswerDisplay({ response }: { response: QueryResponse }) {
    const hasLowConfidence = response.low_confidence;

    return (
        <div className="space-y-4">
            {hasLowConfidence && (
                <div className="flex items-center gap-2 rounded-lg border border-amber-200 bg-amber-50 p-3 dark:border-amber-900 dark:bg-amber-950/20">
                    <AlertTriangle className="h-4 w-4 shrink-0 text-amber-600 dark:text-amber-400" />
                    <p className="text-xs text-amber-700 dark:text-amber-300">
                        Low confidence: the retrieved context may not be relevant to your question.
                        Consider uploading more relevant documents.
                    </p>
                </div>
            )}

            <div className="rounded-lg border bg-card p-4">
                <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-medium text-muted-foreground">{response.model}</span>
                    {response.usage?.total_tokens && (
                        <span className="text-[10px] text-muted-foreground">
                            {response.usage.total_tokens} tokens
                        </span>
                    )}
                </div>
                <p className="text-sm leading-relaxed whitespace-pre-wrap">{response.answer}</p>
            </div>

            {response.sources.length > 0 && (
                <div>
                    <h4 className="mb-2 text-xs font-medium text-muted-foreground">
                        Sources ({response.sources.length})
                    </h4>
                    <div className="space-y-2">
                        {response.sources.map((source) => (
                            <SourceCard key={source.id} source={source} isLowConfidence={hasLowConfidence} />
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}

export function QueryPanel() {
    const [question, setQuestion] = useState('');
    const [selectedModel, setSelectedModel] = useState<Model | null>(null);
    const { mutate, data: response, isPending, error, reset } = useSubmitQuery();

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (!question.trim() || !selectedModel) return;

        reset();
        mutate({
            question: question.trim(),
            model_name: selectedModel.name,
        });
    };

    return (
        <div className="rounded-xl border bg-card p-4">
            <div className="mb-4 flex items-center gap-2">
                <Send className="h-5 w-5" />
                <h2 className="text-lg font-semibold">Ask a Question</h2>
            </div>
            <p className="mb-4 text-sm text-muted-foreground">
                Query your uploaded documents using RAG. Select a model and ask a question.
            </p>

            <form onSubmit={handleSubmit} className="space-y-3">
                <div className="max-w-xs">
                    <ModelSelector
                        selectedModelId={selectedModel?.id ?? null}
                        onSelect={setSelectedModel}
                        label="Model"
                    />
                </div>

                <div className="flex gap-2">
                    <input
                        type="text"
                        value={question}
                        onChange={(e) => setQuestion(e.target.value)}
                        placeholder="Ask a question about your documents..."
                        className="flex-1 rounded-lg border bg-background px-3 py-2.5 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                        disabled={isPending}
                    />
                    <button
                        type="submit"
                        disabled={isPending || !question.trim() || !selectedModel}
                        className="inline-flex items-center gap-2 rounded-lg bg-primary px-4 py-2.5 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90 disabled:opacity-50"
                    >
                        {isPending ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                        ) : (
                            <Send className="h-4 w-4" />
                        )}
                        Ask
                    </button>
                </div>
            </form>

            {error && (
                <div className="mt-4 rounded-lg border border-destructive/50 bg-destructive/10 p-3">
                    <p className="text-xs text-destructive">
                        {error instanceof Error ? error.message : 'An error occurred'}
                    </p>
                </div>
            )}

            {response && (
                <div className="mt-4">
                    <AnswerDisplay response={response} />
                </div>
            )}
        </div>
    );
}
