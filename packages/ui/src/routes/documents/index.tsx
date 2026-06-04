
import { createFileRoute } from '@tanstack/react-router';
import { useRef, useState } from 'react';
import {
    useDocuments,
    useUploadDocument,
    useIngestFromUrl,
    useRetryEmbedding,
    useDeleteDocument,
    usePendingDocumentIngestions,
} from '../../hooks/documents';
import { FileText, Upload, Trash2, Loader2, AlertCircle, Link, RotateCcw } from 'lucide-react';
import { toast } from 'sonner';
import { Skeleton } from '../../components/atoms/skeleton/skeleton';
import type { DocumentResponse } from '../../schemas/documents';
import { DOC_STATUS_COLORS } from '../../lib/status-colors';
import { formatUtcDate } from '../../lib/format';

export const Route = createFileRoute('/documents/')({
    component: DocumentsPage,
});

function formatFileSize(bytes: number | null | undefined): string {
    if (bytes == null) return '--';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function StatusBadge({ status }: { status: string }) {
    return (
        <span
            className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${DOC_STATUS_COLORS[status] ?? DOC_STATUS_COLORS.processing}`}
        >
            {status}
        </span>
    );
}

function PendingIngestionRow({ label, kind }: { label: string; kind: 'file' | 'url' }) {
    return (
        <div className="flex items-center justify-between rounded-lg border border-dashed p-4">
            <div className="flex items-center gap-3">
                <Loader2 className="h-5 w-5 shrink-0 animate-spin text-muted-foreground" />
                <div className="flex min-w-0 flex-col gap-1">
                    <div className="flex flex-wrap items-center gap-2">
                        <span className="truncate font-medium" title={label}>
                            {label}
                        </span>
                        <StatusBadge status="processing" />
                    </div>
                    <span className="text-xs text-muted-foreground">
                        {kind === 'url'
                            ? 'Downloading and processing…'
                            : 'Uploading and processing…'}
                    </span>
                </div>
            </div>
        </div>
    );
}

function DocumentRow({
    doc,
    onDelete,
    onRetryEmbed,
    isDeleting,
    isRetrying,
}: {
    doc: DocumentResponse;
    onDelete: (id: number) => void;
    onRetryEmbed: (id: number) => void;
    isDeleting: boolean;
    isRetrying: boolean;
}) {
    return (
        <div className="flex items-center justify-between rounded-lg border p-4">
            <div className="flex items-center gap-3">
                <FileText className="h-5 w-5 text-muted-foreground" />
                <div className="flex flex-col gap-1">
                    <div className="flex items-center gap-2">
                        <span className="font-medium">{doc.filename}</span>
                        <StatusBadge status={doc.status} />
                    </div>
                    <span className="text-xs text-muted-foreground">
                        {doc.chunk_count} chunks
                        {doc.page_count != null && ` -- ${doc.page_count} pages`}
                        {' -- '}
                        {formatFileSize(doc.file_size_bytes)}
                        {doc.created_at &&
                            ` -- ${formatUtcDate(doc.created_at, 'date')}`}
                    </span>
                    {doc.error_message && (
                        <span className="flex items-center gap-1 text-xs text-destructive">
                            <AlertCircle className="h-3 w-3" />
                            {doc.error_message}
                        </span>
                    )}
                </div>
            </div>
            <div className="flex items-center gap-1">
                {(doc.status === 'embedding_failed' ||
                    (doc.error_message && doc.error_message.includes('missing embeddings'))) && (
                    <button
                        onClick={() => onRetryEmbed(doc.id)}
                        disabled={isRetrying}
                        className="rounded-lg p-2 text-muted-foreground transition-colors hover:bg-primary/10 hover:text-primary disabled:opacity-50"
                        title="Retry embedding generation"
                    >
                        {isRetrying ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                        ) : (
                            <RotateCcw className="h-4 w-4" />
                        )}
                    </button>
                )}
                <button
                    onClick={() => onDelete(doc.id)}
                    disabled={isDeleting}
                    className="rounded-lg p-2 text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive disabled:opacity-50"
                    title="Delete document"
                >
                    {isDeleting ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                        <Trash2 className="h-4 w-4" />
                    )}
                </button>
            </div>
        </div>
    );
}

function UploadForm({ onUploaded }: { onUploaded: () => void }) {
    const uploadMutation = useUploadDocument();
    const urlMutation = useIngestFromUrl();
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [dragOver, setDragOver] = useState(false);
    const [tab, setTab] = useState<'file' | 'url'>('file');
    const [url, setUrl] = useState('');

    const activeMutation = tab === 'file' ? uploadMutation : urlMutation;

    const handleFile = (file: File) => {
        if (!file.name.toLowerCase().endsWith('.pdf')) {
            return;
        }
        uploadMutation.mutate(file, {
            onSuccess: (data) => {
                onUploaded();
                if (fileInputRef.current) fileInputRef.current.value = '';
                toast.success(data.message);
                if (data.embedding_error) {
                    toast.warning(data.embedding_error);
                }
            },
            onError: (err) => {
                toast.error(err.message);
            },
        });
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        setDragOver(false);
        const file = e.dataTransfer.files[0];
        if (file) handleFile(file);
    };

    const handleUrlSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (!url.trim()) return;
        urlMutation.mutate(url.trim(), {
            onSuccess: (data) => {
                onUploaded();
                setUrl('');
                toast.success(data.message);
                if (data.embedding_error) {
                    toast.warning(data.embedding_error);
                }
            },
            onError: (err) => {
                toast.error(err.message);
            },
        });
    };

    return (
        <div className="rounded-xl border bg-card p-6">
            <div className="mb-4 flex items-center gap-1 rounded-lg bg-muted p-1">
                <button
                    onClick={() => setTab('file')}
                    className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
                        tab === 'file'
                            ? 'bg-background text-foreground shadow-sm'
                            : 'text-muted-foreground hover:text-foreground'
                    }`}
                >
                    <Upload className="h-3.5 w-3.5" />
                    Upload File
                </button>
                <button
                    onClick={() => setTab('url')}
                    className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
                        tab === 'url'
                            ? 'bg-background text-foreground shadow-sm'
                            : 'text-muted-foreground hover:text-foreground'
                    }`}
                >
                    <Link className="h-3.5 w-3.5" />
                    From URL
                </button>
            </div>

            {tab === 'file' && (
                <div
                    onDragOver={(e) => {
                        e.preventDefault();
                        setDragOver(true);
                    }}
                    onDragLeave={() => setDragOver(false)}
                    onDrop={handleDrop}
                    className={`flex flex-col items-center justify-center rounded-lg border-2 border-dashed p-8 transition-colors ${
                        dragOver
                            ? 'border-primary bg-primary/5'
                            : 'border-muted-foreground/25'
                    }`}
                >
                    <Upload className="mb-2 h-8 w-8 text-muted-foreground" />
                    <p className="mb-1 text-sm text-muted-foreground">
                        Drag and drop a PDF file here, or
                    </p>
                    <label className="cursor-pointer rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90">
                        Browse files
                        <input
                            ref={fileInputRef}
                            type="file"
                            accept=".pdf"
                            className="hidden"
                            onChange={(e) => {
                                const file = e.target.files?.[0];
                                if (file) handleFile(file);
                            }}
                        />
                    </label>
                    <p className="mt-2 text-xs text-muted-foreground">PDF files only, max 50 MB</p>
                </div>
            )}

            {tab === 'url' && (
                <form onSubmit={handleUrlSubmit} className="space-y-3">
                    <div>
                        <label
                            htmlFor="pdf-url"
                            className="mb-1 block text-sm font-medium text-foreground"
                        >
                            PDF URL
                        </label>
                        <input
                            id="pdf-url"
                            type="url"
                            value={url}
                            onChange={(e) => setUrl(e.target.value)}
                            placeholder="https://example.com/document.pdf"
                            disabled={urlMutation.isPending}
                            className="w-full rounded-lg border bg-background px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary disabled:opacity-50"
                        />
                        <p className="mt-1 text-xs text-muted-foreground">
                            Enter a direct link to a PDF file (max 50 MB)
                        </p>
                    </div>
                    <button
                        type="submit"
                        disabled={!url.trim() || urlMutation.isPending}
                        className="flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90 disabled:opacity-50"
                    >
                        {urlMutation.isPending ? (
                            <>
                                <Loader2 className="h-4 w-4 animate-spin" />
                                Downloading and processing...
                            </>
                        ) : (
                            <>
                                <Link className="h-4 w-4" />
                                Ingest from URL
                            </>
                        )}
                    </button>
                </form>
            )}

            {activeMutation.isPending && tab === 'file' && (
                <div className="mt-3 flex items-center gap-2 text-sm text-muted-foreground">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Uploading and processing...
                </div>
            )}

        </div>
    );
}

function DocumentsPage() {
    const { data: documents, isLoading, error, refetch } = useDocuments();
    const pendingIngestions = usePendingDocumentIngestions();
    const deleteMutation = useDeleteDocument();
    const retryMutation = useRetryEmbedding();
    const [showUpload, setShowUpload] = useState(false);
    const [deletingId, setDeletingId] = useState<number | null>(null);
    const [retryingId, setRetryingId] = useState<number | null>(null);

    const handleDelete = (id: number) => {
        setDeletingId(id);
        deleteMutation.mutate(id, {
            onSuccess: () => toast.success('Document deleted'),
            onError: (err) => toast.error(err.message),
            onSettled: () => setDeletingId(null),
        });
    };

    const handleRetryEmbed = (id: number) => {
        setRetryingId(id);
        retryMutation.mutate(id, {
            onSuccess: () => toast.success('Embedding retry started'),
            onError: (err) => toast.error(err.message),
            onSettled: () => setRetryingId(null),
        });
    };

    return (
        <div className="p-4 sm:p-6 lg:p-8">
            <div className="mx-auto max-w-5xl">
                <div className="mb-6 flex items-center justify-between">
                    <div>
                        <h1 className="text-2xl font-bold tracking-tight">Documents</h1>
                        <p className="text-sm text-muted-foreground">
                            Upload and manage PDF documents for RAG evaluation
                        </p>
                    </div>
                    <button
                        onClick={() => setShowUpload(!showUpload)}
                        className="flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90"
                    >
                        <Upload className="h-4 w-4" />
                        Upload
                    </button>
                </div>

                {showUpload && (
                    <div className="mb-6">
                        <UploadForm
                            onUploaded={() => {
                                refetch();
                            }}
                        />
                    </div>
                )}

                {isLoading && (
                    <div className="space-y-3">
                        {Array.from({ length: 3 }).map((_, i) => (
                            <div key={i} className="flex items-center justify-between rounded-lg border p-4">
                                <div className="flex items-center gap-3">
                                    <Skeleton className="h-5 w-5 rounded" />
                                    <div className="flex flex-col gap-1">
                                        <Skeleton className="h-4 w-48" />
                                        <Skeleton className="h-3 w-32" />
                                    </div>
                                </div>
                                <Skeleton className="h-8 w-8 rounded-lg" />
                            </div>
                        ))}
                    </div>
                )}
                {error && <p className="text-sm text-destructive">{error.message}</p>}

                {documents &&
                    documents.length === 0 &&
                    pendingIngestions.length === 0 &&
                    !showUpload && (
                        <div className="rounded-xl border bg-card p-8 text-center">
                            <FileText className="mx-auto mb-3 h-8 w-8 text-muted-foreground" />
                            <p className="text-sm text-muted-foreground">
                                No documents yet. Click &quot;Upload&quot; to add a PDF.
                            </p>
                        </div>
                    )}

                {(pendingIngestions.length > 0 || (documents && documents.length > 0)) && (
                    <div className="space-y-3">
                        {pendingIngestions.map((p) => (
                            <PendingIngestionRow key={p.rowKey} label={p.label} kind={p.kind} />
                        ))}
                        {documents?.map((doc) => (
                            <DocumentRow
                                key={doc.id}
                                doc={doc}
                                onDelete={handleDelete}
                                onRetryEmbed={handleRetryEmbed}
                                isDeleting={deletingId === doc.id}
                                isRetrying={retryingId === doc.id}
                            />
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}
