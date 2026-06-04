
import {
    useQuery,
    useMutation,
    useQueryClient,
    useMutationState,
    type Mutation,
} from '@tanstack/react-query';
import {
    listDocuments,
    getDocument,
    uploadDocument,
    ingestFromUrl,
    retryEmbedding,
    deleteDocument,
} from '../services/documents';

/** Mutation keys shared with `usePendingDocumentIngestions` for cross-route pending UI. */
export const DOCUMENT_INGEST_FILE_MUTATION_KEY = ['documents', 'ingest', 'file'] as const;
export const DOCUMENT_INGEST_URL_MUTATION_KEY = ['documents', 'ingest', 'url'] as const;

export type PendingDocumentIngestion = {
    rowKey: string;
    label: string;
    kind: 'file' | 'url';
};

function isPendingDocumentIngestMutation(mutation: Mutation<unknown, Error, unknown, unknown>) {
    if (mutation.state.status !== 'pending') return false;
    const key = mutation.options.mutationKey;
    return (
        Array.isArray(key) &&
        key.length >= 3 &&
        key[0] === 'documents' &&
        key[1] === 'ingest' &&
        (key[2] === 'file' || key[2] === 'url')
    );
}

/** In-flight file/URL ingestions from the global mutation cache (visible after navigating away and back). */
export function usePendingDocumentIngestions(): PendingDocumentIngestion[] {
    return useMutationState({
        filters: { predicate: isPendingDocumentIngestMutation },
        select: (mutation) => {
            const key = mutation.options.mutationKey as readonly string[];
            const kind = key[2] === 'url' ? 'url' : 'file';
            const variables = mutation.state.variables;
            let label: string;
            if (kind === 'file' && variables instanceof File) {
                label = variables.name;
            } else if (kind === 'url' && typeof variables === 'string') {
                label = variables.length > 72 ? `${variables.slice(0, 69)}…` : variables;
            } else {
                label = 'Document';
            }
            const rowKey = `${mutation.state.submittedAt ?? 0}-${kind}-${label}`;
            return { rowKey, label, kind };
        },
    });
}

export function useDocuments() {
    return useQuery({
        queryKey: ['documents'],
        queryFn: listDocuments,
        staleTime: 0,
        refetchOnMount: 'always',
        refetchInterval: (query) => {
            const docs = query.state.data;
            const hasProcessing = docs?.some((d) => d.status === 'processing');
            return hasProcessing ? 3000 : false;
        },
    });
}

export function useDocument(id: number) {
    return useQuery({
        queryKey: ['document', id],
        queryFn: () => getDocument(id),
        refetchInterval: (query) => {
            const status = query.state.data?.status;
            return status === 'processing' ? 3000 : false;
        },
    });
}

export function useUploadDocument() {
    const queryClient = useQueryClient();
    return useMutation({
        mutationKey: DOCUMENT_INGEST_FILE_MUTATION_KEY,
        mutationFn: (file: File) => uploadDocument(file),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['documents'] });
        },
    });
}

export function useIngestFromUrl() {
    const queryClient = useQueryClient();
    return useMutation({
        mutationKey: DOCUMENT_INGEST_URL_MUTATION_KEY,
        mutationFn: (url: string) => ingestFromUrl(url),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['documents'] });
        },
    });
}

export function useRetryEmbedding() {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: (id: number) => retryEmbedding(id),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['documents'] });
        },
    });
}

export function useDeleteDocument() {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: (id: number) => deleteDocument(id),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['documents'] });
        },
    });
}
