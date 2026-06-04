
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { waitFor } from '@testing-library/react';
import type { DocumentResponse } from '../../schemas/documents';

const mockDocuments: DocumentResponse[] = [
    {
        id: 1,
        filename: 'test-doc.pdf',
        status: 'ready',
        chunk_count: 12,
        page_count: 3,
        file_size_bytes: 1024 * 500,
        error_message: null,
        created_at: '2026-03-20T10:00:00Z',
    },
    {
        id: 2,
        filename: 'error-doc.pdf',
        status: 'error',
        chunk_count: 0,
        page_count: null,
        file_size_bytes: 2048,
        error_message: 'PDF extraction failed',
        created_at: '2026-03-21T10:00:00Z',
    },
];

vi.mock('../../services/documents', () => ({
    listDocuments: vi.fn(),
    getDocument: vi.fn(),
    uploadDocument: vi.fn(),
    ingestFromUrl: vi.fn(),
    deleteDocument: vi.fn(),
}));

import {
    listDocuments,
    uploadDocument,
    ingestFromUrl,
    deleteDocument,
} from '../../services/documents';

describe('Documents hooks', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('should list documents via useDocuments hook', async () => {
        vi.mocked(listDocuments).mockResolvedValue(mockDocuments);

        const { renderHook } = await import('@testing-library/react');
        const { useDocuments } = await import('../../hooks/documents');
        const { createTestQueryClient } = await import('../../test/test-utils');
        const { QueryClientProvider } = await import('@tanstack/react-query');
        const { createElement } = await import('react');

        const queryClient = createTestQueryClient();
        const wrapper = ({ children }: { children: React.ReactNode }) =>
            createElement(QueryClientProvider, { client: queryClient }, children);

        const { result } = renderHook(() => useDocuments(), { wrapper });

        await waitFor(() => expect(result.current.isSuccess).toBe(true));

        expect(result.current.data).toHaveLength(2);
        expect(result.current.data![0].filename).toBe('test-doc.pdf');
        expect(listDocuments).toHaveBeenCalledOnce();
    });

    it('should upload a document via useUploadDocument hook', async () => {
        const uploadResponse = {
            document_id: 3,
            filename: 'new-doc.pdf',
            status: 'ready',
            message: 'Extracted 5 chunks from 2 pages (with embeddings)',
            embedding_error: null,
        };
        vi.mocked(uploadDocument).mockResolvedValue(uploadResponse);
        vi.mocked(listDocuments).mockResolvedValue(mockDocuments);

        const { renderHook } = await import('@testing-library/react');
        const { useUploadDocument } = await import('../../hooks/documents');
        const { createTestQueryClient } = await import('../../test/test-utils');
        const { QueryClientProvider } = await import('@tanstack/react-query');
        const { createElement } = await import('react');

        const queryClient = createTestQueryClient();
        const wrapper = ({ children }: { children: React.ReactNode }) =>
            createElement(QueryClientProvider, { client: queryClient }, children);

        const { result } = renderHook(() => useUploadDocument(), { wrapper });

        const file = new File(['pdf content'], 'new-doc.pdf', { type: 'application/pdf' });
        result.current.mutate(file);

        await waitFor(() => expect(result.current.isSuccess).toBe(true));

        expect(uploadDocument).toHaveBeenCalledWith(file);
        expect(result.current.data).toEqual(uploadResponse);
    });

    it('should delete a document via useDeleteDocument hook', async () => {
        vi.mocked(deleteDocument).mockResolvedValue(undefined);
        vi.mocked(listDocuments).mockResolvedValue([mockDocuments[1]]);

        const { renderHook } = await import('@testing-library/react');
        const { useDeleteDocument } = await import('../../hooks/documents');
        const { createTestQueryClient } = await import('../../test/test-utils');
        const { QueryClientProvider } = await import('@tanstack/react-query');
        const { createElement } = await import('react');

        const queryClient = createTestQueryClient();
        const wrapper = ({ children }: { children: React.ReactNode }) =>
            createElement(QueryClientProvider, { client: queryClient }, children);

        const { result } = renderHook(() => useDeleteDocument(), { wrapper });

        result.current.mutate(1);

        await waitFor(() => expect(result.current.isSuccess).toBe(true));

        expect(deleteDocument).toHaveBeenCalledWith(1);
    });

    it('should ingest a document from URL via useIngestFromUrl hook', async () => {
        const ingestResponse = {
            document_id: 4,
            filename: 'report.pdf',
            status: 'ready',
            message: 'Extracted 8 chunks from 4 pages (with embeddings)',
            embedding_error: null,
        };
        vi.mocked(ingestFromUrl).mockResolvedValue(ingestResponse);
        vi.mocked(listDocuments).mockResolvedValue(mockDocuments);

        const { renderHook } = await import('@testing-library/react');
        const { useIngestFromUrl } = await import('../../hooks/documents');
        const { createTestQueryClient } = await import('../../test/test-utils');
        const { QueryClientProvider } = await import('@tanstack/react-query');
        const { createElement } = await import('react');

        const queryClient = createTestQueryClient();
        const wrapper = ({ children }: { children: React.ReactNode }) =>
            createElement(QueryClientProvider, { client: queryClient }, children);

        const { result } = renderHook(() => useIngestFromUrl(), { wrapper });

        result.current.mutate('https://example.com/report.pdf');

        await waitFor(() => expect(result.current.isSuccess).toBe(true));

        expect(ingestFromUrl).toHaveBeenCalledWith('https://example.com/report.pdf');
        expect(result.current.data).toEqual(ingestResponse);
    });

    it('should handle URL ingestion error', async () => {
        vi.mocked(ingestFromUrl).mockRejectedValue(new Error('Invalid URL: bad'));

        const { renderHook } = await import('@testing-library/react');
        const { useIngestFromUrl } = await import('../../hooks/documents');
        const { createTestQueryClient } = await import('../../test/test-utils');
        const { QueryClientProvider } = await import('@tanstack/react-query');
        const { createElement } = await import('react');

        const queryClient = createTestQueryClient();
        const wrapper = ({ children }: { children: React.ReactNode }) =>
            createElement(QueryClientProvider, { client: queryClient }, children);

        const { result } = renderHook(() => useIngestFromUrl(), { wrapper });

        result.current.mutate('bad');

        await waitFor(() => expect(result.current.isError).toBe(true));

        expect(result.current.error?.message).toBe('Invalid URL: bad');
    });

    it('should handle upload error', async () => {
        vi.mocked(uploadDocument).mockRejectedValue(new Error('File exceeds 50 MB limit'));

        const { renderHook } = await import('@testing-library/react');
        const { useUploadDocument } = await import('../../hooks/documents');
        const { createTestQueryClient } = await import('../../test/test-utils');
        const { QueryClientProvider } = await import('@tanstack/react-query');
        const { createElement } = await import('react');

        const queryClient = createTestQueryClient();
        const wrapper = ({ children }: { children: React.ReactNode }) =>
            createElement(QueryClientProvider, { client: queryClient }, children);

        const { result } = renderHook(() => useUploadDocument(), { wrapper });

        const file = new File(['x'.repeat(100)], 'big.pdf', { type: 'application/pdf' });
        result.current.mutate(file);

        await waitFor(() => expect(result.current.isError).toBe(true));

        expect(result.current.error?.message).toBe('File exceeds 50 MB limit');
    });
});
