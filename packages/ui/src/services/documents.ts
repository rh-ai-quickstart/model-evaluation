
import {
    DocumentResponseSchema,
    DocumentUploadResponseSchema,
    type DocumentResponse,
    type DocumentUploadResponse,
} from '../schemas/documents';
import { z } from 'zod';

export async function listDocuments(): Promise<DocumentResponse[]> {
    const response = await fetch('/api/documents/');
    if (!response.ok) throw new Error('Failed to fetch documents');
    const data = await response.json();
    return z.array(DocumentResponseSchema).parse(data);
}

export async function getDocument(id: number): Promise<DocumentResponse> {
    const response = await fetch(`/api/documents/${id}`);
    if (!response.ok) throw new Error('Failed to fetch document');
    const data = await response.json();
    return DocumentResponseSchema.parse(data);
}

export async function uploadDocument(file: File): Promise<DocumentUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    const response = await fetch('/api/documents/upload', {
        method: 'POST',
        body: formData,
    });
    if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        throw new Error(errorData?.detail ?? 'Failed to upload document');
    }
    const data = await response.json();
    return DocumentUploadResponseSchema.parse(data);
}

export async function ingestFromUrl(url: string): Promise<DocumentUploadResponse> {
    const response = await fetch('/api/ingest/url', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url }),
    });
    if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        throw new Error(errorData?.detail ?? 'Failed to ingest document from URL');
    }
    const data = await response.json();
    return DocumentUploadResponseSchema.parse(data);
}

export async function retryEmbedding(id: number): Promise<DocumentResponse> {
    const response = await fetch(`/api/documents/${id}/embed`, {
        method: 'POST',
    });
    if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        throw new Error(errorData?.detail ?? 'Failed to generate embeddings');
    }
    const data = await response.json();
    return DocumentResponseSchema.parse(data);
}

export async function deleteDocument(id: number): Promise<void> {
    const response = await fetch(`/api/documents/${id}`, {
        method: 'DELETE',
    });
    if (!response.ok) throw new Error('Failed to delete document');
}
