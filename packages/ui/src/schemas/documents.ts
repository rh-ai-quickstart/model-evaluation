
import { z } from 'zod';

export const DocumentResponseSchema = z.object({
    id: z.number(),
    filename: z.string(),
    status: z.string(),
    chunk_count: z.number(),
    page_count: z.number().nullable().optional(),
    file_size_bytes: z.number().nullable().optional(),
    error_message: z.string().nullable().optional(),
    created_at: z.string().nullable().optional(),
});

export const DocumentUploadResponseSchema = z.object({
    document_id: z.number(),
    filename: z.string(),
    status: z.string(),
    message: z.string(),
    embedding_error: z.string().nullable().optional(),
});

export const DocumentStatusResponseSchema = z.object({
    document_id: z.number(),
    filename: z.string(),
    status: z.string(),
    chunk_count: z.number(),
    page_count: z.number().nullable().optional(),
    error_message: z.string().nullable().optional(),
});

export type DocumentResponse = z.infer<typeof DocumentResponseSchema>;
export type DocumentUploadResponse = z.infer<typeof DocumentUploadResponseSchema>;
export type DocumentStatusResponse = z.infer<typeof DocumentStatusResponseSchema>;
