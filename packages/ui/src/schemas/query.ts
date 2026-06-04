
import { z } from 'zod';

export const SourceChunkSchema = z.object({
    id: z.number(),
    text: z.string(),
    source_document: z.string(),
    page_number: z.string().nullable(),
    score: z.number(),
});

export const UsageInfoSchema = z.object({
    prompt_tokens: z.number().nullable(),
    completion_tokens: z.number().nullable(),
    total_tokens: z.number().nullable(),
});

export const QueryResponseSchema = z.object({
    answer: z.string(),
    model: z.string(),
    sources: z.array(SourceChunkSchema),
    usage: UsageInfoSchema.nullable(),
    low_confidence: z.boolean(),
});

export type SourceChunk = z.infer<typeof SourceChunkSchema>;
export type UsageInfo = z.infer<typeof UsageInfoSchema>;
export type QueryResponse = z.infer<typeof QueryResponseSchema>;

export interface QueryRequest {
    question: string;
    model_name: string;
    top_k?: number;
}
