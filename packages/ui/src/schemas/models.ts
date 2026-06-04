
import { z } from 'zod';

export const ModelSchema = z.object({
    id: z.number(),
    name: z.string(),
    endpoint_url: z.string(),
    deployment_mode: z.string(),
    is_active: z.boolean(),
});

export const ModelStatusSchema = z.object({
    name: z.string(),
    status: z.string(),
    deployment_mode: z.string(),
    endpoint_url: z.string(),
});

export const ModelPricingSchema = z.object({
    input: z.number().nullable().optional(),
    output: z.number().nullable().optional(),
    unit: z.string().nullable().optional(),
});

export const ModelMetadataSchema = z.object({
    id: z.string(),
    name: z.string(),
    context_length: z.number().nullable().optional(),
    max_tokens: z.number().nullable().optional(),
    pricing: ModelPricingSchema.nullable().optional(),
    capabilities: z.array(z.string()).default([]),
    tpm: z.number().nullable().optional(),
    rpm: z.number().nullable().optional(),
    supports_vision: z.boolean().nullable().optional(),
    supports_function_calling: z.boolean().nullable().optional(),
    supports_embeddings: z.boolean().nullable().optional(),
});

export const ModelMetadataResponseSchema = z.object({
    models: z.array(ModelMetadataSchema).default([]),
    available: z.boolean().default(true),
});

export type Model = z.infer<typeof ModelSchema>;
export type ModelStatus = z.infer<typeof ModelStatusSchema>;
export type ModelPricing = z.infer<typeof ModelPricingSchema>;
export type ModelMetadata = z.infer<typeof ModelMetadataSchema>;
export type ModelMetadataResponse = z.infer<typeof ModelMetadataResponseSchema>;
