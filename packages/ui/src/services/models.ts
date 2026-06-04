
import {
    ModelSchema,
    ModelStatusSchema,
    ModelMetadataResponseSchema,
    type Model,
    type ModelStatus,
    type ModelMetadataResponse,
} from '../schemas/models';
import { z } from 'zod';

export const getModels = async (): Promise<Model[]> => {
    const response = await fetch('/api/models/');
    if (!response.ok) {
        throw new Error('Failed to fetch models');
    }
    const data = await response.json();
    return z.array(ModelSchema).parse(data);
};

export const getModelStatus = async (modelId: number): Promise<ModelStatus> => {
    const response = await fetch(`/api/models/${modelId}/status`);
    if (!response.ok) {
        throw new Error('Failed to fetch model status');
    }
    const data = await response.json();
    return ModelStatusSchema.parse(data);
};

export const getModelMetadata = async (): Promise<ModelMetadataResponse> => {
    const response = await fetch('/api/models/metadata');
    if (!response.ok) {
        return { models: [], available: false };
    }
    const data = await response.json();
    return ModelMetadataResponseSchema.parse(data);
};
