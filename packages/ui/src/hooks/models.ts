
import { useQuery } from '@tanstack/react-query';
import { getModels, getModelStatus, getModelMetadata } from '../services/models';

export function useModels() {
    return useQuery({
        queryKey: ['models'],
        queryFn: getModels,
    });
}

export function useModelStatus(modelId: number) {
    return useQuery({
        queryKey: ['model-status', modelId],
        queryFn: () => getModelStatus(modelId),
        refetchInterval: 30_000,
    });
}

export function useModelMetadata() {
    return useQuery({
        queryKey: ['model-metadata'],
        queryFn: getModelMetadata,
        staleTime: 5 * 60 * 1000,
        gcTime: 10 * 60 * 1000,
        refetchOnWindowFocus: false,
    });
}
