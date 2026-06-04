
import { useQuery, UseQueryResult } from '@tanstack/react-query';
import { getReadiness } from '../services/health';
import type { Readiness } from '../schemas/health';

export const useHealth = (): UseQueryResult<Readiness, Error> => {
    return useQuery({
        queryKey: ['health'],
        queryFn: getReadiness,
        // Avoid treating health as stale on every navigation; still refreshed on interval below.
        staleTime: 25_000,
        refetchInterval: 30_000,
    });
};
