
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
    listEvalRuns,
    getEvalRun,
    createEvalRun,
    cancelEvalRun,
    deleteEvalRun,
    rerunEval,
    compareEvalRuns,
    synthesizeQuestions,
    listProfiles,
    type EvalQuestionInput,
} from '../services/evaluation';
import type { EvalRun } from '../schemas/evaluation';

export function useEvalRuns() {
    return useQuery({
        queryKey: ['eval-runs'],
        queryFn: listEvalRuns,
        refetchInterval: (query) => {
            const runs = query.state.data;
            const hasActive = runs?.some((r) => r.status === 'pending' || r.status === 'running');
            return hasActive ? 3000 : false;
        },
    });
}

export function useEvalRun(id: number) {
    return useQuery({
        queryKey: ['eval-run', id],
        queryFn: () => getEvalRun(id),
        refetchInterval: (query) => {
            const status = query.state.data?.status;
            return status === 'pending' || status === 'running' ? 3000 : false;
        },
    });
}

export function useCreateEvalRun() {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: ({
            modelName,
            questions,
            questionSetId,
            profileId,
        }: {
            modelName: string;
            questions: EvalQuestionInput[];
            questionSetId?: number;
            profileId?: string;
        }) => createEvalRun(modelName, questions, questionSetId, profileId),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['eval-runs'] });
        },
    });
}

export function useProfiles() {
    return useQuery({
        queryKey: ['eval-profiles'],
        queryFn: listProfiles,
    });
}

export function useCancelEvalRun() {
    const queryClient = useQueryClient();

    const pollForResults = (cancelledId: number, attempt = 0) => {
        if (attempt >= 12) return; // Stop after ~60s
        setTimeout(async () => {
            const fresh = await listEvalRuns();
            const serverRun = fresh.find((r) => r.id === cancelledId);
            queryClient.setQueryData<EvalRun[]>(
                ['eval-runs'],
                fresh.map((run) =>
                    run.id === cancelledId ? { ...run, status: 'cancelled' } : run,
                ),
            );
            // Keep polling if server still says running (task hasn't finished yet)
            if (serverRun && (serverRun.status === 'running' || serverRun.status === 'pending')) {
                pollForResults(cancelledId, attempt + 1);
            }
        }, 5000);
    };

    return useMutation({
        mutationFn: (id: number) => cancelEvalRun(id),
        onSuccess: (_data, id) => {
            // Optimistically mark the run as cancelled in the cache immediately
            queryClient.setQueryData<EvalRun[]>(['eval-runs'], (old) =>
                old?.map((run) => (run.id === id ? { ...run, status: 'cancelled' } : run)),
            );
            // Poll every 5s until the backend finishes the current question
            // and reports the final cancelled status with partial aggregates
            pollForResults(id);
        },
    });
}

export function useDeleteEvalRun() {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: (id: number) => deleteEvalRun(id),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['eval-runs'] });
        },
    });
}

export function useRerunEval() {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: ({ evalRunId, modelName }: { evalRunId: number; modelName: string }) =>
            rerunEval(evalRunId, modelName),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['eval-runs'] });
        },
    });
}

export function useCompareEvalRuns(runAId: number, runBId: number) {
    return useQuery({
        queryKey: ['eval-compare', runAId, runBId],
        queryFn: () => compareEvalRuns(runAId, runBId),
        enabled: runAId > 0 && runBId > 0,
    });
}

export function useSynthesizeQuestions() {
    return useMutation({
        mutationFn: ({
            maxQuestions,
            documentIds,
        }: {
            maxQuestions?: number;
            documentIds?: number[];
        }) => synthesizeQuestions(maxQuestions, documentIds),
    });
}
