
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { listQuestionSets, createQuestionSet, updateQuestionSet, deleteQuestionSet } from '../services/question-sets';
import type { QuestionSetItem } from '../schemas/question-set';

export function useQuestionSets() {
    return useQuery({
        queryKey: ['question-sets'],
        queryFn: listQuestionSets,
    });
}

export function useCreateQuestionSet() {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: ({
            name,
            questions,
            profileId,
        }: {
            name: string;
            questions: QuestionSetItem[];
            profileId?: string;
        }) => createQuestionSet(name, questions, profileId),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['question-sets'] });
        },
    });
}

export function useUpdateQuestionSet() {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: ({
            id,
            name,
            questions,
            profileId,
        }: {
            id: number;
            name?: string;
            questions?: QuestionSetItem[];
            profileId?: string;
        }) => updateQuestionSet(id, { name, questions, profileId }),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['question-sets'] });
        },
    });
}

export function useDeleteQuestionSet() {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: (id: number) => deleteQuestionSet(id),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['question-sets'] });
        },
    });
}
