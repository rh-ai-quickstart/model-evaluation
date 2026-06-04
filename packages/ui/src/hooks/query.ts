
import { useMutation } from '@tanstack/react-query';
import { submitQuery } from '../services/query';
import type { QueryRequest } from '../schemas/query';

export function useSubmitQuery() {
    return useMutation({
        mutationFn: (request: QueryRequest) => submitQuery(request),
    });
}
