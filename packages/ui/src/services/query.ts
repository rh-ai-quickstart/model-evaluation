
import { QueryResponseSchema, type QueryRequest, type QueryResponse } from '../schemas/query';

export const submitQuery = async (request: QueryRequest): Promise<QueryResponse> => {
    const response = await fetch('/api/query/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
    });
    if (!response.ok) {
        throw new Error('Failed to submit query');
    }
    const data = await response.json();
    return QueryResponseSchema.parse(data);
};
