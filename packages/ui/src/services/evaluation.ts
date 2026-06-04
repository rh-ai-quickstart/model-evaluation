
import {
    ComparisonResponseSchema,
    EvalRunCreateResponseSchema,
    EvalRunDetailSchema,
    EvalRunSchema,
    SynthesizeResponseSchema,
    type ComparisonResponse,
    type EvalRun,
    type EvalRunCreateResponse,
    type EvalRunDetail,
    type SynthesizeResponse,
} from '../schemas/evaluation';
import { z } from 'zod';

export interface EvalQuestionInput {
    question: string;
    expected_answer?: string | null;
    truth?: Record<string, unknown> | null;
}

export async function createEvalRun(
    modelName: string,
    questions: EvalQuestionInput[],
    questionSetId?: number,
    profileId?: string,
): Promise<EvalRunCreateResponse> {
    const response = await fetch('/api/evaluations/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            model_name: modelName,
            questions,
            question_set_id: questionSetId ?? null,
            profile_id: profileId ?? null,
        }),
    });
    if (!response.ok) throw new Error('Failed to create evaluation run');
    const data = await response.json();
    return EvalRunCreateResponseSchema.parse(data);
}

export async function listEvalRuns(): Promise<EvalRun[]> {
    const response = await fetch('/api/evaluations/');
    if (!response.ok) throw new Error('Failed to fetch evaluation runs');
    const data = await response.json();
    return z.array(EvalRunSchema).parse(data);
}

export async function getEvalRun(id: number): Promise<EvalRunDetail> {
    const response = await fetch(`/api/evaluations/${id}`);
    if (!response.ok) throw new Error('Failed to fetch evaluation run');
    const data = await response.json();
    return EvalRunDetailSchema.parse(data);
}

export async function rerunEval(
    evalRunId: number,
    modelName: string,
): Promise<EvalRunCreateResponse> {
    const response = await fetch(`/api/evaluations/${evalRunId}/rerun`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_name: modelName }),
    });
    if (!response.ok) throw new Error('Failed to rerun evaluation');
    const data = await response.json();
    return EvalRunCreateResponseSchema.parse(data);
}

export async function compareEvalRuns(
    runAId: number,
    runBId: number,
): Promise<ComparisonResponse> {
    const response = await fetch(
        `/api/evaluations/compare?run_a_id=${runAId}&run_b_id=${runBId}`,
    );
    if (!response.ok) throw new Error('Failed to compare evaluation runs');
    const data = await response.json();
    return ComparisonResponseSchema.parse(data);
}

export async function cancelEvalRun(id: number): Promise<{ message: string }> {
    const response = await fetch(`/api/evaluations/${id}/cancel`, { method: 'POST' });
    if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        throw new Error(errorData?.detail ?? 'Failed to cancel evaluation run');
    }
    return response.json();
}

export async function deleteEvalRun(id: number): Promise<void> {
    const response = await fetch(`/api/evaluations/${id}`, { method: 'DELETE' });
    if (!response.ok) throw new Error('Failed to delete evaluation run');
}

export interface EvalProfile {
    id: string;
    version: string;
    domain: string;
    description: string;
}

export async function listProfiles(): Promise<EvalProfile[]> {
    const response = await fetch('/api/evaluations/profiles');
    if (!response.ok) throw new Error('Failed to fetch profiles');
    return response.json();
}

export async function synthesizeQuestions(
    maxQuestions: number = 10,
    documentIds?: number[],
): Promise<SynthesizeResponse> {
    const response = await fetch('/api/evaluations/synthesize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            max_questions: maxQuestions,
            document_ids: documentIds ?? null,
        }),
    });
    if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        throw new Error(errorData?.detail ?? 'Failed to synthesize questions');
    }
    const data = await response.json();
    return SynthesizeResponseSchema.parse(data);
}
