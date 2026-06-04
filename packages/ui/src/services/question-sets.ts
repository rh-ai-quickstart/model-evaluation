
import {
    QuestionSetSchema,
    type QuestionSet,
    type QuestionSetItem,
} from '../schemas/question-set';
import { z } from 'zod';

export async function createQuestionSet(
    name: string,
    questions: QuestionSetItem[],
    profileId?: string,
): Promise<QuestionSet> {
    const body: Record<string, unknown> = { name, questions };
    if (profileId) body.profile_id = profileId;
    const response = await fetch('/api/question-sets/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
    if (!response.ok) {
        const err = await response.json().catch(() => null);
        throw new Error(err?.detail ?? `Failed to create question set (${response.status})`);
    }
    const data = await response.json();
    return QuestionSetSchema.parse(data);
}

export async function listQuestionSets(): Promise<QuestionSet[]> {
    const response = await fetch('/api/question-sets/');
    if (!response.ok) throw new Error('Failed to fetch question sets');
    const data = await response.json();
    return z.array(QuestionSetSchema).parse(data);
}

export async function updateQuestionSet(
    id: number,
    data: { name?: string; questions?: QuestionSetItem[]; profileId?: string },
): Promise<QuestionSet> {
    const body: Record<string, unknown> = {};
    if (data.name !== undefined) body.name = data.name;
    if (data.questions !== undefined) body.questions = data.questions;
    if (data.profileId !== undefined) body.profile_id = data.profileId;
    const response = await fetch(`/api/question-sets/${id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
    if (!response.ok) {
        const err = await response.json().catch(() => null);
        throw new Error(err?.detail ?? `Failed to update question set (${response.status})`);
    }
    const json = await response.json();
    return QuestionSetSchema.parse(json);
}

export async function deleteQuestionSet(id: number): Promise<void> {
    const response = await fetch(`/api/question-sets/${id}`, { method: 'DELETE' });
    if (!response.ok) throw new Error('Failed to delete question set');
}
