
import { z } from 'zod';

export const QuestionSetItemSchema = z.object({
    question: z.string(),
    expected_answer: z.string().nullable().optional(),
    truth: z.record(z.unknown()).nullable().optional(),
});

export const QuestionSetSchema = z.object({
    id: z.number(),
    name: z.string(),
    questions: z.array(QuestionSetItemSchema),
    created_at: z.string().nullable().optional(),
    updated_at: z.string().nullable().optional(),
});

export type QuestionSetItem = z.infer<typeof QuestionSetItemSchema>;

export type QuestionSet = z.infer<typeof QuestionSetSchema>;
