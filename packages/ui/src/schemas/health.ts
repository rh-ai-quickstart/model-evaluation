
import { z } from 'zod';

export const ReadinessSchema = z.object({
    status: z.enum(['ready', 'degraded', 'not_ready']),
    service: z.string(),
    timestamp: z.string(),
    dependencies: z.record(z.string()),
    message: z.string().nullable().optional(),
});

export type Readiness = z.infer<typeof ReadinessSchema>;
