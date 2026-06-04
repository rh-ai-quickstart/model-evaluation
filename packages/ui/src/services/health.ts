
import { ReadinessSchema, type Readiness } from '../schemas/health';

export const getReadiness = async (): Promise<Readiness> => {
    const response = await fetch('/api/health/ready');
    if (!response.ok) {
        throw new Error('Failed to fetch health');
    }
    const data = await response.json();
    return ReadinessSchema.parse(data);
};
