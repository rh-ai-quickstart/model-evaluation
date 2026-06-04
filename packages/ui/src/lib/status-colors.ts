
export const EVAL_STATUS_COLORS: Record<string, string> = {
    pending: 'bg-slate-100 text-slate-700 dark:bg-slate-900/60 dark:text-slate-300',
    running: 'bg-blue-100 text-blue-700 dark:bg-blue-950/40 dark:text-blue-300',
    completed: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-300',
    failed: 'bg-rose-100 text-rose-700 dark:bg-rose-950/40 dark:text-rose-300',
    cancelled: 'bg-amber-100 text-amber-700 dark:bg-amber-950/40 dark:text-amber-300',
};

export const HEALTH_STATUS_COLORS: Record<string, { label: string; classes: string }> = {
    healthy: { label: 'Healthy', classes: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-300' },
    unhealthy: { label: 'Unhealthy', classes: 'bg-rose-100 text-rose-700 dark:bg-rose-950/40 dark:text-rose-300' },
    unknown: { label: 'Unknown', classes: 'bg-slate-100 text-slate-700 dark:bg-slate-900/60 dark:text-slate-300' },
};

export const DOC_STATUS_COLORS: Record<string, string> = {
    processing: 'bg-blue-100 text-blue-700 dark:bg-blue-950/40 dark:text-blue-300',
    ready: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-300',
    embedding_failed: 'bg-amber-100 text-amber-700 dark:bg-amber-950/40 dark:text-amber-300',
    error: 'bg-rose-100 text-rose-700 dark:bg-rose-950/40 dark:text-rose-300',
};

export const MODEL_STATUS_COLORS: Record<string, string> = {
    available: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-300',
    unavailable: 'bg-rose-100 text-rose-700 dark:bg-rose-950/40 dark:text-rose-300',
};
