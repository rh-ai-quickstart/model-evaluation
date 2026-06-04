
export function formatScore(val: number | null | undefined): string {
    if (val == null) return '--';
    return (val * 100).toFixed(0) + '%';
}

export function formatLatency(val: number | null | undefined): string {
    if (val == null) return '--';
    return val.toFixed(0) + 'ms';
}

export function formatMetricValue(metric: string, val: number | null | undefined): string {
    if (val == null) return '--';
    if (metric === 'latency_ms') return val.toFixed(0) + 'ms';
    return (val * 100).toFixed(0) + '%';
}

export function formatContextLength(val: number | null | undefined): string {
    if (val == null || val === 0) return '--';
    if (val >= 1_000_000) return `${(val / 1_000_000).toFixed(1)}M tokens`;
    if (val >= 1_000) return `${(val / 1_000).toFixed(0)}K tokens`;
    return `${val} tokens`;
}

export function formatTokenPrice(val: number | null | undefined): string {
    if (val == null) return '--';
    const perMillion = val * 1_000_000;
    return `$${perMillion.toFixed(2)}/1M`;
}

export function formatUtcDate(val: string | null | undefined, style: 'date' | 'datetime' = 'datetime'): string {
    if (!val) return '';
    // API returns UTC timestamps without timezone suffix; append Z so
    // the browser interprets them correctly in the user's local timezone.
    const iso = val.endsWith('Z') || val.includes('+') ? val : val + 'Z';
    const d = new Date(iso);
    return style === 'date' ? d.toLocaleDateString() : d.toLocaleString();
}
