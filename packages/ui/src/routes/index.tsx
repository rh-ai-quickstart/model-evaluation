
import { createFileRoute } from '@tanstack/react-router';
import { useState } from 'react';
import { OverviewPanel } from '../components/dashboard/overview-panel';
import { ChatPanel } from '../components/chat-panel/chat-panel';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const Route = createFileRoute('/' as any)({
    component: Index,
});

function Index() {
    const [selectedModelId, setSelectedModelId] = useState<number | null>(null);

    return (
        <div className="grid h-[calc(100vh-128px)] grid-cols-1 lg:grid-cols-[3fr_2fr]">
            <div className="border-r overflow-y-auto">
                <OverviewPanel selectedModelId={selectedModelId} />
            </div>
            <div className="min-h-0">
                <ChatPanel selectedModelId={selectedModelId} onSelectedModelIdChange={setSelectedModelId} />
            </div>
        </div>
    );
}
