
import { SquarePen } from 'lucide-react';
import { ModelSelector } from '../model-selector/model-selector';
import type { Model } from '../../schemas/models';

interface ChatHeaderProps {
    selectedModelId: number | null;
    onSelectModel: (model: Model) => void;
    onNewChat: () => void;
}

export function ChatHeader({ selectedModelId, onSelectModel, onNewChat }: ChatHeaderProps) {
    return (
        <div className="border-b bg-card px-4 py-3">
            <div className="flex items-end gap-3">
                <div className="flex-1">
                    <ModelSelector
                        selectedModelId={selectedModelId}
                        onSelect={onSelectModel}
                        label="Chat Model"
                    />
                </div>
                <button
                    onClick={onNewChat}
                    className="mb-0.5 inline-flex items-center gap-1.5 rounded-lg border px-3 py-2 text-sm font-medium transition-colors hover:bg-accent"
                >
                    <SquarePen className="h-4 w-4" />
                    New Chat
                </button>
            </div>
        </div>
    );
}
