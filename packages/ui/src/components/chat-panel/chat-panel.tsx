
import { useState, useEffect, useCallback, useRef } from 'react';
import { useModels } from '../../hooks/models';
import { useSubmitQuery } from '../../hooks/query';
import { ChatHeader } from './chat-header';
import { ChatMessageList } from './chat-message-list';
import { ChatInput } from './chat-input';
import type { ChatMessageEntry } from './chat-message';
import type { Model } from '../../schemas/models';

let nextId = 1;

interface ChatPanelProps {
    selectedModelId: number | null;
    onSelectedModelIdChange: (id: number | null) => void;
}

export function ChatPanel({ selectedModelId, onSelectedModelIdChange }: ChatPanelProps) {
    const { data: models } = useModels();
    const submitQuery = useSubmitQuery();

    const [messages, setMessages] = useState<ChatMessageEntry[]>([]);
    const [inputResetKey, setInputResetKey] = useState(0);

    const selectedModel = models?.find((m) => m.id === selectedModelId);

    useEffect(() => {
        if (!selectedModelId && models && models.length > 0) {
            onSelectedModelIdChange(models[0].id);
        }
    }, [models, selectedModelId, onSelectedModelIdChange]);

    const handleNewChat = useCallback(() => {
        setMessages([]);
        setInputResetKey((k) => k + 1);
    }, []);

    const handleSend = useCallback(
        (text: string) => {
            if (!selectedModel) return;

            const userMsg: ChatMessageEntry = {
                id: `msg-${nextId++}`,
                role: 'user',
                content: text,
            };

            const loadingId = `msg-${nextId++}`;
            const loadingMsg: ChatMessageEntry = {
                id: loadingId,
                role: 'assistant',
                content: '',
                isLoading: true,
            };

            setMessages((prev) => [...prev, userMsg, loadingMsg]);

            submitQuery.mutate(
                { question: text, model_name: selectedModel.name },
                {
                    onSuccess: (data) => {
                        const assistantMsg: ChatMessageEntry = {
                            id: loadingId,
                            role: 'assistant',
                            content: data.answer,
                            lowConfidence: data.low_confidence,
                        };
                        setMessages((prev) =>
                            prev.map((m) => (m.id === loadingId ? assistantMsg : m)),
                        );
                    },
                    onError: (error) => {
                        const errorMsg: ChatMessageEntry = {
                            id: loadingId,
                            role: 'assistant',
                            content: '',
                            error: error instanceof Error ? error.message : 'Something went wrong.',
                        };
                        setMessages((prev) =>
                            prev.map((m) => (m.id === loadingId ? errorMsg : m)),
                        );
                    },
                },
            );
        },
        [selectedModel, submitQuery],
    );

    const pendingReask = useRef<string | null>(null);

    const handleSelectModel = useCallback((model: Model) => {
        const lastUserMsg = [...messages].reverse().find((m) => m.role === 'user');
        if (lastUserMsg) {
            pendingReask.current = lastUserMsg.content;
            setMessages([]);
        }
        onSelectedModelIdChange(model.id);
    }, [onSelectedModelIdChange, messages]);

    useEffect(() => {
        if (pendingReask.current && selectedModel) {
            const question = pendingReask.current;
            pendingReask.current = null;
            handleSend(question);
        }
    }, [selectedModel, handleSend]);

    return (
        <div className="flex h-full flex-col">
            <ChatHeader
                selectedModelId={selectedModelId}
                onSelectModel={handleSelectModel}
                onNewChat={handleNewChat}
            />
            <ChatMessageList
                messages={messages}
                modelName={selectedModel?.name ?? null}
                onSuggestedClick={handleSend}
            />
            <ChatInput key={inputResetKey} onSend={handleSend} disabled={!selectedModel} />
        </div>
    );
}
