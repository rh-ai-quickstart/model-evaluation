// This project was developed with assistance from AI tools.

import { useEffect, useRef } from 'react';
import { Bot, MessageSquare } from 'lucide-react';
import { ChatMessage } from './chat-message';
import type { ChatMessageEntry } from './chat-message';

const SUGGESTED_QUESTIONS = [
    'What is the estimated cost for all 1,735 ETFs that can rely on rule 6c-11 to comply with the disclosure requirement?',
    'What information must be disclosed about each person providing transfer agency services to the Fund?',
    'What are the requirements for disclosing changes in valuation methods for the Registrant\'s assets during the reporting period?',
];

interface ChatMessageListProps {
    messages: ChatMessageEntry[];
    modelName: string | null;
    onSuggestedClick: (question: string) => void;
}

export function ChatMessageList({ messages, modelName, onSuggestedClick }: ChatMessageListProps) {
    const bottomRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        bottomRef.current?.scrollIntoView?.({ behavior: 'smooth' });
    }, [messages]);

    if (messages.length === 0) {
        return (
            <div className="flex flex-1 flex-col items-center justify-center px-6 text-center">
                <div className="mb-4 grid h-12 w-12 place-items-center rounded-full bg-primary/10">
                    <Bot className="h-6 w-6 text-primary" />
                </div>
                <h3 className="text-lg font-semibold">Hi! I&apos;m your FSI Compliance Assistant.</h3>
                <p className="mt-2 max-w-sm text-sm text-muted-foreground">
                    {modelName
                        ? `I'm powered by ${modelName} to help you with compliance questions based on your uploaded documents.`
                        : 'Select a model above to start asking compliance questions about your uploaded documents.'}
                </p>
                <p className="mt-1 max-w-sm text-sm text-muted-foreground">
                    You can ask me anything related to policies, regulations, procedures, and
                    compliance requirements.
                </p>

                <div className="mt-6 w-full max-w-md">
                    <div className="mb-3 flex items-center justify-center gap-2 text-xs text-muted-foreground">
                        <MessageSquare className="h-3.5 w-3.5" />
                        <span>Try asking something like:</span>
                    </div>
                    <div className="space-y-2">
                        {SUGGESTED_QUESTIONS.map((q) => (
                            <button
                                key={q}
                                onClick={() => onSuggestedClick(q)}
                                disabled={!modelName}
                                className="w-full rounded-lg border bg-card px-4 py-2.5 text-left text-sm transition-colors hover:bg-accent disabled:opacity-50"
                            >
                                {q}
                            </button>
                        ))}
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="flex-1 space-y-4 overflow-y-auto px-4 py-4">
            {messages.map((msg) => (
                <ChatMessage key={msg.id} message={msg} />
            ))}
            <div ref={bottomRef} />
        </div>
    );
}
