
import { AlertTriangle, Loader2, User, Bot } from 'lucide-react';

export interface ChatMessageEntry {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    lowConfidence?: boolean;
    isLoading?: boolean;
    error?: string;
}

export function ChatMessage({ message }: { message: ChatMessageEntry }) {
    const isUser = message.role === 'user';

    if (message.isLoading) {
        return (
            <div className="flex items-start gap-3">
                <div className="grid h-8 w-8 shrink-0 place-items-center rounded-full bg-primary/10">
                    <Bot className="h-4 w-4 text-primary" />
                </div>
                <div className="flex items-center gap-2 rounded-2xl rounded-tl-sm bg-muted px-4 py-3">
                    <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
                    <span className="text-sm text-muted-foreground">Thinking...</span>
                </div>
            </div>
        );
    }

    if (isUser) {
        return (
            <div className="flex items-start justify-end gap-3">
                <div className="max-w-[80%] rounded-2xl rounded-tr-sm bg-primary px-4 py-2.5">
                    <p className="text-sm text-primary-foreground">{message.content}</p>
                </div>
                <div className="grid h-8 w-8 shrink-0 place-items-center rounded-full bg-muted">
                    <User className="h-4 w-4 text-muted-foreground" />
                </div>
            </div>
        );
    }

    return (
        <div className="flex items-start gap-3">
            <div className="grid h-8 w-8 shrink-0 place-items-center rounded-full bg-primary/10">
                <Bot className="h-4 w-4 text-primary" />
            </div>
            <div className="max-w-[85%] space-y-2">
                {message.error ? (
                    <div className="rounded-2xl rounded-tl-sm border border-destructive/50 bg-destructive/10 px-4 py-3">
                        <p className="text-sm text-destructive">{message.error}</p>
                    </div>
                ) : (
                    <>
                        {message.lowConfidence && (
                            <div className="flex items-center gap-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 dark:border-amber-900 dark:bg-amber-950/20">
                                <AlertTriangle className="h-3.5 w-3.5 shrink-0 text-amber-600 dark:text-amber-400" />
                                <p className="text-xs text-amber-700 dark:text-amber-300">
                                    Low confidence -- the retrieved context may not fully cover this topic.
                                </p>
                            </div>
                        )}
                        <div className="rounded-2xl rounded-tl-sm bg-muted px-4 py-3">
                            <p className="text-sm leading-relaxed whitespace-pre-wrap">{message.content}</p>
                        </div>
                    </>
                )}
            </div>
        </div>
    );
}
