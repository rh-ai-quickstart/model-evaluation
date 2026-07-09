
import { useState, useRef, useEffect } from 'react';
import { SendHorizontal, Info } from 'lucide-react';

interface ChatInputProps {
    onSend: (message: string) => void;
    disabled?: boolean;
}

export function ChatInput({ onSend, disabled }: ChatInputProps) {
    const [value, setValue] = useState('');
    const textareaRef = useRef<HTMLTextAreaElement>(null);

    useEffect(() => {
        const el = textareaRef.current;
        if (el) {
            el.style.height = 'auto';
            el.style.height = Math.min(el.scrollHeight, 120) + 'px';
        }
    }, [value]);

    const handleSubmit = () => {
        const trimmed = value.trim();
        if (!trimmed || disabled) return;
        onSend(trimmed);
        setValue('');
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit();
        }
    };

    return (
        <div className="border-t bg-card px-4 py-3">
            <div className="flex items-end gap-2">
                <textarea
                    ref={textareaRef}
                    value={value}
                    onChange={(e) => setValue(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Ask a compliance question..."
                    disabled={disabled}
                    rows={1}
                    className="max-h-[120px] min-h-[40px] flex-1 resize-none rounded-lg border bg-background px-3 py-2.5 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring disabled:opacity-50"
                />
                <button
                    onClick={handleSubmit}
                    disabled={disabled || !value.trim()}
                    className="grid h-10 w-10 shrink-0 place-items-center rounded-lg bg-primary text-primary-foreground transition-colors hover:bg-primary/90 disabled:opacity-50"
                >
                    <SendHorizontal className="h-4 w-4" />
                </button>
            </div>
            <div className="mt-2 flex items-start gap-2 rounded-lg border border-blue-200 bg-blue-50/60 px-3 py-2 text-xs text-blue-800 dark:border-blue-800 dark:bg-blue-950/30 dark:text-blue-300">
                <Info className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                <span>Responses are generated from your uploaded documents and are for informational purposes only. This is not legal or compliance advice.</span>
            </div>
        </div>
    );
}
