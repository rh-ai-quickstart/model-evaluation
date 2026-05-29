// This project was developed with assistance from AI tools.

import { useState, useEffect, useRef } from 'react';
import { useModels } from '../../hooks/models';
import { Server, ChevronDown, Check } from 'lucide-react';
import type { Model } from '../../schemas/models';
import { Skeleton } from '../atoms/skeleton/skeleton';

interface ModelSelectorProps {
    selectedModelId: number | null;
    onSelect: (model: Model) => void;
    label: string;
}

export function ModelSelector({ selectedModelId, onSelect, label }: ModelSelectorProps) {
    const { data: models, isLoading, error } = useModels();
    const [isOpen, setIsOpen] = useState(false);
    const containerRef = useRef<HTMLDivElement>(null);

    const selectedModel = models?.find((m) => m.id === selectedModelId);

    useEffect(() => {
        function handleClickOutside(event: MouseEvent) {
            if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
                setIsOpen(false);
            }
        }

        if (isOpen) {
            document.addEventListener('mousedown', handleClickOutside);
            return () => {
                document.removeEventListener('mousedown', handleClickOutside);
            };
        }
    }, [isOpen]);

    return (
        <div className="relative" ref={containerRef}>
            <label className="mb-1.5 block text-xs font-medium text-muted-foreground">{label}</label>
            <button
                onClick={() => setIsOpen(!isOpen)}
                disabled={isLoading || !!error}
                className="flex w-full items-center justify-between gap-2 rounded-lg border bg-card px-3 py-2.5 text-sm transition-colors hover:bg-accent disabled:opacity-50"
            >
                <div className="flex items-center gap-2">
                    <Server className="h-4 w-4 text-muted-foreground" />
                    {isLoading ? (
                        <Skeleton className="h-4 w-24" />
                    ) : (
                        <span>{selectedModel?.name ?? 'Select a model'}</span>
                    )}
                </div>
                <ChevronDown className={`h-4 w-4 text-muted-foreground transition-transform ${isOpen ? 'rotate-180' : ''}`} />
            </button>

            {isOpen && models && (
                <div className="absolute z-10 mt-1 w-full rounded-lg border bg-card shadow-lg">
                    {models.map((model) => (
                        <button
                            key={model.id}
                            onClick={() => {
                                onSelect(model);
                                setIsOpen(false);
                            }}
                            className="flex w-full items-center gap-2 px-3 py-2 text-sm hover:bg-accent first:rounded-t-lg last:rounded-b-lg"
                        >
                            <span className="flex-1 text-left">{model.name}</span>
                            <span className="text-[10px] text-muted-foreground">{model.deployment_mode}</span>
                            {model.id === selectedModelId && <Check className="h-3.5 w-3.5 text-primary" />}
                        </button>
                    ))}
                </div>
            )}

            {error && <p className="mt-1 text-xs text-destructive">Failed to load models</p>}
        </div>
    );
}
