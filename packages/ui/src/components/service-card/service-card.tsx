
import React from "react";
import {
    Folder,
} from "lucide-react";
import { useState } from "react";

interface PackageInfo {
    path: string;
    devUrl: string;
    quickActions: { label: string; url: string; external?: boolean }[];
    commands: { label: string; cmd: string }[];
    gettingStarted: string[];
}

export interface ServiceCardProps {
    name: string;
    description: string;
    icon: React.ReactNode;
}

const PACKAGE_INFO: Record<string, PackageInfo> = {
    UI: {
        path: "packages/ui/",
        devUrl: "http://localhost:3000",
        quickActions: [
            { label: "Storybook", url: "http://localhost:6006", external: true },
        ],
        commands: [
            { label: "Dev", cmd: "pnpm dev" },
            { label: "Build", cmd: "pnpm build" },
            { label: "Test", cmd: "pnpm test" },
            { label: "Lint", cmd: "pnpm lint" },
        ],
        gettingStarted: [
            "Create route in `src/routes/`",
            "Add components in `src/components/`",
            "Add API calls in `src/services/`",
        ],
    },
    API: {
        path: "packages/api/",
        devUrl: "http://localhost:8000",
        quickActions: [
            { label: "API Docs", url: "http://localhost:8000/docs", external: true },
            { label: "DB Admin", url: "http://localhost:8000/admin", external: true },
        ],
        commands: [
            { label: "Dev", cmd: "pnpm dev" },
            { label: "Test", cmd: "pnpm test" },
            { label: "Lint", cmd: "pnpm lint" },
        ],
        gettingStarted: [
            "Create schema in `src/schemas/`",
            "Add route in `src/routes/`",
            "Register router in `main.py`",
        ],
    },
    Database: {
        path: "packages/db/",
        devUrl: "postgresql://localhost:5432",
        quickActions: [],
        commands: [
            { label: "Start DB", cmd: "pnpm db:start" },
            { label: "Migrate", cmd: "pnpm migrate" },
        ],
        gettingStarted: [
            "Add models in `src/db/models.py`",
            "Run `pnpm migrate:new` then `pnpm migrate`",
        ],
    },
};

function formatWithCode(text: string) {
    const parts = text.split(/(`[^`]+`)/g);
    return parts.map((part, idx) => {
        if (part.startsWith('`') && part.endsWith('`')) {
            return (
                <code key={idx} className="rounded bg-muted px-1 py-0.5 font-mono text-foreground">
                    {part.slice(1, -1)}
                </code>
            );
        }
        return <span key={idx}>{part}</span>;
    });
}

function DevInfo({ serviceName }: { serviceName: string }) {
    const info = PACKAGE_INFO[serviceName];
    const [showSteps, setShowSteps] = useState(false);

    if (!info) return null;

    return (
        <div className="mt-3 space-y-3">
            <div className="flex flex-wrap gap-2">
                {info.quickActions.map((action, idx) => (
                    <a
                        key={idx}
                        href={action.url}
                        target={action.external ? "_blank" : undefined}
                        rel={action.external ? "noopener noreferrer" : undefined}
                        className="inline-flex items-center gap-1 rounded-md bg-primary/10 px-2 py-1 text-xs font-medium text-primary hover:bg-primary/20 transition-colors"
                    >
                        {action.label}
                    </a>
                ))}
            </div>

            <div className="flex flex-wrap gap-1.5">
                {info.commands.map((cmd, idx) => (
                    <div
                        key={idx}
                        className="inline-flex items-center gap-1.5 rounded bg-muted px-2 py-1 font-mono text-[11px]"
                    >
                        <span className="text-muted-foreground">{cmd.label}:</span>
                        <code className="text-foreground">{cmd.cmd}</code>
                    </div>
                ))}
            </div>

            <div>
                <button
                    onClick={() => setShowSteps(!showSteps)}
                    className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
                >
                    <span>{showSteps ? "v" : ">"}</span>
                    <span>Getting Started</span>
                </button>
                {showSteps && (
                    <ol className="mt-2 ml-4 space-y-1 text-xs text-muted-foreground list-decimal list-outside">
                        {info.gettingStarted.map((step, idx) => (
                            <li key={idx} className="pl-1">{formatWithCode(step)}</li>
                        ))}
                    </ol>
                )}
            </div>

            <div className="flex flex-col gap-1 text-[11px] font-mono pt-2 border-t border-border">
                <div className="flex items-center gap-1.5 text-muted-foreground">
                    <Folder className="h-3 w-3 text-amber-500" />
                    <span>{info.path}</span>
                </div>
                <div className="flex items-center gap-1.5">
                    <span className="text-muted-foreground">Dev:</span>
                    {info.devUrl.startsWith("http") ? (
                        <a
                            href={info.devUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-sky-500 hover:underline"
                        >
                            {info.devUrl}
                        </a>
                    ) : (
                        <span className="text-foreground">{info.devUrl}</span>
                    )}
                </div>
            </div>
        </div>
    );
}

export function ServiceCard({ name, description, icon }: ServiceCardProps) {
    return (
        <div className="group relative overflow-hidden rounded-xl border bg-card p-4 transition-all duration-300 hover:-translate-y-0.5 hover:shadow-md">
            <div className="pointer-events-none absolute inset-x-0 top-0 h-0.5 bg-gradient-to-r from-sky-500/0 via-sky-500/60 to-fuchsia-500/0 opacity-0 transition-opacity duration-300 group-hover:opacity-100" />
            <div className="flex items-start justify-between">
                <div className="flex items-center gap-3">
                    <div className="grid h-10 w-10 place-items-center rounded-lg bg-muted ring-1 ring-border">
                        {icon}
                    </div>
                    <div className="flex flex-col">
                        <span className="text-sm font-medium text-foreground">{name}</span>
                        <span className="text-xs text-muted-foreground">{description}</span>
                    </div>
                </div>
            </div>
            <DevInfo serviceName={name} />
        </div>
    );
}
