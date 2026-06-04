
# Model Evaluation UI

React frontend for OpenShift AI Model Evaluation.

This package provides the product-facing workflow for:

- document ingestion and indexing status
- creating/running evaluation jobs
- viewing run details
- comparing two completed runs with decision context

> Setup, env, and full local run instructions live in the [root README](../../README.md).

## Tech Stack

- React 19 + TypeScript
- TanStack Router (file-based routes)
- TanStack Query (server state)
- Tailwind CSS + shadcn-style UI primitives
- Vite + Storybook + Vitest

## Route Map

| Route | File | Purpose |
| --- | --- | --- |
| `/` | `src/routes/index.tsx` | Dashboard, system health, comparison readiness |
| `/documents` | `src/routes/documents/index.tsx` | Upload/ingest PDFs and manage corpus |
| `/evaluations` | `src/routes/evaluations/index.tsx` | Create runs, synthesize questions, manage runs |
| `/evaluations/$id` | `src/routes/evaluations/$id.tsx` | Per-run deep dive |
| `/evaluations/compare` | `src/routes/evaluations/compare.tsx` | Side-by-side run comparison |

## API Integration Pattern

The UI uses a consistent `services -> hooks -> routes/components` pattern:

```text
Route/Component -> Hook (TanStack Query) -> Service (fetch) -> /api/*
```

Key modules:

- `src/services/documents.ts` + `src/hooks/documents.ts`
- `src/services/evaluation.ts` + `src/hooks/evaluation.ts`
- `src/services/question-sets.ts` + `src/hooks/question-sets.ts`
- `src/services/models.ts` + `src/hooks/models.ts`
- `src/services/health.ts` + `src/hooks/health.ts`

## Networking Behavior

- Services call relative `/api/...` paths.
- In local dev, Vite proxies `/api` to `http://localhost:8000`.
- This keeps browser calls same-origin in development and simplifies deployment routing.

## Project Layout

```text
src/
  routes/        TanStack file routes
  components/    UI components (atoms + feature components)
  hooks/         React Query hooks
  services/      API client calls
  schemas/       Zod API response schemas
  lib/           Formatting/status helpers
```

## Commands

Run these from `packages/ui`:

```bash
pnpm dev             # Vite + Storybook (concurrently)
pnpm dev:vite        # Vite only
pnpm dev:storybook   # Storybook only

pnpm build           # Build app + storybook
pnpm type-check      # TypeScript check

pnpm test            # Vitest
pnpm lint            # ESLint
pnpm format          # Prettier write
```

## Notes

- `src/components/query-panel/query-panel.tsx` exists but is not currently mounted on a route.
- Navigation is defined in `src/components/header/header.tsx`.
