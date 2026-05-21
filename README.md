<!-- This project was developed with assistance from AI tools. -->

# OpenShift AI Model Evaluation

Evaluate and compare candidate LLMs for RAG workloads with a repeatable, score-driven workflow.  
This project ingests your domain PDFs, runs evaluation questions through selected models, computes DeepEval-style metrics, and returns a decision-oriented comparison.

## What It Does

- Builds a document corpus in PostgreSQL + pgvector from uploaded or ingested PDFs.
- Runs asynchronous evaluation jobs (question by question) against selected models.
- Scores answer quality and retrieval quality, then computes run-level and comparison verdicts.
- Supports optional profile-driven thresholds and deterministic checks for stricter domains.
- Exposes a UI workflow for documents, evaluations, run details, and side-by-side comparisons.

## How It Works (End-to-End)

1. Upload documents in the UI (`/documents`) or ingest from URL/S3 via API.
2. Documents are parsed, chunked, and embedded into pgvector.
3. Create or synthesize evaluation questions in `/evaluations`.
4. Start an evaluation run for a model (and optional profile).
5. For each question, the API retrieves relevant chunks, generates an answer, computes metrics, and stores a result.
6. Compare two completed runs in `/evaluations/compare` to get metric winners, warnings, and a final decision summary.

## Architecture

| Layer | Technology | Purpose |
| --- | --- | --- |
| UI | React 19 + Vite + TanStack Router/Query | Document and evaluation workflow |
| API | FastAPI | Orchestration for ingestion, retrieval, generation, scoring, and comparison |
| DB | PostgreSQL + pgvector + SQLAlchemy + Alembic | Persistent corpus, runs, and results |
| Evaluation | DeepEval-style metrics + deterministic checks + profile verdicts | Model quality assessment |
| Deploy | Helm (OpenShift) | UI/API/DB deployment with migration job |
| Monorepo | Turborepo + pnpm + uv | Unified dev/build/test workflows |

## Local Development

### Prerequisites

- Node.js 18+
- pnpm 9+
- Python 3.11+
- uv
- Podman + podman-compose

### Setup

```bash
git clone <your-repo-url>
cd openshift-ai-model-evaluation
pnpm install
pnpm -r --if-present install:deps
```

### Configure Environment

```bash
cp .env.example .env
```

Minimum required variables:

- `MAAS_ENDPOINT`
- `API_TOKEN`
- `MODEL_A_NAME`
- `MODEL_B_NAME`
- `EMBEDDING_MODEL`
- `JUDGE_MODEL_NAME`

Recommended for local UI dev:

- set `ALLOWED_HOSTS=["http://localhost:3000"]` in `.env` (default UI dev server is port `3000`)

### Run Locally

```bash
make db-start
make db-upgrade
make dev
```

Local URLs:

- UI: `http://localhost:3000`
- API: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

## OpenShift Deployment

Helm chart and deployment options are documented in:

- [`deploy/helm/ai-quickstart-template/README.md`](deploy/helm/ai-quickstart-template/README.md)

Quick summary:

- Build and push API/UI images.
- Configure chart values (registry, model names, `API_TOKEN`, DB settings).
- Install with Helm in your target namespace.
- Migration job runs automatically.

## Common Commands

```bash
make dev                # Run dev servers (UI + API)
make build              # Build all packages
make test               # Run tests
make lint               # Run linters
make db-start           # Start PostgreSQL container
make db-stop            # Stop PostgreSQL container
make db-upgrade         # Apply Alembic migrations
make containers-build   # Build compose images
make containers-up      # Start full compose stack
make containers-down    # Stop compose stack
```

## Repository Layout

```text
packages/
  ui/        React application
  api/       FastAPI application
  db/        SQLAlchemy models + Alembic migrations
  configs/   Shared lint/format configs
deploy/helm/ai-quickstart-template/   Helm chart
compose.yml                            Local pgvector database
Makefile                               Top-level workflow commands
```

## Package Docs

- [`packages/ui/README.md`](packages/ui/README.md)
- [`packages/api/README.md`](packages/api/README.md)
- [`packages/db/README.md`](packages/db/README.md)

## Notes

- Some internal package names still use `ai-quickstart-template` as a historical monorepo identifier.
- The shipped product behavior and UI naming are aligned to **OpenShift AI Model Evaluation**.
