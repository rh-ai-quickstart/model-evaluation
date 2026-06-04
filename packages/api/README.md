
# Model Evaluation API

FastAPI backend for OpenShift AI Model Evaluation.

The API orchestrates:

- document ingestion and embedding
- RAG retrieval and answer generation
- per-question scoring and verdicting
- run-level comparison decisions

> Setup and environment details are in the [root README](../../README.md).

## API Surface

Base app and router registration live in `src/main.py`.

| Prefix | Main capabilities |
| --- | --- |
| `/health` | Liveness/readiness |
| `/models` | List configured model A/B and model status |
| `/documents` | Upload/list/get/delete documents, retry embeddings |
| `/ingest` | Ingest PDFs from URL/S3 or batch URL ingestion |
| `/question-sets` | Create/list/get/delete reusable question sets |
| `/query` | RAG answer and retrieval debug endpoint |
| `/evaluations` | Create/list/get/delete/cancel/rerun runs, list profiles, synthesize questions, compare runs |

Swagger UI: `http://localhost:8000/docs` (local dev)

## Architecture

```text
Request -> Router -> Schema Validation -> Service Layer -> DB / MaaS
```

Key design points:

- async FastAPI handlers
- pydantic schemas for request/response contracts
- SQLAlchemy async session dependency (`db` package)
- environment-driven model configuration
- profile-based verdict logic from YAML profiles in `src/profiles/`

## Core Modules

```text
src/
  main.py
  core/config.py
  routes/
  schemas/
  services/
  profiles/
  admin.py
```

Important service groups:

- Retrieval and generation: `retrieval.py`, `query_decomposition.py`, `generation.py`
- Evaluation/scoring: `scoring.py`, `verdicts.py`, `deterministic_checks.py`, `coverage.py`
- Ingestion and parsing: `ingestion.py`, `chunking.py`, `document_parser.py`, `embedding.py`
- Question synthesis and truth: `synthesizer.py`, `truth_generation.py`

## Evaluation Flow

1. Create run at `POST /evaluations/`.
2. Background task processes each question with bounded concurrency.
3. For each question: retrieval -> generation -> scoring -> deterministic checks -> persisted result.
4. Aggregate run metrics and compute run verdict.
5. Compare two runs at `GET /evaluations/compare`.

## Configuration

Configuration lives in `src/core/config.py` and loads from repo root `.env`.

Common variables:

- `MAAS_ENDPOINT`, `API_TOKEN`
- `MODEL_A_NAME`, `MODEL_B_NAME`, `EMBEDDING_MODEL`, `JUDGE_MODEL_NAME`
- `QUESTION_SYNTHESIS_MODEL_NAME` (optional override for synthesis)
- `DATABASE_URL`, `ALLOWED_HOSTS`, `DEBUG`
- optional S3 and safety settings (`S3_*`, `SAFETY_*`, `DOCLING_ENABLED`)

## Commands

Run from `packages/api`:

```bash
pnpm install:deps     # uv sync + editable install of ../db
pnpm dev              # uvicorn with reload
pnpm start            # uvicorn (no reload)

pnpm test             # pytest
pnpm lint             # ruff check
pnpm lint:fix         # ruff autofix
pnpm format           # ruff format
pnpm type-check       # mypy
```

## Notes

- SQLAlchemy models are owned by `packages/db` (not under `packages/api/src/models`).
- `/models` is currently environment-backed (the `ModelConfig` table exists for future expansion).
