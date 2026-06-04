
# Model Evaluation Database

Database layer for OpenShift AI Model Evaluation, built on PostgreSQL + pgvector.

This package owns SQLAlchemy models, async DB session utilities, and Alembic migrations.

> Setup and local run instructions are in the [root README](../../README.md).

## Main Data Entities

- `Document`: uploaded source PDFs and ingestion status
- `Chunk`: parsed text chunks and vector embeddings (`EMBEDDING_DIMENSION = 768`)
- `QuestionSet`: reusable evaluation question collections
- `EvalRun`: run-level metadata, aggregate scores, profile metadata, verdict summary
- `EvalResult`: per-question outputs, metric scores, deterministic checks, truth payload
- `ModelConfig`: model metadata table (available for extension)

## Package Layout

```text
src/db/
  database.py       async engine/session, health service, dependencies
  models.py         SQLAlchemy models
  __init__.py       package exports

alembic/
  env.py
  versions/         migration history
```

## Database Defaults

- host: `localhost` (local dev) / `model-evaluation-db` (OpenShift)
- port: `5432`
- db: `model-evaluation`
- user: `user`

Configure `DATABASE_URL` and credentials via environment variables or Helm values.

## Migrations

Run from `packages/db`:

```bash
pnpm db:start                     # start postgres container
pnpm migrate                      # alembic upgrade head
pnpm migrate:down                 # rollback one migration
pnpm migrate:history              # show migration history
pnpm migrate:new -- -m "message"  # autogenerate new migration
```

Or from repository root:

```bash
pnpm db:start
pnpm db:migrate
pnpm db:migrate:down
pnpm db:migrate:new -- -m "message"
```

## Engine and Session Notes

- Async engine uses `asyncpg`.
- SQL logging is controlled by `DB_ECHO=true|false`.
- Pool configuration is set in `src/db/database.py`.
- API routes typically consume sessions through `get_db()`.

## Example Usage from API

```python
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from db import get_db

async def endpoint(session: AsyncSession = Depends(get_db)):
    ...
```

## Notes

- Do not hardcode migration count in docs; use `alembic/versions/` as source of truth.
- Keep database credentials aligned between your environment and migration config before running migrations.
