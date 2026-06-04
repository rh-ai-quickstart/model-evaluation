"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .admin import setup_admin
from .core.config import settings
from .routes import documents, evaluation, health, ingestion, models, query, question_sets

app = FastAPI(
    title="OpenShift AI Model Evaluation API",
    description="API for evaluating and comparing AI models on OpenShift AI",
    version="0.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=False,  # NOTE: credentials disabled; enable when auth is implemented with specific origins
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(models.router, prefix="/models", tags=["models"])
app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(query.router, prefix="/query", tags=["query"])
app.include_router(evaluation.router, prefix="/evaluations", tags=["evaluations"])
app.include_router(question_sets.router, prefix="/question-sets", tags=["question-sets"])
app.include_router(ingestion.router, prefix="/ingest", tags=["ingestion"])

setup_admin(app)


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {"message": "Welcome to OpenShift AI Model Evaluation API"}
