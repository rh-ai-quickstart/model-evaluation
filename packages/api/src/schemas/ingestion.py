"""Pydantic schemas for document ingestion endpoints."""

from pydantic import BaseModel, Field


class IngestUrlRequest(BaseModel):
    """Request to ingest a PDF from a URL."""

    url: str = Field(..., min_length=1)


class IngestS3Request(BaseModel):
    """Request to ingest a PDF from S3/MinIO."""

    bucket: str = Field(..., min_length=1)
    key: str = Field(..., min_length=1)


class IngestBatchUrlRequest(BaseModel):
    """Request to ingest multiple PDFs from URLs."""

    urls: list[str] = Field(..., min_length=1, max_length=20)


class IngestResult(BaseModel):
    """Result for a single ingested document."""

    document_id: int
    filename: str
    status: str
    message: str
    chunk_count: int = 0
    page_count: int | None = None
    embedding_error: str | None = None


class IngestBatchResponse(BaseModel):
    """Response for a batch ingestion request."""

    results: list[IngestResult] = []
    total: int = 0
    succeeded: int = 0
    failed: int = 0
