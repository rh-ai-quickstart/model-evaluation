"""Pydantic schemas for document management endpoints."""

from pydantic import BaseModel


class DocumentResponse(BaseModel):
    """Response schema for a document."""

    id: int
    filename: str
    status: str
    chunk_count: int
    page_count: int | None = None
    file_size_bytes: int | None = None
    error_message: str | None = None
    created_at: str | None = None

    model_config = {"from_attributes": True}


class DocumentUploadResponse(BaseModel):
    """Response after uploading a document."""

    document_id: int
    filename: str
    status: str
    message: str
    embedding_error: str | None = None


class DocumentStatusResponse(BaseModel):
    """Response for document processing status."""

    document_id: int
    filename: str
    status: str
    chunk_count: int
    page_count: int | None = None
    error_message: str | None = None
