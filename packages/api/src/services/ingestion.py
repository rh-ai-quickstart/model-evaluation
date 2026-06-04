"""Document ingestion service -- download and process PDFs from URLs and S3."""

import logging
import os
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx
from db import Chunk, Document
from fastapi import BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import settings
from .document_parser import parse_pdf

logger = logging.getLogger(__name__)

MAX_DOWNLOAD_SIZE = 50 * 1024 * 1024  # 50 MB
DOWNLOAD_TIMEOUT = 60.0  # seconds


@dataclass
class IngestionResult:
    """Result of ingesting a single document."""

    document_id: int
    filename: str
    status: str  # "processing" | "ready" | "error"
    message: str
    chunk_count: int = 0
    page_count: int | None = None
    embedding_error: str | None = None


async def download_from_url(url: str) -> tuple[bytes, str]:
    """Download a PDF from a URL.

    Args:
        url: URL to download from.

    Returns:
        Tuple of (content bytes, filename).

    Raises:
        ValueError: If the URL is invalid or response is not a PDF.
        httpx.HTTPStatusError: If the download fails.
    """
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid URL: {url}")

    filename = os.path.basename(parsed.path) or "downloaded.pdf"
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"

    async with httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()

    content = response.content
    if len(content) > MAX_DOWNLOAD_SIZE:
        raise ValueError(f"File exceeds {MAX_DOWNLOAD_SIZE // (1024 * 1024)} MB limit")

    if not content.startswith(b"%PDF"):
        raise ValueError("Downloaded file is not a valid PDF")

    return content, filename


async def download_from_s3(bucket: str, key: str) -> tuple[bytes, str]:
    """Download a PDF from S3/MinIO.

    Uses settings for S3 configuration (endpoint, credentials).

    Args:
        bucket: S3 bucket name.
        key: Object key (path) within the bucket.

    Returns:
        Tuple of (content bytes, filename).

    Raises:
        RuntimeError: If S3 is not configured.
        ValueError: If the downloaded file is not a valid PDF.
    """
    if not settings.S3_ENDPOINT_URL:
        raise RuntimeError(
            "S3 is not configured. Set S3_ENDPOINT_URL, S3_ACCESS_KEY, and S3_SECRET_KEY."
        )

    try:
        import boto3
        from botocore.config import Config as BotoConfig
    except ImportError as e:
        raise RuntimeError(
            "boto3 is required for S3 ingestion. Install it with: pip install boto3"
        ) from e

    s3_client = boto3.client(
        "s3",
        endpoint_url=settings.S3_ENDPOINT_URL,
        aws_access_key_id=settings.S3_ACCESS_KEY,
        aws_secret_access_key=settings.S3_SECRET_KEY,
        config=BotoConfig(signature_version="s3v4"),
    )

    response = s3_client.get_object(Bucket=bucket, Key=key)
    content = response["Body"].read()

    if len(content) > MAX_DOWNLOAD_SIZE:
        raise ValueError(f"File exceeds {MAX_DOWNLOAD_SIZE // (1024 * 1024)} MB limit")

    if not content.startswith(b"%PDF"):
        raise ValueError("Downloaded file is not a valid PDF")

    filename = os.path.basename(key) or "s3_document.pdf"
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"

    return content, filename


async def process_and_store(
    content: bytes,
    filename: str,
    session: AsyncSession,
    background_tasks: BackgroundTasks,
) -> IngestionResult:
    """Parse a PDF, store chunks, and queue embedding generation.

    Chunks are stored without embeddings. Embedding generation runs in a
    serialized background task (shared semaphore with document upload).

    Args:
        content: Raw PDF bytes.
        filename: Filename to store.
        session: Database session.
        background_tasks: FastAPI background tasks for embedding work.

    Returns:
        IngestionResult with document details.
    """
    safe_filename = os.path.basename(filename).replace("..", "")

    doc = Document(
        filename=safe_filename,
        status="processing",
        file_size_bytes=len(content),
    )
    session.add(doc)
    await session.flush()

    try:
        parse_result = parse_pdf(content, safe_filename)
    except Exception as e:
        logger.error("Failed to parse %s: %s", safe_filename, e)
        doc.status = "error"
        doc.error_message = f"PDF parsing failed: {e}"
        doc_id = doc.id
        await session.commit()
        return IngestionResult(
            document_id=doc_id,
            filename=safe_filename,
            status="error",
            message=doc.error_message,
        )

    db_chunks = []
    for chunk_data in parse_result.chunks:
        chunk = Chunk(
            document_id=doc.id,
            text=chunk_data.text,
            source_document=chunk_data.source_document,
            page_number=chunk_data.page_number,
            section_path=chunk_data.section_path,
            element_type=chunk_data.element_type,
            token_count=chunk_data.token_count,
            embedding=None,
        )
        db_chunks.append(chunk)

    session.add_all(db_chunks)
    doc.chunk_count = len(db_chunks)
    doc.page_count = parse_result.page_count

    doc_id = doc.id
    num_chunks = len(db_chunks)
    num_pages = parse_result.page_count
    parser = parse_result.parser_used
    await session.commit()

    # Queue embedding generation (serialized via shared semaphore)
    from ..routes.documents import _generate_embeddings_for_document

    background_tasks.add_task(_generate_embeddings_for_document, doc_id)

    return IngestionResult(
        document_id=doc_id,
        filename=safe_filename,
        status="processing",
        message=(
            f"Parsed {num_chunks} chunks from {num_pages} pages (parser: {parser}). "
            "Embedding generation in progress."
        ),
        chunk_count=num_chunks,
        page_count=num_pages,
    )


async def ingest_from_url(
    url: str,
    session: AsyncSession,
    background_tasks: BackgroundTasks,
) -> IngestionResult:
    """Download a PDF from a URL and ingest it.

    Args:
        url: URL to download from.
        session: Database session.
        background_tasks: FastAPI background tasks.

    Returns:
        IngestionResult with document details.
    """
    content, filename = await download_from_url(url)
    return await process_and_store(content, filename, session, background_tasks)


async def ingest_from_s3(
    bucket: str,
    key: str,
    session: AsyncSession,
    background_tasks: BackgroundTasks,
) -> IngestionResult:
    """Download a PDF from S3/MinIO and ingest it.

    Args:
        bucket: S3 bucket name.
        key: Object key within the bucket.
        session: Database session.
        background_tasks: FastAPI background tasks.

    Returns:
        IngestionResult with document details.
    """
    content, filename = await download_from_s3(bucket, key)
    return await process_and_store(content, filename, session, background_tasks)
