# This project was developed with assistance from AI tools.
"""Document management endpoints -- upload, list, and status."""

import asyncio
import logging
import os

from db import Chunk, Document, get_db
from db.database import SessionLocal
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..schemas.documents import (
    DocumentResponse,
    DocumentStatusResponse,
    DocumentUploadResponse,
)
from ..services.document_parser import parse_pdf
from ..services.embedding import DOCUMENT_PREFIX, generate_embeddings

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

# Serialize embedding generation so concurrent uploads don't flood the API
_embedding_semaphore = asyncio.Semaphore(1)


async def _generate_embeddings_for_document(document_id: int) -> None:
    """Background task: generate embeddings for all chunks of a document.

    Acquires a semaphore so only one document is embedded at a time,
    preventing concurrent uploads from overwhelming the embedding API.
    """
    async with _embedding_semaphore:
        async with SessionLocal() as session:
            doc = await session.get(Document, document_id)
            if not doc or doc.deleted_at is not None:
                return

            result = await session.execute(
                select(Chunk)
                .where(Chunk.document_id == document_id, Chunk.embedding.is_(None))
                .order_by(Chunk.id)
            )
            chunks = result.scalars().all()
            if not chunks:
                doc.status = "ready"
                await session.commit()
                return

            texts = [c.text for c in chunks]
            embed_out = await generate_embeddings(texts, prefix=DOCUMENT_PREFIX)
            embeddings = embed_out.vectors

            if not embeddings:
                doc.status = "embedding_failed"
                doc.error_message = embed_out.error or "Embedding generation failed"
                await session.commit()
                logger.error(
                    "Embedding failed for document %d (%s): %s",
                    document_id,
                    doc.filename,
                    embed_out.error,
                )
                return

            updated = 0
            for i, chunk in enumerate(chunks):
                vec = embeddings[i]
                if vec is not None:
                    chunk.embedding = vec
                    updated += 1

            if updated == len(chunks):
                doc.status = "ready"
                doc.error_message = None
            elif updated > 0:
                doc.status = "ready"
                doc.error_message = (
                    f"{len(chunks) - updated}/{len(chunks)} chunks missing embeddings"
                )
            else:
                doc.status = "embedding_failed"
                doc.error_message = embed_out.error or "Embedding generation failed"

            await session.commit()
            logger.info(
                "Embedded document %d (%s): %d/%d chunks",
                document_id,
                doc.filename,
                updated,
                len(chunks),
            )


@router.post("/upload", response_model=DocumentUploadResponse, status_code=201)
async def upload_document(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_db),
) -> DocumentUploadResponse:
    """Upload a PDF document for processing.

    Parses the PDF and stores chunks immediately. Embedding generation
    runs in the background so the upload returns quickly. The document
    status will transition from 'processing' to 'ready' (or
    'embedding_failed') once embeddings are generated.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    if file.content_type and file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid content type")

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File exceeds 50 MB limit")

    # Validate PDF magic bytes
    if not content.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="File is not a valid PDF")

    # Sanitize filename
    safe_filename = os.path.basename(file.filename).replace("..", "")

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
        logger.error("Failed to parse %s: %s", file.filename, e)
        doc.status = "error"
        doc.error_message = f"PDF parsing failed: {e}"
        doc_id = doc.id
        doc_filename = doc.filename
        error_msg = doc.error_message
        await session.commit()
        return DocumentUploadResponse(
            document_id=doc_id,
            filename=doc_filename,
            status="error",
            message=error_msg,
        )

    # Store chunks without embeddings (background task will add them)
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

    # Capture values before commit
    doc_id = doc.id
    doc_filename = doc.filename
    num_chunks = len(db_chunks)
    num_pages = parse_result.page_count
    parser = parse_result.parser_used
    await session.commit()

    # Generate embeddings in background (serialized via semaphore)
    background_tasks.add_task(_generate_embeddings_for_document, doc_id)

    return DocumentUploadResponse(
        document_id=doc_id,
        filename=doc_filename,
        status="processing",
        message=(
            f"Parsed {num_chunks} chunks from {num_pages} pages (parser: {parser}). "
            "Embedding generation in progress."
        ),
    )


@router.get("/", response_model=list[DocumentResponse])
async def list_documents(
    session: AsyncSession = Depends(get_db),
) -> list[DocumentResponse]:
    """List all non-deleted documents."""
    result = await session.execute(
        select(Document).where(Document.deleted_at.is_(None)).order_by(Document.created_at.desc())
    )
    docs = result.scalars().all()
    return [
        DocumentResponse(
            id=doc.id,
            filename=doc.filename,
            status=doc.status,
            chunk_count=doc.chunk_count,
            page_count=doc.page_count,
            file_size_bytes=doc.file_size_bytes,
            error_message=doc.error_message,
            created_at=doc.created_at.isoformat() if doc.created_at else None,
        )
        for doc in docs
    ]


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: int,
    session: AsyncSession = Depends(get_db),
) -> DocumentResponse:
    """Get a single document by ID."""
    doc = await session.get(Document, document_id)
    if not doc or doc.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentResponse(
        id=doc.id,
        filename=doc.filename,
        status=doc.status,
        chunk_count=doc.chunk_count,
        page_count=doc.page_count,
        file_size_bytes=doc.file_size_bytes,
        error_message=doc.error_message,
        created_at=doc.created_at.isoformat() if doc.created_at else None,
    )


@router.delete("/{document_id}", status_code=204)
async def delete_document(
    document_id: int,
    session: AsyncSession = Depends(get_db),
) -> None:
    """Delete a document and all its chunks/embeddings."""
    result = await session.execute(
        select(Document)
        .options(selectinload(Document.chunks))
        .where(Document.id == document_id)
    )
    doc = result.scalars().first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    await session.delete(doc)
    await session.commit()


@router.post("/{document_id}/embed", response_model=DocumentStatusResponse)
async def embed_document(
    document_id: int,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_db),
) -> DocumentStatusResponse:
    """Retry embedding generation for a document with missing embeddings.

    Queues the embedding work as a background task (serialized with other
    embedding work). The document status changes to 'processing' immediately;
    poll the status endpoint to see when it completes.
    """
    doc = await session.get(Document, document_id)
    if not doc or doc.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Document not found")

    result = await session.execute(
        select(Chunk.id)
        .where(
            Chunk.document_id == document_id,
            Chunk.embedding.is_(None),
        )
        .limit(1)
    )
    has_missing = result.scalar_one_or_none() is not None

    if not has_missing:
        return DocumentStatusResponse(
            document_id=doc.id,
            filename=doc.filename,
            status=doc.status,
            chunk_count=doc.chunk_count,
            page_count=doc.page_count,
            error_message="All chunks already have embeddings",
        )

    doc.status = "processing"
    doc.error_message = None

    # Capture values before commit (async session expires ORM attributes)
    doc_id = doc.id
    doc_filename = doc.filename
    doc_chunk_count = doc.chunk_count
    doc_page_count = doc.page_count
    await session.commit()

    background_tasks.add_task(_generate_embeddings_for_document, document_id)

    return DocumentStatusResponse(
        document_id=doc_id,
        filename=doc_filename,
        status="processing",
        chunk_count=doc_chunk_count,
        page_count=doc_page_count,
        error_message="Embedding generation queued",
    )


@router.get("/{document_id}/status", response_model=DocumentStatusResponse)
async def get_document_status(
    document_id: int,
    session: AsyncSession = Depends(get_db),
) -> DocumentStatusResponse:
    """Get processing status for a document."""
    doc = await session.get(Document, document_id)
    if not doc or doc.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentStatusResponse(
        document_id=doc.id,
        filename=doc.filename,
        status=doc.status,
        chunk_count=doc.chunk_count,
        page_count=doc.page_count,
        error_message=doc.error_message,
    )
