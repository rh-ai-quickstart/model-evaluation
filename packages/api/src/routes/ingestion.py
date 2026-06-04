"""Ingestion endpoints -- ingest documents from URLs and S3/MinIO."""

import logging

from db import get_db
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from ..schemas.ingestion import (
    IngestBatchResponse,
    IngestBatchUrlRequest,
    IngestResult,
    IngestS3Request,
    IngestUrlRequest,
)
from ..services.ingestion import ingest_from_s3, ingest_from_url

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/url", response_model=IngestResult, status_code=201)
async def ingest_url(
    request: IngestUrlRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_db),
) -> IngestResult:
    """Ingest a PDF document from a URL.

    Downloads the PDF and processes it through the standard pipeline
    (parse, store, embed in background). Same result as uploading via
    /documents/upload.
    """
    try:
        result = await ingest_from_url(request.url, session, background_tasks)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("URL ingestion failed for %s", request.url)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)[:200]}")

    return IngestResult(
        document_id=result.document_id,
        filename=result.filename,
        status=result.status,
        message=result.message,
        chunk_count=result.chunk_count,
        page_count=result.page_count,
        embedding_error=result.embedding_error,
    )


@router.post("/s3", response_model=IngestResult, status_code=201)
async def ingest_s3(
    request: IngestS3Request,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_db),
) -> IngestResult:
    """Ingest a PDF document from S3/MinIO.

    Downloads the PDF from the configured S3-compatible endpoint and
    processes it through the standard pipeline.
    """
    try:
        result = await ingest_from_s3(request.bucket, request.key, session, background_tasks)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("S3 ingestion failed for s3://%s/%s", request.bucket, request.key)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)[:200]}")

    return IngestResult(
        document_id=result.document_id,
        filename=result.filename,
        status=result.status,
        message=result.message,
        chunk_count=result.chunk_count,
        page_count=result.page_count,
        embedding_error=result.embedding_error,
    )


@router.post("/batch", response_model=IngestBatchResponse, status_code=201)
async def ingest_batch(
    request: IngestBatchUrlRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_db),
) -> IngestBatchResponse:
    """Ingest multiple PDF documents from URLs.

    Processes each URL sequentially. Partial failures do not stop
    the batch -- failed items are included with error status.
    """
    results: list[IngestResult] = []
    succeeded = 0
    failed = 0

    for url in request.urls:
        try:
            result = await ingest_from_url(url, session, background_tasks)
            results.append(
                IngestResult(
                    document_id=result.document_id,
                    filename=result.filename,
                    status=result.status,
                    message=result.message,
                    chunk_count=result.chunk_count,
                    page_count=result.page_count,
                    embedding_error=result.embedding_error,
                )
            )
            if result.status != "error":
                succeeded += 1
            else:
                failed += 1
        except Exception as e:
            logger.error("Batch ingestion failed for %s: %s", url, e)
            results.append(
                IngestResult(
                    document_id=0,
                    filename=url.rsplit("/", 1)[-1] or "unknown",
                    status="error",
                    message=f"Download failed: {str(e)[:200]}",
                )
            )
            failed += 1

    return IngestBatchResponse(
        results=results,
        total=len(request.urls),
        succeeded=succeeded,
        failed=failed,
    )
