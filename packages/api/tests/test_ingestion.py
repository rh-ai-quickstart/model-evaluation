"""Tests for document ingestion service."""

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import BackgroundTasks

from src.services.ingestion import (
    IngestionResult,
    download_from_s3,
    download_from_url,
    process_and_store,
)

# Minimal valid PDF header for validation tests
_VALID_PDF = b"%PDF-1.4 fake content"


# --- download_from_url tests ---


@pytest.mark.asyncio
async def test_download_url_returns_content_and_filename():
    """Should return PDF bytes and extract filename from URL path."""
    mock_response = MagicMock()
    mock_response.content = _VALID_PDF
    mock_response.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("src.services.ingestion.httpx.AsyncClient", return_value=mock_client):
        content, filename = await download_from_url("https://example.com/docs/report.pdf")

    assert content == _VALID_PDF
    assert filename == "report.pdf"


@pytest.mark.asyncio
async def test_download_url_appends_pdf_extension():
    """Should append .pdf when URL path has no extension."""
    mock_response = MagicMock()
    mock_response.content = _VALID_PDF
    mock_response.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("src.services.ingestion.httpx.AsyncClient", return_value=mock_client):
        _, filename = await download_from_url("https://example.com/docs/report")

    assert filename == "report.pdf"


@pytest.mark.asyncio
async def test_download_url_rejects_invalid_url():
    """Should raise ValueError for malformed URLs."""
    with pytest.raises(ValueError, match="Invalid URL"):
        await download_from_url("not-a-url")


@pytest.mark.asyncio
async def test_download_url_rejects_non_pdf():
    """Should raise ValueError when downloaded content is not a PDF."""
    mock_response = MagicMock()
    mock_response.content = b"<html>not a pdf</html>"
    mock_response.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("src.services.ingestion.httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(ValueError, match="not a valid PDF"):
            await download_from_url("https://example.com/page.html")


@pytest.mark.asyncio
async def test_download_url_rejects_oversized_file():
    """Should raise ValueError when file exceeds size limit."""
    mock_response = MagicMock()
    mock_response.content = b"%PDF" + b"x" * (51 * 1024 * 1024)
    mock_response.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("src.services.ingestion.httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(ValueError, match="exceeds"):
            await download_from_url("https://example.com/huge.pdf")


# --- download_from_s3 tests ---


@pytest.mark.asyncio
async def test_download_s3_raises_when_not_configured():
    """Should raise RuntimeError when S3 is not configured."""
    with patch("src.services.ingestion.settings") as mock_settings:
        mock_settings.S3_ENDPOINT_URL = ""
        with pytest.raises(RuntimeError, match="S3 is not configured"):
            await download_from_s3("bucket", "key.pdf")


@pytest.mark.asyncio
async def test_download_s3_returns_content_and_filename():
    """Should download from S3 and return content with filename from key."""
    mock_body = MagicMock()
    mock_body.read.return_value = _VALID_PDF

    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = {"Body": mock_body}

    with patch("src.services.ingestion.settings") as mock_settings:
        mock_settings.S3_ENDPOINT_URL = "https://minio.local:9000"
        mock_settings.S3_ACCESS_KEY = "access"
        mock_settings.S3_SECRET_KEY = "secret"

        with patch("boto3.client", return_value=mock_s3):
            content, filename = await download_from_s3("my-bucket", "docs/filing.pdf")

    assert content == _VALID_PDF
    assert filename == "filing.pdf"
    mock_s3.get_object.assert_called_once_with(Bucket="my-bucket", Key="docs/filing.pdf")


# --- process_and_store tests ---


@pytest.mark.asyncio
async def test_process_and_store_success(_setup_db):
    """Should parse and store a document with processing status."""
    _, async_session = _setup_db

    @dataclass
    class FakeChunk:
        text: str = "chunk text"
        source_document: str = "test.pdf"
        page_number: str | None = "1"
        section_path: str | None = None
        element_type: str = "paragraph"
        token_count: int = 5

    @dataclass
    class FakeParseResult:
        chunks: list = None
        page_count: int = 1
        parser_used: str = "pypdf"
        error: str | None = None

        def __post_init__(self):
            if self.chunks is None:
                self.chunks = [FakeChunk()]

    bg_tasks = BackgroundTasks()

    with (
        patch("src.services.ingestion.parse_pdf", return_value=FakeParseResult()),
        patch("src.routes.documents._generate_embeddings_for_document", new_callable=AsyncMock),
    ):
        async with async_session() as session:
            result = await process_and_store(_VALID_PDF, "test.pdf", session, bg_tasks)

    assert result.status == "processing"
    assert result.chunk_count == 1
    assert result.document_id >= 1
    assert "1 chunks" in result.message


@pytest.mark.asyncio
async def test_process_and_store_handles_parse_error(_setup_db):
    """Should return error status when parsing fails."""
    _, async_session = _setup_db

    bg_tasks = BackgroundTasks()

    with patch("src.services.ingestion.parse_pdf", side_effect=RuntimeError("bad pdf")):
        async with async_session() as session:
            result = await process_and_store(b"%PDF-broken", "bad.pdf", session, bg_tasks)

    assert result.status == "error"
    assert "parsing failed" in result.message.lower()


# --- Endpoint tests ---


def _mock_ingest_url_success():
    """Patch ingest_from_url to return a successful result."""
    return patch(
        "src.routes.ingestion.ingest_from_url",
        new_callable=AsyncMock,
        return_value=IngestionResult(
            document_id=1,
            filename="report.pdf",
            status="processing",
            message="Parsed 5 chunks from 3 pages (parser: pypdf). Embedding generation in progress.",
            chunk_count=5,
            page_count=3,
        ),
    )


def test_ingest_url_endpoint(client):
    """Should ingest a PDF from URL and return document details."""
    with _mock_ingest_url_success():
        response = client.post("/ingest/url", json={"url": "https://example.com/report.pdf"})

    assert response.status_code == 201
    data = response.json()
    assert data["document_id"] == 1
    assert data["filename"] == "report.pdf"
    assert data["status"] == "processing"
    assert data["chunk_count"] == 5


def test_ingest_url_rejects_invalid_url(client):
    """Should return 400 for download validation errors."""
    with patch(
        "src.routes.ingestion.ingest_from_url",
        new_callable=AsyncMock,
        side_effect=ValueError("Invalid URL: bad"),
    ):
        response = client.post("/ingest/url", json={"url": "bad"})

    assert response.status_code == 400
    assert "Invalid URL" in response.json()["detail"]


def test_ingest_url_validates_empty_url(client):
    """Should return 422 for empty URL."""
    response = client.post("/ingest/url", json={"url": ""})
    assert response.status_code == 422


def test_ingest_s3_endpoint(client):
    """Should ingest a PDF from S3 and return document details."""
    with patch(
        "src.routes.ingestion.ingest_from_s3",
        new_callable=AsyncMock,
        return_value=IngestionResult(
            document_id=2,
            filename="filing.pdf",
            status="processing",
            message="Parsed 10 chunks from 5 pages (parser: pypdf). Embedding generation in progress.",
            chunk_count=10,
            page_count=5,
        ),
    ):
        response = client.post("/ingest/s3", json={"bucket": "my-bucket", "key": "docs/filing.pdf"})

    assert response.status_code == 201
    data = response.json()
    assert data["document_id"] == 2
    assert data["filename"] == "filing.pdf"
    assert data["chunk_count"] == 10


def test_ingest_s3_rejects_when_not_configured(client):
    """Should return 400 when S3 is not configured."""
    with patch(
        "src.routes.ingestion.ingest_from_s3",
        new_callable=AsyncMock,
        side_effect=RuntimeError("S3 is not configured"),
    ):
        response = client.post("/ingest/s3", json={"bucket": "bucket", "key": "key.pdf"})

    assert response.status_code == 400
    assert "S3 is not configured" in response.json()["detail"]


def test_ingest_batch_endpoint(client):
    """Should ingest multiple URLs and return batch results."""
    with _mock_ingest_url_success():
        response = client.post(
            "/ingest/batch",
            json={"urls": ["https://example.com/a.pdf", "https://example.com/b.pdf"]},
        )

    assert response.status_code == 201
    data = response.json()
    assert data["total"] == 2
    assert data["succeeded"] == 2
    assert data["failed"] == 0
    assert len(data["results"]) == 2


def test_ingest_batch_partial_failure(client):
    """Should continue batch when individual URLs fail."""
    call_count = 0

    async def _alternate_results(url, session, background_tasks):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return IngestionResult(
                document_id=1,
                filename="good.pdf",
                status="processing",
                message="OK",
                chunk_count=3,
                page_count=1,
            )
        raise ValueError("Download failed")

    with patch("src.routes.ingestion.ingest_from_url", side_effect=_alternate_results):
        response = client.post(
            "/ingest/batch",
            json={"urls": ["https://example.com/good.pdf", "https://example.com/bad.pdf"]},
        )

    assert response.status_code == 201
    data = response.json()
    assert data["succeeded"] == 1
    assert data["failed"] == 1
    assert data["results"][0]["status"] == "processing"
    assert data["results"][1]["status"] == "error"


def test_ingest_batch_validates_empty_urls(client):
    """Should return 422 for empty URL list."""
    response = client.post("/ingest/batch", json={"urls": []})
    assert response.status_code == 422
