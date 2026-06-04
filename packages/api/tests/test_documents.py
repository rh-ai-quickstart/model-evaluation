"""Tests for document management endpoints (/documents)."""

import io
from unittest.mock import AsyncMock, patch

from src.services.document_parser import ChunkData, ParseResult

# --- Helpers ---


def _make_pdf_bytes() -> bytes:
    """Create a minimal valid PDF file."""
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _mock_parse(chunks: list[ChunkData], page_count: int = 1, parser: str = "pypdf"):
    """Return a patch that replaces parse_pdf with a mock ParseResult."""
    return patch(
        "src.routes.documents.parse_pdf",
        return_value=ParseResult(chunks=chunks, page_count=page_count, parser_used=parser),
    )


def _mock_bg_embed_noop():
    """Return a patch that makes background embedding a no-op."""
    return patch(
        "src.routes.documents._generate_embeddings_for_document",
        new_callable=AsyncMock,
    )


# --- Upload tests ---


def test_upload_pdf_success(client):
    """Should upload a PDF and return document info with processing status."""
    pdf = _make_pdf_bytes()
    mock_chunks = [
        ChunkData(
            text="Page one content", source_document="test.pdf", page_number="1", token_count=3
        ),
        ChunkData(
            text="Page two content", source_document="test.pdf", page_number="2", token_count=3
        ),
    ]
    with _mock_parse(mock_chunks, page_count=2), _mock_bg_embed_noop():
        response = client.post(
            "/documents/upload",
            files={"file": ("test.pdf", pdf, "application/pdf")},
        )

    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "processing"
    assert data["filename"] == "test.pdf"
    assert data["document_id"] >= 1
    assert "2 chunks" in data["message"]


def test_upload_rejects_non_pdf(client):
    """Should return 400 when uploading a non-PDF file."""
    response = client.post(
        "/documents/upload",
        files={"file": ("readme.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


def test_upload_handles_parsing_error(client):
    """Should return error status when PDF parsing fails."""
    pdf = _make_pdf_bytes()
    with patch(
        "src.routes.documents.parse_pdf",
        side_effect=Exception("corrupt pdf"),
    ):
        response = client.post(
            "/documents/upload",
            files={"file": ("bad.pdf", pdf, "application/pdf")},
        )

    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "error"
    assert "corrupt pdf" in data["message"]


# --- List tests ---


def test_list_documents_empty(client):
    """Should return empty list when no documents exist."""
    response = client.get("/documents/")
    assert response.status_code == 200
    assert response.json() == []


def test_list_documents_after_upload(client):
    """Should include uploaded document in list."""
    pdf = _make_pdf_bytes()
    mock_chunks = [
        ChunkData(text="content", source_document="doc.pdf", page_number="1", token_count=1)
    ]
    with _mock_parse(mock_chunks), _mock_bg_embed_noop():
        client.post(
            "/documents/upload",
            files={"file": ("doc.pdf", pdf, "application/pdf")},
        )

    response = client.get("/documents/")
    assert response.status_code == 200
    docs = response.json()
    assert len(docs) == 1
    assert docs[0]["filename"] == "doc.pdf"
    assert docs[0]["status"] == "processing"


# --- Get / Status / Delete tests ---


def test_get_document_by_id(client):
    """Should return a specific document by ID."""
    pdf = _make_pdf_bytes()
    mock_chunks = [
        ChunkData(text="content", source_document="doc.pdf", page_number="1", token_count=1)
    ]
    with _mock_parse(mock_chunks), _mock_bg_embed_noop():
        upload = client.post(
            "/documents/upload",
            files={"file": ("doc.pdf", pdf, "application/pdf")},
        )
    doc_id = upload.json()["document_id"]

    response = client.get(f"/documents/{doc_id}")
    assert response.status_code == 200
    assert response.json()["filename"] == "doc.pdf"


def test_get_document_not_found(client):
    """Should return 404 for non-existent document."""
    response = client.get("/documents/999")
    assert response.status_code == 404


def test_document_status(client):
    """Should return processing status for a document."""
    pdf = _make_pdf_bytes()
    mock_chunks = [
        ChunkData(text="content", source_document="doc.pdf", page_number="1", token_count=1)
    ]
    with _mock_parse(mock_chunks), _mock_bg_embed_noop():
        upload = client.post(
            "/documents/upload",
            files={"file": ("doc.pdf", pdf, "application/pdf")},
        )
    doc_id = upload.json()["document_id"]

    response = client.get(f"/documents/{doc_id}/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "processing"
    assert data["chunk_count"] == 1


def test_delete_document(client):
    """Should soft-delete a document so it no longer appears in list."""
    pdf = _make_pdf_bytes()
    mock_chunks = [
        ChunkData(text="content", source_document="doc.pdf", page_number="1", token_count=1)
    ]
    with _mock_parse(mock_chunks), _mock_bg_embed_noop():
        upload = client.post(
            "/documents/upload",
            files={"file": ("doc.pdf", pdf, "application/pdf")},
        )
    doc_id = upload.json()["document_id"]

    response = client.delete(f"/documents/{doc_id}")
    assert response.status_code == 204

    # Should no longer appear in list
    response = client.get("/documents/")
    assert response.json() == []

    # Should return 404 on direct get
    response = client.get(f"/documents/{doc_id}")
    assert response.status_code == 404


def test_delete_document_not_found(client):
    """Should return 404 when deleting non-existent document."""
    response = client.delete("/documents/999")
    assert response.status_code == 404
