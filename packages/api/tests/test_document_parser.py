"""Tests for the document parser service (Docling + pypdf)."""

import io
from unittest.mock import MagicMock, patch

from src.services.document_parser import (
    ChunkData,
    ParseResult,
    _docling_available,
    _parse_with_pypdf,
    parse_pdf,
)


def _make_pdf_bytes(text: str = "Hello world from test PDF.") -> bytes:
    """Create a minimal valid PDF with text content."""
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    # pypdf PdfWriter doesn't support adding text directly,
    # so we create via reportlab-free approach: use a real PDF with content
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


# --- ChunkData / ParseResult dataclass tests ---


def test_chunk_data_defaults():
    """Should create ChunkData with sensible defaults."""
    chunk = ChunkData(text="hello", source_document="doc.pdf")
    assert chunk.text == "hello"
    assert chunk.source_document == "doc.pdf"
    assert chunk.page_number is None
    assert chunk.section_path is None
    assert chunk.token_count == 0
    assert chunk.element_type == "chunk"
    assert chunk.metadata == {}


def test_parse_result_defaults():
    """Should create ParseResult with required fields."""
    result = ParseResult(chunks=[], page_count=5, parser_used="pypdf")
    assert result.chunks == []
    assert result.page_count == 5
    assert result.parser_used == "pypdf"
    assert result.error is None


# --- _docling_available tests ---


def test_docling_available_when_installed():
    """Should return True when docling is importable."""
    mock_module = MagicMock()
    with patch.dict("sys.modules", {"docling": mock_module}):
        assert _docling_available() is True


def test_docling_unavailable_when_not_installed():
    """Should return False when docling is not importable."""
    with patch.dict("sys.modules", {"docling": None}):
        assert _docling_available() is False


# --- _parse_with_pypdf tests ---


def test_pypdf_parser_returns_chunks():
    """Should extract text and produce chunks via pypdf fallback.

    Pages are merged before chunking, so short texts produce a single chunk.
    """
    with patch("src.services.document_parser.PdfReader") as mock_reader_cls:
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Revenue was $1 billion in Q4 2024."
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "Operating expenses increased by 15%."

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page1, mock_page2]
        mock_reader_cls.return_value = mock_reader

        result = _parse_with_pypdf(b"%PDF-fake", "report.pdf")

    assert result.parser_used == "pypdf"
    assert result.page_count == 2
    assert len(result.chunks) >= 1
    assert result.chunks[0].source_document == "report.pdf"
    assert "Revenue" in result.chunks[0].text
    assert "Operating expenses" in result.chunks[0].text


def test_pypdf_parser_skips_empty_pages():
    """Should skip pages with no text content."""
    with patch("src.services.document_parser.PdfReader") as mock_reader_cls:
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Some content here."
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = ""
        mock_page3 = MagicMock()
        mock_page3.extract_text.return_value = "   "

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page1, mock_page2, mock_page3]
        mock_reader_cls.return_value = mock_reader

        result = _parse_with_pypdf(b"%PDF-fake", "report.pdf")

    assert result.page_count == 3  # total pages in PDF
    assert len(result.chunks) == 1  # only one page had content


def test_pypdf_parser_preserves_section_path():
    """Should preserve section_path from section-aware chunking."""
    # Text with a detectable heading (ALL CAPS)
    text_with_heading = "FINANCIAL SUMMARY\nRevenue was strong this quarter."

    with patch("src.services.document_parser.PdfReader") as mock_reader_cls:
        mock_page = MagicMock()
        mock_page.extract_text.return_value = text_with_heading

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]
        mock_reader_cls.return_value = mock_reader

        result = _parse_with_pypdf(b"%PDF-fake", "report.pdf")

    assert len(result.chunks) >= 1
    # The section chunker should detect the heading
    has_section = any(c.section_path for c in result.chunks)
    assert has_section


def test_pypdf_parser_sets_token_count():
    """Should set token_count based on word count."""
    with patch("src.services.document_parser.PdfReader") as mock_reader_cls:
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "one two three four five"

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]
        mock_reader_cls.return_value = mock_reader

        result = _parse_with_pypdf(b"%PDF-fake", "doc.pdf")

    assert result.chunks[0].token_count == 5


# --- parse_pdf integration tests ---


def test_parse_pdf_uses_pypdf_when_docling_disabled():
    """Should use pypdf when DOCLING_ENABLED is False."""
    with (
        patch("src.services.document_parser.settings") as mock_settings,
        patch("src.services.document_parser.PdfReader") as mock_reader_cls,
    ):
        mock_settings.DOCLING_ENABLED = False

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Test content."
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]
        mock_reader_cls.return_value = mock_reader

        result = parse_pdf(b"%PDF-fake", "test.pdf")

    assert result.parser_used == "pypdf"


def test_parse_pdf_uses_pypdf_when_docling_not_installed():
    """Should fall back to pypdf when docling is not importable."""
    with (
        patch("src.services.document_parser.settings") as mock_settings,
        patch("src.services.document_parser._docling_available", return_value=False),
        patch("src.services.document_parser.PdfReader") as mock_reader_cls,
    ):
        mock_settings.DOCLING_ENABLED = True

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Test content."
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]
        mock_reader_cls.return_value = mock_reader

        result = parse_pdf(b"%PDF-fake", "test.pdf")

    assert result.parser_used == "pypdf"


def test_parse_pdf_falls_back_on_docling_error():
    """Should fall back to pypdf when Docling raises an exception."""
    with (
        patch("src.services.document_parser.settings") as mock_settings,
        patch("src.services.document_parser._docling_available", return_value=True),
        patch(
            "src.services.document_parser._parse_with_docling",
            side_effect=RuntimeError("Docling crashed"),
        ),
        patch("src.services.document_parser.PdfReader") as mock_reader_cls,
    ):
        mock_settings.DOCLING_ENABLED = True

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Fallback content."
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]
        mock_reader_cls.return_value = mock_reader

        result = parse_pdf(b"%PDF-fake", "test.pdf")

    assert result.parser_used == "pypdf"
    assert len(result.chunks) == 1


def test_parse_pdf_uses_docling_when_available():
    """Should use Docling when enabled and available."""
    mock_docling_result = ParseResult(
        chunks=[
            ChunkData(
                text="Docling-parsed content",
                source_document="test.pdf",
                page_number="1",
                token_count=2,
            ),
        ],
        page_count=1,
        parser_used="docling",
    )

    with (
        patch("src.services.document_parser.settings") as mock_settings,
        patch("src.services.document_parser._docling_available", return_value=True),
        patch(
            "src.services.document_parser._parse_with_docling",
            return_value=mock_docling_result,
        ),
    ):
        mock_settings.DOCLING_ENABLED = True

        result = parse_pdf(b"%PDF-fake", "test.pdf")

    assert result.parser_used == "docling"
    assert result.chunks[0].text == "Docling-parsed content"


def test_parse_pdf_empty_pdf():
    """Should return zero chunks for an empty PDF."""
    with (
        patch("src.services.document_parser.settings") as mock_settings,
        patch("src.services.document_parser.PdfReader") as mock_reader_cls,
    ):
        mock_settings.DOCLING_ENABLED = False

        mock_reader = MagicMock()
        mock_reader.pages = []
        mock_reader_cls.return_value = mock_reader

        result = parse_pdf(b"%PDF-fake", "empty.pdf")

    assert result.parser_used == "pypdf"
    assert result.chunks == []
    assert result.page_count == 0


def test_parse_pdf_respects_chunk_size():
    """Should pass chunk_size through to the parser."""
    long_text = " ".join(f"word{i}" for i in range(300))

    with (
        patch("src.services.document_parser.settings") as mock_settings,
        patch("src.services.document_parser.PdfReader") as mock_reader_cls,
    ):
        mock_settings.DOCLING_ENABLED = False

        mock_page = MagicMock()
        mock_page.extract_text.return_value = long_text
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]
        mock_reader_cls.return_value = mock_reader

        result_small = parse_pdf(b"%PDF-fake", "test.pdf", chunk_size=50)

    # With 300 words and chunk_size=50, we should get multiple chunks
    assert len(result_small.chunks) > 1
    for chunk in result_small.chunks:
        assert chunk.token_count <= 50
