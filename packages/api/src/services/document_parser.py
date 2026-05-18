# This project was developed with assistance from AI tools.
"""Document parsing with Docling (layout-aware) and pypdf (fallback).

Provides a unified ``parse_pdf()`` entry point that returns structured
chunks from a PDF byte stream.  When ``DOCLING_ENABLED`` is True and the
docling library is importable, the parser uses Docling's
``DocumentConverter`` + ``HybridChunker`` for layout-aware extraction
that preserves tables, multi-column layouts, and figure context.  When
Docling is unavailable or disabled, it falls back to pypdf text
extraction with section-aware chunking.
"""

from __future__ import annotations

import io
import logging
import tempfile
from dataclasses import dataclass, field

from pypdf import PdfReader

from ..core.config import settings
from ..services.chunking import CHUNK_SIZE, section_chunk_text

logger = logging.getLogger(__name__)


@dataclass
class ChunkData:
    """A chunk produced by the document parser."""

    text: str
    source_document: str
    page_number: str | None = None
    section_path: str | None = None
    token_count: int = 0
    element_type: str = "chunk"
    metadata: dict = field(default_factory=dict)


@dataclass
class ParseResult:
    """Result of parsing a PDF document."""

    chunks: list[ChunkData]
    page_count: int
    parser_used: str  # "docling" or "pypdf"
    error: str | None = None


def _docling_available() -> bool:
    """Check whether the docling library can be imported."""
    try:
        import docling  # noqa: F401

        return True
    except ImportError:
        return False


def _parse_with_docling(
    content: bytes,
    source_document: str,
    chunk_size: int = CHUNK_SIZE,
) -> ParseResult:
    """Parse a PDF using Docling's DocumentConverter + HybridChunker.

    Args:
        content: Raw PDF bytes.
        source_document: Filename for provenance tracking.
        chunk_size: Target token count per chunk (passed to HybridChunker).

    Returns:
        ParseResult with layout-aware chunks.
    """
    from docling.chunking import HybridChunker
    from docling.document_converter import DocumentConverter

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
        tmp.write(content)
        tmp.flush()

        converter = DocumentConverter()
        doc_result = converter.convert(tmp.name)

    doc = doc_result.document

    chunker = HybridChunker(max_tokens=chunk_size)
    raw_chunks = list(chunker.chunk(doc))

    chunks: list[ChunkData] = []
    for chunk in raw_chunks:
        text = chunk.text.strip()
        if not text:
            continue

        page_num = None
        heading = None
        if hasattr(chunk, "meta") and chunk.meta:
            meta = chunk.meta
            if hasattr(meta, "headings") and meta.headings:
                heading = " > ".join(meta.headings)
            if hasattr(meta, "page") and meta.page is not None:
                page_num = str(meta.page)

        word_count = len(text.split())
        chunks.append(
            ChunkData(
                text=text,
                source_document=source_document,
                page_number=page_num,
                section_path=heading,
                token_count=word_count,
                element_type="chunk",
            )
        )

    page_count = doc.num_pages() if hasattr(doc, "num_pages") else 0

    return ParseResult(
        chunks=chunks,
        page_count=page_count,
        parser_used="docling",
    )


def _parse_with_pypdf(
    content: bytes,
    source_document: str,
    chunk_size: int = CHUNK_SIZE,
) -> ParseResult:
    """Parse a PDF using pypdf text extraction + section-aware chunking.

    Args:
        content: Raw PDF bytes.
        source_document: Filename for provenance tracking.
        chunk_size: Target number of words per chunk.

    Returns:
        ParseResult with text-based chunks.
    """
    reader = PdfReader(io.BytesIO(content))
    page_texts: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            page_texts.append(text.strip())

    full_text = "\n\n".join(page_texts)
    raw_chunks = section_chunk_text(
        text=full_text,
        source_document=source_document,
        chunk_size=chunk_size,
    )

    all_chunks: list[ChunkData] = []
    for rc in raw_chunks:
        all_chunks.append(
            ChunkData(
                text=rc["text"],
                source_document=rc["source_document"],
                page_number=rc["page_number"],
                section_path=rc.get("section_path"),
                token_count=rc["token_count"],
            )
        )

    return ParseResult(
        chunks=all_chunks,
        page_count=len(reader.pages),
        parser_used="pypdf",
    )


def parse_pdf(
    content: bytes,
    source_document: str,
    chunk_size: int = CHUNK_SIZE,
) -> ParseResult:
    """Parse a PDF document and return structured chunks.

    Uses Docling when ``DOCLING_ENABLED`` is True and the library is
    installed; otherwise falls back to pypdf + section-aware chunking.
    If Docling fails at runtime, the error is logged and pypdf is used.

    Args:
        content: Raw PDF bytes.
        source_document: Filename for provenance tracking.
        chunk_size: Target word/token count per chunk.

    Returns:
        ParseResult containing chunks and metadata.
    """
    if settings.DOCLING_ENABLED and _docling_available():
        try:
            result = _parse_with_docling(content, source_document, chunk_size)
            logger.info(
                "Docling parsed %s: %d chunks from %d pages",
                source_document,
                len(result.chunks),
                result.page_count,
            )
            return result
        except Exception:
            logger.exception("Docling failed for %s, falling back to pypdf", source_document)

    result = _parse_with_pypdf(content, source_document, chunk_size)
    logger.info(
        "pypdf parsed %s: %d chunks from %d pages",
        source_document,
        len(result.chunks),
        result.page_count,
    )
    return result
