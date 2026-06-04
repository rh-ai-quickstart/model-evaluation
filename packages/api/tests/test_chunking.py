"""Tests for text chunking service."""

from src.services.chunking import CHUNK_SIZE, _is_heading, chunk_text, section_chunk_text


def test_short_text_returns_single_chunk():
    """Should return one chunk when text is shorter than chunk_size."""
    result = chunk_text("Hello world", source_document="test.pdf")
    assert len(result) == 1
    assert result[0]["text"] == "Hello world"
    assert result[0]["token_count"] == 2
    assert result[0]["source_document"] == "test.pdf"


def test_empty_text_returns_empty():
    """Should return no chunks for empty text."""
    assert chunk_text("", source_document="test.pdf") == []
    assert chunk_text("   ", source_document="test.pdf") == []


def test_long_text_produces_multiple_chunks():
    """Should split long text into overlapping chunks."""
    words = [f"word{i}" for i in range(1000)]
    text = " ".join(words)
    result = chunk_text(text, source_document="test.pdf")

    assert len(result) > 1
    # Each chunk should be at most chunk_size words
    for chunk in result:
        assert chunk["token_count"] <= CHUNK_SIZE


def test_chunks_overlap():
    """Consecutive chunks should share overlapping words."""
    words = [f"w{i}" for i in range(1024)]
    text = " ".join(words)
    result = chunk_text(text, source_document="test.pdf", chunk_size=512, chunk_overlap=64)

    assert len(result) >= 2
    first_words = set(result[0]["text"].split())
    second_words = set(result[1]["text"].split())
    overlap = first_words & second_words
    assert len(overlap) >= 64


def test_page_number_preserved():
    """Should preserve page number in chunk metadata."""
    result = chunk_text("Some text", source_document="doc.pdf", page_number="3")
    assert result[0]["page_number"] == "3"


def test_all_text_covered():
    """Every word from the input should appear in at least one chunk."""
    words = [f"w{i}" for i in range(800)]
    text = " ".join(words)
    result = chunk_text(text, source_document="test.pdf")

    all_chunk_words = set()
    for chunk in result:
        all_chunk_words.update(chunk["text"].split())

    assert all_chunk_words == set(words)


# --- Structure-aware chunking tests ---


def test_is_heading_all_caps():
    """Should detect ALL CAPS lines as headings."""
    assert _is_heading("RISK FACTORS") is True
    assert _is_heading("ITEM 1A RISK FACTORS") is True
    assert _is_heading("This is a normal sentence.") is False


def test_is_heading_numbered_section():
    """Should detect numbered section patterns as headings."""
    assert _is_heading("1.1 Overview") is True
    assert _is_heading("2.3.1 Capital Requirements") is True
    assert _is_heading("Section 4 Compliance") is True
    assert _is_heading("Article III Definitions") is True


def test_is_heading_short_title():
    """Should detect short title-like lines as headings."""
    assert _is_heading("Executive Summary") is True
    assert (
        _is_heading("This is a much longer line that is clearly body text and not a heading.")
        is False
    )


def test_section_chunk_preserves_section_path():
    """Should tag chunks with section_path from detected headings."""
    text = """RISK FACTORS
The company faces several risks including market volatility and regulatory changes.
These risks could materially affect our financial condition.

FINANCIAL STATEMENTS
Revenue for the quarter was $100M representing a 15% increase.
Operating expenses totaled $80M."""

    result = section_chunk_text(text, source_document="10k.pdf", page_number="5")
    assert len(result) >= 2

    risk_chunks = [c for c in result if c.get("section_path") == "RISK FACTORS"]
    fin_chunks = [c for c in result if c.get("section_path") == "FINANCIAL STATEMENTS"]
    assert len(risk_chunks) >= 1
    assert len(fin_chunks) >= 1
    assert "risks" in risk_chunks[0]["text"].lower()
    assert "revenue" in fin_chunks[0]["text"].lower()


def test_section_chunk_falls_back_for_plain_text():
    """Should fall back to simple chunking when no sections are detected."""
    text = "This is just a plain paragraph with no headings or structure at all."
    result = section_chunk_text(text, source_document="plain.pdf")
    assert len(result) >= 1
    assert result[0].get("section_path") is None


def test_section_chunk_numbered_sections():
    """Should handle numbered section patterns."""
    text = """1.1 Purpose
This document defines the compliance requirements for all trading activities.

1.2 Scope
These requirements apply to all registered representatives and their supervisors."""

    result = section_chunk_text(text, source_document="policy.pdf")
    assert any(c.get("section_path") == "1.1 Purpose" for c in result)
    assert any(c.get("section_path") == "1.2 Scope" for c in result)
