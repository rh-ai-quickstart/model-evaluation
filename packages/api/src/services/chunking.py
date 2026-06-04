"""Text chunking for RAG pipeline.

Provides two chunking strategies:
- chunk_text(): Simple word-count splitting with overlap (fallback).
- section_chunk_text(): Structure-aware splitting that detects headers
  from typography/format patterns and preserves section hierarchy.
"""

import re

# nomic-embed-text-v1.5 supports up to 8192 tokens (~3000 words).
# 400 words ≈ 600 tokens: large enough to capture full regulatory paragraphs.
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80

# Average body line length threshold: lines significantly shorter are likely headings.
_HEADING_MAX_WORDS = 12


def chunk_text(
    text: str,
    source_document: str,
    page_number: str | None = None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """Split text into overlapping chunks.

    Args:
        text: The source text to chunk.
        source_document: Filename for provenance tracking.
        page_number: Optional page number string.
        chunk_size: Target number of words per chunk.
        chunk_overlap: Number of overlapping words between chunks.

    Returns:
        List of chunk dicts with 'text', 'source_document',
        'page_number', 'token_count' keys.
    """
    words = text.split()
    if not words:
        return []

    # If text fits in a single chunk, return it as-is
    if len(words) <= chunk_size:
        return [
            {
                "text": text,
                "source_document": source_document,
                "page_number": page_number,
                "token_count": len(words),
            }
        ]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunks.append(
            {
                "text": " ".join(chunk_words),
                "source_document": source_document,
                "page_number": page_number,
                "token_count": len(chunk_words),
            }
        )
        # Advance by (chunk_size - overlap), but at least 1 word
        step = max(chunk_size - chunk_overlap, 1)
        start += step

    return chunks


def _is_heading(line: str) -> bool:
    """Detect if a line is likely a heading based on typography/format.

    Heuristics (not regulatory-specific):
    - ALL CAPS lines (3+ words, no lowercase)
    - Numbered section patterns (1.0, 1.1, Article I, Section 2, etc.)
    - Short lines that look like titles (<=12 words, no trailing period)
    """
    stripped = line.strip()
    if not stripped or len(stripped) < 3:
        return False

    words = stripped.split()

    # ALL CAPS: at least 2 alpha words, all uppercase
    alpha_words = [w for w in words if any(c.isalpha() for c in w)]
    if len(alpha_words) >= 2 and all(w == w.upper() for w in alpha_words):
        return True

    # Numbered section patterns: "1.0", "1.1.2", "Article I", "Section 2"
    if re.match(r"^\d+(\.\d+)+\.?\s", stripped):
        return True
    if re.match(r"^(Article|Section|Part|Chapter|Item)\s+[\dIVXLCDMivxlcdm]+", stripped, re.I):
        return True

    # Short line heuristic: title-like if short, doesn't end with period,
    # and has some capitalized words
    if len(words) <= _HEADING_MAX_WORDS and not stripped.endswith("."):
        cap_count = sum(1 for w in words if w[0].isupper())
        if cap_count >= len(words) * 0.5:
            return True

    return False


def section_chunk_text(
    text: str,
    source_document: str,
    page_number: str | None = None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """Split text into chunks using section-aware boundaries.

    Detects headings from typography/format patterns (ALL CAPS, numbered
    sections, short title-like lines) and uses them as chunk boundaries.
    Preserves section_path metadata on each chunk.

    Falls back to chunk_text() if no sections are detected.

    Args:
        text: The source text to chunk.
        source_document: Filename for provenance tracking.
        page_number: Optional page number string.
        chunk_size: Target number of words per chunk.
        chunk_overlap: Number of overlapping words between chunks.

    Returns:
        List of chunk dicts with 'text', 'source_document',
        'page_number', 'section_path', 'token_count' keys.
    """
    lines = text.split("\n")
    sections: list[tuple[str, list[str]]] = []  # (heading, body_lines)
    current_heading = ""
    current_body: list[str] = []

    for line in lines:
        if _is_heading(line) and line.strip():
            # Flush previous section
            if current_body or current_heading:
                sections.append((current_heading, current_body))
            current_heading = line.strip()
            current_body = []
        else:
            if line.strip():
                current_body.append(line.strip())

    # Flush last section
    if current_body or current_heading:
        sections.append((current_heading, current_body))

    # If no sections detected (or only one with no heading), fall back
    if len(sections) <= 1 and not sections[0][0] if sections else True:
        return chunk_text(text, source_document, page_number, chunk_size, chunk_overlap)

    result: list[dict] = []
    for heading, body_lines in sections:
        body_text = " ".join(body_lines)
        if not body_text.strip():
            continue

        sub_chunks = chunk_text(body_text, source_document, page_number, chunk_size, chunk_overlap)
        for chunk in sub_chunks:
            chunk["section_path"] = heading
            if heading:
                chunk["text"] = f"{heading}\n{chunk['text']}"
        result.extend(sub_chunks)

    return (
        result
        if result
        else chunk_text(text, source_document, page_number, chunk_size, chunk_overlap)
    )
