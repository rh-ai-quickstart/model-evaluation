"""Pydantic schemas for structured truth payloads.

Truth is a frozen, immutable object attached to each question at creation time.
It splits into two concerns: what the answer must contain (answer_truth) and
what evidence should support it (retrieval_truth).
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

EvidenceMode = Literal[
    "traced_from_synthesis",
    "grounded_from_synthesis",
    "grounded_from_manual_answer",
]


class AnswerTruth(BaseModel):
    """What the answer must contain."""

    required_concepts: list[str] = Field(
        ..., description="Key concepts extracted from the expected answer."
    )
    abstention_expected: bool = Field(
        default=False, description="Whether the model should abstain from answering."
    )


class RetrievalTruth(BaseModel):
    """What evidence should support the answer."""

    required_documents: list[str] = Field(
        default_factory=list, description="Source document filenames that must be retrieved."
    )
    expected_chunk_refs: list[str] = Field(
        default_factory=list, description='Canonical chunk references in "chunk:{id}" format.'
    )
    expected_chunk_texts: list[str] = Field(
        default_factory=list,
        description="Text content of expected chunks for text-based fallback matching.",
    )
    supporting_documents: list[str] = Field(
        default_factory=list,
        description="Documents with marginal relevance (background, examples, secondary details).",
    )
    supporting_chunk_refs: list[str] = Field(
        default_factory=list,
        description='Chunk refs from supporting documents, in "chunk:{id}" format.',
    )
    evidence_mode: EvidenceMode = Field(
        ..., description="How retrieval truth was produced."
    )


class TruthMetadata(BaseModel):
    """Versioning and provenance for the truth payload."""

    truth_schema_version: str = "1.1"
    concept_extraction_version: str = "v1"
    evidence_alignment_version: str = "v1"
    generated_by_model: str = Field(..., description="Model used for concept extraction.")
    generated_at: datetime = Field(..., description="When truth was generated.")
    source_chunk_ids: list[int] = Field(
        default_factory=list, description="Chunk IDs that informed this truth."
    )


class TruthPayload(BaseModel):
    """Complete structured truth for a question with an expected answer."""

    answer_truth: AnswerTruth
    retrieval_truth: RetrievalTruth
    metadata: TruthMetadata
