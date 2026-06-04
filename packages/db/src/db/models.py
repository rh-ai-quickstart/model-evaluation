
"""Database models for model evaluation."""

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from .database import Base

EMBEDDING_DIMENSION = 768


class ModelConfig(Base):
    """Configuration for a served model available for evaluation."""

    __tablename__ = "model_config"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200), nullable=False, unique=True)
    endpoint_url = Column(String(500), nullable=False)
    deployment_mode = Column(String(50), nullable=False, default="maas")
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    def __repr__(self) -> str:
        return f"<ModelConfig(id={self.id}, name='{self.name}', mode='{self.deployment_mode}')>"


class Document(Base):
    """An uploaded document in the RAG knowledge base."""

    __tablename__ = "document"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(500), nullable=False)
    status = Column(String(50), nullable=False, default="processing")
    chunk_count = Column(Integer, nullable=False, default=0)
    page_count = Column(Integer, nullable=True)
    file_size_bytes = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    deleted_at = Column(DateTime, nullable=True)

    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Document(id={self.id}, filename='{self.filename}', status='{self.status}')>"


class Chunk(Base):
    """A text chunk extracted from a document, with optional embedding."""

    __tablename__ = "chunk"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("document.id", ondelete="CASCADE"), nullable=False, index=True)
    text = Column(Text, nullable=False)
    source_document = Column(String(500), nullable=False)
    page_number = Column(String(20), nullable=True)
    section_path = Column(Text, nullable=True)
    element_type = Column(String(50), nullable=False, default="paragraph")
    token_count = Column(Integer, nullable=False, default=0)
    embedding = Column(
        Vector(EMBEDDING_DIMENSION).with_variant(LargeBinary, "sqlite"),
        nullable=True,
    )
    created_at = Column(DateTime, server_default=func.now())

    document = relationship("Document", back_populates="chunks")


class QuestionSet(Base):
    """A reusable set of evaluation questions."""

    __tablename__ = "question_set"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200), nullable=False)
    questions = Column(JSON().with_variant(JSONB, "postgresql"), nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    eval_runs = relationship("EvalRun", back_populates="question_set", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<QuestionSet(id={self.id}, name='{self.name}', count={len(self.questions)})>"


class EvalRun(Base):
    """A single evaluation run against a model using a set of test questions."""

    __tablename__ = "eval_run"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(200), nullable=False)
    question_set_id = Column(Integer, ForeignKey("question_set.id", ondelete="CASCADE"), nullable=True, index=True)
    status = Column(String(50), nullable=False, default="pending")
    total_questions = Column(Integer, nullable=False, default=0)
    completed_questions = Column(Integer, nullable=False, default=0)
    avg_latency_ms = Column(Float, nullable=True)
    avg_relevancy = Column(Float, nullable=True)
    avg_groundedness = Column(Float, nullable=True)
    avg_context_precision = Column(Float, nullable=True)
    avg_context_relevancy = Column(Float, nullable=True)
    avg_completeness = Column(Float, nullable=True)
    avg_correctness = Column(Float, nullable=True)
    avg_compliance_accuracy = Column(Float, nullable=True)
    avg_abstention = Column(Float, nullable=True)
    hallucination_rate = Column(Float, nullable=True)
    avg_chunk_alignment = Column(Float, nullable=True)
    total_tokens = Column(Integer, nullable=True)
    profile_id = Column(String(200), nullable=True)
    profile_version = Column(String(100), nullable=True)
    judge_model_name = Column(String(200), nullable=True)
    synthesis_model_name = Column(String(200), nullable=True)
    retrieval_config = Column(JSON().with_variant(JSONB, "postgresql"), nullable=True)
    corpus_snapshot = Column(JSON().with_variant(JSONB, "postgresql"), nullable=True)
    overall_verdict = Column(String(50), nullable=True)
    pass_count = Column(Integer, nullable=True)
    fail_count = Column(Integer, nullable=True)
    review_count = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    completed_at = Column(DateTime, nullable=True)

    question_set = relationship("QuestionSet", back_populates="eval_runs", lazy="joined")
    results = relationship("EvalResult", back_populates="eval_run", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<EvalRun(id={self.id}, model='{self.model_name}', status='{self.status}')>"


class EvalResult(Base):
    """Result for a single question within an evaluation run."""

    __tablename__ = "eval_result"

    id = Column(Integer, primary_key=True, autoincrement=True)
    eval_run_id = Column(Integer, ForeignKey("eval_run.id", ondelete="CASCADE"), nullable=False, index=True)
    question = Column(Text, nullable=False)
    expected_answer = Column(Text, nullable=True)
    answer = Column(Text, nullable=True)
    contexts = Column(Text, nullable=True)
    latency_ms = Column(Float, nullable=True)
    relevancy_score = Column(Float, nullable=True)
    groundedness_score = Column(Float, nullable=True)
    context_precision_score = Column(Float, nullable=True)
    context_relevancy_score = Column(Float, nullable=True)
    completeness_score = Column(Float, nullable=True)
    correctness_score = Column(Float, nullable=True)
    compliance_accuracy_score = Column(Float, nullable=True)
    abstention_score = Column(Float, nullable=True)
    is_hallucination = Column(Boolean, nullable=True)
    chunk_alignment_score = Column(Float, nullable=True)
    verdict = Column(String(50), nullable=True)
    fail_reasons = Column(JSON().with_variant(JSONB, "postgresql"), nullable=True)
    coverage_gaps = Column(JSON().with_variant(JSONB, "postgresql"), nullable=True)
    deterministic_checks = Column(JSON().with_variant(JSONB, "postgresql"), nullable=True)
    truth_payload = Column(JSON().with_variant(JSONB, "postgresql"), nullable=True)
    total_tokens = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())

    eval_run = relationship("EvalRun", back_populates="results")
