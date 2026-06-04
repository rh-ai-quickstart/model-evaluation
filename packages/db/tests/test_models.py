
"""Model schema tests (no database required)."""

# NOTE: These tests verify ORM metadata configuration. They complement (but don't replace) integration tests that exercise actual database operations.

from db.models import ModelConfig


def test_model_config_table_name():
    """ModelConfig should map to model_config table."""
    assert ModelConfig.__tablename__ == "model_config"


def test_model_config_has_required_columns():
    """ModelConfig should have all expected columns."""
    columns = {c.name for c in ModelConfig.__table__.columns}
    expected = {"id", "name", "endpoint_url", "deployment_mode", "is_active", "created_at", "updated_at"}
    assert expected == columns


def test_model_config_name_is_unique():
    """Model name should be unique to prevent duplicate entries."""
    name_col = ModelConfig.__table__.c.name
    assert name_col.unique is True


def test_model_config_repr():
    """ModelConfig repr should show key fields."""
    m = ModelConfig(id=1, name="test-model", deployment_mode="maas")
    assert "test-model" in repr(m)
    assert "maas" in repr(m)


# --- Document model tests ---


def test_document_table_name():
    """Document should map to document table."""
    from db.models import Document

    assert Document.__tablename__ == "document"


def test_document_has_required_columns():
    """Document should have all expected columns."""
    from db.models import Document

    columns = {c.name for c in Document.__table__.columns}
    expected = {
        "id", "filename", "status", "chunk_count", "page_count",
        "file_size_bytes", "error_message", "created_at", "deleted_at",
    }
    assert expected == columns


def test_document_repr():
    """Document repr should show key fields."""
    from db.models import Document

    d = Document(id=1, filename="test.pdf", status="ready")
    assert "test.pdf" in repr(d)
    assert "ready" in repr(d)


# --- Chunk model tests ---


def test_chunk_table_name():
    """Chunk should map to chunk table."""
    from db.models import Chunk

    assert Chunk.__tablename__ == "chunk"


def test_chunk_has_required_columns():
    """Chunk should have all expected columns."""
    from db.models import Chunk

    columns = {c.name for c in Chunk.__table__.columns}
    expected = {
        "id", "document_id", "text", "source_document", "page_number",
        "section_path", "element_type", "token_count", "embedding", "created_at",
    }
    assert expected == columns


def test_chunk_document_id_has_foreign_key():
    """Chunk.document_id should reference document.id."""
    from db.models import Chunk

    fk = list(Chunk.__table__.c.document_id.foreign_keys)
    assert len(fk) == 1
    assert str(fk[0].column) == "document.id"


# --- EvalRun model tests ---


def test_eval_run_table_name():
    """EvalRun should map to eval_run table."""
    from db.models import EvalRun

    assert EvalRun.__tablename__ == "eval_run"


def test_eval_run_has_required_columns():
    """EvalRun should have all expected columns."""
    from db.models import EvalRun

    columns = {c.name for c in EvalRun.__table__.columns}
    expected = {
        "id", "model_name", "question_set_id", "status",
        "total_questions", "completed_questions",
        "avg_latency_ms", "avg_relevancy", "avg_groundedness",
        "avg_context_precision", "avg_context_relevancy",
        "avg_completeness", "avg_correctness", "avg_compliance_accuracy",
        "avg_abstention", "avg_chunk_alignment",
        "hallucination_rate", "total_tokens",
        "profile_id", "profile_version",
        "overall_verdict", "pass_count", "fail_count", "review_count",
        "error_message", "created_at", "completed_at",
    }
    assert expected == columns


def test_eval_run_repr():
    """EvalRun repr should show key fields."""
    from db.models import EvalRun

    r = EvalRun(id=1, model_name="granite", status="completed")
    assert "granite" in repr(r)
    assert "completed" in repr(r)


# --- EvalResult model tests ---


def test_eval_result_table_name():
    """EvalResult should map to eval_result table."""
    from db.models import EvalResult

    assert EvalResult.__tablename__ == "eval_result"


def test_eval_result_has_required_columns():
    """EvalResult should have all expected columns."""
    from db.models import EvalResult

    columns = {c.name for c in EvalResult.__table__.columns}
    expected = {
        "id", "eval_run_id", "question", "expected_answer", "answer", "contexts",
        "latency_ms", "relevancy_score", "groundedness_score",
        "context_precision_score", "context_relevancy_score",
        "completeness_score", "correctness_score", "compliance_accuracy_score",
        "abstention_score", "chunk_alignment_score",
        "is_hallucination", "verdict", "fail_reasons", "total_tokens",
        "error_message", "created_at",
    }
    assert expected == columns


def test_eval_result_has_foreign_key():
    """EvalResult.eval_run_id should reference eval_run.id."""
    from db.models import EvalResult

    fk = list(EvalResult.__table__.c.eval_run_id.foreign_keys)
    assert len(fk) == 1
    assert str(fk[0].column) == "eval_run.id"
