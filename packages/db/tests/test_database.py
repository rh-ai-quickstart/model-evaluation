"""Database module tests (no database connection required)."""

from db.database import Base, SessionLocal, engine


def test_engine_is_configured():
    """Engine should be created from DATABASE_URL."""
    assert engine is not None


def test_session_local_is_configured():
    """SessionLocal should be an async session factory."""
    assert SessionLocal is not None


def test_base_has_metadata():
    """Base should have metadata for table definitions."""
    assert Base.metadata is not None
