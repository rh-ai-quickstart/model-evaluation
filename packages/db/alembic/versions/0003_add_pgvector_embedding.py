
"""Add pgvector extension and embedding column to chunk table.

Revision ID: 0003
Revises: 0002
Create Date: 2026-03-24
"""

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

EMBEDDING_DIMENSION = 768


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.add_column("chunk", sa.Column("embedding", Vector(EMBEDDING_DIMENSION), nullable=True))
    op.create_index(
        "idx_chunk_embedding",
        "chunk",
        ["embedding"],
        postgresql_using="ivfflat",
        postgresql_with={"lists": 100},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_index("idx_chunk_embedding", table_name="chunk")
    op.drop_column("chunk", "embedding")
