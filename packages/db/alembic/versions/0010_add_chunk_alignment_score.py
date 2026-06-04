"""Add chunk alignment score columns.

Stores per-question retrieval quality (chunk_alignment_score on eval_result)
and run-level average (avg_chunk_alignment on eval_run). Score measures
how well retrieved chunks match expected source chunks.

Revision ID: 0010
Revises: 0009
"""

import sqlalchemy as sa
from alembic import op

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("eval_result", sa.Column("chunk_alignment_score", sa.Float(), nullable=True))
    op.add_column("eval_run", sa.Column("avg_chunk_alignment", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("eval_run", "avg_chunk_alignment")
    op.drop_column("eval_result", "chunk_alignment_score")
