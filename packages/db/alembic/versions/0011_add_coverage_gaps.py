"""Add coverage_gaps JSON column to eval_result.

Stores per-question coverage gap analysis: which key concepts from the
expected answer are covered vs missing in the model's actual answer.
Format: {"concepts": [...], "covered": [...], "missing": [...]}

Revision ID: 0011
Revises: 0010
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "eval_result",
        sa.Column("coverage_gaps", sa.JSON().with_variant(JSONB, "postgresql"), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("eval_result", "coverage_gaps")
