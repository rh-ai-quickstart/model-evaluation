"""Add truth_payload column to eval_result.

Stores the frozen structured truth (answer_truth, retrieval_truth, metadata)
alongside each evaluation result so it persists through reruns and is
available for UI display without re-generation.

Revision ID: 0014
Revises: 0013
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "eval_result",
        sa.Column(
            "truth_payload",
            sa.JSON().with_variant(JSONB, "postgresql"),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("eval_result", "truth_payload")
