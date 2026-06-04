"""Add run metadata columns to eval_run for reproducibility.

Stores judge_model_name, synthesis_model_name, retrieval_config,
and corpus_snapshot so that evaluation conditions can be compared
and reproduced.

Revision ID: 0013
Revises: 0012
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0013"
down_revision = "0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "eval_run",
        sa.Column("judge_model_name", sa.String(200), nullable=True),
    )
    op.add_column(
        "eval_run",
        sa.Column("synthesis_model_name", sa.String(200), nullable=True),
    )
    op.add_column(
        "eval_run",
        sa.Column(
            "retrieval_config",
            sa.JSON().with_variant(JSONB, "postgresql"),
            nullable=True,
        ),
    )
    op.add_column(
        "eval_run",
        sa.Column(
            "corpus_snapshot",
            sa.JSON().with_variant(JSONB, "postgresql"),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("eval_run", "corpus_snapshot")
    op.drop_column("eval_run", "retrieval_config")
    op.drop_column("eval_run", "synthesis_model_name")
    op.drop_column("eval_run", "judge_model_name")
