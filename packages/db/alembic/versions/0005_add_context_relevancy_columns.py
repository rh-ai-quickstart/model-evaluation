"""Add context_relevancy_score to eval_result and avg_context_relevancy to eval_run.

Revision ID: 0005
Revises: 0004
"""

import sqlalchemy as sa
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("eval_result", sa.Column("context_relevancy_score", sa.Float(), nullable=True))
    op.add_column("eval_run", sa.Column("avg_context_relevancy", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("eval_result", "context_relevancy_score")
    op.drop_column("eval_run", "avg_context_relevancy")
