"""Add expected_answer column to eval_result.

Stores the expected answer used during scoring so it can be displayed
in comparison views alongside the model's actual answer.

Revision ID: 0009
Revises: 0008
"""

import sqlalchemy as sa
from alembic import op

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("eval_result", sa.Column("expected_answer", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("eval_result", "expected_answer")
