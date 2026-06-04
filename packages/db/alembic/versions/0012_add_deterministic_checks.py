"""Add deterministic_checks JSON column to eval_result.

Stores per-question deterministic check results: document presence,
chunk alignment, abstention validation, and source reference checks.
Format: [{"check_name": "...", "passed": bool, "detail": "...", "category": "..."}]

Revision ID: 0012
Revises: 0011
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "eval_result",
        sa.Column("deterministic_checks", sa.JSON().with_variant(JSONB, "postgresql"), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("eval_result", "deterministic_checks")
