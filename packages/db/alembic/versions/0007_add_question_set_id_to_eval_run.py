
"""Add question_set_id to eval_run.

Revision ID: 0007
Revises: 0006
Create Date: 2026-03-30
"""

from alembic import op
import sqlalchemy as sa


revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "eval_run",
        sa.Column("question_set_id", sa.Integer(), nullable=True),
    )
    op.create_foreign_key(
        "fk_eval_run_question_set_id",
        "eval_run",
        "question_set",
        ["question_set_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index("idx_eval_run_question_set_id", "eval_run", ["question_set_id"])


def downgrade() -> None:
    op.drop_index("idx_eval_run_question_set_id", table_name="eval_run")
    op.drop_constraint("fk_eval_run_question_set_id", "eval_run", type_="foreignkey")
    op.drop_column("eval_run", "question_set_id")
