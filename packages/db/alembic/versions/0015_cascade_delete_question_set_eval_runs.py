"""Change eval_run.question_set_id FK from SET NULL to CASCADE.

When a question set is deleted, all associated eval runs (and their
results via the existing eval_result FK cascade) are now deleted
instead of orphaned.

Revision ID: 0015
Revises: 0014
"""

from alembic import op

revision = "0015"
down_revision = "0014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint("fk_eval_run_question_set_id", "eval_run", type_="foreignkey")
    op.create_foreign_key(
        "fk_eval_run_question_set_id",
        "eval_run",
        "question_set",
        ["question_set_id"],
        ["id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    op.drop_constraint("fk_eval_run_question_set_id", "eval_run", type_="foreignkey")
    op.create_foreign_key(
        "fk_eval_run_question_set_id",
        "eval_run",
        "question_set",
        ["question_set_id"],
        ["id"],
        ondelete="SET NULL",
    )
