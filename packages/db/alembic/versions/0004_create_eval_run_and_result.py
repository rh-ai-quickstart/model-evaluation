
"""Create eval_run and eval_result tables.

Revision ID: 0004
Revises: 0003
Create Date: 2026-03-24
"""

from alembic import op
import sqlalchemy as sa


revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "eval_run",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("model_name", sa.String(length=200), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False, server_default="pending"),
        sa.Column("total_questions", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("completed_questions", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("avg_latency_ms", sa.Float(), nullable=True),
        sa.Column("avg_relevancy", sa.Float(), nullable=True),
        sa.Column("avg_groundedness", sa.Float(), nullable=True),
        sa.Column("avg_context_precision", sa.Float(), nullable=True),
        sa.Column("hallucination_rate", sa.Float(), nullable=True),
        sa.Column("total_tokens", sa.Integer(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "eval_result",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("eval_run_id", sa.Integer(), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column("contexts", sa.Text(), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("relevancy_score", sa.Float(), nullable=True),
        sa.Column("groundedness_score", sa.Float(), nullable=True),
        sa.Column("context_precision_score", sa.Float(), nullable=True),
        sa.Column("is_hallucination", sa.Boolean(), nullable=True),
        sa.Column("total_tokens", sa.Integer(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["eval_run_id"], ["eval_run.id"]),
    )
    op.create_index("idx_eval_result_run_id", "eval_result", ["eval_run_id"])


def downgrade() -> None:
    op.drop_index("idx_eval_result_run_id", table_name="eval_result")
    op.drop_table("eval_result")
    op.drop_table("eval_run")
