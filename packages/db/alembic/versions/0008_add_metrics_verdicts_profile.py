"""Add new metric scores, verdict columns, and profile fields.

New EvalResult columns: completeness_score, correctness_score,
compliance_accuracy_score, abstention_score, verdict, fail_reasons.

New EvalRun columns: avg_completeness, avg_correctness,
avg_compliance_accuracy, avg_abstention, profile_id, profile_version,
overall_verdict, pass_count, fail_count, review_count.

Revision ID: 0008
Revises: 0007
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # EvalResult: new metric scores
    op.add_column("eval_result", sa.Column("completeness_score", sa.Float(), nullable=True))
    op.add_column("eval_result", sa.Column("correctness_score", sa.Float(), nullable=True))
    op.add_column(
        "eval_result", sa.Column("compliance_accuracy_score", sa.Float(), nullable=True)
    )
    op.add_column("eval_result", sa.Column("abstention_score", sa.Float(), nullable=True))

    # EvalResult: verdict columns
    op.add_column("eval_result", sa.Column("verdict", sa.String(50), nullable=True))
    op.add_column(
        "eval_result",
        sa.Column("fail_reasons", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )

    # EvalRun: new aggregate scores
    op.add_column("eval_run", sa.Column("avg_completeness", sa.Float(), nullable=True))
    op.add_column("eval_run", sa.Column("avg_correctness", sa.Float(), nullable=True))
    op.add_column("eval_run", sa.Column("avg_compliance_accuracy", sa.Float(), nullable=True))
    op.add_column("eval_run", sa.Column("avg_abstention", sa.Float(), nullable=True))

    # EvalRun: profile tracking
    op.add_column("eval_run", sa.Column("profile_id", sa.String(200), nullable=True))
    op.add_column("eval_run", sa.Column("profile_version", sa.String(100), nullable=True))

    # EvalRun: verdict summary
    op.add_column("eval_run", sa.Column("overall_verdict", sa.String(50), nullable=True))
    op.add_column("eval_run", sa.Column("pass_count", sa.Integer(), nullable=True))
    op.add_column("eval_run", sa.Column("fail_count", sa.Integer(), nullable=True))
    op.add_column("eval_run", sa.Column("review_count", sa.Integer(), nullable=True))


def downgrade() -> None:
    # EvalRun: verdict summary
    op.drop_column("eval_run", "review_count")
    op.drop_column("eval_run", "fail_count")
    op.drop_column("eval_run", "pass_count")
    op.drop_column("eval_run", "overall_verdict")

    # EvalRun: profile tracking
    op.drop_column("eval_run", "profile_version")
    op.drop_column("eval_run", "profile_id")

    # EvalRun: new aggregate scores
    op.drop_column("eval_run", "avg_abstention")
    op.drop_column("eval_run", "avg_compliance_accuracy")
    op.drop_column("eval_run", "avg_correctness")
    op.drop_column("eval_run", "avg_completeness")

    # EvalResult: verdict columns
    op.drop_column("eval_result", "fail_reasons")
    op.drop_column("eval_result", "verdict")

    # EvalResult: new metric scores
    op.drop_column("eval_result", "abstention_score")
    op.drop_column("eval_result", "compliance_accuracy_score")
    op.drop_column("eval_result", "correctness_score")
    op.drop_column("eval_result", "completeness_score")
