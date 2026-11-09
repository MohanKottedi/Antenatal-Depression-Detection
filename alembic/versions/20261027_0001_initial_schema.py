"""Initial Sneha schema

Revision ID: 20261027_0001
Revises: 
Create Date: 2026-10-27 12:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "20261027_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("full_name", sa.String(length=255), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("role", sa.String(length=30), nullable=False, server_default="patient"),
        sa.Column("account_status", sa.String(length=30), nullable=False, server_default="active"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("last_login_at", sa.DateTime(), nullable=True),
        sa.UniqueConstraint("email", name="uq_users_email"),
    )
    op.create_index(op.f("ix_users_email"), "users", ["email"], unique=True)
    op.create_index(op.f("ix_users_id"), "users", ["id"], unique=False)

    op.create_table(
        "patient_profiles",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("patient_reference", sa.String(length=64), nullable=False),
        sa.Column("age", sa.Integer(), nullable=True),
        sa.Column("gestational_age", sa.Float(), nullable=True),
        sa.Column("contact_phone", sa.String(length=40), nullable=True),
        sa.Column("city", sa.String(length=120), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.UniqueConstraint("user_id", name="uq_patient_user"),
        sa.UniqueConstraint("patient_reference", name="uq_patient_reference"),
    )
    op.create_index(op.f("ix_patient_profiles_id"), "patient_profiles", ["id"], unique=False)
    op.create_index(op.f("ix_patient_profiles_patient_reference"), "patient_profiles", ["patient_reference"], unique=True)

    op.create_table(
        "screening_sessions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("patient_id", sa.Integer(), nullable=True),
        sa.Column("screening_date", sa.DateTime(), nullable=False),
        sa.Column("screening_status", sa.String(length=30), nullable=False, server_default="in_progress"),
        sa.Column("completion_percentage", sa.Float(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["patient_id"], ["patient_profiles.id"]),
    )
    op.create_index(op.f("ix_screening_sessions_id"), "screening_sessions", ["id"], unique=False)

    op.create_table(
        "screening_responses",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("screening_session_id", sa.Integer(), nullable=False),
        sa.Column("question_key", sa.String(length=255), nullable=False),
        sa.Column("response_value", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["screening_session_id"], ["screening_sessions.id"]),
    )
    op.create_index(op.f("ix_screening_responses_id"), "screening_responses", ["id"], unique=False)

    op.create_table(
        "screening_results",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("screening_session_id", sa.Integer(), nullable=False),
        sa.Column("model_name", sa.String(length=80), nullable=False),
        sa.Column("model_version", sa.String(length=80), nullable=True),
        sa.Column("predicted_class", sa.String(length=80), nullable=False),
        sa.Column("risk_score", sa.Float(), nullable=True),
        sa.Column("screening_summary", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["screening_session_id"], ["screening_sessions.id"]),
    )
    op.create_index(op.f("ix_screening_results_id"), "screening_results", ["id"], unique=False)

    op.create_table(
        "audit_logs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("actor_user_id", sa.Integer(), nullable=True),
        sa.Column("action", sa.String(length=80), nullable=False),
        sa.Column("resource_type", sa.String(length=80), nullable=True),
        sa.Column("resource_id", sa.String(length=120), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["actor_user_id"], ["users.id"]),
    )
    op.create_index(op.f("ix_audit_logs_id"), "audit_logs", ["id"], unique=False)


def downgrade() -> None:
    op.drop_table("audit_logs")
    op.drop_table("screening_results")
    op.drop_table("screening_responses")
    op.drop_table("screening_sessions")
    op.drop_table("patient_profiles")
    op.drop_table("users")
