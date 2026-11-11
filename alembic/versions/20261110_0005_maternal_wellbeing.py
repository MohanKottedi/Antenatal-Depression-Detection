"""Add patient-owned pregnancy, mood, assistant, and meditation records.

Revision ID: 20261110_0005
Revises: 20261110_0004
Create Date: 2026-11-10
"""

from alembic import op
import sqlalchemy as sa


revision = "20261110_0005"
down_revision = "20261110_0004"
branch_labels = None
depends_on = None


def _create_table_if_missing(name: str, *columns_and_constraints) -> None:
    if not sa.inspect(op.get_bind()).has_table(name):
        op.create_table(name, *columns_and_constraints)


def _create_index_if_missing(
    name: str,
    table_name: str,
    columns: list[str],
    unique: bool = False,
) -> None:
    indexes = {
        index["name"]
        for index in sa.inspect(op.get_bind()).get_indexes(table_name)
    }
    if name not in indexes:
        op.create_index(name, table_name, columns, unique=unique)


def upgrade() -> None:
    _create_table_if_missing(
        "pregnancy_profiles",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("patient_id", sa.Integer(), nullable=False),
        sa.Column("lmp_date", sa.Date(), nullable=True),
        sa.Column("confirmed_due_date", sa.Date(), nullable=True),
        sa.Column("follow_up_reminder_date", sa.Date(), nullable=True),
        sa.Column(
            "dietary_preference",
            sa.String(length=30),
            nullable=False,
            server_default="no_preference",
        ),
        sa.Column("food_allergies", sa.String(length=1000), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["patient_id"],
            ["patient_profiles.id"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("patient_id", name="uq_pregnancy_profile_patient"),
    )
    _create_index_if_missing(
        "ix_pregnancy_profiles_id",
        "pregnancy_profiles",
        ["id"],
        unique=False,
    )
    _create_index_if_missing(
        "ix_pregnancy_profiles_patient_id",
        "pregnancy_profiles",
        ["patient_id"],
        unique=False,
    )

    _create_table_if_missing(
        "mood_entries",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("patient_id", sa.Integer(), nullable=False),
        sa.Column("entry_date", sa.Date(), nullable=False),
        sa.Column("mood", sa.String(length=30), nullable=False),
        sa.Column("stress_level", sa.Integer(), nullable=True),
        sa.Column("sleep_hours", sa.Float(), nullable=True),
        sa.Column("journal", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["patient_id"],
            ["patient_profiles.id"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "patient_id",
            "entry_date",
            name="uq_mood_patient_date",
        ),
    )
    _create_index_if_missing("ix_mood_entries_id", "mood_entries", ["id"], unique=False)
    _create_index_if_missing(
        "ix_mood_entries_patient_id",
        "mood_entries",
        ["patient_id"],
        unique=False,
    )
    _create_index_if_missing(
        "ix_mood_entries_entry_date",
        "mood_entries",
        ["entry_date"],
        unique=False,
    )

    _create_table_if_missing(
        "assistant_conversations",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("patient_id", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(length=120), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["patient_id"],
            ["patient_profiles.id"],
            ondelete="CASCADE",
        ),
    )
    _create_index_if_missing(
        "ix_assistant_conversations_id",
        "assistant_conversations",
        ["id"],
        unique=False,
    )
    _create_index_if_missing(
        "ix_assistant_conversations_patient_id",
        "assistant_conversations",
        ["patient_id"],
        unique=False,
    )

    _create_table_if_missing(
        "assistant_messages",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("conversation_id", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(length=20), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["assistant_conversations.id"],
            ondelete="CASCADE",
        ),
    )
    _create_index_if_missing(
        "ix_assistant_messages_id",
        "assistant_messages",
        ["id"],
        unique=False,
    )
    _create_index_if_missing(
        "ix_assistant_messages_conversation_id",
        "assistant_messages",
        ["conversation_id"],
        unique=False,
    )

    _create_table_if_missing(
        "meditation_sessions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("patient_id", sa.Integer(), nullable=False),
        sa.Column("activity_id", sa.String(length=50), nullable=False),
        sa.Column("duration_seconds", sa.Integer(), nullable=False),
        sa.Column("completed_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["patient_id"],
            ["patient_profiles.id"],
            ondelete="CASCADE",
        ),
    )
    _create_index_if_missing(
        "ix_meditation_sessions_id",
        "meditation_sessions",
        ["id"],
        unique=False,
    )
    _create_index_if_missing(
        "ix_meditation_sessions_patient_id",
        "meditation_sessions",
        ["patient_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_meditation_sessions_patient_id", table_name="meditation_sessions")
    op.drop_index("ix_meditation_sessions_id", table_name="meditation_sessions")
    op.drop_table("meditation_sessions")
    op.drop_index("ix_assistant_messages_conversation_id", table_name="assistant_messages")
    op.drop_index("ix_assistant_messages_id", table_name="assistant_messages")
    op.drop_table("assistant_messages")
    op.drop_index(
        "ix_assistant_conversations_patient_id",
        table_name="assistant_conversations",
    )
    op.drop_index("ix_assistant_conversations_id", table_name="assistant_conversations")
    op.drop_table("assistant_conversations")
    op.drop_index("ix_mood_entries_entry_date", table_name="mood_entries")
    op.drop_index("ix_mood_entries_patient_id", table_name="mood_entries")
    op.drop_index("ix_mood_entries_id", table_name="mood_entries")
    op.drop_table("mood_entries")
    op.drop_index(
        "ix_pregnancy_profiles_patient_id",
        table_name="pregnancy_profiles",
    )
    op.drop_index("ix_pregnancy_profiles_id", table_name="pregnancy_profiles")
    op.drop_table("pregnancy_profiles")
