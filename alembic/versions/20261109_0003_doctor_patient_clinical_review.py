"""Add consent-based doctor/patient relationships and clinical review.

Revision ID: 20261109_0003
Revises: 20261109_0002
Create Date: 2026-11-09
"""

from alembic import op
import sqlalchemy as sa


revision = "20261109_0003"
down_revision = "20261109_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if not sa.inspect(op.get_bind()).has_table("doctor_patients"):
        op.create_table(
            "doctor_patients",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("doctor_id", sa.Integer(), nullable=False),
            sa.Column("patient_id", sa.Integer(), nullable=False),
            sa.Column("consent_status", sa.String(length=30), nullable=False, server_default="pending"),
            sa.Column("relationship_status", sa.String(length=30), nullable=False, server_default="active"),
            sa.Column("requested_at", sa.DateTime(), nullable=False),
            sa.Column("consented_at", sa.DateTime(), nullable=True),
            sa.Column("revoked_at", sa.DateTime(), nullable=True),
            sa.ForeignKeyConstraint(["doctor_id"], ["doctor_profiles.id"]),
            sa.ForeignKeyConstraint(["patient_id"], ["patient_profiles.id"]),
            sa.UniqueConstraint("doctor_id", "patient_id", name="uq_doctor_patient_relationship"),
        )
    if not sa.inspect(op.get_bind()).has_table("clinical_notes"):
        op.create_table(
            "clinical_notes",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("doctor_id", sa.Integer(), nullable=False),
            sa.Column("patient_id", sa.Integer(), nullable=False),
            sa.Column("screening_session_id", sa.Integer(), nullable=True),
            sa.Column("note", sa.Text(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(["doctor_id"], ["doctor_profiles.id"]),
            sa.ForeignKeyConstraint(["patient_id"], ["patient_profiles.id"]),
            sa.ForeignKeyConstraint(["screening_session_id"], ["screening_sessions.id"]),
        )
    if not sa.inspect(op.get_bind()).has_table("recommendations"):
        op.create_table(
            "recommendations",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("doctor_id", sa.Integer(), nullable=False),
            sa.Column("patient_id", sa.Integer(), nullable=False),
            sa.Column("screening_session_id", sa.Integer(), nullable=True),
            sa.Column("recommendation", sa.Text(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(["doctor_id"], ["doctor_profiles.id"]),
            sa.ForeignKeyConstraint(["patient_id"], ["patient_profiles.id"]),
            sa.ForeignKeyConstraint(["screening_session_id"], ["screening_sessions.id"]),
        )

    indexes_to_ensure = (
        ("doctor_patients", op.f("ix_doctor_patients_id"), ["id"]),
        ("doctor_patients", op.f("ix_doctor_patients_doctor_id"), ["doctor_id"]),
        ("doctor_patients", op.f("ix_doctor_patients_patient_id"), ["patient_id"]),
        ("clinical_notes", op.f("ix_clinical_notes_id"), ["id"]),
        ("recommendations", op.f("ix_recommendations_id"), ["id"]),
    )
    for table, name, columns in indexes_to_ensure:
        indexes = {
            index["name"] for index in sa.inspect(op.get_bind()).get_indexes(table)
        }
        if name not in indexes:
            op.create_index(name, table, columns, unique=False)


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    for table, index_names in (
        ("recommendations", (op.f("ix_recommendations_id"),)),
        ("clinical_notes", (op.f("ix_clinical_notes_id"),)),
        (
            "doctor_patients",
            (
                op.f("ix_doctor_patients_patient_id"),
                op.f("ix_doctor_patients_doctor_id"),
                op.f("ix_doctor_patients_id"),
            ),
        ),
    ):
        if inspector.has_table(table):
            indexes = {index["name"] for index in inspector.get_indexes(table)}
            for name in index_names:
                if name in indexes:
                    op.drop_index(name, table_name=table)
            op.drop_table(table)
