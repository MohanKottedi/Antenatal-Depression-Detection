"""Add doctor verification profiles and patient registration details.

Revision ID: 20261109_0002
Revises: 20261027_0001
Create Date: 2026-11-09
"""

from alembic import op
import sqlalchemy as sa


revision = "20261109_0002"
down_revision = "20261027_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    patient_columns = {
        column["name"] for column in inspector.get_columns("patient_profiles")
    }
    for name, column in (
        ("date_of_birth", sa.Column("date_of_birth", sa.Date(), nullable=True)),
        ("gender", sa.Column("gender", sa.String(length=40), nullable=True)),
        ("address", sa.Column("address", sa.Text(), nullable=True)),
    ):
        if name not in patient_columns:
            op.add_column("patient_profiles", column)

    if not inspector.has_table("doctor_profiles"):
        op.create_table(
            "doctor_profiles",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("user_id", sa.Integer(), nullable=False),
            sa.Column("medical_registration_number", sa.String(length=120), nullable=False),
            sa.Column("qualification", sa.String(length=255), nullable=False),
            sa.Column("specialization", sa.String(length=255), nullable=False),
            sa.Column("hospital", sa.String(length=255), nullable=False),
            sa.Column("years_of_experience", sa.Integer(), nullable=False),
            sa.Column("contact_phone", sa.String(length=40), nullable=True),
            sa.Column("verification_status", sa.String(length=30), nullable=False, server_default="pending"),
            sa.Column("verified_at", sa.DateTime(), nullable=True),
            sa.Column("verified_by_user_id", sa.Integer(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
            sa.ForeignKeyConstraint(["verified_by_user_id"], ["users.id"]),
            sa.UniqueConstraint("user_id", name="uq_doctor_profile_user"),
            sa.UniqueConstraint(
                "medical_registration_number",
                name="uq_doctor_medical_registration_number",
            ),
        )

    inspector = sa.inspect(op.get_bind())
    doctor_indexes = {
        index["name"] for index in inspector.get_indexes("doctor_profiles")
    }
    if op.f("ix_doctor_profiles_id") not in doctor_indexes:
        op.create_index(
            op.f("ix_doctor_profiles_id"),
            "doctor_profiles",
            ["id"],
            unique=False,
        )


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if inspector.has_table("doctor_profiles"):
        indexes = {
            index["name"] for index in inspector.get_indexes("doctor_profiles")
        }
        if op.f("ix_doctor_profiles_id") in indexes:
            op.drop_index(op.f("ix_doctor_profiles_id"), table_name="doctor_profiles")
        op.drop_table("doctor_profiles")
    patient_columns = {
        column["name"] for column in sa.inspect(op.get_bind()).get_columns("patient_profiles")
    }
    for name in ("address", "gender", "date_of_birth"):
        if name in patient_columns:
            op.drop_column("patient_profiles", name)
