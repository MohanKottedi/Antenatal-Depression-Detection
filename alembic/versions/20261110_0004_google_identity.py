"""Persist verified Google identities for account sign-in.

Revision ID: 20261110_0004
Revises: 20261109_0003
Create Date: 2026-11-10
"""

from alembic import op
import sqlalchemy as sa


revision = "20261110_0004"
down_revision = "20261109_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    columns = {column["name"] for column in inspector.get_columns("users")}
    if "google_subject" not in columns:
        op.add_column(
            "users",
            sa.Column("google_subject", sa.String(length=255), nullable=True),
        )

    indexes = {
        index["name"] for index in sa.inspect(op.get_bind()).get_indexes("users")
    }
    if "ix_users_google_subject" not in indexes:
        op.create_index(
            "ix_users_google_subject",
            "users",
            ["google_subject"],
            unique=True,
        )


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    indexes = {index["name"] for index in inspector.get_indexes("users")}
    if "ix_users_google_subject" in indexes:
        op.drop_index("ix_users_google_subject", table_name="users")
    columns = {column["name"] for column in sa.inspect(op.get_bind()).get_columns("users")}
    if "google_subject" in columns:
        op.drop_column("users", "google_subject")
