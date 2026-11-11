"""Add configured tracker push reminders.

Revision ID: 20261110_0006
Revises: 20261110_0005
Create Date: 2026-11-10
"""

from alembic import op
import sqlalchemy as sa


revision = "20261110_0006"
down_revision = "20261110_0005"
branch_labels = None
depends_on = None


def _create_index_if_missing(name: str, table_name: str, columns: list[str]) -> None:
    indexes = {
        item["name"] for item in sa.inspect(op.get_bind()).get_indexes(table_name)
    }
    if name not in indexes:
        op.create_index(name, table_name, columns, unique=False)


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    pregnancy_columns = {
        item["name"] for item in inspector.get_columns("pregnancy_profiles")
    }
    if "reminder_time" not in pregnancy_columns:
        op.add_column(
            "pregnancy_profiles",
            sa.Column(
                "reminder_time",
                sa.String(length=5),
                nullable=False,
                server_default="15:41",
            ),
        )
    if "reminder_timezone" not in pregnancy_columns:
        op.add_column(
            "pregnancy_profiles",
            sa.Column("reminder_timezone", sa.String(length=100), nullable=True),
        )
    if "weekly_tracker_reminders" not in pregnancy_columns:
        op.add_column(
            "pregnancy_profiles",
            sa.Column(
                "weekly_tracker_reminders",
                sa.Boolean(),
                nullable=False,
                server_default=sa.true(),
            ),
        )
    if "screening_reminders" not in pregnancy_columns:
        op.add_column(
            "pregnancy_profiles",
            sa.Column(
                "screening_reminders",
                sa.Boolean(),
                nullable=False,
                server_default=sa.true(),
            ),
        )

    if not sa.inspect(op.get_bind()).has_table("push_subscriptions"):
        op.create_table(
            "push_subscriptions",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("user_id", sa.Integer(), nullable=False),
            sa.Column("endpoint", sa.Text(), nullable=False),
            sa.Column("endpoint_hash", sa.String(length=64), nullable=False),
            sa.Column("p256dh_key", sa.String(length=255), nullable=False),
            sa.Column("auth_key", sa.String(length=255), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
            sa.UniqueConstraint(
                "endpoint_hash",
                name="uq_push_subscription_endpoint_hash",
            ),
        )
    _create_index_if_missing("ix_push_subscriptions_id", "push_subscriptions", ["id"])
    _create_index_if_missing(
        "ix_push_subscriptions_user_id",
        "push_subscriptions",
        ["user_id"],
    )

    if not sa.inspect(op.get_bind()).has_table("notification_deliveries"):
        op.create_table(
            "notification_deliveries",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("user_id", sa.Integer(), nullable=False),
            sa.Column("reminder_key", sa.String(length=120), nullable=False),
            sa.Column("status", sa.String(length=20), nullable=False),
            sa.Column("last_attempt_at", sa.DateTime(), nullable=False),
            sa.Column("sent_at", sa.DateTime(), nullable=True),
            sa.Column("last_error", sa.String(length=500), nullable=True),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
            sa.UniqueConstraint(
                "user_id",
                "reminder_key",
                name="uq_notification_user_reminder",
            ),
        )
    _create_index_if_missing(
        "ix_notification_deliveries_id",
        "notification_deliveries",
        ["id"],
    )
    _create_index_if_missing(
        "ix_notification_deliveries_user_id",
        "notification_deliveries",
        ["user_id"],
    )


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if inspector.has_table("notification_deliveries"):
        indexes = {
            item["name"]
            for item in inspector.get_indexes("notification_deliveries")
        }
        for index_name in (
            "ix_notification_deliveries_user_id",
            "ix_notification_deliveries_id",
        ):
            if index_name in indexes:
                op.drop_index(index_name, table_name="notification_deliveries")
        op.drop_table("notification_deliveries")
    if inspector.has_table("push_subscriptions"):
        indexes = {
            item["name"] for item in inspector.get_indexes("push_subscriptions")
        }
        for index_name in ("ix_push_subscriptions_user_id", "ix_push_subscriptions_id"):
            if index_name in indexes:
                op.drop_index(index_name, table_name="push_subscriptions")
        op.drop_table("push_subscriptions")

    pregnancy_columns = {
        item["name"] for item in sa.inspect(op.get_bind()).get_columns("pregnancy_profiles")
    }
    if "reminder_timezone" in pregnancy_columns:
        op.drop_column("pregnancy_profiles", "reminder_timezone")
    if "reminder_time" in pregnancy_columns:
        op.drop_column("pregnancy_profiles", "reminder_time")
    if "weekly_tracker_reminders" in pregnancy_columns:
        op.drop_column("pregnancy_profiles", "weekly_tracker_reminders")
    if "screening_reminders" in pregnancy_columns:
        op.drop_column("pregnancy_profiles", "screening_reminders")
