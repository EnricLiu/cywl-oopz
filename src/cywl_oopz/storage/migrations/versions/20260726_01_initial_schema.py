"""Create initial application tables.

Revision ID: 20260726_01
Revises:
Create Date: 2026-07-26 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260726_01"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create settings, conversation, and rate-limit tables."""
    op.create_table(
        "channel_settings",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("area_id", sa.String(length=128), nullable=False),
        sa.Column("channel_id", sa.String(length=128), nullable=False),
        sa.Column("chat_enabled", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("area_id", "channel_id"),
    )
    op.create_table(
        "conversation_sessions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("scope", sa.String(length=32), nullable=False),
        sa.Column("area_id", sa.String(length=128), nullable=False),
        sa.Column("channel_id", sa.String(length=128), nullable=False),
        sa.Column("person_id", sa.String(length=128), nullable=False),
        sa.Column("selected_model", sa.String(length=256), nullable=True),
        sa.Column("messages", sa.JSON(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("scope", "area_id", "channel_id", "person_id"),
    )
    op.create_table(
        "rate_limit_buckets",
        sa.Column("bucket_key", sa.String(length=256), nullable=False),
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("hit_count", sa.Integer(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("bucket_key"),
    )


def downgrade() -> None:
    """Remove initial application tables."""
    op.drop_table("rate_limit_buckets")
    op.drop_table("conversation_sessions")
    op.drop_table("channel_settings")
