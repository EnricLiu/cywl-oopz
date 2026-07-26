"""Add user-controlled long-term Agent memory.

Revision ID: 20260727_04
Revises: 20260727_03
Create Date: 2026-07-27 02:40:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260727_04"
down_revision: str | Sequence[str] | None = "20260727_03"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create preference and owner-scoped expiring memory tables."""
    op.create_table(
        "agent_memory_preferences",
        sa.Column("person_id", sa.String(length=128), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("person_id"),
    )
    op.create_table(
        "agent_memory_items",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("owner_person_id", sa.String(length=128), nullable=False),
        sa.Column("namespace", sa.String(length=64), nullable=False),
        sa.Column(
            "content",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column("source_thread_id", sa.Uuid(), nullable=True),
        sa.Column("source_message_sequence", sa.BigInteger(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["source_thread_id"],
            ["agent_threads.id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_agent_memory_items_owner_updated",
        "agent_memory_items",
        ["owner_person_id", "updated_at"],
        unique=False,
    )


def downgrade() -> None:
    """Remove only long-term Agent memory storage."""
    op.drop_index(
        "ix_agent_memory_items_owner_updated",
        table_name="agent_memory_items",
    )
    op.drop_table("agent_memory_items")
    op.drop_table("agent_memory_preferences")
