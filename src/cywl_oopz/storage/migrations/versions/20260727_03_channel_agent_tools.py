"""Add the Agent tool allow-list and semantic idempotency constraint.

Revision ID: 20260727_03
Revises: 20260726_02
Create Date: 2026-07-27 01:20:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260727_03"
down_revision: str | Sequence[str] | None = "20260726_02"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_DEFAULT_TOOLS = '["get_agent_status", "get_channel_settings", "react_to_message"]'


def upgrade() -> None:
    """Add channel visibility and prevent duplicate semantic tool effects."""
    op.add_column(
        "channel_settings",
        sa.Column(
            "enabled_agent_tools",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text(f"'{_DEFAULT_TOOLS}'::jsonb"),
            nullable=False,
        ),
    )
    op.alter_column(
        "channel_settings",
        "enabled_agent_tools",
        server_default=None,
    )
    op.create_index(
        "uq_agent_tool_executions_idempotency",
        "agent_tool_executions",
        ["run_id", "idempotency_key"],
        unique=True,
    )


def downgrade() -> None:
    """Remove the A3 visibility and idempotency additions."""
    op.drop_index(
        "uq_agent_tool_executions_idempotency",
        table_name="agent_tool_executions",
    )
    op.drop_column("channel_settings", "enabled_agent_tools")
