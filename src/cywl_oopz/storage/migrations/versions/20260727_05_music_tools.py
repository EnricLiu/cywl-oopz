"""Enable the first music Agent tools for existing channel settings.

Revision ID: 20260727_05
Revises: 20260727_04
Create Date: 2026-07-27 04:40:00.000000
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260727_05"
down_revision: str | Sequence[str] | None = "20260727_04"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TOOLS = (
    "search_music_catalog",
    "enqueue_music",
    "get_music_queue",
    "skip_music",
    "pause_music",
    "resume_music",
)


def upgrade() -> None:
    """Append each music tool once without changing other channel choices."""
    for tool in _TOOLS:
        op.execute(
            f"""
            UPDATE channel_settings
            SET enabled_agent_tools = enabled_agent_tools || '["{tool}"]'::jsonb
            WHERE NOT enabled_agent_tools ? '{tool}'
            """
        )


def downgrade() -> None:
    """Remove only A5 music tools from channel allow-lists."""
    for tool in _TOOLS:
        op.execute(
            f"""
            UPDATE channel_settings
            SET enabled_agent_tools = enabled_agent_tools - '{tool}'
            """
        )
