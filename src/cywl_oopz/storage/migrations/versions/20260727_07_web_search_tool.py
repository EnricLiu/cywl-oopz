"""Enable the public web-search Agent tool.

Revision ID: 20260727_07
Revises: 20260727_06
Create Date: 2026-07-27 17:00:00.000000
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260727_07"
down_revision: str | Sequence[str] | None = "20260727_06"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TOOL = "search_web"
_DEFAULT_WITH_SEARCH = (
    "'["
    '"get_agent_status",'
    '"get_channel_settings",'
    '"react_to_message",'
    '"search_music_catalog",'
    '"enqueue_music",'
    '"get_music_queue",'
    '"skip_music",'
    '"pause_music",'
    '"resume_music",'
    '"search_web"'
    "]'::jsonb"
)
_DEFAULT_WITHOUT_SEARCH = (
    "'["
    '"get_agent_status",'
    '"get_channel_settings",'
    '"react_to_message",'
    '"search_music_catalog",'
    '"enqueue_music",'
    '"get_music_queue",'
    '"skip_music",'
    '"pause_music",'
    '"resume_music"'
    "]'::jsonb"
)


def upgrade() -> None:
    """Append search once and update the default for newly created channels."""
    op.execute(
        f"""
        UPDATE channel_settings
        SET enabled_agent_tools = enabled_agent_tools || '["{_TOOL}"]'::jsonb
        WHERE NOT enabled_agent_tools ? '{_TOOL}'
        """
    )
    op.execute(
        "ALTER TABLE channel_settings "
        f"ALTER COLUMN enabled_agent_tools SET DEFAULT {_DEFAULT_WITH_SEARCH}"
    )


def downgrade() -> None:
    """Remove only the web-search tool and restore the previous default."""
    op.execute(
        f"""
        UPDATE channel_settings
        SET enabled_agent_tools = enabled_agent_tools - '{_TOOL}'
        """
    )
    op.execute(
        "ALTER TABLE channel_settings "
        f"ALTER COLUMN enabled_agent_tools SET DEFAULT {_DEFAULT_WITHOUT_SEARCH}"
    )
