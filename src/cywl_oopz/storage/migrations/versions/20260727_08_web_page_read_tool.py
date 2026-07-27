"""Enable bounded webpage reading for existing channel settings.

Revision ID: 20260727_08
Revises: 20260727_07
Create Date: 2026-07-27 19:00:00.000000
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260727_08"
down_revision: str | Sequence[str] | None = "20260727_07"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TOOL = "read_web_page"
_DEFAULT_WITH_READ = (
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
    '"search_web",'
    '"read_web_page"'
    "]'::jsonb"
)
_DEFAULT_WITHOUT_READ = (
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


def upgrade() -> None:
    """Append read_web_page once and update the new-channel default."""
    op.execute(
        f"""
        UPDATE channel_settings
        SET enabled_agent_tools = enabled_agent_tools || '["{_TOOL}"]'::jsonb
        WHERE NOT enabled_agent_tools ? '{_TOOL}'
        """
    )
    op.execute(
        "ALTER TABLE channel_settings "
        f"ALTER COLUMN enabled_agent_tools SET DEFAULT {_DEFAULT_WITH_READ}"
    )


def downgrade() -> None:
    """Remove only read_web_page and restore the W1 default."""
    op.execute(
        f"""
        UPDATE channel_settings
        SET enabled_agent_tools = enabled_agent_tools - '{_TOOL}'
        """
    )
    op.execute(
        "ALTER TABLE channel_settings "
        f"ALTER COLUMN enabled_agent_tools SET DEFAULT {_DEFAULT_WITHOUT_READ}"
    )
