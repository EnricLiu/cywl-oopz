"""Enable the music playback-mode Agent tool.

Revision ID: 20260727_09
Revises: 20260727_08
Create Date: 2026-07-27 21:00:00.000000
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260727_09"
down_revision: str | Sequence[str] | None = "20260727_08"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TOOL = "set_music_playback_mode"
_DEFAULT_WITH_MODE = (
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
    '"read_web_page",'
    '"set_music_playback_mode"'
    "]'::jsonb"
)
_DEFAULT_WITHOUT_MODE = (
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


def upgrade() -> None:
    """Append the mode tool once and update the new-channel default."""
    op.execute(
        f"""
        UPDATE channel_settings
        SET enabled_agent_tools = enabled_agent_tools || '["{_TOOL}"]'::jsonb
        WHERE NOT enabled_agent_tools ? '{_TOOL}'
        """
    )
    op.execute(
        "ALTER TABLE channel_settings "
        f"ALTER COLUMN enabled_agent_tools SET DEFAULT {_DEFAULT_WITH_MODE}"
    )


def downgrade() -> None:
    """Remove only the mode tool and restore the previous default."""
    op.execute(
        f"""
        UPDATE channel_settings
        SET enabled_agent_tools = enabled_agent_tools - '{_TOOL}'
        """
    )
    op.execute(
        "ALTER TABLE channel_settings "
        f"ALTER COLUMN enabled_agent_tools SET DEFAULT {_DEFAULT_WITHOUT_MODE}"
    )
