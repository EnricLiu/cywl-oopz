"""Add area-shared music playlists and Agent tools.

Revision ID: 20260727_10
Revises: 20260727_09
Create Date: 2026-07-27 23:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260727_10"
down_revision: str | Sequence[str] | None = "20260727_09"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TOOLS = (
    "create_music_playlist",
    "list_music_playlists",
    "get_music_playlist",
    "add_music_playlist_track",
    "remove_music_playlist_track",
    "load_music_playlist",
)
_DEFAULT_WITH_PLAYLISTS = (
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
    '"set_music_playback_mode",'
    '"create_music_playlist",'
    '"list_music_playlists",'
    '"get_music_playlist",'
    '"add_music_playlist_track",'
    '"remove_music_playlist_track",'
    '"load_music_playlist"'
    "]'::jsonb"
)
_DEFAULT_WITHOUT_PLAYLISTS = (
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


def upgrade() -> None:
    """Create playlist storage and enable the curated playlist tools."""
    op.create_table(
        "music_playlists",
        sa.Column(
            "id",
            sa.Uuid(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("area_id", sa.String(length=128), nullable=False),
        sa.Column("name", sa.String(length=80), nullable=False),
        sa.Column("normalized_name", sa.String(length=80), nullable=False),
        sa.Column("created_by_person_id", sa.String(length=128), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("area_id", "normalized_name"),
    )
    op.create_index(
        "ix_music_playlists_area_updated",
        "music_playlists",
        ["area_id", "updated_at"],
        unique=False,
    )
    op.create_table(
        "music_playlist_tracks",
        sa.Column(
            "id",
            sa.Uuid(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("playlist_id", sa.Uuid(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("source", sa.String(length=64), nullable=False),
        sa.Column("source_id", sa.String(length=256), nullable=False),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column(
            "artists",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("added_by_person_id", sa.String(length=128), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "duration_ms IS NULL OR duration_ms >= 0",
            name="ck_music_playlist_tracks_duration_nonnegative",
        ),
        sa.CheckConstraint(
            "position > 0",
            name="ck_music_playlist_tracks_position_positive",
        ),
        sa.ForeignKeyConstraint(
            ["playlist_id"],
            ["music_playlists.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "playlist_id",
            "position",
            deferrable=True,
            initially="DEFERRED",
        ),
    )
    op.execute(
        """
        CREATE TRIGGER trg_music_playlists_set_updated_at
        BEFORE UPDATE ON music_playlists
        FOR EACH ROW
        EXECUTE FUNCTION cywl_set_updated_at()
        """
    )
    for tool in _TOOLS:
        op.execute(
            f"""
            UPDATE channel_settings
            SET enabled_agent_tools = enabled_agent_tools || '["{tool}"]'::jsonb
            WHERE NOT enabled_agent_tools ? '{tool}'
            """
        )
    op.execute(
        "ALTER TABLE channel_settings "
        f"ALTER COLUMN enabled_agent_tools SET DEFAULT {_DEFAULT_WITH_PLAYLISTS}"
    )


def downgrade() -> None:
    """Remove playlist tools and storage without changing unrelated music state."""
    for tool in reversed(_TOOLS):
        op.execute(
            f"""
            UPDATE channel_settings
            SET enabled_agent_tools = enabled_agent_tools - '{tool}'
            """
        )
    op.execute(
        "ALTER TABLE channel_settings "
        f"ALTER COLUMN enabled_agent_tools SET DEFAULT {_DEFAULT_WITHOUT_PLAYLISTS}"
    )
    op.execute("DROP TRIGGER trg_music_playlists_set_updated_at ON music_playlists")
    op.drop_table("music_playlist_tracks")
    op.drop_index("ix_music_playlists_area_updated", table_name="music_playlists")
    op.drop_table("music_playlists")
