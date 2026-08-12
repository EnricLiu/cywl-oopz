"""Add playlist management and transient queue clear tools.

Revision ID: 20260811_22
Revises: 20260804_21
Create Date: 2026-08-11 18:00:00.000000
"""

import json
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260811_22"
down_revision: str | Sequence[str] | None = "20260804_21"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TOOLS = (
    "clear_music_queue",
    "rename_music_playlist",
    "delete_music_playlist",
    "clear_music_playlist",
)
_DEFAULT_TOOLS = (
    "get_agent_status",
    "get_channel_settings",
    "react_to_message",
    "search_music_catalog",
    "enqueue_music",
    "get_music_queue",
    "skip_music",
    "pause_music",
    "resume_music",
    "clear_music_queue",
    "search_web",
    "read_web_page",
    "browser_open",
    "browser_snapshot",
    "browser_wait",
    "browser_close",
    "browser_click",
    "browser_fill",
    "browser_press",
    "set_music_playback_mode",
    "create_music_playlist",
    "list_music_playlists",
    "get_music_playlist",
    "add_music_playlist_track",
    "remove_music_playlist_track",
    "rename_music_playlist",
    "delete_music_playlist",
    "clear_music_playlist",
    "load_music_playlist",
    "load_agent_skill",
    "read_agent_skill_resource",
    "list_agent_skill_library",
    "inspect_agent_skill",
    "create_agent_skill",
    "update_agent_skill",
    "manage_agent_skill_resource",
    "set_agent_skill_state",
    "invite_agent_skill_share",
    "respond_agent_skill_share",
    "revoke_agent_skill_share",
    "preview_netease_playlist",
    "import_netease_playlist",
)
_CURATOR_ADDITION = """

9. 用户明确要求重命名、清空或删除共享歌单时，分别调用 `rename_music_playlist`、
   `clear_music_playlist` 或 `delete_music_playlist`。清空会保留歌单，删除会移除歌单本身。
10. 用户明确要求停止播放并清空临时队列时调用 `clear_music_queue`；它不会修改共享歌单。
播放顺序和循环策略是两个维度；随机列表循环应同时设置 `order=shuffle` 与 `repeat=all`。"""
_CURATOR_TOOLS = (
    "rename_music_playlist",
    "delete_music_playlist",
    "clear_music_playlist",
    "clear_music_queue",
)


def _default_sql(tools: tuple[str, ...]) -> str:
    value = json.dumps(tools, ensure_ascii=True, separators=(",", ":"))
    return f"'{value}'::jsonb"


def upgrade() -> None:
    for tool in _TOOLS:
        op.execute(
            f"""
            UPDATE channel_settings
            SET enabled_agent_tools = enabled_agent_tools || '["{tool}"]'::jsonb
            WHERE NOT enabled_agent_tools ? '{tool}'
            """
        )
    op.execute(
        "ALTER TABLE channel_settings ALTER COLUMN enabled_agent_tools "
        f"SET DEFAULT {_default_sql(_DEFAULT_TOOLS)}"
    )
    connection = op.get_bind()
    connection.execute(
        sa.text(
            """
            UPDATE agent_skills
            SET instructions = instructions || :addition,
                required_tools = required_tools || CAST(:tools AS jsonb),
                version = '1.1.0',
                updated_at = CURRENT_TIMESTAMP
            WHERE name = 'music-curator'
              AND version = '1.0.0'
              AND metadata->>'builtin_seed' = '20260729_12'
            """
        ),
        {
            "addition": _CURATOR_ADDITION,
            "tools": json.dumps(_CURATOR_TOOLS, ensure_ascii=False),
        },
    )


def downgrade() -> None:
    connection = op.get_bind()
    connection.execute(
        sa.text(
            """
            UPDATE agent_skills
            SET instructions = replace(instructions, :addition, ''),
                required_tools = required_tools
                    - 'rename_music_playlist'
                    - 'delete_music_playlist'
                    - 'clear_music_playlist'
                    - 'clear_music_queue',
                version = '1.0.0',
                updated_at = CURRENT_TIMESTAMP
            WHERE name = 'music-curator'
              AND version = '1.1.0'
              AND metadata->>'builtin_seed' = '20260729_12'
            """
        ),
        {"addition": _CURATOR_ADDITION},
    )
    for tool in _TOOLS:
        op.execute(
            f"""
            UPDATE channel_settings
            SET enabled_agent_tools = enabled_agent_tools - '{tool}'
            WHERE enabled_agent_tools ? '{tool}'
            """
        )
    without_controls = tuple(tool for tool in _DEFAULT_TOOLS if tool not in _TOOLS)
    op.execute(
        "ALTER TABLE channel_settings ALTER COLUMN enabled_agent_tools "
        f"SET DEFAULT {_default_sql(without_controls)}"
    )
