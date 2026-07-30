"""Enable user-to-user Agent Skill sharing tools.

Revision ID: 20260730_17
Revises: 20260729_16
Create Date: 2026-07-30 00:30:00.000000
"""

import json
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260730_17"
down_revision: str | Sequence[str] | None = "20260729_16"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TOOLS = (
    "invite_agent_skill_share",
    "respond_agent_skill_share",
    "revoke_agent_skill_share",
)
_DEFAULT_WITH_SHARING = (
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
    '"load_music_playlist",'
    '"load_agent_skill",'
    '"read_agent_skill_resource",'
    '"list_agent_skill_library",'
    '"inspect_agent_skill",'
    '"create_agent_skill",'
    '"update_agent_skill",'
    '"manage_agent_skill_resource",'
    '"set_agent_skill_state",'
    '"invite_agent_skill_share",'
    '"respond_agent_skill_share",'
    '"revoke_agent_skill_share",'
    '"preview_netease_playlist",'
    '"import_netease_playlist"'
    "]'::jsonb"
)
_DEFAULT_WITHOUT_SHARING = (
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
    '"load_music_playlist",'
    '"load_agent_skill",'
    '"read_agent_skill_resource",'
    '"list_agent_skill_library",'
    '"inspect_agent_skill",'
    '"create_agent_skill",'
    '"update_agent_skill",'
    '"manage_agent_skill_resource",'
    '"set_agent_skill_state",'
    '"preview_netease_playlist",'
    '"import_netease_playlist"'
    "]'::jsonb"
)
_SHARING_INSTRUCTIONS = """

10. 只有用户当前明确要求分享时才调用分享工具。邀请对象只来自当前消息真实 `@` 提及，
    不接受或猜测 person ID；没有有效提及时请用户在一条新消息中重新提及目标。
11. 分享是只读实时授权。说明接收方必须先接受，接受后从下一轮可用；接收方不能修改或二次分享。
12. 接受、拒绝或撤销前先列出技能库取得真实 invitation/share ID，不要从历史猜 ID。
"""


def upgrade() -> None:
    """Expose sharing tools and extend the builtin authoring methodology."""
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
        f"ALTER COLUMN enabled_agent_tools SET DEFAULT {_DEFAULT_WITH_SHARING}"
    )
    op.get_bind().execute(
        sa.text(
            """
            UPDATE agent_skills
            SET required_tools = required_tools || CAST(:tools AS jsonb),
                instructions = instructions || CAST(:instructions AS text),
                description = :description
            WHERE ownership_kind = 'builtin'
              AND name = 'skill-authoring'
            """
        ),
        {
            "tools": json.dumps(_TOOLS),
            "instructions": _SHARING_INSTRUCTIONS,
            "description": (
                "创建、修改、整理、归档、恢复或分享个人 Skill，并维护其按需资料时使用。"
            ),
        },
    )


def downgrade() -> None:
    """Remove sharing tools while preserving the authoring seed."""
    op.get_bind().execute(
        sa.text(
            """
            UPDATE agent_skills
            SET required_tools = (required_tools - :invite - :respond - :revoke),
                instructions = replace(
                    instructions,
                    CAST(:instructions AS text),
                    ''
                ),
                description = :description
            WHERE ownership_kind = 'builtin'
              AND name = 'skill-authoring'
            """
        ),
        {
            "invite": _TOOLS[0],
            "respond": _TOOLS[1],
            "revoke": _TOOLS[2],
            "instructions": _SHARING_INSTRUCTIONS,
            "description": ("创建、修改、整理、归档或恢复个人 Skill，并维护其按需资料时使用。"),
        },
    )
    for tool in reversed(_TOOLS):
        op.execute(
            f"""
            UPDATE channel_settings
            SET enabled_agent_tools = enabled_agent_tools - '{tool}'
            """
        )
    op.execute(
        "ALTER TABLE channel_settings "
        f"ALTER COLUMN enabled_agent_tools SET DEFAULT {_DEFAULT_WITHOUT_SHARING}"
    )
