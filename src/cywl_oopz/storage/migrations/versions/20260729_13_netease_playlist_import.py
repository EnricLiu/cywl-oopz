"""Add Netease playlist import tools and Skill.

Revision ID: 20260729_13
Revises: 20260729_12
Create Date: 2026-07-29 18:00:00.000000
"""

import json
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260729_13"
down_revision: str | Sequence[str] | None = "20260729_12"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TOOLS = (
    "preview_netease_playlist",
    "import_netease_playlist",
)
_SEED_MARKER = "20260729_13"
_DEFAULT_WITH_IMPORT = (
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
    '"preview_netease_playlist",'
    '"import_netease_playlist"'
    "]'::jsonb"
)
_DEFAULT_WITHOUT_IMPORT = (
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
    '"read_agent_skill_resource"'
    "]'::jsonb"
)

_INSTRUCTIONS = """\
# 网易云歌单导入

当用户要把网易云音乐歌单保存为当前 OOPZ area 的共享歌单时，按以下流程行动。

1. 接受网易云歌单数字 ID 或 `music.163.com` 的标准歌单链接，不自行猜测或拼接 ID。
2. 先调用 `list_music_playlists` 查看当前 area 的歌单，识别同名歌单和用户可能指定的目标名称。
3. 必须调用 `preview_netease_playlist`。向用户说明源歌单名称、声明歌曲数、可见歌曲数，
   以及是否完整；不要把预览成功说成已经导入。
4. 完整且没有同名冲突时，用户已经明确要求导入即可调用 `import_netease_playlist`。
   省略 `name` 会沿用网易云歌单名称；用户指定新名称时原样传入规范化前的名称。
5. 若预览显示 `requires_partial_confirmation=true`，先说明缺失或容量截断的歌曲数。
   只有用户明确接受部分导入后，才可把 `allow_partial` 设为 true；不得替用户默认同意。
6. 不要先调用 `create_music_playlist`，也不要逐首搜索和追加。导入工具会在一个数据库事务中
   创建新歌单并按网易云顺序写入全部可见曲目，失败时不会留下半成品。
7. 工具返回同名冲突时，列出当前歌单并询问新名称，不要删除或覆盖已有 area 歌单。
8. 成功后报告新 area 歌单 ID、导入数和跳过数。只有用户另外要求播放时，
   才使用 `load_music_playlist`；导入本身不会改变当前播放队列。

私有或受权限限制的网易云歌单可能只能返回部分歌曲。始终以工具报告的计数为准。\
"""

_RESOURCE = """\
# 网易云 API 行为说明

- `/playlist/detail` 可返回歌单名称和完整 `trackIds`，但其中的 `tracks` 可能不完整。
- 完整歌曲元数据应通过 `/playlist/track/all` 获取；接口支持 `limit` 和从 0 开始的 `offset`。
- 未登录、私有歌单、地区或版权限制仍可能导致可见歌曲少于歌单声明数量。
- CYWL 只保存歌曲 ID、标题、歌手和时长；播放时再通过现有音乐目录解析临时播放地址。
- area 歌单有配置容量。源歌单超过容量时，默认拒绝；用户明确同意后才能导入可见前缀。
- 导入创建一个新的 area 歌单，不会覆盖同名歌单，也不会自动加入或替换播放队列。\
"""


def upgrade() -> None:
    """Enable import tools and seed their progressive-disclosure workflow."""
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
        f"ALTER COLUMN enabled_agent_tools SET DEFAULT {_DEFAULT_WITH_IMPORT}"
    )
    connection = op.get_bind()
    connection.execute(
        sa.text(
            """
            INSERT INTO agent_skills (
                name,
                display_name,
                description,
                instructions,
                version,
                required_tools,
                metadata
            )
            VALUES (
                'netease-playlist-importer',
                '网易云歌单导入',
                :description,
                :instructions,
                '1.0.0',
                CAST(:required_tools AS jsonb),
                jsonb_build_object(
                    'builtin_seed', CAST(:seed_marker AS text),
                    'managed_by', 'alembic'
                )
            )
            ON CONFLICT (name) DO NOTHING
            """
        ),
        {
            "instructions": _INSTRUCTIONS,
            "description": (
                "把网易云歌单预览并原子导入为当前 area 的共享歌单；用户提供歌单 ID 或链接时使用。"
            ),
            "required_tools": json.dumps(
                (
                    "list_music_playlists",
                    "preview_netease_playlist",
                    "import_netease_playlist",
                )
            ),
            "seed_marker": _SEED_MARKER,
        },
    )
    connection.execute(
        sa.text(
            """
            INSERT INTO agent_skill_resources (
                skill_id,
                key,
                display_name,
                description,
                kind,
                media_type,
                content,
                position
            )
            SELECT
                id,
                'netease-api-behavior',
                '网易云 API 行为说明',
                '预览结果不完整或需要解释导入边界时读取。',
                'reference',
                'text/markdown',
                :content,
                1
            FROM agent_skills
            WHERE name = 'netease-playlist-importer'
              AND metadata ->> 'builtin_seed' = CAST(:seed_marker AS text)
            ON CONFLICT DO NOTHING
            """
        ),
        {
            "content": _RESOURCE,
            "seed_marker": _SEED_MARKER,
        },
    )


def downgrade() -> None:
    """Remove only this seed and restore prior channel tool defaults."""
    op.get_bind().execute(
        sa.text(
            """
            DELETE FROM agent_skills
            WHERE metadata ->> 'builtin_seed' = CAST(:seed_marker AS text)
            """
        ),
        {"seed_marker": _SEED_MARKER},
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
        f"ALTER COLUMN enabled_agent_tools SET DEFAULT {_DEFAULT_WITHOUT_IMPORT}"
    )
