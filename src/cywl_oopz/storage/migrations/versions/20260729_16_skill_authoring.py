"""Enable conversational Skill authoring tools and workflow.

Revision ID: 20260729_16
Revises: 20260729_15
Create Date: 2026-07-29 23:30:00.000000
"""

import json
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260729_16"
down_revision: str | Sequence[str] | None = "20260729_15"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TOOLS = (
    "list_agent_skill_library",
    "inspect_agent_skill",
    "create_agent_skill",
    "update_agent_skill",
    "manage_agent_skill_resource",
    "set_agent_skill_state",
)
_SEED_MARKER = "20260729_16"
_DEFAULT_WITH_AUTHORING = (
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
_DEFAULT_WITHOUT_AUTHORING = (
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

_INSTRUCTIONS = """\
# Skill 创作与维护

只有用户当前明确要求创建、修改、整理、归档或恢复 Skill 时，才维护其技能库。

1. 先调用 `list_agent_skill_library` 理解现状，避免创建重复 Skill。编辑、维护资料或改变状态前，
   必须调用 `inspect_agent_skill` 获取最新正文和 revision；不要使用对话历史中的旧 revision。
2. 把 Skill 设计为可重复使用的方法：description 只简短说明“何时使用”，instructions 说明
   “怎样做”、必要工具、失败处理和停止条件。不要把一次性事实、短期对话或用户隐私写进 Skill；
   这些内容属于当前对话或 memory。
3. required_tools 只能选本轮真实存在的工具，不得虚构能力。Skill 本身不授予额外工具权限。
4. 创建时使用稳定的 lowercase kebab-case name。用途或会实质改变行为的要求仍不明确时，
   先询问用户，不要自行补全关键意图。
5. 更新时完整替换需要改变的字段，并传入 fresh inspect 返回的 expected_revision。
   revision 冲突后重新 inspect，再按用户原意重新整理；绝不覆盖并发修改。
6. 长参考、模板或示例放进 resource，让运行时按需读取；instructions 应可独立理解，
   不应依赖未说明的资料。一次只维护一份 resource。
7. builtin 和 shared Skill 都是只读的。不要尝试修改、归档或转售它们。
8. 归档是可恢复的软状态。归档前确认用户确实要停用；恢复后原共享授权会重新可见。
9. 创建或修改成功后，只简洁报告改动和“下一轮生效”；不要声称当前 Agent run 已热加载新内容。

保持内容直接、步骤有限、错误时不声称成功。\
"""


def upgrade() -> None:
    """Enable authoring tools and seed their builtin methodology Skill."""
    op.execute(
        """
        UPDATE agent_skills
        SET archived_at = CURRENT_TIMESTAMP
        WHERE ownership_kind = 'personal'
          AND enabled = false
          AND archived_at IS NULL
        """
    )
    op.execute(
        """
        UPDATE agent_skills
        SET enabled = false
        WHERE ownership_kind = 'personal'
          AND archived_at IS NOT NULL
          AND enabled = true
        """
    )
    op.create_check_constraint(
        "ck_agent_skills_personal_state",
        "agent_skills",
        "ownership_kind = 'builtin' OR enabled = (archived_at IS NULL)",
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
        f"ALTER COLUMN enabled_agent_tools SET DEFAULT {_DEFAULT_WITH_AUTHORING}"
    )
    op.get_bind().execute(
        sa.text(
            """
            INSERT INTO agent_skills (
                name,
                display_name,
                description,
                instructions,
                version,
                required_tools,
                metadata,
                ownership_kind
            )
            VALUES (
                'skill-authoring',
                '技能创作',
                :description,
                :instructions,
                '1.0.0',
                CAST(:required_tools AS jsonb),
                jsonb_build_object(
                    'builtin_seed', CAST(:seed_marker AS text),
                    'managed_by', 'alembic'
                ),
                'builtin'
            )
            ON CONFLICT DO NOTHING
            """
        ),
        {
            "description": ("创建、修改、整理、归档或恢复个人 Skill，并维护其按需资料时使用。"),
            "instructions": _INSTRUCTIONS,
            "required_tools": json.dumps(_TOOLS),
            "seed_marker": _SEED_MARKER,
        },
    )


def downgrade() -> None:
    """Remove only the authoring seed and restore previous tool defaults."""
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
        f"ALTER COLUMN enabled_agent_tools SET DEFAULT {_DEFAULT_WITHOUT_AUTHORING}"
    )
    op.drop_constraint(
        "ck_agent_skills_personal_state",
        "agent_skills",
        type_="check",
    )
