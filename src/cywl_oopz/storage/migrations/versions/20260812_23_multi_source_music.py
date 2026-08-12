"""Teach the built-in music curator about multiple catalog sources.

Revision ID: 20260812_23
Revises: 20260811_22
Create Date: 2026-08-12 18:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260812_23"
down_revision: str | Sequence[str] | None = "20260811_22"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_CURATOR_ADDITION = """

11. 普通关键词使用 `source=auto`，让 Bot 采用部署配置的默认来源；用户明确指定
    网易云、YouTube 或 Bilibili 时必须保留该选择，不要静默换源。
12. 单曲 URL 直接交给 `add_music_playlist_track`，由工具识别来源，不要降级成关键词搜索。
13. 现场版、翻唱、分 P、MV 等版本敏感请求先调用 `search_music_catalog`，再将候选中
    真实的 `source` 与 `source_id` 作为精确 `track` 提交；绝不猜测来源 ID。
14. area 共享歌单允许混合来源。整理、读取和加载歌单时保留每首曲目的来源，
    不要为了统一来源而重新搜索或替换用户已经确认的版本。"""


def upgrade() -> None:
    """Update only the Alembic-managed curator at the preceding version."""
    op.get_bind().execute(
        sa.text(
            """
            UPDATE agent_skills
            SET instructions = instructions || :addition,
                version = '1.2.0',
                updated_at = CURRENT_TIMESTAMP
            WHERE name = 'music-curator'
              AND version = '1.1.0'
              AND metadata->>'builtin_seed' = '20260729_12'
            """
        ),
        {"addition": _CURATOR_ADDITION},
    )


def downgrade() -> None:
    """Restore only the exact version managed by this migration."""
    op.get_bind().execute(
        sa.text(
            """
            UPDATE agent_skills
            SET instructions = replace(instructions, :addition, ''),
                version = '1.1.0',
                updated_at = CURRENT_TIMESTAMP
            WHERE name = 'music-curator'
              AND version = '1.2.0'
              AND metadata->>'builtin_seed' = '20260729_12'
            """
        ),
        {"addition": _CURATOR_ADDITION},
    )
