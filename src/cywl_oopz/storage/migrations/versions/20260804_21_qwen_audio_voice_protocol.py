"""Add the Qwen-Audio realtime voice Provider protocol.

Revision ID: 20260804_21
Revises: 20260804_20
Create Date: 2026-08-04 22:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260804_21"
down_revision: str | Sequence[str] | None = "20260804_20"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

previous_protocol = postgresql.ENUM(
    "qwen_omni_realtime_ws",
    "volc_realtime_dialogue_ws",
    name="voice_provider_protocol",
    create_type=False,
)


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("ALTER TYPE voice_provider_protocol ADD VALUE 'qwen_audio_realtime_ws'")


def downgrade() -> None:
    bind = op.get_bind()
    using_audio = bind.scalar(
        sa.text(
            "SELECT EXISTS (SELECT 1 FROM voice_providers "
            "WHERE protocol::text = 'qwen_audio_realtime_ws')"
        )
    )
    if using_audio:
        raise RuntimeError(
            "Cannot remove qwen_audio_realtime_ws while voice providers still use it"
        )
    op.execute("ALTER TABLE voice_providers ALTER COLUMN protocol TYPE text USING protocol::text")
    op.execute("ALTER TYPE voice_provider_protocol RENAME TO voice_provider_protocol_old")
    previous_protocol.create(bind, checkfirst=False)
    op.execute(
        "ALTER TABLE voice_providers ALTER COLUMN protocol TYPE voice_provider_protocol "
        "USING protocol::voice_provider_protocol"
    )
    op.execute("DROP TYPE voice_provider_protocol_old")
