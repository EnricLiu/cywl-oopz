"""Add realtime voice configuration, session, and final transcript storage.

Revision ID: 20260804_19
Revises: 20260730_18
Create Date: 2026-08-04 12:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260804_19"
down_revision: str | Sequence[str] | None = "20260730_18"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

JSON_OBJECT = sa.text("'{}'::jsonb")
NOW = sa.text("CURRENT_TIMESTAMP")
UUID_DEFAULT = sa.text("gen_random_uuid()")

provider_protocol = postgresql.ENUM(
    "qwen_omni_realtime_ws",
    "volc_realtime_dialogue_ws",
    name="voice_provider_protocol",
    create_type=False,
)
model_mode = postgresql.ENUM(
    "native_realtime", "agent_cascade", name="voice_model_mode", create_type=False
)
duplex_mode = postgresql.ENUM("full", "half", name="voice_duplex_mode", create_type=False)
session_status = postgresql.ENUM(
    "starting",
    "active",
    "recovering",
    "ended",
    "failed",
    name="voice_session_status",
    create_type=False,
)
turn_role = postgresql.ENUM("user", "assistant", name="voice_turn_role", create_type=False)


def _timestamps() -> tuple[sa.Column, sa.Column]:
    return (
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
    )


def _updated_trigger(table: str) -> None:
    op.execute(
        f"""
        CREATE TRIGGER trg_{table}_set_updated_at
        BEFORE UPDATE ON {table}
        FOR EACH ROW EXECUTE FUNCTION cywl_set_updated_at()
        """
    )


def upgrade() -> None:
    """Create the complete first realtime voice persistence slice."""
    bind = op.get_bind()
    for enum in (provider_protocol, model_mode, duplex_mode, session_status, turn_role):
        enum.create(bind, checkfirst=False)

    op.create_table(
        "voice_providers",
        sa.Column("id", sa.Uuid(), server_default=UUID_DEFAULT, nullable=False),
        sa.Column("alias", sa.String(128), nullable=False),
        sa.Column("display_name", sa.String(256), nullable=False),
        sa.Column("protocol", provider_protocol, nullable=False),
        sa.Column("endpoint", sa.String(2048), nullable=False),
        sa.Column("credentials", postgresql.JSONB(), server_default=JSON_OBJECT, nullable=False),
        sa.Column("config", postgresql.JSONB(), server_default=JSON_OBJECT, nullable=False),
        sa.Column("enabled", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column("user_selectable", sa.Boolean(), server_default=sa.true(), nullable=False),
        *_timestamps(),
        sa.CheckConstraint(
            "jsonb_typeof(credentials) = 'object'", name="ck_voice_providers_credentials_object"
        ),
        sa.CheckConstraint(
            "jsonb_typeof(config) = 'object'", name="ck_voice_providers_config_object"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("alias"),
    )
    op.create_table(
        "voice_models",
        sa.Column("id", sa.Uuid(), server_default=UUID_DEFAULT, nullable=False),
        sa.Column("provider_id", sa.Uuid(), nullable=False),
        sa.Column("alias", sa.String(128), nullable=False),
        sa.Column("remote_model_name", sa.String(256), nullable=False),
        sa.Column("display_name", sa.String(256), nullable=False),
        sa.Column("mode", model_mode, server_default="native_realtime", nullable=False),
        sa.Column("enabled", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column("is_provider_default", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column(
            "is_application_default", sa.Boolean(), server_default=sa.false(), nullable=False
        ),
        sa.Column("capabilities", postgresql.JSONB(), server_default=JSON_OBJECT, nullable=False),
        sa.Column("audio_config", postgresql.JSONB(), server_default=JSON_OBJECT, nullable=False),
        sa.Column("prompt_config", postgresql.JSONB(), server_default=JSON_OBJECT, nullable=False),
        sa.Column("limits", postgresql.JSONB(), server_default=JSON_OBJECT, nullable=False),
        *_timestamps(),
        sa.CheckConstraint(
            "jsonb_typeof(capabilities) = 'object'", name="ck_voice_models_capabilities_object"
        ),
        sa.CheckConstraint(
            "jsonb_typeof(audio_config) = 'object'", name="ck_voice_models_audio_config_object"
        ),
        sa.CheckConstraint(
            "jsonb_typeof(prompt_config) = 'object'", name="ck_voice_models_prompt_config_object"
        ),
        sa.CheckConstraint("jsonb_typeof(limits) = 'object'", name="ck_voice_models_limits_object"),
        sa.ForeignKeyConstraint(["provider_id"], ["voice_providers.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("provider_id", "alias"),
        sa.UniqueConstraint("provider_id", "remote_model_name"),
    )
    op.create_index(
        "ux_voice_models_one_provider_default",
        "voice_models",
        ["provider_id"],
        unique=True,
        postgresql_where=sa.text("is_provider_default"),
    )
    op.create_index(
        "ux_voice_models_one_application_default",
        "voice_models",
        ["is_application_default"],
        unique=True,
        postgresql_where=sa.text("is_application_default"),
    )
    op.create_table(
        "voice_user_preferences",
        sa.Column("owner_person_id", sa.String(128), nullable=False),
        sa.Column("preferred_model_id", sa.Uuid(), nullable=True),
        sa.Column("voice_id", sa.String(128), server_default="", nullable=False),
        sa.Column("duplex_mode", duplex_mode, server_default="full", nullable=False),
        sa.Column("delegated_agent_model_id", sa.Uuid(), nullable=True),
        *_timestamps(),
        sa.ForeignKeyConstraint(["preferred_model_id"], ["voice_models.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(
            ["delegated_agent_model_id"], ["llm_models.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("owner_person_id"),
    )
    op.create_table(
        "voice_channel_settings",
        sa.Column("area_id", sa.String(128), nullable=False),
        sa.Column("voice_channel_id", sa.String(128), nullable=False),
        sa.Column("enabled", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column(
            "delegated_task_profile",
            sa.String(128),
            server_default="voice_readonly_v1",
            nullable=False,
        ),
        sa.Column("idle_timeout_seconds", sa.Integer(), server_default="300", nullable=False),
        *_timestamps(),
        sa.CheckConstraint(
            "idle_timeout_seconds > 0", name="ck_voice_channel_settings_idle_timeout_positive"
        ),
        sa.PrimaryKeyConstraint("area_id", "voice_channel_id"),
    )
    op.create_table(
        "voice_sessions",
        sa.Column("id", sa.Uuid(), server_default=UUID_DEFAULT, nullable=False),
        sa.Column("owner_person_id", sa.String(128), nullable=False),
        sa.Column("area_id", sa.String(128), nullable=False),
        sa.Column("voice_channel_id", sa.String(128), nullable=False),
        sa.Column("text_channel_id", sa.String(128), nullable=False),
        sa.Column("model_id", sa.Uuid(), nullable=False),
        sa.Column("voice_id", sa.String(128), server_default="", nullable=False),
        sa.Column("duplex_mode", duplex_mode, nullable=False),
        sa.Column("status", session_status, server_default="starting", nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("stop_reason", sa.String(128), server_default="", nullable=False),
        sa.Column("usage", postgresql.JSONB(), server_default=JSON_OBJECT, nullable=False),
        sa.Column("summary", sa.Text(), server_default="", nullable=False),
        *_timestamps(),
        sa.CheckConstraint("jsonb_typeof(usage) = 'object'", name="ck_voice_sessions_usage_object"),
        sa.CheckConstraint(
            "ended_at IS NULL OR ended_at >= started_at",
            name="ck_voice_sessions_ended_after_started",
        ),
        sa.ForeignKeyConstraint(["model_id"], ["voice_models.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_voice_sessions_owner_started", "voice_sessions", ["owner_person_id", "started_at"]
    )
    op.create_table(
        "voice_turns",
        sa.Column("id", sa.Uuid(), server_default=UUID_DEFAULT, nullable=False),
        sa.Column("session_id", sa.Uuid(), nullable=False),
        sa.Column("sequence", sa.BigInteger(), nullable=False),
        sa.Column("role", turn_role, nullable=False),
        sa.Column("transcript", sa.Text(), nullable=False),
        sa.Column("provider_item_id", sa.String(256), server_default="", nullable=False),
        sa.Column("usage", postgresql.JSONB(), server_default=JSON_OBJECT, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
        sa.CheckConstraint("sequence > 0", name="ck_voice_turns_sequence_positive"),
        sa.CheckConstraint(
            "char_length(btrim(transcript)) > 0", name="ck_voice_turns_transcript_nonempty"
        ),
        sa.CheckConstraint("jsonb_typeof(usage) = 'object'", name="ck_voice_turns_usage_object"),
        sa.ForeignKeyConstraint(["session_id"], ["voice_sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("session_id", "sequence"),
    )
    for table in (
        "voice_providers",
        "voice_models",
        "voice_user_preferences",
        "voice_channel_settings",
        "voice_sessions",
    ):
        _updated_trigger(table)


def downgrade() -> None:
    """Remove voice persistence without touching the shared timestamp function."""
    for table in (
        "voice_sessions",
        "voice_channel_settings",
        "voice_user_preferences",
        "voice_models",
        "voice_providers",
    ):
        op.execute(f"DROP TRIGGER trg_{table}_set_updated_at ON {table}")
    op.drop_table("voice_turns")
    op.drop_index("ix_voice_sessions_owner_started", table_name="voice_sessions")
    op.drop_table("voice_sessions")
    op.drop_table("voice_channel_settings")
    op.drop_table("voice_user_preferences")
    op.drop_index("ux_voice_models_one_application_default", table_name="voice_models")
    op.drop_index("ux_voice_models_one_provider_default", table_name="voice_models")
    op.drop_table("voice_models")
    op.drop_table("voice_providers")
    bind = op.get_bind()
    for enum in (turn_role, session_status, duplex_mode, model_mode, provider_protocol):
        enum.drop(bind, checkfirst=False)
