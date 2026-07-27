"""Add database-owned defaults, timestamps, and lifecycle enums.

Revision ID: 20260727_06
Revises: 20260727_05
Create Date: 2026-07-27 15:30:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260727_06"
down_revision: str | Sequence[str] | None = "20260727_05"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_AGENT_RUN_STATUS = postgresql.ENUM(
    "pending",
    "running",
    "succeeded",
    "failed",
    "cancelled",
    "abandoned",
    name="agent_run_status",
    create_type=False,
)
_AGENT_STOP_REASON = postgresql.ENUM(
    "completed",
    "cancelled",
    "timeout",
    "model_request_limit",
    "tool_call_limit",
    "token_limit",
    "tool_denied",
    "provider_error",
    "tool_error",
    "invalid_output",
    "stale_run_abandoned",
    name="agent_stop_reason",
    create_type=False,
)
_MODEL_SELECTION_SOURCE = postgresql.ENUM(
    "thread",
    "user",
    "channel",
    "application",
    name="model_selection_source",
    create_type=False,
)
_TOOL_EFFECT = postgresql.ENUM(
    "read",
    "write",
    "admin",
    name="tool_effect",
    create_type=False,
)
_TOOL_EXECUTION_STATUS = postgresql.ENUM(
    "started",
    "succeeded",
    "failed",
    "denied",
    "cancelled",
    name="tool_execution_status",
    create_type=False,
)

_DEFAULT_AGENT_TOOLS_SQL = (
    "'["
    '"get_agent_status",'
    '"get_channel_settings",'
    '"react_to_message",'
    '"search_music_catalog",'
    '"enqueue_music",'
    '"get_music_queue",'
    '"skip_music",'
    '"pause_music",'
    '"resume_music"'
    "]'::jsonb"
)

_DEFAULTS = (
    ("channel_settings", "id", "gen_random_uuid()"),
    ("channel_settings", "chat_enabled", "false"),
    (
        "channel_settings",
        "enabled_agent_tools",
        _DEFAULT_AGENT_TOOLS_SQL,
    ),
    ("channel_settings", "created_at", "CURRENT_TIMESTAMP"),
    ("channel_settings", "updated_at", "CURRENT_TIMESTAMP"),
    ("conversation_sessions", "id", "gen_random_uuid()"),
    ("conversation_sessions", "area_id", "''"),
    ("conversation_sessions", "channel_id", "''"),
    ("conversation_sessions", "messages", "'[]'::json"),
    ("conversation_sessions", "created_at", "CURRENT_TIMESTAMP"),
    ("conversation_sessions", "updated_at", "CURRENT_TIMESTAMP"),
    ("rate_limit_buckets", "updated_at", "CURRENT_TIMESTAMP"),
    ("rate_limit_buckets", "hit_count", "0"),
    ("rate_limit_buckets", "reason", "''"),
    ("llm_providers", "id", "gen_random_uuid()"),
    ("llm_providers", "user_selectable", "true"),
    ("llm_providers", "enabled", "true"),
    ("llm_providers", "config", "'{}'::jsonb"),
    ("llm_providers", "created_at", "CURRENT_TIMESTAMP"),
    ("llm_providers", "updated_at", "CURRENT_TIMESTAMP"),
    ("llm_models", "id", "gen_random_uuid()"),
    ("llm_models", "enabled", "true"),
    ("llm_models", "is_provider_default", "false"),
    ("llm_models", "is_application_default", "false"),
    ("llm_models", "capabilities", "'[]'::jsonb"),
    ("llm_models", "limits", "'{}'::jsonb"),
    ("llm_models", "pricing", "'{}'::jsonb"),
    ("llm_models", "created_at", "CURRENT_TIMESTAMP"),
    ("llm_models", "updated_at", "CURRENT_TIMESTAMP"),
    ("user_llm_preferences", "updated_at", "CURRENT_TIMESTAMP"),
    ("agent_threads", "id", "gen_random_uuid()"),
    ("agent_threads", "area_id", "''"),
    ("agent_threads", "channel_id", "''"),
    ("agent_threads", "summary", "''"),
    ("agent_threads", "summary_through_sequence", "0"),
    ("agent_threads", "summary_version", "0"),
    ("agent_threads", "version", "1"),
    ("agent_threads", "created_at", "CURRENT_TIMESTAMP"),
    ("agent_threads", "updated_at", "CURRENT_TIMESTAMP"),
    ("agent_memory_preferences", "enabled", "true"),
    ("agent_memory_preferences", "updated_at", "CURRENT_TIMESTAMP"),
    ("agent_memory_items", "id", "gen_random_uuid()"),
    ("agent_memory_items", "namespace", "'explicit'"),
    ("agent_memory_items", "created_at", "CURRENT_TIMESTAMP"),
    ("agent_memory_items", "updated_at", "CURRENT_TIMESTAMP"),
    ("agent_runs", "id", "gen_random_uuid()"),
    ("agent_runs", "status", "'pending'"),
    ("agent_runs", "limits", "'{}'::jsonb"),
    ("agent_runs", "usage", "'{}'::jsonb"),
    ("agent_runs", "error_code", "''"),
    ("agent_runs", "started_at", "CURRENT_TIMESTAMP"),
    ("agent_runs", "heartbeat_at", "CURRENT_TIMESTAMP"),
    ("agent_messages", "id", "gen_random_uuid()"),
    ("agent_messages", "created_at", "CURRENT_TIMESTAMP"),
    ("agent_tool_executions", "id", "gen_random_uuid()"),
    ("agent_tool_executions", "status", "'started'"),
    ("agent_tool_executions", "error_code", "''"),
    ("agent_tool_executions", "started_at", "CURRENT_TIMESTAMP"),
)

_UPDATED_AT_TABLES = (
    "channel_settings",
    "conversation_sessions",
    "rate_limit_buckets",
    "llm_providers",
    "llm_models",
    "user_llm_preferences",
    "agent_threads",
    "agent_memory_preferences",
    "agent_memory_items",
)


def upgrade() -> None:
    """Move stable defaults and lifecycle validation into PostgreSQL."""
    bind = op.get_bind()
    for enum_type in (
        _AGENT_RUN_STATUS,
        _AGENT_STOP_REASON,
        _MODEL_SELECTION_SOURCE,
        _TOOL_EFFECT,
        _TOOL_EXECUTION_STATUS,
    ):
        enum_type.create(bind, checkfirst=True)

    op.alter_column(
        "agent_runs",
        "status",
        existing_type=sa.String(length=32),
        type_=_AGENT_RUN_STATUS,
        postgresql_using="status::text::agent_run_status",
    )
    op.alter_column(
        "agent_runs",
        "stop_reason",
        existing_type=sa.String(length=64),
        type_=_AGENT_STOP_REASON,
        postgresql_using="stop_reason::text::agent_stop_reason",
    )
    op.alter_column(
        "agent_runs",
        "selection_source",
        existing_type=sa.String(length=32),
        type_=_MODEL_SELECTION_SOURCE,
        postgresql_using="selection_source::text::model_selection_source",
    )
    op.alter_column(
        "agent_tool_executions",
        "effect",
        existing_type=sa.String(length=32),
        type_=_TOOL_EFFECT,
        postgresql_using="effect::text::tool_effect",
    )
    op.alter_column(
        "agent_tool_executions",
        "status",
        existing_type=sa.String(length=32),
        type_=_TOOL_EXECUTION_STATUS,
        postgresql_using="status::text::tool_execution_status",
    )

    for table, column, expression in _DEFAULTS:
        op.execute(f'ALTER TABLE "{table}" ALTER COLUMN "{column}" SET DEFAULT {expression}')

    op.execute(
        """
        CREATE FUNCTION cywl_set_updated_at()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$
        """
    )
    for table in _UPDATED_AT_TABLES:
        op.execute(
            f"""
            CREATE TRIGGER trg_{table}_set_updated_at
            BEFORE UPDATE ON "{table}"
            FOR EACH ROW
            EXECUTE FUNCTION cywl_set_updated_at()
            """
        )


def downgrade() -> None:
    """Restore application-owned defaults and string lifecycle columns."""
    for table in reversed(_UPDATED_AT_TABLES):
        op.execute(f'DROP TRIGGER trg_{table}_set_updated_at ON "{table}"')
    op.execute("DROP FUNCTION cywl_set_updated_at()")

    for table, column, _ in reversed(_DEFAULTS):
        op.execute(f'ALTER TABLE "{table}" ALTER COLUMN "{column}" DROP DEFAULT')

    op.alter_column(
        "agent_tool_executions",
        "status",
        existing_type=_TOOL_EXECUTION_STATUS,
        type_=sa.String(length=32),
        postgresql_using="status::text",
    )
    op.alter_column(
        "agent_tool_executions",
        "effect",
        existing_type=_TOOL_EFFECT,
        type_=sa.String(length=32),
        postgresql_using="effect::text",
    )
    op.alter_column(
        "agent_runs",
        "selection_source",
        existing_type=_MODEL_SELECTION_SOURCE,
        type_=sa.String(length=32),
        postgresql_using="selection_source::text",
    )
    op.alter_column(
        "agent_runs",
        "stop_reason",
        existing_type=_AGENT_STOP_REASON,
        type_=sa.String(length=64),
        postgresql_using="stop_reason::text",
    )
    op.alter_column(
        "agent_runs",
        "status",
        existing_type=_AGENT_RUN_STATUS,
        type_=sa.String(length=32),
        postgresql_using="status::text",
    )

    for enum_type in (
        _TOOL_EXECUTION_STATUS,
        _TOOL_EFFECT,
        _MODEL_SELECTION_SOURCE,
        _AGENT_STOP_REASON,
        _AGENT_RUN_STATUS,
    ):
        enum_type.drop(op.get_bind(), checkfirst=True)
