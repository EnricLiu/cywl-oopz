"""Add the AI Agent provider catalog and runtime tables.

Revision ID: 20260726_02
Revises: 20260726_01
Create Date: 2026-07-26 23:40:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260726_02"
down_revision: str | Sequence[str] | None = "20260726_01"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create additive provider, thread, run, message, and tool tables."""
    op.create_table(
        "llm_providers",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("alias", sa.String(length=128), nullable=False),
        sa.Column("display_name", sa.String(length=256), nullable=False),
        sa.Column("protocol", sa.String(length=64), nullable=False),
        sa.Column("base_url", sa.String(length=2048), nullable=False),
        sa.Column("api_key", sa.Text(), nullable=False),
        sa.Column("user_selectable", sa.Boolean(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("config", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("alias"),
    )
    op.create_table(
        "llm_models",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("provider_id", sa.Uuid(), nullable=False),
        sa.Column("alias", sa.String(length=128), nullable=False),
        sa.Column("remote_model_name", sa.String(length=256), nullable=False),
        sa.Column("display_name", sa.String(length=256), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("is_provider_default", sa.Boolean(), nullable=False),
        sa.Column("is_application_default", sa.Boolean(), nullable=False),
        sa.Column("capabilities", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("limits", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("fallback_model_id", sa.Uuid(), nullable=True),
        sa.Column("pricing", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["fallback_model_id"],
            ["llm_models.id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["provider_id"],
            ["llm_providers.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("provider_id", "alias"),
    )
    op.create_index(
        "uq_llm_models_provider_default",
        "llm_models",
        ["provider_id"],
        unique=True,
        postgresql_where=sa.text("is_provider_default"),
    )
    op.create_index(
        "uq_llm_models_application_default",
        "llm_models",
        ["is_application_default"],
        unique=True,
        postgresql_where=sa.text("is_application_default"),
    )
    op.add_column("channel_settings", sa.Column("default_model_id", sa.Uuid(), nullable=True))
    op.create_foreign_key(
        "fk_channel_settings_default_model_id",
        "channel_settings",
        "llm_models",
        ["default_model_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_table(
        "user_llm_preferences",
        sa.Column("person_id", sa.String(length=128), nullable=False),
        sa.Column("preferred_model_id", sa.Uuid(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["preferred_model_id"],
            ["llm_models.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("person_id"),
    )
    op.create_table(
        "agent_threads",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("scope", sa.String(length=32), nullable=False),
        sa.Column("area_id", sa.String(length=128), nullable=False),
        sa.Column("channel_id", sa.String(length=128), nullable=False),
        sa.Column("person_id", sa.String(length=128), nullable=False),
        sa.Column("selected_model_id", sa.Uuid(), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("summary_through_sequence", sa.BigInteger(), nullable=False),
        sa.Column("summary_version", sa.Integer(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["selected_model_id"],
            ["llm_models.id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("scope", "area_id", "channel_id", "person_id"),
    )
    op.create_table(
        "agent_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("thread_id", sa.Uuid(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("stop_reason", sa.String(length=64), nullable=True),
        sa.Column("provider_id", sa.Uuid(), nullable=False),
        sa.Column("model_id", sa.Uuid(), nullable=False),
        sa.Column("selection_source", sa.String(length=32), nullable=False),
        sa.Column("limits", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("usage", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("error_code", sa.String(length=128), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["model_id"], ["llm_models.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(
            ["provider_id"],
            ["llm_providers.id"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(["thread_id"], ["agent_threads.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "agent_messages",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("thread_id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=True),
        sa.Column("sequence", sa.BigInteger(), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("content", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["agent_runs.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["thread_id"], ["agent_threads.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("thread_id", "sequence"),
    )
    op.create_table(
        "agent_tool_executions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("tool_call_id", sa.String(length=256), nullable=False),
        sa.Column("tool_name", sa.String(length=128), nullable=False),
        sa.Column("tool_version", sa.String(length=64), nullable=False),
        sa.Column("effect", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("idempotency_key", sa.String(length=256), nullable=False),
        sa.Column("input", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("output", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("error_code", sa.String(length=128), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["run_id"], ["agent_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", "tool_call_id"),
    )


def downgrade() -> None:
    """Remove only the additive Agent schema."""
    op.drop_table("agent_tool_executions")
    op.drop_table("agent_messages")
    op.drop_table("agent_runs")
    op.drop_table("agent_threads")
    op.drop_table("user_llm_preferences")
    op.drop_constraint(
        "fk_channel_settings_default_model_id",
        "channel_settings",
        type_="foreignkey",
    )
    op.drop_column("channel_settings", "default_model_id")
    op.drop_index("uq_llm_models_application_default", table_name="llm_models")
    op.drop_index("uq_llm_models_provider_default", table_name="llm_models")
    op.drop_table("llm_models")
    op.drop_table("llm_providers")
