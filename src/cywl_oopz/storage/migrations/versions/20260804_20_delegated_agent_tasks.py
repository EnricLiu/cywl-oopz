"""Add durable Agent tasks delegated from realtime voice sessions.

Revision ID: 20260804_20
Revises: 20260804_19
Create Date: 2026-08-04 18:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260804_20"
down_revision: str | Sequence[str] | None = "20260804_19"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

NOW = sa.text("CURRENT_TIMESTAMP")
UUID_DEFAULT = sa.text("gen_random_uuid()")

result_style = postgresql.ENUM(
    "brief", "detailed", name="delegated_result_style", create_type=False
)
task_status = postgresql.ENUM(
    "queued",
    "running",
    "waiting_retry",
    "cancel_requested",
    "succeeded",
    "failed",
    "cancelled",
    "interrupted",
    name="delegated_task_status",
    create_type=False,
)
task_lane = postgresql.ENUM(
    "read_parallel", "mutation_serial", name="delegated_task_lane", create_type=False
)
notification_state = postgresql.ENUM(
    "pending",
    "claimed",
    "presented",
    "deferred",
    name="task_notification_state",
    create_type=False,
)


def upgrade() -> None:
    bind = op.get_bind()
    for enum in (result_style, task_status, task_lane, notification_state):
        enum.create(bind, checkfirst=False)

    op.create_table(
        "delegated_agent_tasks",
        sa.Column("id", sa.Uuid(), server_default=UUID_DEFAULT, nullable=False),
        sa.Column("owner_person_id", sa.String(128), nullable=False),
        sa.Column("area_id", sa.String(128), nullable=False),
        sa.Column("text_channel_id", sa.String(128), nullable=False),
        sa.Column("voice_channel_id", sa.String(128), nullable=False),
        sa.Column("origin_voice_session_id", sa.Uuid(), nullable=False),
        sa.Column("session_sequence", sa.Integer(), nullable=False),
        sa.Column("provider_call_id", sa.String(256), nullable=False),
        sa.Column("objective", sa.Text(), nullable=False),
        sa.Column("result_style", result_style, server_default="brief", nullable=False),
        sa.Column("status", task_status, server_default="queued", nullable=False),
        sa.Column("lane", task_lane, server_default="read_parallel", nullable=False),
        sa.Column("conflict_key", sa.String(256), server_default="", nullable=False),
        sa.Column(
            "notification_state", notification_state, server_default="pending", nullable=False
        ),
        sa.Column("agent_model_id", sa.Uuid(), nullable=False),
        sa.Column(
            "allowed_tool_names",
            postgresql.ARRAY(sa.Text()),
            server_default=sa.text("'{}'::text[]"),
            nullable=False,
        ),
        sa.Column("agent_thread_id", sa.Uuid(), nullable=True),
        sa.Column("agent_run_id", sa.Uuid(), nullable=True),
        sa.Column("progress_stage", sa.String(64), server_default="", nullable=False),
        sa.Column("progress_summary", sa.String(512), server_default="", nullable=False),
        sa.Column("result_summary", sa.Text(), server_default="", nullable=False),
        sa.Column("result_text", sa.Text(), server_default="", nullable=False),
        sa.Column("error_code", sa.String(128), server_default="", nullable=False),
        sa.Column("error_message", sa.Text(), server_default="", nullable=False),
        sa.Column("retry_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("worker_id", sa.String(128), server_default="", nullable=False),
        sa.Column("cancel_requested_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("next_attempt_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("presented_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
        sa.CheckConstraint("session_sequence > 0", name="ck_delegated_tasks_sequence_positive"),
        sa.CheckConstraint("retry_count >= 0", name="ck_delegated_tasks_retry_nonnegative"),
        sa.CheckConstraint(
            "char_length(btrim(objective)) BETWEEN 1 AND 4000",
            name="ck_delegated_tasks_objective_length",
        ),
        sa.CheckConstraint(
            "char_length(progress_stage) <= 64 AND char_length(progress_summary) <= 512",
            name="ck_delegated_tasks_progress_length",
        ),
        sa.CheckConstraint(
            "char_length(result_summary) <= 1000 AND char_length(result_text) <= 16000",
            name="ck_delegated_tasks_result_length",
        ),
        sa.CheckConstraint(
            "char_length(error_code) <= 128 AND char_length(error_message) <= 1000",
            name="ck_delegated_tasks_error_length",
        ),
        sa.ForeignKeyConstraint(
            ["origin_voice_session_id"], ["voice_sessions.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(["agent_model_id"], ["llm_models.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["agent_thread_id"], ["agent_threads.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["agent_run_id"], ["agent_runs.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("origin_voice_session_id", "provider_call_id"),
        sa.UniqueConstraint("origin_voice_session_id", "session_sequence"),
    )
    op.create_index(
        "ix_delegated_tasks_owner_created",
        "delegated_agent_tasks",
        ["owner_person_id", sa.text("created_at DESC")],
    )
    op.create_index(
        "ix_delegated_tasks_worker_claim",
        "delegated_agent_tasks",
        ["status", "next_attempt_at", "created_at"],
        postgresql_where=sa.text("status IN ('queued', 'waiting_retry')"),
    )
    op.create_index(
        "ix_delegated_tasks_mailbox",
        "delegated_agent_tasks",
        ["origin_voice_session_id", "notification_state", "finished_at"],
        postgresql_where=sa.text(
            "status IN ('succeeded', 'failed', 'cancelled', 'interrupted') "
            "AND notification_state IN ('pending', 'deferred')"
        ),
    )
    op.execute(
        """
        CREATE TRIGGER trg_delegated_agent_tasks_set_updated_at
        BEFORE UPDATE ON delegated_agent_tasks
        FOR EACH ROW EXECUTE FUNCTION cywl_set_updated_at()
        """
    )


def downgrade() -> None:
    op.execute(
        "DROP TRIGGER IF EXISTS trg_delegated_agent_tasks_set_updated_at ON delegated_agent_tasks"
    )
    op.drop_index("ix_delegated_tasks_mailbox", table_name="delegated_agent_tasks")
    op.drop_index("ix_delegated_tasks_worker_claim", table_name="delegated_agent_tasks")
    op.drop_index("ix_delegated_tasks_owner_created", table_name="delegated_agent_tasks")
    op.drop_table("delegated_agent_tasks")
    bind = op.get_bind()
    for enum in (notification_state, task_lane, task_status, result_style):
        enum.drop(bind, checkfirst=False)
