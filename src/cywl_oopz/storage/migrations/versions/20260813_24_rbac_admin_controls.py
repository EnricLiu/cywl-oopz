"""Add scoped RBAC and outbound-message diagnostic linkage.

Revision ID: 20260813_24
Revises: 20260812_23
Create Date: 2026-08-13 12:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260813_24"
down_revision: str | Sequence[str] | None = "20260812_23"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

NOW = sa.text("CURRENT_TIMESTAMP")
UUID_DEFAULT = sa.text("gen_random_uuid()")
EMPTY_STRING = sa.text("''")
JSON_OBJECT = sa.text("'{}'::jsonb")

rbac_role = postgresql.ENUM(
    "owner",
    "admin",
    "moderator",
    name="rbac_role",
    create_type=False,
)
rbac_scope = postgresql.ENUM(
    "global",
    "area",
    "channel",
    name="rbac_scope",
    create_type=False,
)
oopz_message_scope = postgresql.ENUM(
    "channel",
    "private",
    name="oopz_message_scope",
    create_type=False,
)
oopz_outbound_message_kind = postgresql.ENUM(
    "agent_response",
    "command_reply",
    "status",
    "notification",
    name="oopz_outbound_message_kind",
    create_type=False,
)
oopz_outbound_message_state = postgresql.ENUM(
    "active",
    "final",
    "recalled",
    "superseded",
    name="oopz_outbound_message_state",
    create_type=False,
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
    """Create the complete RBAC persistence foundation."""
    bind = op.get_bind()
    for enum in (
        rbac_role,
        rbac_scope,
        oopz_message_scope,
        oopz_outbound_message_kind,
        oopz_outbound_message_state,
    ):
        enum.create(bind, checkfirst=False)

    op.create_table(
        "rbac_role_bindings",
        sa.Column("id", sa.Uuid(), server_default=UUID_DEFAULT, nullable=False),
        sa.Column("subject_person_id", sa.String(length=128), nullable=False),
        sa.Column("role", rbac_role, nullable=False),
        sa.Column("scope", rbac_scope, nullable=False),
        sa.Column("area_id", sa.String(length=128), server_default=EMPTY_STRING, nullable=False),
        sa.Column("channel_id", sa.String(length=128), server_default=EMPTY_STRING, nullable=False),
        sa.Column(
            "granted_by_person_id",
            sa.String(length=128),
            server_default=EMPTY_STRING,
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
        sa.CheckConstraint(
            "(scope = 'global' AND area_id = '' AND channel_id = '') OR "
            "(scope = 'area' AND area_id <> '' AND channel_id = '') OR "
            "(scope = 'channel' AND area_id <> '' AND channel_id <> '')",
            name="ck_rbac_role_bindings_scope_address",
        ),
        sa.CheckConstraint(
            "role <> 'owner' OR scope = 'global'",
            name="ck_rbac_role_bindings_owner_global",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "subject_person_id",
            "role",
            "scope",
            "area_id",
            "channel_id",
            name="uq_rbac_role_bindings_assignment",
        ),
    )
    op.create_index(
        "ix_rbac_role_bindings_subject_scope",
        "rbac_role_bindings",
        ["subject_person_id", "scope", "area_id", "channel_id"],
    )
    op.create_index(
        "ix_rbac_role_bindings_resource",
        "rbac_role_bindings",
        ["scope", "area_id", "channel_id"],
    )

    op.add_column(
        "agent_runs",
        sa.Column(
            "diagnostics",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=JSON_OBJECT,
            nullable=False,
        ),
    )
    op.create_check_constraint(
        "ck_agent_runs_diagnostics_object",
        "agent_runs",
        "jsonb_typeof(diagnostics) = 'object'",
    )

    op.create_table(
        "oopz_outbound_messages",
        sa.Column("id", sa.Uuid(), server_default=UUID_DEFAULT, nullable=False),
        sa.Column("message_id", sa.String(length=256), nullable=False),
        sa.Column(
            "message_timestamp",
            sa.String(length=64),
            server_default=EMPTY_STRING,
            nullable=False,
        ),
        sa.Column("kind", oopz_outbound_message_kind, nullable=False),
        sa.Column(
            "state",
            oopz_outbound_message_state,
            server_default=sa.text("'final'"),
            nullable=False,
        ),
        sa.Column("scope", oopz_message_scope, nullable=False),
        sa.Column("area_id", sa.String(length=128), server_default=EMPTY_STRING, nullable=False),
        sa.Column("channel_id", sa.String(length=128), nullable=False),
        sa.Column(
            "target_person_id",
            sa.String(length=128),
            server_default=EMPTY_STRING,
            nullable=False,
        ),
        sa.Column(
            "in_reply_to_message_id",
            sa.String(length=256),
            server_default=EMPTY_STRING,
            nullable=False,
        ),
        sa.Column(
            "owner_person_id",
            sa.String(length=128),
            server_default=EMPTY_STRING,
            nullable=False,
        ),
        sa.Column("agent_run_id", sa.Uuid(), nullable=True),
        sa.Column(
            "diagnostic_snapshot",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=JSON_OBJECT,
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
        sa.Column("recalled_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "(scope = 'channel' AND area_id <> '' AND channel_id <> '' "
            "AND target_person_id = '') OR "
            "(scope = 'private' AND area_id = '' AND channel_id <> '' "
            "AND target_person_id <> '')",
            name="ck_oopz_outbound_messages_scope_address",
        ),
        sa.CheckConstraint(
            "jsonb_typeof(diagnostic_snapshot) = 'object'",
            name="ck_oopz_outbound_messages_diagnostic_object",
        ),
        sa.ForeignKeyConstraint(["agent_run_id"], ["agent_runs.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("message_id"),
    )
    op.create_index(
        "ix_oopz_outbound_messages_address",
        "oopz_outbound_messages",
        ["scope", "area_id", "channel_id", "target_person_id"],
    )
    op.create_index(
        "ix_oopz_outbound_messages_agent_run",
        "oopz_outbound_messages",
        ["agent_run_id"],
    )

    _updated_trigger("rbac_role_bindings")
    _updated_trigger("oopz_outbound_messages")


def downgrade() -> None:
    """Remove RBAC and outbound-message records without touching shared functions."""
    op.execute("DROP TRIGGER trg_oopz_outbound_messages_set_updated_at ON oopz_outbound_messages")
    op.execute("DROP TRIGGER trg_rbac_role_bindings_set_updated_at ON rbac_role_bindings")

    op.drop_index("ix_oopz_outbound_messages_agent_run", table_name="oopz_outbound_messages")
    op.drop_index("ix_oopz_outbound_messages_address", table_name="oopz_outbound_messages")
    op.drop_table("oopz_outbound_messages")

    op.drop_constraint("ck_agent_runs_diagnostics_object", "agent_runs", type_="check")
    op.drop_column("agent_runs", "diagnostics")

    op.drop_index("ix_rbac_role_bindings_resource", table_name="rbac_role_bindings")
    op.drop_index("ix_rbac_role_bindings_subject_scope", table_name="rbac_role_bindings")
    op.drop_table("rbac_role_bindings")

    bind = op.get_bind()
    for enum in reversed(
        (
            rbac_role,
            rbac_scope,
            oopz_message_scope,
            oopz_outbound_message_kind,
            oopz_outbound_message_state,
        )
    ):
        enum.drop(bind, checkfirst=False)
