"""Add user ownership and sharing to PostgreSQL Agent Skills.

Revision ID: 20260729_15
Revises: 20260729_14
Create Date: 2026-07-29 22:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260729_15"
down_revision: str | Sequence[str] | None = "20260729_14"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_OWNERSHIP_KIND = postgresql.ENUM(
    "builtin",
    "personal",
    name="agent_skill_ownership_kind",
    create_type=False,
)
_SHARE_STATUS = postgresql.ENUM(
    "pending",
    "accepted",
    "declined",
    name="agent_skill_share_status",
    create_type=False,
)


def upgrade() -> None:
    """Classify existing Skills as builtin and add personal sharing grants."""
    bind = op.get_bind()
    _OWNERSHIP_KIND.create(bind, checkfirst=True)
    _SHARE_STATUS.create(bind, checkfirst=True)
    op.add_column(
        "agent_skills",
        sa.Column(
            "ownership_kind",
            _OWNERSHIP_KIND,
            server_default=sa.text("'builtin'"),
            nullable=False,
        ),
    )
    op.add_column(
        "agent_skills",
        sa.Column("owner_person_id", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "agent_skills",
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_check_constraint(
        "ck_agent_skills_ownership",
        "agent_skills",
        "(ownership_kind = 'builtin' AND owner_person_id IS NULL) OR "
        "(ownership_kind = 'personal' AND "
        "char_length(btrim(owner_person_id)) > 0)",
    )
    op.drop_constraint("agent_skills_name_key", "agent_skills", type_="unique")
    op.create_index(
        "ux_agent_skills_builtin_name",
        "agent_skills",
        ["name"],
        unique=True,
        postgresql_where=sa.text("ownership_kind = 'builtin'"),
    )
    op.create_index(
        "ux_agent_skills_personal_owner_name",
        "agent_skills",
        ["owner_person_id", "name"],
        unique=True,
        postgresql_where=sa.text("ownership_kind = 'personal'"),
    )
    op.create_table(
        "agent_skill_shares",
        sa.Column(
            "id",
            sa.Uuid(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("skill_id", sa.Uuid(), nullable=False),
        sa.Column("recipient_person_id", sa.String(length=128), nullable=False),
        sa.Column(
            "status",
            _SHARE_STATUS,
            server_default=sa.text("'pending'"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("responded_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "(status = 'pending' AND responded_at IS NULL) OR "
            "(status IN ('accepted', 'declined') AND responded_at IS NOT NULL)",
            name="ck_agent_skill_shares_response",
        ),
        sa.ForeignKeyConstraint(
            ["skill_id"],
            ["agent_skills.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("skill_id", "recipient_person_id"),
    )
    op.create_index(
        "ix_agent_skill_shares_recipient_status",
        "agent_skill_shares",
        ["recipient_person_id", "status"],
    )
    op.create_index(
        "ix_agent_skill_shares_skill_status",
        "agent_skill_shares",
        ["skill_id", "status"],
    )
    op.execute(
        """
        CREATE TRIGGER trg_agent_skill_shares_set_updated_at
        BEFORE UPDATE ON agent_skill_shares
        FOR EACH ROW
        EXECUTE FUNCTION cywl_set_updated_at()
        """
    )


def downgrade() -> None:
    """Remove personal Skills because their names may conflict globally."""
    op.execute("DROP TRIGGER trg_agent_skill_shares_set_updated_at ON agent_skill_shares")
    op.drop_index("ix_agent_skill_shares_skill_status", table_name="agent_skill_shares")
    op.drop_index(
        "ix_agent_skill_shares_recipient_status",
        table_name="agent_skill_shares",
    )
    op.drop_table("agent_skill_shares")
    op.drop_index("ux_agent_skills_personal_owner_name", table_name="agent_skills")
    op.drop_index("ux_agent_skills_builtin_name", table_name="agent_skills")
    op.execute("DELETE FROM agent_skills WHERE ownership_kind = 'personal'")
    op.create_unique_constraint("agent_skills_name_key", "agent_skills", ["name"])
    op.drop_constraint("ck_agent_skills_ownership", "agent_skills", type_="check")
    op.drop_column("agent_skills", "archived_at")
    op.drop_column("agent_skills", "owner_person_id")
    op.drop_column("agent_skills", "ownership_kind")
    _SHARE_STATUS.drop(op.get_bind(), checkfirst=True)
    _OWNERSHIP_KIND.drop(op.get_bind(), checkfirst=True)
