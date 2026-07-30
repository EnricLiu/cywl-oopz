"""Remove the obsolete global Agent Skill catalog generation state.

Revision ID: 20260730_18
Revises: 20260730_17
Create Date: 2026-07-30 03:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260730_18"
down_revision: str | Sequence[str] | None = "20260730_17"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Drop snapshot-only triggers, function, and singleton state."""
    for table in ("agent_skill_resources", "agent_skills"):
        op.execute(f"DROP TRIGGER trg_{table}_bump_generation ON {table}")
    op.execute(
        "DROP TRIGGER trg_agent_skill_catalog_state_set_updated_at ON agent_skill_catalog_state"
    )
    op.execute("DROP FUNCTION cywl_bump_agent_skill_catalog_generation()")
    op.drop_table("agent_skill_catalog_state")


def downgrade() -> None:
    """Restore the generation mechanism required by revision 17 code."""
    op.create_table(
        "agent_skill_catalog_state",
        sa.Column(
            "singleton_id",
            sa.SmallInteger(),
            server_default=sa.text("1"),
            nullable=False,
        ),
        sa.Column(
            "generation",
            sa.BigInteger(),
            server_default=sa.text("1"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "singleton_id = 1",
            name="ck_agent_skill_catalog_state_singleton",
        ),
        sa.CheckConstraint(
            "generation > 0",
            name="ck_agent_skill_catalog_state_generation_positive",
        ),
        sa.PrimaryKeyConstraint("singleton_id"),
    )
    op.execute("INSERT INTO agent_skill_catalog_state (singleton_id, generation) VALUES (1, 1)")
    op.execute(
        """
        CREATE FUNCTION cywl_bump_agent_skill_catalog_generation()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        BEGIN
            INSERT INTO agent_skill_catalog_state (
                singleton_id, generation, updated_at
            )
            VALUES (1, 2, CURRENT_TIMESTAMP)
            ON CONFLICT (singleton_id) DO UPDATE
            SET generation = agent_skill_catalog_state.generation + 1,
                updated_at = CURRENT_TIMESTAMP;
            RETURN NULL;
        END;
        $$
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_agent_skill_catalog_state_set_updated_at
        BEFORE UPDATE ON agent_skill_catalog_state
        FOR EACH ROW
        EXECUTE FUNCTION cywl_set_updated_at()
        """
    )
    for table in ("agent_skills", "agent_skill_resources"):
        op.execute(
            f"""
            CREATE TRIGGER trg_{table}_bump_generation
            AFTER INSERT OR UPDATE OR DELETE ON {table}
            FOR EACH STATEMENT
            EXECUTE FUNCTION cywl_bump_agent_skill_catalog_generation()
            """
        )
