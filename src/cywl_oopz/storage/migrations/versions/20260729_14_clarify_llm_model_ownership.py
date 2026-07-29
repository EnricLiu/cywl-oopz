"""Clarify the many-to-one Provider/model relationship.

Revision ID: 20260729_14
Revises: 20260729_13
Create Date: 2026-07-29 21:00:00.000000
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260729_14"
down_revision: str | Sequence[str] | None = "20260729_13"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Rename partial indexes and document that one Provider owns many models."""
    op.execute(
        "ALTER INDEX uq_llm_models_provider_default RENAME TO ux_llm_models_one_provider_default"
    )
    op.execute(
        "ALTER INDEX uq_llm_models_application_default "
        "RENAME TO ux_llm_models_one_application_default"
    )
    op.execute(
        """
        COMMENT ON COLUMN llm_models.provider_id IS
        'Many-to-one owner; multiple models may reference the same provider.'
        """
    )
    op.execute(
        """
        COMMENT ON INDEX ux_llm_models_one_provider_default IS
        'Partial unique index: at most one row with is_provider_default=true per provider.'
        """
    )
    op.execute(
        """
        COMMENT ON INDEX ux_llm_models_one_application_default IS
        'Partial unique index: at most one row with is_application_default=true globally.'
        """
    )


def downgrade() -> None:
    """Restore the former index names and remove explanatory comments."""
    op.execute("COMMENT ON COLUMN llm_models.provider_id IS NULL")
    op.execute("COMMENT ON INDEX ux_llm_models_one_provider_default IS NULL")
    op.execute("COMMENT ON INDEX ux_llm_models_one_application_default IS NULL")
    op.execute(
        "ALTER INDEX ux_llm_models_one_provider_default RENAME TO uq_llm_models_provider_default"
    )
    op.execute(
        "ALTER INDEX ux_llm_models_one_application_default "
        "RENAME TO uq_llm_models_application_default"
    )
