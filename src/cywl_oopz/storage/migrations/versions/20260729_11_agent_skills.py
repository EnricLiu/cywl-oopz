"""Add PostgreSQL-backed Agent skills and catalog generation.

Revision ID: 20260729_11
Revises: 20260727_10
Create Date: 2026-07-29 10:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260729_11"
down_revision: str | Sequence[str] | None = "20260727_10"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_RESOURCE_KIND = postgresql.ENUM(
    "reference",
    "template",
    "example",
    name="agent_skill_resource_kind",
    create_type=False,
)
_SKILL_TOOLS = (
    "load_agent_skill",
    "read_agent_skill_resource",
)
_DEFAULT_WITH_SKILLS = (
    "'["
    '"get_agent_status",'
    '"get_channel_settings",'
    '"react_to_message",'
    '"search_music_catalog",'
    '"enqueue_music",'
    '"get_music_queue",'
    '"skip_music",'
    '"pause_music",'
    '"resume_music",'
    '"search_web",'
    '"read_web_page",'
    '"set_music_playback_mode",'
    '"create_music_playlist",'
    '"list_music_playlists",'
    '"get_music_playlist",'
    '"add_music_playlist_track",'
    '"remove_music_playlist_track",'
    '"load_music_playlist",'
    '"load_agent_skill",'
    '"read_agent_skill_resource"'
    "]'::jsonb"
)
_DEFAULT_WITHOUT_SKILLS = (
    "'["
    '"get_agent_status",'
    '"get_channel_settings",'
    '"react_to_message",'
    '"search_music_catalog",'
    '"enqueue_music",'
    '"get_music_queue",'
    '"skip_music",'
    '"pause_music",'
    '"resume_music",'
    '"search_web",'
    '"read_web_page",'
    '"set_music_playback_mode",'
    '"create_music_playlist",'
    '"list_music_playlists",'
    '"get_music_playlist",'
    '"add_music_playlist_track",'
    '"remove_music_playlist_track",'
    '"load_music_playlist"'
    "]'::jsonb"
)


def upgrade() -> None:
    """Create skill bundles, resource text, and change-detection triggers."""
    _RESOURCE_KIND.create(op.get_bind(), checkfirst=True)
    op.execute(
        """
        CREATE FUNCTION cywl_valid_agent_skill_required_tools(value jsonb)
        RETURNS boolean
        LANGUAGE plpgsql
        IMMUTABLE
        STRICT
        AS $$
        BEGIN
            IF jsonb_typeof(value) <> 'array' THEN
                RETURN false;
            END IF;
            RETURN NOT EXISTS (
                SELECT 1
                FROM jsonb_array_elements(value) AS item
                WHERE jsonb_typeof(item) <> 'string'
                   OR btrim(item #>> '{}') = ''
                   OR (item #>> '{}') !~ '^[a-z][a-z0-9_]{0,127}$'
            )
            AND jsonb_array_length(value) = (
                SELECT count(DISTINCT item)
                FROM jsonb_array_elements_text(value) AS item
            );
        END;
        $$
        """
    )
    op.create_table(
        "agent_skills",
        sa.Column(
            "id",
            sa.Uuid(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.Column("display_name", sa.String(length=80), nullable=False),
        sa.Column("description", sa.String(length=1024), nullable=False),
        sa.Column("instructions", sa.Text(), nullable=False),
        sa.Column("version", sa.String(length=64), nullable=False),
        sa.Column(
            "revision",
            sa.BigInteger(),
            server_default=sa.text("1"),
            nullable=False,
        ),
        sa.Column(
            "required_tools",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "enabled",
            sa.Boolean(),
            server_default=sa.text("true"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "name ~ '^[a-z][a-z0-9-]{0,63}$'",
            name="ck_agent_skills_name",
        ),
        sa.CheckConstraint(
            "char_length(btrim(display_name)) > 0",
            name="ck_agent_skills_display_name",
        ),
        sa.CheckConstraint(
            "char_length(btrim(description)) > 0 AND char_length(description) <= 1024",
            name="ck_agent_skills_description",
        ),
        sa.CheckConstraint(
            "char_length(btrim(instructions)) > 0 AND char_length(instructions) <= 20000",
            name="ck_agent_skills_instructions",
        ),
        sa.CheckConstraint(
            "char_length(btrim(version)) > 0",
            name="ck_agent_skills_version",
        ),
        sa.CheckConstraint(
            "revision > 0",
            name="ck_agent_skills_revision_positive",
        ),
        sa.CheckConstraint(
            "cywl_valid_agent_skill_required_tools(required_tools)",
            name="ck_agent_skills_required_tools",
        ),
        sa.CheckConstraint(
            "jsonb_typeof(metadata) = 'object'",
            name="ck_agent_skills_metadata_object",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_table(
        "agent_skill_resources",
        sa.Column(
            "id",
            sa.Uuid(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("skill_id", sa.Uuid(), nullable=False),
        sa.Column("key", sa.String(length=160), nullable=False),
        sa.Column("display_name", sa.String(length=120), nullable=False),
        sa.Column("description", sa.String(length=500), nullable=False),
        sa.Column("kind", _RESOURCE_KIND, nullable=False),
        sa.Column("media_type", sa.String(length=100), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "key ~ '^[a-z][a-z0-9-]{0,159}$'",
            name="ck_agent_skill_resources_key",
        ),
        sa.CheckConstraint(
            "char_length(btrim(display_name)) > 0",
            name="ck_agent_skill_resources_display_name",
        ),
        sa.CheckConstraint(
            "char_length(btrim(description)) > 0",
            name="ck_agent_skill_resources_description",
        ),
        sa.CheckConstraint(
            "char_length(btrim(content)) > 0 AND char_length(content) <= 20000",
            name="ck_agent_skill_resources_content",
        ),
        sa.CheckConstraint(
            "media_type IN ('text/markdown', 'text/plain', 'application/json')",
            name="ck_agent_skill_resources_media_type",
        ),
        sa.CheckConstraint(
            "position > 0",
            name="ck_agent_skill_resources_position_positive",
        ),
        sa.ForeignKeyConstraint(
            ["skill_id"],
            ["agent_skills.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("skill_id", "key"),
        sa.UniqueConstraint("skill_id", "position"),
    )
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
    _create_skill_triggers()
    for tool in _SKILL_TOOLS:
        op.execute(
            f"""
            UPDATE channel_settings
            SET enabled_agent_tools = enabled_agent_tools || '["{tool}"]'::jsonb
            WHERE NOT enabled_agent_tools ? '{tool}'
            """
        )
    op.execute(
        "ALTER TABLE channel_settings "
        f"ALTER COLUMN enabled_agent_tools SET DEFAULT {_DEFAULT_WITH_SKILLS}"
    )


def downgrade() -> None:
    """Remove skill storage and its channel tool defaults."""
    for tool in reversed(_SKILL_TOOLS):
        op.execute(
            f"""
            UPDATE channel_settings
            SET enabled_agent_tools = enabled_agent_tools - '{tool}'
            """
        )
    op.execute(
        "ALTER TABLE channel_settings "
        f"ALTER COLUMN enabled_agent_tools SET DEFAULT {_DEFAULT_WITHOUT_SKILLS}"
    )
    _drop_skill_triggers()
    op.drop_table("agent_skill_catalog_state")
    op.drop_table("agent_skill_resources")
    op.drop_table("agent_skills")
    op.execute("DROP FUNCTION cywl_valid_agent_skill_required_tools(jsonb)")
    _RESOURCE_KIND.drop(op.get_bind(), checkfirst=True)


def _create_skill_triggers() -> None:
    op.execute(
        """
        CREATE FUNCTION cywl_bump_agent_skill_revision()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        BEGIN
            NEW.revision = OLD.revision + 1;
            RETURN NEW;
        END;
        $$
        """
    )
    op.execute(
        """
        CREATE FUNCTION cywl_touch_agent_skill_from_resource()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        BEGIN
            IF TG_OP = 'DELETE' THEN
                UPDATE agent_skills SET updated_at = CURRENT_TIMESTAMP
                WHERE id = OLD.skill_id;
            ELSIF TG_OP = 'UPDATE' AND OLD.skill_id <> NEW.skill_id THEN
                UPDATE agent_skills SET updated_at = CURRENT_TIMESTAMP
                WHERE id IN (OLD.skill_id, NEW.skill_id);
            ELSE
                UPDATE agent_skills SET updated_at = CURRENT_TIMESTAMP
                WHERE id = NEW.skill_id;
            END IF;
            RETURN NULL;
        END;
        $$
        """
    )
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
        CREATE TRIGGER trg_agent_skills_bump_revision
        BEFORE UPDATE ON agent_skills
        FOR EACH ROW
        EXECUTE FUNCTION cywl_bump_agent_skill_revision()
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_agent_skills_set_updated_at
        BEFORE UPDATE ON agent_skills
        FOR EACH ROW
        EXECUTE FUNCTION cywl_set_updated_at()
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_agent_skill_resources_set_updated_at
        BEFORE UPDATE ON agent_skill_resources
        FOR EACH ROW
        EXECUTE FUNCTION cywl_set_updated_at()
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_agent_skill_resources_touch_skill
        AFTER INSERT OR UPDATE OR DELETE ON agent_skill_resources
        FOR EACH ROW
        EXECUTE FUNCTION cywl_touch_agent_skill_from_resource()
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


def _drop_skill_triggers() -> None:
    for table in ("agent_skill_resources", "agent_skills"):
        op.execute(f"DROP TRIGGER trg_{table}_bump_generation ON {table}")
    op.execute(
        "DROP TRIGGER trg_agent_skill_catalog_state_set_updated_at ON agent_skill_catalog_state"
    )
    op.execute("DROP TRIGGER trg_agent_skill_resources_touch_skill ON agent_skill_resources")
    op.execute("DROP TRIGGER trg_agent_skill_resources_set_updated_at ON agent_skill_resources")
    op.execute("DROP TRIGGER trg_agent_skills_set_updated_at ON agent_skills")
    op.execute("DROP TRIGGER trg_agent_skills_bump_revision ON agent_skills")
    op.execute("DROP FUNCTION cywl_bump_agent_skill_catalog_generation()")
    op.execute("DROP FUNCTION cywl_touch_agent_skill_from_resource()")
    op.execute("DROP FUNCTION cywl_bump_agent_skill_revision()")
