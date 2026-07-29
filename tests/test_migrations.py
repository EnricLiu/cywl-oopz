from alembic.config import Config
from alembic.script import ScriptDirectory

from cywl_oopz.storage.models import Base


def test_initial_schema_models_and_migration_head_are_present() -> None:
    config = Config("alembic.ini")
    revisions = ScriptDirectory.from_config(config)

    assert revisions.get_current_head() == "20260729_16"
    assert set(Base.metadata.tables) == {
        "agent_memory_items",
        "agent_memory_preferences",
        "agent_messages",
        "agent_runs",
        "agent_skill_catalog_state",
        "agent_skill_resources",
        "agent_skill_shares",
        "agent_skills",
        "agent_threads",
        "agent_tool_executions",
        "channel_settings",
        "conversation_sessions",
        "llm_models",
        "llm_providers",
        "music_playlist_tracks",
        "music_playlists",
        "rate_limit_buckets",
        "user_llm_preferences",
    }

    assert Base.metadata.tables["channel_settings"].c.id.server_default is not None
    assert Base.metadata.tables["channel_settings"].c.created_at.server_default is not None
    assert Base.metadata.tables["channel_settings"].c.updated_at.server_default is not None
    assert Base.metadata.tables["agent_runs"].c.status.type.name == "agent_run_status"
    assert (
        Base.metadata.tables["agent_tool_executions"].c.status.type.name == "tool_execution_status"
    )
    assert (
        Base.metadata.tables["agent_skill_resources"].c.kind.type.name
        == "agent_skill_resource_kind"
    )
    agent_skills = Base.metadata.tables["agent_skills"]
    assert agent_skills.c.ownership_kind.type.name == "agent_skill_ownership_kind"
    assert {index.name for index in agent_skills.indexes} >= {
        "ux_agent_skills_builtin_name",
        "ux_agent_skills_personal_owner_name",
    }
    assert {constraint.name for constraint in agent_skills.constraints} >= {
        "ck_agent_skills_personal_state",
    }
    assert (
        Base.metadata.tables["agent_skill_shares"].c.status.type.name == "agent_skill_share_status"
    )
    llm_models = Base.metadata.tables["llm_models"]
    assert llm_models.c.provider_id.unique is not True
    assert {index.name for index in llm_models.indexes} >= {
        "ux_llm_models_one_provider_default",
        "ux_llm_models_one_application_default",
    }
