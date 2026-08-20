from alembic.config import Config
from alembic.script import ScriptDirectory

from cywl_oopz.storage.models import Base


def test_initial_schema_models_and_migration_head_are_present() -> None:
    config = Config("alembic.ini")
    revisions = ScriptDirectory.from_config(config)

    assert revisions.get_current_head() == "20260820_25"
    assert set(Base.metadata.tables) == {
        "agent_memory_items",
        "agent_memory_preferences",
        "agent_media_assets",
        "agent_messages",
        "agent_runs",
        "agent_skill_resources",
        "agent_skill_shares",
        "agent_skills",
        "agent_threads",
        "agent_tool_executions",
        "channel_settings",
        "conversation_sessions",
        "delegated_agent_tasks",
        "llm_models",
        "llm_providers",
        "music_playlist_tracks",
        "music_playlists",
        "oopz_outbound_messages",
        "rate_limit_buckets",
        "rbac_role_bindings",
        "user_llm_preferences",
        "voice_channel_settings",
        "voice_models",
        "voice_providers",
        "voice_sessions",
        "voice_turns",
        "voice_user_preferences",
    }

    assert Base.metadata.tables["channel_settings"].c.id.server_default is not None
    assert Base.metadata.tables["channel_settings"].c.created_at.server_default is not None
    assert Base.metadata.tables["channel_settings"].c.updated_at.server_default is not None
    assert Base.metadata.tables["agent_runs"].c.status.type.name == "agent_run_status"
    assert Base.metadata.tables["agent_runs"].c.diagnostics.server_default is not None
    media_assets = Base.metadata.tables["agent_media_assets"]
    assert media_assets.c.id.server_default is not None
    assert media_assets.c.created_at.server_default is not None
    assert {index.name for index in media_assets.indexes} >= {
        "ix_agent_media_assets_message_id",
    }
    assert Base.metadata.tables["rbac_role_bindings"].c.role.type.name == "rbac_role"
    assert Base.metadata.tables["rbac_role_bindings"].c.scope.type.name == "rbac_scope"
    outbound = Base.metadata.tables["oopz_outbound_messages"]
    assert outbound.c.scope.type.name == "oopz_message_scope"
    assert outbound.c.kind.type.name == "oopz_outbound_message_kind"
    assert outbound.c.state.type.name == "oopz_outbound_message_state"
    assert outbound.c.diagnostic_snapshot.server_default is not None
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
    voice_providers = Base.metadata.tables["voice_providers"]
    assert voice_providers.c.id.server_default is not None
    assert voice_providers.c.credentials.server_default is not None
    assert voice_providers.c.protocol.type.name == "voice_provider_protocol"
    assert "qwen_audio_realtime_ws" in voice_providers.c.protocol.type.enums
    voice_models = Base.metadata.tables["voice_models"]
    assert voice_models.c.provider_id.unique is not True
    assert voice_models.c.mode.type.name == "voice_model_mode"
    assert {index.name for index in voice_models.indexes} >= {
        "ux_voice_models_one_provider_default",
        "ux_voice_models_one_application_default",
    }
    voice_preferences = Base.metadata.tables["voice_user_preferences"]
    assert voice_preferences.c.preferred_model_id.nullable is True
    assert voice_preferences.c.duplex_mode.server_default is not None
    voice_channels = Base.metadata.tables["voice_channel_settings"]
    assert voice_channels.c.enabled.server_default is not None
    assert voice_channels.c.updated_at.server_default is not None
    voice_sessions = Base.metadata.tables["voice_sessions"]
    assert voice_sessions.c.status.type.name == "voice_session_status"
    assert voice_sessions.c.usage.server_default is not None
    voice_turns = Base.metadata.tables["voice_turns"]
    assert voice_turns.c.role.type.name == "voice_turn_role"
    assert voice_turns.c.created_at.server_default is not None
    delegated_tasks = Base.metadata.tables["delegated_agent_tasks"]
    assert delegated_tasks.c.id.server_default is not None
    assert delegated_tasks.c.status.type.name == "delegated_task_status"
    assert delegated_tasks.c.lane.type.name == "delegated_task_lane"
    assert delegated_tasks.c.notification_state.type.name == "task_notification_state"
    assert delegated_tasks.c.allowed_tool_names.server_default is not None
    assert {index.name for index in delegated_tasks.indexes} >= {
        "ix_delegated_tasks_owner_created",
        "ix_delegated_tasks_worker_claim",
        "ix_delegated_tasks_mailbox",
    }
