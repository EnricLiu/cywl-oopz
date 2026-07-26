from alembic.config import Config
from alembic.script import ScriptDirectory

from cywl_oopz.storage.models import Base


def test_initial_schema_models_and_migration_head_are_present() -> None:
    config = Config("alembic.ini")
    revisions = ScriptDirectory.from_config(config)

    assert revisions.get_current_head() == "20260727_03"
    assert set(Base.metadata.tables) == {
        "agent_messages",
        "agent_runs",
        "agent_threads",
        "agent_tool_executions",
        "channel_settings",
        "conversation_sessions",
        "llm_models",
        "llm_providers",
        "rate_limit_buckets",
        "user_llm_preferences",
    }
