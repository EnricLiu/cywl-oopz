from __future__ import annotations

import pytest
from oopz_sdk import OopzConfig

from cywl_oopz.core.errors import ConfigurationError
from cywl_oopz.settings import (
    AgentMode,
    AppSettings,
    ChatSettings,
    DatabaseSettings,
    MusicSettings,
)


def valid_environment() -> dict[str, str]:
    return {
        "OOPZ_DEVICE_ID": "device",
        "OOPZ_PERSON_UID": "bot",
        "OOPZ_JWT_TOKEN": "token",
        "DATABASE_URL": "postgresql://user:secret@localhost:5432/cywl",
        "CYWL_CHAT_ENABLED": "false",
    }


def test_database_url_is_normalized_to_asyncpg() -> None:
    settings = DatabaseSettings.from_mapping(valid_environment())

    assert settings.url == "postgresql+asyncpg://user:secret@localhost:5432/cywl"


def test_app_settings_use_the_injected_oopz_credentials(monkeypatch) -> None:
    for name in ("OOPZ_DEVICE_ID", "OOPZ_PERSON_UID", "OOPZ_JWT_TOKEN"):
        monkeypatch.delenv(name, raising=False)

    settings = AppSettings.from_mapping(valid_environment())

    assert settings.oopz.device_id == "device"
    assert settings.oopz.person_uid == "bot"
    assert settings.oopz.jwt_token == "token"
    assert settings.agent.mode is AgentMode.LEGACY
    assert settings.agent.enabled_tools == (
        "get_agent_status",
        "get_channel_settings",
        "react_to_message",
        "search_music_catalog",
        "enqueue_music",
        "get_music_queue",
        "skip_music",
        "pause_music",
        "resume_music",
    )
    assert settings.agent.summary_enabled is True
    assert settings.agent.memory_enabled_by_default is True
    assert settings.agent.live_display is False
    assert settings.agent.display_edit_interval_seconds == 0.8
    assert settings.music.enabled is False


def test_agent_mode_uses_database_catalog_without_legacy_llm_credentials() -> None:
    settings = AppSettings.from_mapping(valid_environment() | {"CYWL_AGENT_MODE": "agent"})

    assert settings.agent.enabled is True
    assert settings.chat.enabled is False
    assert "初音未来" in settings.agent.system_prompt
    assert "♪" in settings.agent.system_prompt
    assert "把准确、清楚和实际完成目标放在角色表现之前" in settings.agent.system_prompt


def test_agent_system_prompt_keeps_custom_base_instructions() -> None:
    settings = AppSettings.from_mapping(
        valid_environment()
        | {
            "CYWL_AGENT_SYSTEM_PROMPT": "你是社区里的点歌搭子。",
        }
    )

    assert settings.agent.system_prompt == "你是社区里的点歌搭子。"


def test_agent_live_display_settings_are_validated() -> None:
    settings = AppSettings.from_mapping(
        valid_environment()
        | {
            "CYWL_AGENT_LIVE_DISPLAY": "true",
            "CYWL_AGENT_DISPLAY_EDIT_INTERVAL_SECONDS": "1.25",
        }
    )

    assert settings.agent.live_display is True
    assert settings.agent.display_edit_interval_seconds == 1.25

    with pytest.raises(
        ConfigurationError,
        match="CYWL_AGENT_DISPLAY_EDIT_INTERVAL_SECONDS",
    ):
        AppSettings.from_mapping(
            valid_environment() | {"CYWL_AGENT_DISPLAY_EDIT_INTERVAL_SECONDS": "0"}
        )


def test_agent_tools_can_be_disabled_independently() -> None:
    settings = AppSettings.from_mapping(
        valid_environment()
        | {
            "CYWL_AGENT_MODE": "agent",
            "CYWL_AGENT_ENABLED_TOOLS": "",
        }
    )

    assert settings.agent.enabled_tools == ()


def test_agent_summary_and_memory_limits_are_consistent() -> None:
    with pytest.raises(ConfigurationError, match="SUMMARY_RETAIN_MESSAGES"):
        AppSettings.from_mapping(
            valid_environment()
            | {
                "CYWL_AGENT_SUMMARY_TRIGGER_MESSAGES": "4",
                "CYWL_AGENT_SUMMARY_RETAIN_MESSAGES": "4",
            }
        )
    with pytest.raises(ConfigurationError, match="MEMORY_CONTEXT_ITEMS"):
        AppSettings.from_mapping(
            valid_environment()
            | {
                "CYWL_AGENT_MEMORY_MAX_ITEMS": "2",
                "CYWL_AGENT_MEMORY_CONTEXT_ITEMS": "3",
            }
        )


def test_settings_require_database_url_without_echoing_secret() -> None:
    values = valid_environment()
    values.pop("DATABASE_URL")

    with pytest.raises(ConfigurationError, match="DATABASE_URL") as error:
        AppSettings.from_mapping(values)

    assert "secret" not in str(error.value)


def test_enabled_chat_requires_provider_credentials() -> None:
    values = valid_environment() | {"CYWL_CHAT_ENABLED": "true"}

    with pytest.raises(ConfigurationError, match="CYWL_LLM_BASE_URL"):
        ChatSettings.from_mapping(values)


def test_enabled_music_requires_http_catalog_and_loads_bounds() -> None:
    with pytest.raises(ConfigurationError, match="MUSIC_CATALOG_BASE_URL"):
        MusicSettings.from_mapping({"CYWL_MUSIC_ENABLED": "true"})

    settings = MusicSettings.from_mapping(
        {
            "CYWL_MUSIC_ENABLED": "true",
            "CYWL_MUSIC_CATALOG_BASE_URL": "http://music.example/",
            "CYWL_MUSIC_SEARCH_LIMIT": "7",
            "CYWL_MUSIC_MAX_QUEUE_LENGTH": "12",
        }
    )

    assert settings.catalog_base_url == "http://music.example"
    assert settings.search_limit == 7
    assert settings.max_queue_length == 12


def test_from_environment_loads_dotenv_from_the_current_directory(tmp_path, monkeypatch) -> None:
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "OOPZ_DEVICE_ID=device",
                "OOPZ_PERSON_UID=bot",
                "OOPZ_JWT_TOKEN=token",
                "DATABASE_URL=postgresql://user:secret@localhost:5432/cywl",
                "CYWL_CHAT_ENABLED=false",
            ]
        )
    )
    for name in valid_environment():
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)

    settings = AppSettings.from_environment()

    assert settings.database.url.startswith("postgresql+asyncpg://")


@pytest.mark.asyncio
async def test_from_environment_async_uses_async_oopz_loading(
    tmp_path,
    monkeypatch,
) -> None:
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "DATABASE_URL=postgresql://user:secret@localhost:5432/cywl",
                "CYWL_CHAT_ENABLED=false",
            ]
        )
    )
    for name in valid_environment():
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    expected = AppSettings.from_mapping(valid_environment()).oopz
    called = False

    async def load_oopz() -> OopzConfig:
        nonlocal called
        called = True
        return expected

    monkeypatch.setattr(OopzConfig, "from_env_async", staticmethod(load_oopz))

    settings = await AppSettings.from_environment_async()

    assert called is True
    assert settings.oopz is expected
    assert settings.database.url.startswith("postgresql+asyncpg://")
