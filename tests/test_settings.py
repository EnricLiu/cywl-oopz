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
    WebSearchSafeSearch,
    WebToolsSettings,
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
        "search_web",
        "read_web_page",
        "browser_open",
        "browser_snapshot",
        "browser_wait",
        "browser_close",
        "browser_click",
        "browser_fill",
        "browser_press",
        "set_music_playback_mode",
        "create_music_playlist",
        "list_music_playlists",
        "get_music_playlist",
        "add_music_playlist_track",
        "remove_music_playlist_track",
        "load_music_playlist",
        "load_agent_skill",
        "read_agent_skill_resource",
        "list_agent_skill_library",
        "inspect_agent_skill",
        "create_agent_skill",
        "update_agent_skill",
        "manage_agent_skill_resource",
        "set_agent_skill_state",
        "invite_agent_skill_share",
        "respond_agent_skill_share",
        "revoke_agent_skill_share",
        "preview_netease_playlist",
        "import_netease_playlist",
    )
    assert settings.agent.summary_enabled is True
    assert settings.agent.memory_enabled_by_default is True
    assert settings.agent.live_display is False
    assert settings.agent.display_edit_interval_seconds == 0.8
    assert settings.agent.provider_max_retries == 2
    assert settings.agent.skills_enabled is True
    assert settings.agent.skill_catalog_refresh_seconds == 30
    assert settings.agent.max_available_skills == 32
    assert settings.agent.skill_authoring_enabled is True
    assert settings.agent.max_personal_skills == 16
    assert settings.agent.max_resources_per_skill == 8
    assert settings.agent.max_accepted_shared_skills == 8
    assert settings.agent.max_skill_share_recipients_per_call == 5
    assert settings.agent.max_skill_activations == 3
    assert settings.agent.max_skill_resources == 4
    assert settings.agent.max_skill_instruction_characters == 12000
    assert settings.agent.max_skill_resource_characters == 12000
    assert settings.agent.max_skill_context_characters == 24000
    assert settings.music.enabled is False
    assert settings.web.search_enabled is True
    assert settings.web.search_safesearch is WebSearchSafeSearch.MODERATE
    assert settings.web.browser_enabled is False


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


def test_agent_provider_retry_limit_is_configurable_and_bounded() -> None:
    settings = AppSettings.from_mapping(
        valid_environment() | {"CYWL_AGENT_PROVIDER_MAX_RETRIES": "0"}
    )

    assert settings.agent.provider_max_retries == 0

    with pytest.raises(ConfigurationError, match="PROVIDER_MAX_RETRIES"):
        AppSettings.from_mapping(valid_environment() | {"CYWL_AGENT_PROVIDER_MAX_RETRIES": "6"})


def test_agent_tools_can_be_disabled_independently() -> None:
    settings = AppSettings.from_mapping(
        valid_environment()
        | {
            "CYWL_AGENT_MODE": "agent",
            "CYWL_AGENT_ENABLED_TOOLS": "",
        }
    )

    assert settings.agent.enabled_tools == ()


def test_agent_skill_catalog_settings_are_validated() -> None:
    settings = AppSettings.from_mapping(
        valid_environment()
        | {
            "CYWL_AGENT_SKILLS_ENABLED": "false",
            "CYWL_AGENT_SKILL_CATALOG_REFRESH_SECONDS": "12.5",
            "CYWL_AGENT_MAX_AVAILABLE_SKILLS": "8",
            "CYWL_AGENT_MAX_SKILL_ACTIVATIONS": "2",
            "CYWL_AGENT_MAX_SKILL_RESOURCES": "3",
            "CYWL_AGENT_MAX_SKILL_INSTRUCTION_CHARACTERS": "4000",
            "CYWL_AGENT_MAX_SKILL_RESOURCE_CHARACTERS": "5000",
            "CYWL_AGENT_MAX_SKILL_CONTEXT_CHARACTERS": "9000",
            "CYWL_AGENT_SKILL_AUTHORING_ENABLED": "false",
            "CYWL_AGENT_MAX_PERSONAL_SKILLS": "7",
            "CYWL_AGENT_MAX_RESOURCES_PER_SKILL": "5",
            "CYWL_AGENT_MAX_ACCEPTED_SHARED_SKILLS": "4",
            "CYWL_AGENT_MAX_SKILL_SHARE_RECIPIENTS_PER_CALL": "3",
        }
    )

    assert settings.agent.skills_enabled is False
    assert settings.agent.skill_catalog_refresh_seconds == 12.5
    assert settings.agent.max_available_skills == 8
    assert settings.agent.max_skill_activations == 2
    assert settings.agent.max_skill_resources == 3
    assert settings.agent.max_skill_instruction_characters == 4000
    assert settings.agent.max_skill_resource_characters == 5000
    assert settings.agent.max_skill_context_characters == 9000
    assert settings.agent.skill_authoring_enabled is False
    assert settings.agent.max_personal_skills == 7
    assert settings.agent.max_resources_per_skill == 5
    assert settings.agent.max_accepted_shared_skills == 4
    assert settings.agent.max_skill_share_recipients_per_call == 3

    with pytest.raises(ConfigurationError, match="SKILL_CATALOG_REFRESH_SECONDS"):
        AppSettings.from_mapping(
            valid_environment()
            | {
                "CYWL_AGENT_SKILL_CATALOG_REFRESH_SECONDS": "0",
            }
        )


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
    with pytest.raises(ConfigurationError, match="SKILL_INSTRUCTION_CHARACTERS"):
        AppSettings.from_mapping(
            valid_environment()
            | {
                "CYWL_AGENT_MAX_SKILL_INSTRUCTION_CHARACTERS": "100",
                "CYWL_AGENT_MAX_SKILL_CONTEXT_CHARACTERS": "99",
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
    assert settings.max_playlist_tracks == 1000


def test_web_search_settings_validate_provider_bounds() -> None:
    settings = WebToolsSettings.from_mapping(
        {
            "CYWL_WEB_SEARCH_REGION": "us-en",
            "CYWL_WEB_SEARCH_SAFESEARCH": "off",
            "CYWL_WEB_SEARCH_MAX_RESULTS": "8",
            "CYWL_WEB_SEARCH_TIMEOUT_SECONDS": "3.5",
        }
    )

    assert settings.search_region == "us-en"
    assert settings.search_safesearch is WebSearchSafeSearch.OFF
    assert settings.search_max_results == 8
    assert settings.search_timeout_seconds == 3.5

    with pytest.raises(ConfigurationError, match="SAFESEARCH"):
        WebToolsSettings.from_mapping({"CYWL_WEB_SEARCH_SAFESEARCH": "sometimes"})
    with pytest.raises(ConfigurationError, match="MAX_RESULTS"):
        WebToolsSettings.from_mapping({"CYWL_WEB_SEARCH_MAX_RESULTS": "11"})
    with pytest.raises(ConfigurationError, match="MAX_QUERY_CHARACTERS"):
        WebToolsSettings.from_mapping({"CYWL_WEB_SEARCH_MAX_QUERY_CHARACTERS": "301"})
    with pytest.raises(ConfigurationError, match="MAX_CONCURRENCY"):
        WebToolsSettings.from_mapping({"CYWL_WEB_SEARCH_MAX_CONCURRENCY": "9"})


def test_web_browser_settings_require_the_base_feature_and_bound_concurrency() -> None:
    settings = WebToolsSettings.from_mapping(
        {
            "CYWL_WEB_BROWSER_ENABLED": "true",
            "CYWL_WEB_BROWSER_SESSION_IDLE_SECONDS": "120",
            "CYWL_WEB_BROWSER_MAX_CONCURRENCY": "4",
            "CYWL_WEB_BROWSER_MCP_CALL_TIMEOUT_SECONDS": "7.5",
        }
    )

    assert settings.browser_enabled is True
    assert settings.browser_session_idle_seconds == 120
    assert settings.browser_max_concurrency == 4
    assert settings.browser_mcp_call_timeout_seconds == 7.5

    with pytest.raises(ConfigurationError, match="requires"):
        WebToolsSettings.from_mapping({"CYWL_WEB_BROWSER_INTERACTION_ENABLED": "true"})
    with pytest.raises(ConfigurationError, match="MAX_CONCURRENCY"):
        WebToolsSettings.from_mapping(
            {
                "CYWL_WEB_BROWSER_ENABLED": "true",
                "CYWL_WEB_BROWSER_MAX_CONCURRENCY": "9",
            }
        )


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
