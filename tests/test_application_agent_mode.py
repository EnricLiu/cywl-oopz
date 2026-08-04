from __future__ import annotations

from uuid import UUID

import pytest
from oopz_sdk import VoiceCapabilities

import cywl_oopz.application as application_module
from cywl_oopz.application import BotApplication
from cywl_oopz.core.errors import ConfigurationError
from cywl_oopz.features.agent.catalog import ProviderCatalog
from cywl_oopz.features.agent.commands import AgentModelCommand
from cywl_oopz.features.agent.models import LlmModel, LlmProvider, ProviderProtocol
from cywl_oopz.features.chat.commands import ModelCommand
from cywl_oopz.features.web.errors import BrowserUnavailableError
from cywl_oopz.settings import AppSettings

PROVIDER_ID = UUID("10000000-0000-0000-0000-000000000001")
MODEL_ID = UUID("10000000-0000-0000-0000-000000000002")


class FakeOopzBot:
    def __init__(self, config) -> None:
        self.config = config

    def on_ready(self, handler) -> None:
        self.ready_handler = handler

    def on_message(self, handler) -> None:
        self.message_handler = handler

    async def run(self) -> None:
        self.did_run = True


class IncompleteVoiceContractBot(FakeOopzBot):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.voice = type(
            "IncompleteVoiceBackend",
            (),
            {
                "capabilities": VoiceCapabilities(
                    feature_version=0,
                    remote_audio_subscription=True,
                    person_audio_subscription=False,
                    streaming_pcm_output=False,
                    playback_cursor=False,
                    typed_playback_handle=False,
                )
            },
        )()


def settings(mode: str, **overrides: str) -> AppSettings:
    return AppSettings.from_mapping(
        {
            "OOPZ_DEVICE_ID": "device",
            "OOPZ_PERSON_UID": "bot",
            "OOPZ_JWT_TOKEN": "token",
            "DATABASE_URL": "postgresql://user:secret@localhost:5432/cywl",
            "CYWL_CHAT_ENABLED": "false",
            "CYWL_AGENT_MODE": mode,
            **overrides,
        }
    )


async def no_stale_voice_sessions(now) -> int:
    del now
    return 0


@pytest.mark.asyncio
async def test_composition_root_routes_chat_and_provider_command_by_agent_flag(
    monkeypatch,
) -> None:
    monkeypatch.setattr(application_module, "OopzBot", FakeOopzBot)
    agent_application = BotApplication(settings("agent"))
    legacy_application = BotApplication(settings("legacy"))

    assert agent_application.chat is agent_application.agent_chat
    assert isinstance(
        next(command for command in agent_application.commands.commands if command.name == "model"),
        AgentModelCommand,
    )
    assert "provider" in {command.name for command in agent_application.commands.commands}
    assert "tools" in {command.name for command in agent_application.commands.commands}
    assert "tool" in {command.name for command in agent_application.commands.commands}
    assert "memory" in {command.name for command in agent_application.commands.commands}
    assert "skills" in {command.name for command in agent_application.commands.commands}
    assert {
        "load_agent_skill",
        "read_agent_skill_resource",
        "list_agent_skill_library",
        "inspect_agent_skill",
        "create_agent_skill",
        "update_agent_skill",
        "manage_agent_skill_resource",
        "set_agent_skill_state",
    }.issubset(agent_application.agent_tool_registry.names)
    assert legacy_application.chat is legacy_application.legacy_chat
    assert isinstance(
        next(
            command for command in legacy_application.commands.commands if command.name == "model"
        ),
        ModelCommand,
    )
    assert "provider" not in {command.name for command in legacy_application.commands.commands}
    assert "tools" not in {command.name for command in legacy_application.commands.commands}
    assert "tool" not in {command.name for command in legacy_application.commands.commands}
    assert "memory" not in {command.name for command in legacy_application.commands.commands}
    assert "skills" not in {command.name for command in legacy_application.commands.commands}

    for application in (agent_application, legacy_application):
        if application.music is not None:
            await application.music.aclose()
        await application.agent_engine.aclose()
        await application._provider.aclose()
        await application.database.close()


@pytest.mark.asyncio
async def test_composition_root_registers_music_tools_only_when_music_is_enabled(
    monkeypatch,
) -> None:
    monkeypatch.setattr(application_module, "OopzBot", FakeOopzBot)
    disabled = BotApplication(settings("agent"))
    enabled = BotApplication(
        settings(
            "agent",
            CYWL_MUSIC_ENABLED="true",
            CYWL_MUSIC_CATALOG_BASE_URL="https://music.example",
        )
    )

    assert disabled.music is None
    assert disabled.music_playlists is None
    assert "enqueue_music" not in disabled.agent_tool_registry.names
    assert enabled.music is not None
    assert enabled.music_playlists is not None
    assert enabled.music._voice._leases is enabled.voice_leases
    assert {
        "search_music_catalog",
        "enqueue_music",
        "get_music_queue",
        "skip_music",
        "pause_music",
        "resume_music",
        "set_music_playback_mode",
        "create_music_playlist",
        "list_music_playlists",
        "get_music_playlist",
        "add_music_playlist_track",
        "remove_music_playlist_track",
        "load_music_playlist",
        "preview_netease_playlist",
        "import_netease_playlist",
    }.issubset(enabled.agent_tool_registry.names)

    await enabled.music.aclose()
    for application in (disabled, enabled):
        await application.agent_engine.aclose()
        await application._provider.aclose()
        await application.database.close()


@pytest.mark.asyncio
async def test_composition_root_registers_experimental_voice_command_only_when_enabled(
    monkeypatch,
) -> None:
    monkeypatch.setattr(application_module, "OopzBot", FakeOopzBot)
    disabled = BotApplication(settings("legacy"))
    enabled = BotApplication(settings("legacy", CYWL_VOICE_ENABLED="true"))

    assert "voice" not in {command.name for command in disabled.commands.commands}
    assert "voice" in {command.name for command in enabled.commands.commands}
    assert enabled.voice_conversations._access._leases is enabled.voice_leases
    assert enabled.voice_media._bot is enabled.bot
    assert {check.name: check.state.value for check in enabled.health.snapshot()}[
        "voice"
    ] == "pending"

    for application in (disabled, enabled):
        await application.voice_conversations.aclose()
        await application.agent_engine.aclose()
        await application._provider.aclose()
        await application.database.close()


@pytest.mark.asyncio
async def test_composition_root_removes_skill_tools_when_skills_are_disabled(
    monkeypatch,
) -> None:
    monkeypatch.setattr(application_module, "OopzBot", FakeOopzBot)
    application = BotApplication(settings("agent", CYWL_AGENT_SKILLS_ENABLED="false"))

    assert {
        "load_agent_skill",
        "read_agent_skill_resource",
        "list_agent_skill_library",
        "inspect_agent_skill",
        "create_agent_skill",
        "update_agent_skill",
        "manage_agent_skill_resource",
        "set_agent_skill_state",
    }.isdisjoint(application.agent_tool_registry.names)
    assert "skills" not in {command.name for command in application.commands.commands}
    skills_health = {check.name: check for check in application.health.snapshot()}["skills"]
    assert skills_health.state.value == "disabled"

    await application.agent_engine.aclose()
    await application._provider.aclose()
    await application.database.close()


@pytest.mark.asyncio
async def test_composition_root_can_disable_only_skill_authoring(monkeypatch) -> None:
    monkeypatch.setattr(application_module, "OopzBot", FakeOopzBot)
    application = BotApplication(settings("agent", CYWL_AGENT_SKILL_AUTHORING_ENABLED="false"))

    assert {
        "load_agent_skill",
        "read_agent_skill_resource",
    }.issubset(application.agent_tool_registry.names)
    assert {
        "list_agent_skill_library",
        "inspect_agent_skill",
        "create_agent_skill",
        "update_agent_skill",
        "manage_agent_skill_resource",
        "set_agent_skill_state",
    }.isdisjoint(application.agent_tool_registry.names)

    await application.agent_engine.aclose()
    await application._provider.aclose()
    await application.database.close()


@pytest.mark.asyncio
async def test_composition_root_registers_web_search_only_when_enabled(monkeypatch) -> None:
    monkeypatch.setattr(application_module, "OopzBot", FakeOopzBot)
    enabled = BotApplication(settings("agent"))
    disabled = BotApplication(settings("agent", CYWL_WEB_SEARCH_ENABLED="false"))

    assert enabled.web_search is not None
    assert "search_web" in enabled.agent_tool_registry.names
    assert disabled.web_search is None
    assert "search_web" not in disabled.agent_tool_registry.names

    for application in (enabled, disabled):
        await application.agent_engine.aclose()
        await application._provider.aclose()
        await application.database.close()


@pytest.mark.asyncio
async def test_composition_root_registers_browser_read_tools_only_when_enabled(
    monkeypatch,
) -> None:
    monkeypatch.setattr(application_module, "OopzBot", FakeOopzBot)
    disabled = BotApplication(settings("agent"))
    enabled = BotApplication(
        settings(
            "agent",
            CYWL_WEB_BROWSER_ENABLED="true",
        )
    )
    interactions = BotApplication(
        settings(
            "agent",
            CYWL_WEB_BROWSER_ENABLED="true",
            CYWL_WEB_BROWSER_INTERACTION_ENABLED="true",
        )
    )

    browser_tools = {
        "read_web_page",
        "browser_open",
        "browser_snapshot",
        "browser_wait",
        "browser_close",
    }
    assert disabled.browser is None
    assert browser_tools.isdisjoint(disabled.agent_tool_registry.names)
    assert enabled.browser is not None
    assert browser_tools.issubset(enabled.agent_tool_registry.names)
    assert {
        "browser_click",
        "browser_fill",
        "browser_press",
    }.isdisjoint(enabled.agent_tool_registry.names)
    assert {
        "browser_click",
        "browser_fill",
        "browser_press",
    }.issubset(interactions.agent_tool_registry.names)

    await enabled.browser.aclose()
    assert interactions.browser is not None
    await interactions.browser.aclose()
    for application in (disabled, enabled, interactions):
        await application.agent_engine.aclose()
        await application._provider.aclose()
        await application.database.close()


@pytest.mark.asyncio
async def test_browser_startup_failure_degrades_health_without_stopping_bot(
    monkeypatch,
) -> None:
    monkeypatch.setattr(application_module, "OopzBot", FakeOopzBot)
    application = BotApplication(settings("legacy", CYWL_WEB_BROWSER_ENABLED="true"))

    async def no_op() -> None:
        return None

    async def unavailable() -> None:
        raise BrowserUnavailableError

    monkeypatch.setattr(application.database, "start", no_op)
    monkeypatch.setattr(application.database, "close", no_op)
    monkeypatch.setattr(application.voice_sessions, "recover_stale", no_stale_voice_sessions)
    assert application.browser is not None
    monkeypatch.setattr(application.browser, "start", unavailable)

    await application.run()

    browser_health = {check.name: check for check in application.health.snapshot()}["browser"]
    assert application.bot.did_run is True
    assert browser_health.state.value == "degraded"
    assert browser_health.detail == "MCP initialization failed"


@pytest.mark.asyncio
async def test_application_recovers_stale_voice_sessions_before_oopz_even_when_disabled(
    monkeypatch,
) -> None:
    monkeypatch.setattr(application_module, "OopzBot", FakeOopzBot)
    application = BotApplication(settings("legacy"))
    events: list[str] = []

    async def no_op() -> None:
        return None

    async def recover_stale(now) -> int:
        assert now.tzinfo is not None
        events.append("recover_voice")
        return 2

    async def run_bot() -> None:
        events.append("run_oopz")
        application.bot.did_run = True

    monkeypatch.setattr(application.database, "start", no_op)
    monkeypatch.setattr(application.database, "close", no_op)
    monkeypatch.setattr(application.voice_sessions, "recover_stale", recover_stale)
    monkeypatch.setattr(application.bot, "run", run_bot)

    await application.run()

    assert events == ["recover_voice", "run_oopz"]
    assert application.bot.did_run is True


@pytest.mark.asyncio
async def test_voice_sdk_contract_is_validated_before_database_or_oopz_start(
    monkeypatch,
) -> None:
    monkeypatch.setattr(application_module, "OopzBot", IncompleteVoiceContractBot)
    application = BotApplication(settings("legacy", CYWL_VOICE_ENABLED="true"))
    database_started = False

    async def start_database() -> None:
        nonlocal database_started
        database_started = True

    async def no_op() -> None:
        return None

    monkeypatch.setattr(application.database, "start", start_database)
    monkeypatch.setattr(application.database, "close", no_op)

    with pytest.raises(ConfigurationError, match="newer OOPZ SDK voice contract"):
        await application.run()

    assert database_started is False
    assert not hasattr(application.bot, "did_run")


@pytest.mark.asyncio
async def test_agent_mode_fails_before_oopz_when_catalog_has_no_application_default(
    monkeypatch,
) -> None:
    monkeypatch.setattr(application_module, "OopzBot", FakeOopzBot)
    application = BotApplication(settings("agent"))

    async def no_op() -> None:
        return None

    monkeypatch.setattr(application.database, "start", no_op)
    monkeypatch.setattr(application.database, "close", no_op)
    monkeypatch.setattr(application.agent_models, "reload", no_op)
    monkeypatch.setattr(application.voice_sessions, "recover_stale", no_stale_voice_sessions)

    with pytest.raises(ConfigurationError, match="application-default"):
        await application.run()

    assert not hasattr(application.bot, "did_run")


@pytest.mark.asyncio
async def test_agent_mode_rejects_disabled_application_default(monkeypatch) -> None:
    monkeypatch.setattr(application_module, "OopzBot", FakeOopzBot)
    application = BotApplication(settings("agent"))
    application.agent_catalog._catalog = ProviderCatalog.build(
        (
            LlmProvider(
                id=PROVIDER_ID,
                alias="disabled",
                display_name="Disabled",
                protocol=ProviderProtocol.OPENAI_CHAT_COMPATIBLE,
                base_url="https://llm.example/v1",
                api_key="",
                user_selectable=True,
                enabled=False,
            ),
        ),
        (
            LlmModel(
                id=MODEL_ID,
                provider_id=PROVIDER_ID,
                alias="model",
                remote_model_name="model",
                display_name="Model",
                enabled=True,
                is_provider_default=True,
                is_application_default=True,
            ),
        ),
    )

    async def no_op() -> None:
        return None

    monkeypatch.setattr(application.database, "start", no_op)
    monkeypatch.setattr(application.database, "close", no_op)
    monkeypatch.setattr(application.agent_models, "reload", no_op)
    monkeypatch.setattr(application.voice_sessions, "recover_stale", no_stale_voice_sessions)

    with pytest.raises(ConfigurationError, match="enabled application-default"):
        await application.run()

    assert not hasattr(application.bot, "did_run")


@pytest.mark.asyncio
async def test_agent_tools_require_a_tool_calling_application_default(monkeypatch) -> None:
    monkeypatch.setattr(application_module, "OopzBot", FakeOopzBot)
    application = BotApplication(settings("agent"))
    application.agent_catalog._catalog = ProviderCatalog.build(
        (
            LlmProvider(
                id=PROVIDER_ID,
                alias="text-only",
                display_name="Text only",
                protocol=ProviderProtocol.OPENAI_CHAT_COMPATIBLE,
                base_url="https://llm.example/v1",
                api_key="database-key",
                user_selectable=True,
                enabled=True,
            ),
        ),
        (
            LlmModel(
                id=MODEL_ID,
                provider_id=PROVIDER_ID,
                alias="model",
                remote_model_name="model",
                display_name="Model",
                enabled=True,
                is_provider_default=True,
                is_application_default=True,
            ),
        ),
    )

    async def no_op() -> None:
        return None

    monkeypatch.setattr(application.database, "start", no_op)
    monkeypatch.setattr(application.database, "close", no_op)
    monkeypatch.setattr(application.agent_models, "reload", no_op)
    monkeypatch.setattr(application.voice_sessions, "recover_stale", no_stale_voice_sessions)

    with pytest.raises(ConfigurationError, match="application-default"):
        await application.run()

    assert not hasattr(application.bot, "did_run")
