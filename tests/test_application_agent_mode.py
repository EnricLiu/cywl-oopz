from __future__ import annotations

from uuid import UUID

import pytest

import cywl_oopz.application as application_module
from cywl_oopz.application import BotApplication
from cywl_oopz.core.errors import ConfigurationError
from cywl_oopz.features.agent.catalog import ProviderCatalog
from cywl_oopz.features.agent.models import LlmModel, LlmProvider, ProviderProtocol
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


@pytest.mark.asyncio
async def test_composition_root_routes_chat_and_provider_command_by_agent_flag(
    monkeypatch,
) -> None:
    monkeypatch.setattr(application_module, "OopzBot", FakeOopzBot)
    agent_application = BotApplication(settings("agent"))
    legacy_application = BotApplication(settings("legacy"))

    assert agent_application.chat is agent_application.agent_chat
    assert "provider" in {command.name for command in agent_application.commands.commands}
    assert "tools" in {command.name for command in agent_application.commands.commands}
    assert "memory" in {command.name for command in agent_application.commands.commands}
    assert legacy_application.chat is legacy_application.legacy_chat
    assert "provider" not in {command.name for command in legacy_application.commands.commands}
    assert "tools" not in {command.name for command in legacy_application.commands.commands}
    assert "memory" not in {command.name for command in legacy_application.commands.commands}

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
    assert "enqueue_music" not in disabled.agent_tool_registry.names
    assert enabled.music is not None
    assert {
        "search_music_catalog",
        "enqueue_music",
        "get_music_queue",
        "skip_music",
        "pause_music",
        "resume_music",
    }.issubset(enabled.agent_tool_registry.names)

    await enabled.music.aclose()
    for application in (disabled, enabled):
        await application.agent_engine.aclose()
        await application._provider.aclose()
        await application.database.close()


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

    with pytest.raises(ConfigurationError, match="application-default"):
        await application.run()

    assert not hasattr(application.bot, "did_run")
