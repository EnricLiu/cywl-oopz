from __future__ import annotations

import pytest

from cywl_oopz.core.errors import ConfigurationError
from cywl_oopz.settings import AgentMode, AppSettings, ChatSettings, DatabaseSettings


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
    )


def test_agent_mode_uses_database_catalog_without_legacy_llm_credentials() -> None:
    settings = AppSettings.from_mapping(valid_environment() | {"CYWL_AGENT_MODE": "agent"})

    assert settings.agent.enabled is True
    assert settings.chat.enabled is False


def test_agent_tools_can_be_disabled_independently() -> None:
    settings = AppSettings.from_mapping(
        valid_environment()
        | {
            "CYWL_AGENT_MODE": "agent",
            "CYWL_AGENT_ENABLED_TOOLS": "",
        }
    )

    assert settings.agent.enabled_tools == ()


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
