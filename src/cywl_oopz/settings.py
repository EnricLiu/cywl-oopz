"""Validated settings loaded from the deployment-local environment."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlparse

from dotenv import find_dotenv, load_dotenv
from oopz_sdk import OopzConfig

from .core.errors import ConfigurationError
from .storage.url import normalize_asyncpg_url


def _required(values: Mapping[str, str], name: str) -> str:
    value = values.get(name, "").strip()
    if not value:
        raise ConfigurationError(f"{name} environment variable is required")
    return value


def _boolean(values: Mapping[str, str], name: str, default: bool) -> bool:
    raw = values.get(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{name} must be a boolean")


def _positive_integer(
    values: Mapping[str, str],
    name: str,
    default: int,
    *,
    allow_zero: bool = False,
) -> int:
    raw = values.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise ConfigurationError(f"{name} must be at least {minimum}")
    return value


def _positive_float(
    values: Mapping[str, str],
    name: str,
    default: float,
    *,
    allow_zero: bool = False,
) -> float:
    raw = values.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number") from exc
    minimum = 0.0 if allow_zero else 0.000001
    if value < minimum:
        raise ConfigurationError(f"{name} must be at least {minimum}")
    return value


def _csv(values: Mapping[str, str], name: str) -> tuple[str, ...]:
    """Read a comma-separated list while preserving declaration order."""
    raw = values.get(name, "")
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _oopz_config(values: Mapping[str, str]) -> OopzConfig:
    """Build SDK credentials from the same injectable mapping as project settings."""
    options: dict[str, str] = {
        "device_id": _required(values, "OOPZ_DEVICE_ID"),
        "person_uid": _required(values, "OOPZ_PERSON_UID"),
        "jwt_token": _required(values, "OOPZ_JWT_TOKEN"),
    }
    private_key = values.get("OOPZ_PRIVATE_KEY", "")
    if private_key.strip():
        options["private_key"] = private_key
    app_version = values.get("OOPZ_APP_VERSION", "").strip()
    if app_version:
        options["app_version"] = app_version
    return OopzConfig(**options)


@dataclass(frozen=True, slots=True)
class DatabaseSettings:
    """Connection-pool settings for PostgreSQL."""

    url: str
    pool_size: int
    max_overflow: int
    pool_timeout_seconds: int
    pool_recycle_seconds: int

    @classmethod
    def from_mapping(cls, values: Mapping[str, str]) -> DatabaseSettings:
        """Build pool settings without exposing the URL in validation errors."""
        return cls(
            url=normalize_asyncpg_url(_required(values, "DATABASE_URL")),
            pool_size=_positive_integer(values, "CYWL_DB_POOL_SIZE", 5),
            max_overflow=_positive_integer(
                values,
                "CYWL_DB_MAX_OVERFLOW",
                10,
                allow_zero=True,
            ),
            pool_timeout_seconds=_positive_integer(values, "CYWL_DB_POOL_TIMEOUT_SECONDS", 30),
            pool_recycle_seconds=_positive_integer(values, "CYWL_DB_POOL_RECYCLE_SECONDS", 1800),
        )


@dataclass(frozen=True, slots=True)
class ChatSettings:
    """Provider, context, and local rate-limit policy for text chat."""

    enabled: bool
    base_url: str
    api_key: str
    model: str
    allowed_models: tuple[str, ...]
    model_selection_users: frozenset[str]
    system_prompt: str
    request_timeout_seconds: float
    stream_responses: bool
    session_ttl_seconds: int
    max_history_messages: int
    max_history_characters: int
    user_cooldown_seconds: float
    max_global_concurrency: int
    max_channel_concurrency: int
    max_user_concurrency: int

    @classmethod
    def from_mapping(cls, values: Mapping[str, str]) -> ChatSettings:
        """Build chat settings and require provider details only when enabled."""
        enabled = _boolean(values, "CYWL_CHAT_ENABLED", False)
        if not enabled:
            return cls(
                enabled=False,
                base_url="",
                api_key="",
                model="",
                allowed_models=(),
                model_selection_users=frozenset(),
                system_prompt="",
                request_timeout_seconds=30.0,
                stream_responses=False,
                session_ttl_seconds=86400,
                max_history_messages=12,
                max_history_characters=12000,
                user_cooldown_seconds=1.0,
                max_global_concurrency=8,
                max_channel_concurrency=2,
                max_user_concurrency=1,
            )

        base_url = _required(values, "CYWL_LLM_BASE_URL").rstrip("/")
        parsed_base_url = urlparse(base_url)
        if parsed_base_url.scheme not in {"http", "https"} or not parsed_base_url.netloc:
            raise ConfigurationError("CYWL_LLM_BASE_URL must be an HTTP(S) URL")
        model = _required(values, "CYWL_LLM_MODEL")
        allowed_models = _csv(values, "CYWL_LLM_ALLOWED_MODELS") or (model,)
        if model not in allowed_models:
            raise ConfigurationError("CYWL_LLM_MODEL must appear in CYWL_LLM_ALLOWED_MODELS")

        system_prompt = values.get("CYWL_LLM_SYSTEM_PROMPT", "").strip() or (
            "你是 CYWL，一个友好、简洁的 OOPZ 社区助手。"
        )

        return cls(
            enabled=True,
            base_url=base_url,
            api_key=_required(values, "CYWL_LLM_API_KEY"),
            model=model,
            allowed_models=allowed_models,
            model_selection_users=frozenset(_csv(values, "CYWL_CHAT_MODEL_SELECTION_USERS")),
            system_prompt=system_prompt,
            request_timeout_seconds=_positive_float(
                values,
                "CYWL_LLM_TIMEOUT_SECONDS",
                30.0,
            ),
            stream_responses=_boolean(values, "CYWL_LLM_STREAM", True),
            session_ttl_seconds=_positive_integer(values, "CYWL_CHAT_SESSION_TTL_SECONDS", 86400),
            max_history_messages=_positive_integer(values, "CYWL_CHAT_MAX_HISTORY_MESSAGES", 12),
            max_history_characters=_positive_integer(
                values, "CYWL_CHAT_MAX_HISTORY_CHARACTERS", 12000
            ),
            user_cooldown_seconds=_positive_float(
                values,
                "CYWL_CHAT_USER_COOLDOWN_SECONDS",
                1.0,
                allow_zero=True,
            ),
            max_global_concurrency=_positive_integer(
                values,
                "CYWL_CHAT_MAX_GLOBAL_CONCURRENCY",
                8,
            ),
            max_channel_concurrency=_positive_integer(
                values,
                "CYWL_CHAT_MAX_CHANNEL_CONCURRENCY",
                2,
            ),
            max_user_concurrency=_positive_integer(
                values,
                "CYWL_CHAT_MAX_USER_CONCURRENCY",
                1,
            ),
        )


@dataclass(frozen=True, slots=True)
class AppSettings:
    """All settings owned by the CYWL application."""

    oopz: OopzConfig
    database: DatabaseSettings
    chat: ChatSettings
    command_prefix: str = "!"
    environment: str = "development"

    @classmethod
    def from_environment(cls) -> AppSettings:
        """Load ignored `.env` values, then validate the effective environment."""
        load_dotenv(find_dotenv(usecwd=True), override=False)
        return cls.from_mapping(os.environ)

    @classmethod
    def from_mapping(cls, values: Mapping[str, str]) -> AppSettings:
        """Build settings from an injectable mapping for deterministic tests."""
        command_prefix = values.get("CYWL_COMMAND_PREFIX", "!").strip()
        if not command_prefix:
            raise ConfigurationError("CYWL_COMMAND_PREFIX must not be empty")

        return cls(
            oopz=_oopz_config(values),
            database=DatabaseSettings.from_mapping(values),
            chat=ChatSettings.from_mapping(values),
            command_prefix=command_prefix,
            environment=values.get("CYWL_ENV", "development").strip() or "development",
        )
