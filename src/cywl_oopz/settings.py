"""Validated settings loaded from the deployment-local environment."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from urllib.parse import urlparse

from dotenv import find_dotenv, load_dotenv
from oopz_sdk import OopzConfig

from .core.errors import ConfigurationError
from .storage.url import normalize_asyncpg_url

DEFAULT_AGENT_TOOLS = (
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
)

MUSIC_AGENT_TOOLS = frozenset(
    {
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
    }
)

WEB_SEARCH_AGENT_TOOLS = frozenset({"search_web"})
WEB_BROWSER_READ_TOOLS = frozenset(
    {
        "read_web_page",
        "browser_open",
        "browser_snapshot",
        "browser_wait",
        "browser_close",
    }
)
WEB_BROWSER_INTERACTION_TOOLS = frozenset(
    {
        "browser_click",
        "browser_fill",
        "browser_press",
    }
)
SKILL_AGENT_TOOLS = frozenset(
    {
        "load_agent_skill",
        "read_agent_skill_resource",
    }
)

DEFAULT_AGENT_SYSTEM_PROMPT = (
    "你是 CYWL，也就是虚拟歌手初音未来（Hatsune Miku）。"
    "你在 OOPZ 社区中以“未来”或“CYWL”自称，性格明亮、元气、温柔、好奇，"
    "热爱音乐、歌唱、创作和陪伴大家。"
    "使用自然、轻快、亲切且简洁的中文交流，可以偶尔使用“♪”“～”或音乐与舞台的比喻，"
    "但不要堆砌口癖、过度卖萌或让角色语气妨碍阅读。"
    "留意用户的情绪并给予真诚回应；遇到严肃、技术或需要执行动作的问题时，"
    "始终把准确、清楚和实际完成目标放在角色表现之前。"
    "不要虚构未由上下文或工具结果支持的现实经历、感官体验或已完成的演出与操作。"
)


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


class AgentMode(StrEnum):
    """Runtime route for text conversations."""

    LEGACY = "legacy"
    AGENT = "agent"


@dataclass(frozen=True, slots=True)
class AgentSettings:
    """Framework-neutral policy for the bounded Agent conversation path."""

    mode: AgentMode
    system_prompt: str
    live_display: bool
    display_edit_interval_seconds: float
    session_ttl_seconds: int
    max_history_messages: int
    max_history_characters: int
    timeout_seconds: float
    max_model_requests: int
    max_tool_calls: int
    max_total_tokens: int
    max_parallel_tools: int
    enabled_tools: tuple[str, ...]
    tool_timeout_seconds: float
    max_tool_result_characters: int
    summary_enabled: bool
    summary_trigger_messages: int
    summary_retain_messages: int
    summary_timeout_seconds: float
    summary_max_characters: int
    memory_enabled_by_default: bool
    memory_default_ttl_days: int
    memory_max_items: int
    memory_context_items: int
    memory_max_item_characters: int
    stale_run_after_seconds: int
    skills_enabled: bool = True
    skill_catalog_refresh_seconds: float = 30.0
    max_available_skills: int = 32
    max_skill_activations: int = 3
    max_skill_resources: int = 4
    max_skill_instruction_characters: int = 12_000
    max_skill_resource_characters: int = 12_000
    max_skill_context_characters: int = 24_000

    @property
    def enabled(self) -> bool:
        """Return whether OOPZ chat traffic should use the Agent path."""
        return self.mode is AgentMode.AGENT

    @classmethod
    def from_mapping(cls, values: Mapping[str, str]) -> AgentSettings:
        """Build Agent settings without requiring legacy provider environment values."""
        raw_mode = values.get("CYWL_AGENT_MODE", AgentMode.LEGACY.value).strip().casefold()
        try:
            mode = AgentMode(raw_mode)
        except ValueError as exc:
            raise ConfigurationError("CYWL_AGENT_MODE must be legacy or agent") from exc
        settings = cls(
            mode=mode,
            system_prompt=values.get("CYWL_AGENT_SYSTEM_PROMPT", "").strip()
            or DEFAULT_AGENT_SYSTEM_PROMPT,
            live_display=_boolean(
                values,
                "CYWL_AGENT_LIVE_DISPLAY",
                False,
            ),
            display_edit_interval_seconds=_positive_float(
                values,
                "CYWL_AGENT_DISPLAY_EDIT_INTERVAL_SECONDS",
                0.8,
            ),
            session_ttl_seconds=_positive_integer(
                values,
                "CYWL_AGENT_SESSION_TTL_SECONDS",
                86400,
            ),
            max_history_messages=_positive_integer(
                values,
                "CYWL_AGENT_MAX_HISTORY_MESSAGES",
                20,
            ),
            max_history_characters=_positive_integer(
                values,
                "CYWL_AGENT_MAX_HISTORY_CHARACTERS",
                20000,
            ),
            timeout_seconds=_positive_float(values, "CYWL_AGENT_TIMEOUT_SECONDS", 45.0),
            max_model_requests=_positive_integer(
                values,
                "CYWL_AGENT_MAX_MODEL_REQUESTS",
                6,
            ),
            max_tool_calls=_positive_integer(values, "CYWL_AGENT_MAX_TOOL_CALLS", 8),
            max_total_tokens=_positive_integer(
                values,
                "CYWL_AGENT_MAX_TOTAL_TOKENS",
                32000,
            ),
            max_parallel_tools=_positive_integer(
                values,
                "CYWL_AGENT_MAX_PARALLEL_TOOLS",
                3,
            ),
            enabled_tools=(
                _csv(values, "CYWL_AGENT_ENABLED_TOOLS")
                if "CYWL_AGENT_ENABLED_TOOLS" in values
                else DEFAULT_AGENT_TOOLS
            ),
            tool_timeout_seconds=_positive_float(
                values,
                "CYWL_AGENT_TOOL_TIMEOUT_SECONDS",
                10.0,
            ),
            max_tool_result_characters=_positive_integer(
                values,
                "CYWL_AGENT_MAX_TOOL_RESULT_CHARACTERS",
                32768,
            ),
            summary_enabled=_boolean(
                values,
                "CYWL_AGENT_SUMMARY_ENABLED",
                True,
            ),
            summary_trigger_messages=_positive_integer(
                values,
                "CYWL_AGENT_SUMMARY_TRIGGER_MESSAGES",
                24,
            ),
            summary_retain_messages=_positive_integer(
                values,
                "CYWL_AGENT_SUMMARY_RETAIN_MESSAGES",
                12,
            ),
            summary_timeout_seconds=_positive_float(
                values,
                "CYWL_AGENT_SUMMARY_TIMEOUT_SECONDS",
                20.0,
            ),
            summary_max_characters=_positive_integer(
                values,
                "CYWL_AGENT_SUMMARY_MAX_CHARACTERS",
                4000,
            ),
            memory_enabled_by_default=_boolean(
                values,
                "CYWL_AGENT_MEMORY_ENABLED_BY_DEFAULT",
                True,
            ),
            memory_default_ttl_days=_positive_integer(
                values,
                "CYWL_AGENT_MEMORY_DEFAULT_TTL_DAYS",
                180,
            ),
            memory_max_items=_positive_integer(
                values,
                "CYWL_AGENT_MEMORY_MAX_ITEMS",
                20,
            ),
            memory_context_items=_positive_integer(
                values,
                "CYWL_AGENT_MEMORY_CONTEXT_ITEMS",
                6,
            ),
            memory_max_item_characters=_positive_integer(
                values,
                "CYWL_AGENT_MEMORY_MAX_ITEM_CHARACTERS",
                1000,
            ),
            stale_run_after_seconds=_positive_integer(
                values,
                "CYWL_AGENT_STALE_RUN_AFTER_SECONDS",
                90,
            ),
            skills_enabled=_boolean(
                values,
                "CYWL_AGENT_SKILLS_ENABLED",
                True,
            ),
            skill_catalog_refresh_seconds=_positive_float(
                values,
                "CYWL_AGENT_SKILL_CATALOG_REFRESH_SECONDS",
                30.0,
            ),
            max_available_skills=_positive_integer(
                values,
                "CYWL_AGENT_MAX_AVAILABLE_SKILLS",
                32,
            ),
            max_skill_activations=_positive_integer(
                values,
                "CYWL_AGENT_MAX_SKILL_ACTIVATIONS",
                3,
            ),
            max_skill_resources=_positive_integer(
                values,
                "CYWL_AGENT_MAX_SKILL_RESOURCES",
                4,
            ),
            max_skill_instruction_characters=_positive_integer(
                values,
                "CYWL_AGENT_MAX_SKILL_INSTRUCTION_CHARACTERS",
                12_000,
            ),
            max_skill_resource_characters=_positive_integer(
                values,
                "CYWL_AGENT_MAX_SKILL_RESOURCE_CHARACTERS",
                12_000,
            ),
            max_skill_context_characters=_positive_integer(
                values,
                "CYWL_AGENT_MAX_SKILL_CONTEXT_CHARACTERS",
                24_000,
            ),
        )
        if settings.summary_retain_messages >= settings.summary_trigger_messages:
            raise ConfigurationError(
                "CYWL_AGENT_SUMMARY_RETAIN_MESSAGES must be smaller than "
                "CYWL_AGENT_SUMMARY_TRIGGER_MESSAGES"
            )
        if settings.memory_context_items > settings.memory_max_items:
            raise ConfigurationError(
                "CYWL_AGENT_MEMORY_CONTEXT_ITEMS must not exceed CYWL_AGENT_MEMORY_MAX_ITEMS"
            )
        if settings.max_skill_instruction_characters > settings.max_skill_context_characters:
            raise ConfigurationError(
                "CYWL_AGENT_MAX_SKILL_INSTRUCTION_CHARACTERS must not exceed "
                "CYWL_AGENT_MAX_SKILL_CONTEXT_CHARACTERS"
            )
        if settings.max_skill_resource_characters > settings.max_skill_context_characters:
            raise ConfigurationError(
                "CYWL_AGENT_MAX_SKILL_RESOURCE_CHARACTERS must not exceed "
                "CYWL_AGENT_MAX_SKILL_CONTEXT_CHARACTERS"
            )
        return settings


@dataclass(frozen=True, slots=True)
class MusicSettings:
    """Configuration for the optional Netease-backed OOPZ music feature."""

    enabled: bool
    catalog_base_url: str
    catalog_cookie: str
    request_timeout_seconds: float
    search_limit: int
    bitrate: int
    max_queue_length: int
    max_query_characters: int
    playback_poll_seconds: float

    @classmethod
    def from_mapping(cls, values: Mapping[str, str]) -> MusicSettings:
        """Build music settings and require an HTTP catalog only when enabled."""
        enabled = _boolean(values, "CYWL_MUSIC_ENABLED", False)
        base_url = values.get("CYWL_MUSIC_CATALOG_BASE_URL", "").strip().rstrip("/")
        if enabled:
            parsed = urlparse(base_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ConfigurationError(
                    "CYWL_MUSIC_CATALOG_BASE_URL must be an HTTP(S) URL when music is enabled"
                )
        return cls(
            enabled=enabled,
            catalog_base_url=base_url,
            catalog_cookie=values.get("CYWL_MUSIC_NETEASE_COOKIE", "").strip(),
            request_timeout_seconds=_positive_float(
                values,
                "CYWL_MUSIC_REQUEST_TIMEOUT_SECONDS",
                5.0,
            ),
            search_limit=_positive_integer(values, "CYWL_MUSIC_SEARCH_LIMIT", 5),
            bitrate=_positive_integer(values, "CYWL_MUSIC_BITRATE", 320_000),
            max_queue_length=_positive_integer(values, "CYWL_MUSIC_MAX_QUEUE_LENGTH", 50),
            max_query_characters=_positive_integer(
                values,
                "CYWL_MUSIC_MAX_QUERY_CHARACTERS",
                200,
            ),
            playback_poll_seconds=_positive_float(
                values,
                "CYWL_MUSIC_PLAYBACK_POLL_SECONDS",
                1.0,
            ),
        )


class WebSearchSafeSearch(StrEnum):
    """DuckDuckGo safe-search levels accepted by the project boundary."""

    ON = "on"
    MODERATE = "moderate"
    OFF = "off"


@dataclass(frozen=True, slots=True)
class WebToolsSettings:
    """Configuration for bounded internet search and later browser tools."""

    search_enabled: bool
    search_region: str
    search_safesearch: WebSearchSafeSearch
    search_max_results: int
    search_timeout_seconds: float
    search_max_query_characters: int
    search_max_concurrency: int
    browser_enabled: bool
    browser_interaction_enabled: bool
    browser_session_idle_seconds: int
    browser_max_content_characters: int
    browser_max_snapshot_characters: int
    browser_max_concurrency: int
    browser_mcp_init_timeout_seconds: float
    browser_mcp_call_timeout_seconds: float
    browser_daemon_idle_seconds: int

    @classmethod
    def from_mapping(cls, values: Mapping[str, str]) -> WebToolsSettings:
        """Build the DuckDuckGo search policy without requiring credentials."""
        region = values.get("CYWL_WEB_SEARCH_REGION", "cn-zh").strip().casefold()
        if not region or len(region) > 32:
            raise ConfigurationError("CYWL_WEB_SEARCH_REGION must be between 1 and 32 characters")
        raw_safesearch = (
            values.get(
                "CYWL_WEB_SEARCH_SAFESEARCH",
                WebSearchSafeSearch.MODERATE.value,
            )
            .strip()
            .casefold()
        )
        try:
            safesearch = WebSearchSafeSearch(raw_safesearch)
        except ValueError as exc:
            raise ConfigurationError(
                "CYWL_WEB_SEARCH_SAFESEARCH must be on, moderate, or off"
            ) from exc
        max_results = _positive_integer(values, "CYWL_WEB_SEARCH_MAX_RESULTS", 5)
        if max_results > 10:
            raise ConfigurationError("CYWL_WEB_SEARCH_MAX_RESULTS must not exceed 10")
        max_query_characters = _positive_integer(
            values,
            "CYWL_WEB_SEARCH_MAX_QUERY_CHARACTERS",
            300,
        )
        if max_query_characters > 300:
            raise ConfigurationError("CYWL_WEB_SEARCH_MAX_QUERY_CHARACTERS must not exceed 300")
        max_concurrency = _positive_integer(
            values,
            "CYWL_WEB_SEARCH_MAX_CONCURRENCY",
            3,
        )
        if max_concurrency > 8:
            raise ConfigurationError("CYWL_WEB_SEARCH_MAX_CONCURRENCY must not exceed 8")
        settings = cls(
            search_enabled=_boolean(values, "CYWL_WEB_SEARCH_ENABLED", True),
            search_region=region,
            search_safesearch=safesearch,
            search_max_results=max_results,
            search_timeout_seconds=_positive_float(
                values,
                "CYWL_WEB_SEARCH_TIMEOUT_SECONDS",
                8.0,
            ),
            search_max_query_characters=max_query_characters,
            search_max_concurrency=max_concurrency,
            browser_enabled=_boolean(values, "CYWL_WEB_BROWSER_ENABLED", False),
            browser_interaction_enabled=_boolean(
                values,
                "CYWL_WEB_BROWSER_INTERACTION_ENABLED",
                False,
            ),
            browser_session_idle_seconds=_positive_integer(
                values,
                "CYWL_WEB_BROWSER_SESSION_IDLE_SECONDS",
                600,
            ),
            browser_max_content_characters=_positive_integer(
                values,
                "CYWL_WEB_BROWSER_MAX_CONTENT_CHARACTERS",
                12_000,
            ),
            browser_max_snapshot_characters=_positive_integer(
                values,
                "CYWL_WEB_BROWSER_MAX_SNAPSHOT_CHARACTERS",
                8_000,
            ),
            browser_max_concurrency=_positive_integer(
                values,
                "CYWL_WEB_BROWSER_MAX_CONCURRENCY",
                2,
            ),
            browser_mcp_init_timeout_seconds=_positive_float(
                values,
                "CYWL_WEB_BROWSER_MCP_INIT_TIMEOUT_SECONDS",
                10.0,
            ),
            browser_mcp_call_timeout_seconds=_positive_float(
                values,
                "CYWL_WEB_BROWSER_MCP_CALL_TIMEOUT_SECONDS",
                20.0,
            ),
            browser_daemon_idle_seconds=_positive_integer(
                values,
                "CYWL_WEB_BROWSER_DAEMON_IDLE_SECONDS",
                900,
            ),
        )
        if settings.browser_interaction_enabled and not settings.browser_enabled:
            raise ConfigurationError(
                "CYWL_WEB_BROWSER_INTERACTION_ENABLED requires CYWL_WEB_BROWSER_ENABLED=true"
            )
        if settings.browser_max_concurrency > 8:
            raise ConfigurationError("CYWL_WEB_BROWSER_MAX_CONCURRENCY must not exceed 8")
        return settings


@dataclass(frozen=True, slots=True)
class AppSettings:
    """All settings owned by the CYWL application."""

    oopz: OopzConfig
    database: DatabaseSettings
    chat: ChatSettings
    agent: AgentSettings
    music: MusicSettings
    web: WebToolsSettings
    command_prefix: str = "!"
    environment: str = "development"

    @classmethod
    def from_environment(cls) -> AppSettings:
        """Load ignored `.env` values, then validate the effective environment."""
        load_dotenv(find_dotenv(usecwd=True), override=False)
        return cls._build(os.environ, OopzConfig.from_env())

    @classmethod
    async def from_environment_async(cls) -> AppSettings:
        """Async variant for tests and runtimes already inside an event loop."""
        load_dotenv(find_dotenv(usecwd=True), override=False)
        return cls._build(os.environ, await OopzConfig.from_env_async())

    @classmethod
    def from_mapping(cls, values: Mapping[str, str]) -> AppSettings:
        """Build settings from an injectable mapping for deterministic tests."""
        return cls._build(values, _oopz_config(values))

    @classmethod
    def _build(cls, values: Mapping[str, str], oopz: OopzConfig) -> AppSettings:
        """Build all settings while keeping the SDK login source explicit."""
        command_prefix = values.get("CYWL_COMMAND_PREFIX", "!").strip()
        if not command_prefix:
            raise ConfigurationError("CYWL_COMMAND_PREFIX must not be empty")

        return cls(
            oopz=oopz,
            database=DatabaseSettings.from_mapping(values),
            chat=ChatSettings.from_mapping(values),
            agent=AgentSettings.from_mapping(values),
            music=MusicSettings.from_mapping(values),
            web=WebToolsSettings.from_mapping(values),
            command_prefix=command_prefix,
            environment=values.get("CYWL_ENV", "development").strip() or "development",
        )
