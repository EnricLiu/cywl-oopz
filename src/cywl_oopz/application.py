"""Composition root for the bot application."""

from __future__ import annotations

import logging
from collections.abc import Coroutine
from datetime import UTC, datetime, timedelta
from typing import Any

from oopz_sdk import OopzBot
from oopz_sdk.events.context import EventContext
from oopz_sdk.models import Message as OopzMessage

from .commands.builtin import HelpCommand, PingCommand, StatusCommand
from .commands.router import CommandRouter
from .core.errors import ConfigurationError, DatabaseError
from .core.health import HealthRegistry, HealthState
from .core.observability import exception_kind, opaque_ref
from .core.tasks import TaskSupervisor
from .features.agent.catalog import ProviderCatalogAdminService, ReloadableProviderCatalog
from .features.agent.commands import (
    AgentModelCommand,
    MemoryCommand,
    ProviderCommand,
    SkillsCommand,
    ToolCommand,
    ToolsCommand,
)
from .features.agent.context import AgentContextBuilder
from .features.agent.direct_tools import DirectToolService
from .features.agent.memory import MemoryService
from .features.agent.memory_repository import SqlAlchemyMemoryRepository
from .features.agent.models import ModelCapability
from .features.agent.pydantic_ai_engine import PydanticAiAgentEngine
from .features.agent.registry import AgentModelRegistry
from .features.agent.repository import (
    SqlAlchemyAgentMessageRepository,
    SqlAlchemyAgentRunRepository,
    SqlAlchemyAgentThreadRepository,
    SqlAlchemyModelSelectionRepository,
    SqlAlchemyProviderCatalogRepository,
    SqlAlchemyToolExecutionRepository,
)
from .features.agent.selection import ProviderSelectionService
from .features.agent.service import AgentConversationService
from .features.agent.skills.availability import SkillAvailabilityService
from .features.agent.skills.library import AgentSkillLibraryService
from .features.agent.skills.library_tools import (
    SKILL_LIBRARY_TOOL_NAMES,
    skill_library_tools,
)
from .features.agent.skills.repository import SqlAlchemyAgentSkillRepository
from .features.agent.skills.tools import LoadAgentSkillTool, ReadAgentSkillResourceTool
from .features.agent.summarization import (
    PydanticAiThreadSummarizer,
    ThreadSummaryService,
)
from .features.agent.tools.builtin import (
    GetAgentStatusTool,
    GetChannelSettingsTool,
    ReactToMessageTool,
)
from .features.agent.tools.executor import ToolExecutor
from .features.agent.tools.music import (
    EnqueueMusicTool,
    GetMusicQueueTool,
    PauseMusicTool,
    ResumeMusicTool,
    SearchMusicCatalogTool,
    SetMusicPlaybackModeTool,
    SkipMusicTool,
)
from .features.agent.tools.playlists import (
    AddMusicPlaylistTrackTool,
    CreateMusicPlaylistTool,
    GetMusicPlaylistTool,
    ImportNeteasePlaylistTool,
    ListMusicPlaylistsTool,
    LoadMusicPlaylistTool,
    PreviewNeteasePlaylistTool,
    RemoveMusicPlaylistTrackTool,
)
from .features.agent.tools.policy import ToolAvailabilityService, ToolPolicy
from .features.agent.tools.registry import ToolRegistry
from .features.agent.tools.web import (
    BrowserClickTool,
    BrowserCloseTool,
    BrowserFillTool,
    BrowserOpenTool,
    BrowserPressTool,
    BrowserSnapshotTool,
    BrowserWaitTool,
    ReadWebPageTool,
    SearchWebTool,
)
from .features.chat.commands import (
    AmbientChatHandler,
    CancelChatCommand,
    ChatCommand,
    ChatStatusCommand,
    MentionChatHandler,
    ModelCommand,
    NewConversationCommand,
)
from .features.chat.models import ConversationKey
from .features.chat.openai_compatible import OpenAICompatibleChatProvider
from .features.chat.provider import ChatProvider, DisabledChatProvider
from .features.chat.repository import SqlAlchemyConversationRepository
from .features.chat.service import ChatService
from .features.chat.tasks import ChatTaskSupervisor
from .features.music.netease import NeteaseMusicCatalog
from .features.music.playlist_repository import SqlAlchemyMusicPlaylistRepository
from .features.music.playlists import MusicPlaylistService
from .features.music.service import MusicRequestService
from .features.web.browser import BrowserSessionManager
from .features.web.errors import BrowserError
from .features.web.service import WebSearchService
from .integrations.oopz.agent_presenter import OopzAgentPresenterFactory
from .integrations.oopz.chat_invocation import OopzChatInvocationFactory
from .integrations.oopz.editable_messages import OopzEditableMessageGateway
from .integrations.oopz.message_renderer import OopzMessageRenderer
from .integrations.oopz.music import OopzMusicVoiceGateway
from .integrations.oopz.reactions import OopzReactionGateway
from .integrations.oopz.skill_sharing import OopzSkillShareNotifier
from .integrations.web.agent_browser_mcp import AgentBrowserMcpGateway
from .integrations.web.duckduckgo import DuckDuckGoSearchGateway
from .settings import (
    MUSIC_AGENT_TOOLS,
    SKILL_AGENT_TOOLS,
    SKILL_AUTHORING_AGENT_TOOLS,
    WEB_BROWSER_INTERACTION_TOOLS,
    WEB_BROWSER_READ_TOOLS,
    WEB_SEARCH_AGENT_TOOLS,
    AppSettings,
)
from .storage.channel_settings import SqlAlchemyChannelSettingsRepository
from .storage.database import Database

logger = logging.getLogger(__name__)


class BotApplication:
    """Owns OOPZ integration, application resources, and feature services."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.health = HealthRegistry()
        self.database = Database(settings.database)
        self.bot = OopzBot(settings.oopz)
        self.chat_invocations = OopzChatInvocationFactory(settings.oopz.person_uid)
        self.agent_presenters = OopzAgentPresenterFactory(
            OopzEditableMessageGateway(self.bot),
            OopzMessageRenderer(),
            enabled=settings.agent.enabled and settings.agent.live_display,
            edit_interval_seconds=settings.agent.display_edit_interval_seconds,
        )
        self.commands = CommandRouter(settings.command_prefix)
        catalog_repository = SqlAlchemyProviderCatalogRepository(self.database.session_factory)
        self.agent_catalog = ReloadableProviderCatalog(catalog_repository)
        self.agent_catalog_admin = ProviderCatalogAdminService(
            catalog_repository,
            self.agent_catalog,
        )
        self.agent_threads = SqlAlchemyAgentThreadRepository(self.database.session_factory)
        self.agent_runs = SqlAlchemyAgentRunRepository(self.database.session_factory)
        self.agent_messages = SqlAlchemyAgentMessageRepository(self.database.session_factory)
        self.agent_memory_repository = SqlAlchemyMemoryRepository(self.database.session_factory)
        self.agent_memory = MemoryService(settings.agent, self.agent_memory_repository)
        self.agent_context = AgentContextBuilder(
            settings.agent,
            self.agent_messages,
            self.agent_memory,
        )
        selection_repository = SqlAlchemyModelSelectionRepository(self.database.session_factory)
        self.agent_selection = ProviderSelectionService(
            self.agent_catalog,
            selection_repository,
        )
        channel_settings = SqlAlchemyChannelSettingsRepository(self.database.session_factory)
        self.agent_skill_repository = SqlAlchemyAgentSkillRepository(self.database.session_factory)
        self.agent_skill_notifier = OopzSkillShareNotifier(self.bot)
        agent_tools = [
            GetAgentStatusTool(
                timeout_seconds=settings.agent.tool_timeout_seconds,
                max_output_characters=settings.agent.max_tool_result_characters,
            ),
            GetChannelSettingsTool(
                channel_settings,
                timeout_seconds=settings.agent.tool_timeout_seconds,
                max_output_characters=settings.agent.max_tool_result_characters,
            ),
            ReactToMessageTool(
                OopzReactionGateway(self.bot),
                timeout_seconds=settings.agent.tool_timeout_seconds,
                max_output_characters=settings.agent.max_tool_result_characters,
            ),
        ]
        self.music: MusicRequestService | None = None
        self.music_catalog: NeteaseMusicCatalog | None = None
        self.music_playlists: MusicPlaylistService | None = None
        enabled_agent_tools = settings.agent.enabled_tools
        if settings.agent.skills_enabled:
            agent_tools.extend(
                (
                    LoadAgentSkillTool(),
                    ReadAgentSkillResourceTool(),
                )
            )
        else:
            enabled_agent_tools = tuple(
                name for name in enabled_agent_tools if name not in SKILL_AGENT_TOOLS
            )
        if settings.music.enabled:
            self.music_catalog = NeteaseMusicCatalog(settings.music)
            self.music = MusicRequestService(
                settings.music,
                self.music_catalog,
                OopzMusicVoiceGateway(self.bot),
            )
            self.music_playlists = MusicPlaylistService(
                settings.music,
                SqlAlchemyMusicPlaylistRepository(self.database.session_factory),
                self.music,
                self.music_catalog,
            )
            music_tool_options = {
                "timeout_seconds": settings.agent.tool_timeout_seconds,
                "max_output_characters": settings.agent.max_tool_result_characters,
            }
            music_import_tool_options = {
                **music_tool_options,
                "timeout_seconds": max(
                    settings.agent.tool_timeout_seconds,
                    settings.music.request_timeout_seconds * 2 + 2,
                ),
            }
            agent_tools.extend(
                (
                    SearchMusicCatalogTool(self.music, **music_tool_options),
                    EnqueueMusicTool(self.music, **music_tool_options),
                    GetMusicQueueTool(self.music, **music_tool_options),
                    SkipMusicTool(self.music, **music_tool_options),
                    PauseMusicTool(self.music, **music_tool_options),
                    ResumeMusicTool(self.music, **music_tool_options),
                    SetMusicPlaybackModeTool(self.music, **music_tool_options),
                    CreateMusicPlaylistTool(self.music_playlists, **music_tool_options),
                    ListMusicPlaylistsTool(self.music_playlists, **music_tool_options),
                    GetMusicPlaylistTool(self.music_playlists, **music_tool_options),
                    AddMusicPlaylistTrackTool(self.music_playlists, **music_tool_options),
                    RemoveMusicPlaylistTrackTool(
                        self.music_playlists,
                        **music_tool_options,
                    ),
                    LoadMusicPlaylistTool(self.music_playlists, **music_tool_options),
                    PreviewNeteasePlaylistTool(
                        self.music_playlists,
                        **music_import_tool_options,
                    ),
                    ImportNeteasePlaylistTool(
                        self.music_playlists,
                        **music_import_tool_options,
                    ),
                )
            )
        else:
            enabled_agent_tools = tuple(
                name for name in enabled_agent_tools if name not in MUSIC_AGENT_TOOLS
            )
        self.web_search: WebSearchService | None = None
        if settings.web.search_enabled:
            self.web_search = WebSearchService(
                settings.web,
                DuckDuckGoSearchGateway(
                    timeout_seconds=settings.web.search_timeout_seconds,
                    max_concurrency=settings.web.search_max_concurrency,
                ),
            )
            agent_tools.append(
                SearchWebTool(
                    self.web_search,
                    timeout_seconds=settings.agent.tool_timeout_seconds,
                    max_output_characters=settings.agent.max_tool_result_characters,
                )
            )
        else:
            enabled_agent_tools = tuple(
                name for name in enabled_agent_tools if name not in WEB_SEARCH_AGENT_TOOLS
            )
        self.browser: BrowserSessionManager | None = None
        if settings.web.browser_enabled:
            self.browser = BrowserSessionManager(
                settings.web,
                AgentBrowserMcpGateway(settings.web),
            )
            browser_tool_options = {
                "timeout_seconds": max(
                    settings.agent.tool_timeout_seconds,
                    settings.web.browser_mcp_call_timeout_seconds + 2,
                ),
                "max_output_characters": settings.agent.max_tool_result_characters,
            }
            agent_tools.extend(
                (
                    ReadWebPageTool(self.browser, **browser_tool_options),
                    BrowserOpenTool(self.browser, **browser_tool_options),
                    BrowserSnapshotTool(self.browser, **browser_tool_options),
                    BrowserWaitTool(self.browser, **browser_tool_options),
                    BrowserCloseTool(self.browser, **browser_tool_options),
                )
            )
            if settings.web.browser_interaction_enabled:
                agent_tools.extend(
                    (
                        BrowserClickTool(self.browser, **browser_tool_options),
                        BrowserFillTool(self.browser, **browser_tool_options),
                        BrowserPressTool(self.browser, **browser_tool_options),
                    )
                )
        else:
            enabled_agent_tools = tuple(
                name
                for name in enabled_agent_tools
                if name not in WEB_BROWSER_READ_TOOLS and name not in WEB_BROWSER_INTERACTION_TOOLS
            )
        if not settings.web.browser_interaction_enabled:
            enabled_agent_tools = tuple(
                name for name in enabled_agent_tools if name not in WEB_BROWSER_INTERACTION_TOOLS
            )
        self.agent_skill_library: AgentSkillLibraryService | None = None
        if settings.agent.skills_enabled and settings.agent.skill_authoring_enabled:
            registered_skill_tools = (
                frozenset(tool.descriptor.name for tool in agent_tools) | SKILL_LIBRARY_TOOL_NAMES
            )
            self.agent_skill_library = AgentSkillLibraryService(
                self.agent_skill_repository,
                registered_tools=registered_skill_tools,
                max_personal_skills=settings.agent.max_personal_skills,
                max_available_skills=settings.agent.max_available_skills,
                max_resources_per_skill=settings.agent.max_resources_per_skill,
                max_instruction_characters=(settings.agent.max_skill_instruction_characters),
                max_resource_characters=settings.agent.max_skill_resource_characters,
                max_accepted_shared_skills=settings.agent.max_accepted_shared_skills,
                max_share_recipients_per_call=(settings.agent.max_skill_share_recipients_per_call),
                notifier=self.agent_skill_notifier,
            )
            agent_tools.extend(skill_library_tools(self.agent_skill_library))
        else:
            enabled_agent_tools = tuple(
                name for name in enabled_agent_tools if name not in SKILL_AUTHORING_AGENT_TOOLS
            )
        self.agent_tool_registry = ToolRegistry(agent_tools)
        self.agent_skill_availability = SkillAvailabilityService(
            max_available_skills=settings.agent.max_available_skills,
        )
        logger.info(
            "Application configured: agent=%s tools=%s music=%s web_search=%s browser=%s",
            settings.agent.enabled,
            len(self.agent_tool_registry.names),
            self.music is not None,
            self.web_search is not None,
            self.browser is not None,
        )
        self.agent_tool_availability = ToolAvailabilityService(
            self.agent_tool_registry,
            channel_settings,
            enabled_agent_tools,
        )
        self.agent_tool_executor = ToolExecutor(
            self.agent_tool_registry,
            ToolPolicy(),
            SqlAlchemyToolExecutionRepository(self.database.session_factory),
        )
        self.direct_tools = DirectToolService(
            settings.agent,
            self.agent_tool_registry,
            self.agent_tool_availability,
            self.agent_selection,
        )
        self.agent_models = AgentModelRegistry(
            self.agent_catalog,
            default_max_retries=settings.agent.provider_max_retries,
        )
        self.agent_summary_tasks = TaskSupervisor(lambda thread_id: f"agent-summary:{thread_id}")
        self.agent_summary_service = ThreadSummaryService(
            settings.agent,
            PydanticAiThreadSummarizer(self.agent_models, settings.agent),
            self.agent_threads,
            self.agent_messages,
        )
        self.agent_engine = PydanticAiAgentEngine(
            self.agent_models,
            self.agent_tool_executor,
        )
        self.agent_chat = AgentConversationService(
            settings.agent,
            settings.chat,
            self.agent_engine,
            self.agent_catalog,
            self.agent_selection,
            selection_repository,
            self.agent_threads,
            self.agent_runs,
            self.agent_messages,
            self.agent_tool_availability,
            self.agent_skill_repository if settings.agent.skills_enabled else None,
            self.agent_skill_availability if settings.agent.skills_enabled else None,
            context_builder=self.agent_context,
            summary_service=self.agent_summary_service,
            summary_tasks=self.agent_summary_tasks,
            health=self.health,
        )
        self._provider = self._create_chat_provider()
        self.chat_tasks = ChatTaskSupervisor()
        self.legacy_chat = ChatService(
            settings.chat,
            self._provider,
            SqlAlchemyConversationRepository(self.database.session_factory),
            health=self.health,
        )
        self.chat = self.agent_chat if settings.agent.enabled else self.legacy_chat
        self._mention_handler = MentionChatHandler(
            self.chat,
            settings.oopz.person_uid,
            self.agent_presenters,
            self.chat_invocations,
        )
        self._ambient_handler = AmbientChatHandler(
            self.chat,
            channel_settings,
            self.agent_presenters,
            self.chat_invocations,
        )
        self._register_commands()
        self.bot.on_ready(self._on_ready)
        self.bot.on_message(self._on_message)
        self.health.mark("database", HealthState.PENDING)
        self.health.mark(
            "llm",
            HealthState.PENDING
            if settings.chat.enabled or settings.agent.enabled
            else HealthState.DISABLED,
        )
        self.health.mark("oopz", HealthState.PENDING)
        self.health.mark(
            "browser",
            HealthState.PENDING if settings.web.browser_enabled else HealthState.DISABLED,
        )
        self.health.mark(
            "skills",
            HealthState.PENDING
            if settings.agent.enabled and settings.agent.skills_enabled
            else HealthState.DISABLED,
        )

    def _create_chat_provider(self) -> ChatProvider:
        if not self.settings.chat.enabled:
            return DisabledChatProvider()
        return OpenAICompatibleChatProvider(self.settings.chat)

    def _register_commands(self) -> None:
        self.commands.register(PingCommand())
        self.commands.register(HelpCommand(self.commands))
        self.commands.register(StatusCommand(self.health))
        self.commands.register(
            ChatCommand(
                self.chat,
                self.agent_presenters,
                self.chat_invocations,
            )
        )
        self.commands.register(NewConversationCommand(self.chat, self.chat_tasks))
        self.commands.register(
            CancelChatCommand(
                self.chat,
                self.chat_tasks,
                active_message_reports_cancel=(
                    self.settings.agent.enabled and self.settings.agent.live_display
                ),
            )
        )
        if self.settings.agent.enabled:
            self.commands.register(
                AgentModelCommand(
                    self.agent_chat,
                    self.chat_tasks,
                    self.settings.command_prefix,
                )
            )
        else:
            self.commands.register(ModelCommand(self.legacy_chat, self.chat_tasks))
        self.commands.register(ChatStatusCommand(self.chat))
        if self.settings.agent.enabled:
            self.commands.register(
                ProviderCommand(
                    self.agent_chat,
                    self.chat_tasks,
                    self.settings.command_prefix,
                )
            )
            self.commands.register(ToolsCommand(self.agent_chat))
            self.commands.register(
                ToolCommand(
                    self.direct_tools,
                    self.settings.command_prefix,
                    self.chat_invocations,
                )
            )
            self.commands.register(MemoryCommand(self.agent_chat, self.agent_memory))
            if self.settings.agent.skills_enabled:
                self.commands.register(SkillsCommand(self.agent_chat, self.agent_skill_library))

    async def run(self) -> None:
        """Start the database check before entering the long-running OOPZ client."""
        logger.info("Application startup started")
        try:
            try:
                await self.database.start()
            except DatabaseError:
                self.health.mark("database", HealthState.DEGRADED, "connection check failed")
                raise
            self.health.mark("database", HealthState.HEALTHY, "connection check passed")
            logger.info("Database health check passed")
            if self.browser is not None:
                try:
                    await self.browser.start()
                except BrowserError as exc:
                    logger.warning(
                        "Agent-browser MCP initialization failed: error=%s",
                        exception_kind(exc),
                    )
                    self.health.mark(
                        "browser",
                        HealthState.DEGRADED,
                        "MCP initialization failed",
                    )
                else:
                    logger.info("Agent-browser MCP contract validated")
                    self.health.mark(
                        "browser",
                        HealthState.HEALTHY,
                        "MCP contract validated",
                    )
            if self.settings.agent.enabled:
                await self.agent_models.reload()
                catalog = self.agent_catalog.snapshot
                logger.info(
                    "Agent provider catalog loaded: providers=%s models=%s",
                    len(catalog.providers),
                    len(catalog.models),
                )
                default_model_id = catalog.application_default_model_id()
                default_model = (
                    catalog.resolve(
                        default_model_id,
                        required_capabilities=(
                            frozenset({ModelCapability.TOOL_CALLING})
                            if self.settings.agent.enabled_tools
                            else frozenset()
                        ),
                        require_user_selectable=False,
                    )
                    if default_model_id is not None
                    else None
                )
                if default_model is None:
                    raise ConfigurationError(
                        "Agent mode requires an enabled application-default LLM model"
                    )
                now = datetime.now(UTC)
                abandoned = await self.agent_runs.abandon_stale(
                    now - timedelta(seconds=self.settings.agent.stale_run_after_seconds),
                    now,
                )
                if abandoned:
                    logger.warning("Marked stale Agent runs abandoned: count=%s", abandoned)
            logger.info("Starting OOPZ client")
            await self.bot.run()
        finally:
            logger.info("Application shutdown started")
            await self.chat_tasks.close()
            await self.agent_summary_tasks.close()
            if self.music is not None:
                await self.music.aclose()
            if self.browser is not None:
                await self.browser.aclose()
            if self.web_search is not None:
                await self.web_search.aclose()
            await self.agent_engine.aclose()
            await self._provider.aclose()
            await self.database.close()
            logger.info("Application shutdown completed")

    async def _on_ready(self, _: EventContext) -> None:
        self.health.mark("oopz", HealthState.HEALTHY, "websocket connected")
        logger.info("Bot connected; command prefix is %r", self.settings.command_prefix)

    async def _on_message(self, message: OopzMessage, context: EventContext) -> None:
        """Route short commands inline and own slow LLM work in supervised tasks."""
        logger.debug(
            "Received OOPZ message: scope=%s conversation=%s has_text=%s",
            "private" if getattr(context.event, "is_private", False) else "channel",
            self._message_reference(message, context),
            bool(message.plain_text or message.text or message.content),
        )
        command = self.commands.parse(message.plain_text or message.text or message.content)
        if command is not None:
            logger.info(
                "Dispatching command: name=%s conversation=%s",
                command.name,
                self._message_reference(message, context),
            )
            if command.name == "chat":
                await self._start_chat_task(context, self.commands.dispatch(message, context))
                return
            await self.commands.dispatch(message, context)
            return
        if self._mention_handler.matches(message):
            logger.info(
                "Dispatching mention chat: conversation=%s",
                self._message_reference(message, context),
            )
            await self._start_chat_task(context, self._mention_handler.handle(message, context))
            return
        try:
            ambient_enabled = await self._ambient_handler.matches(message, context)
        except DatabaseError as exc:
            logger.warning(
                "Failed to evaluate ambient chat policy: conversation=%s error=%s",
                self._message_reference(message, context),
                exception_kind(exc),
            )
            return
        if ambient_enabled:
            logger.info(
                "Dispatching ambient chat: conversation=%s",
                self._message_reference(message, context),
            )
            await self._start_chat_task(context, self._ambient_handler.handle(message, context))

    async def _start_chat_task(
        self,
        context: EventContext,
        operation: Coroutine[Any, Any, object],
    ) -> None:
        """Register slow work so the SDK receive loop can immediately process later events."""
        try:
            key = ConversationKey.from_oopz_context(context)
        except ValueError as exc:
            operation.close()
            logger.warning("Could not start chat task: error=%s", exception_kind(exc))
            await context.reply("无法识别当前对话的位置，请稍后重试。")
            return
        if not self.chat_tasks.start(key, operation):
            logger.info(
                "Rejected duplicate chat task: conversation=%s",
                opaque_ref(key.scope, key.area_id, key.channel_id, key.person_id),
            )
            await context.reply("当前对话正在生成回复；可使用 !cancel 取消后再试。")

    @staticmethod
    def _message_reference(message: OopzMessage, context: EventContext) -> str:
        """Build a stable correlation token without logging OOPZ identifiers or content."""
        event = context.event
        return opaque_ref(
            "private" if getattr(event, "is_private", False) else "channel",
            getattr(message, "area", ""),
            getattr(message, "channel", ""),
            getattr(message, "sender_id", ""),
        )
