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
from .core.tasks import TaskSupervisor
from .features.agent.catalog import ProviderCatalogAdminService, ReloadableProviderCatalog
from .features.agent.commands import MemoryCommand, ProviderCommand, ToolsCommand
from .features.agent.context import AgentContextBuilder
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
    SkipMusicTool,
)
from .features.agent.tools.policy import ToolAvailabilityService, ToolPolicy
from .features.agent.tools.registry import ToolRegistry
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
from .features.music.service import MusicRequestService
from .integrations.oopz.agent_presenter import OopzAgentPresenterFactory
from .integrations.oopz.editable_messages import OopzEditableMessageGateway
from .integrations.oopz.message_renderer import OopzMessageRenderer
from .integrations.oopz.music import OopzMusicVoiceGateway
from .integrations.oopz.reactions import OopzReactionGateway
from .settings import MUSIC_AGENT_TOOLS, AppSettings
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
        enabled_agent_tools = settings.agent.enabled_tools
        if settings.music.enabled:
            self.music = MusicRequestService(
                settings.music,
                NeteaseMusicCatalog(settings.music),
                OopzMusicVoiceGateway(self.bot),
            )
            music_tool_options = {
                "timeout_seconds": settings.agent.tool_timeout_seconds,
                "max_output_characters": settings.agent.max_tool_result_characters,
            }
            agent_tools.extend(
                (
                    SearchMusicCatalogTool(self.music, **music_tool_options),
                    EnqueueMusicTool(self.music, **music_tool_options),
                    GetMusicQueueTool(self.music, **music_tool_options),
                    SkipMusicTool(self.music, **music_tool_options),
                    PauseMusicTool(self.music, **music_tool_options),
                    ResumeMusicTool(self.music, **music_tool_options),
                )
            )
        else:
            enabled_agent_tools = tuple(
                name for name in enabled_agent_tools if name not in MUSIC_AGENT_TOOLS
            )
        self.agent_tool_registry = ToolRegistry(agent_tools)
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
        self.agent_models = AgentModelRegistry(self.agent_catalog)
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
        )
        self._ambient_handler = AmbientChatHandler(
            self.chat,
            channel_settings,
            self.agent_presenters,
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

    def _create_chat_provider(self) -> ChatProvider:
        if not self.settings.chat.enabled:
            return DisabledChatProvider()
        return OpenAICompatibleChatProvider(self.settings.chat)

    def _register_commands(self) -> None:
        self.commands.register(PingCommand())
        self.commands.register(HelpCommand(self.commands))
        self.commands.register(StatusCommand(self.health))
        self.commands.register(ChatCommand(self.chat, self.agent_presenters))
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
        self.commands.register(ModelCommand(self.chat, self.chat_tasks))
        self.commands.register(ChatStatusCommand(self.chat))
        if self.settings.agent.enabled:
            self.commands.register(ProviderCommand(self.agent_chat, self.chat_tasks))
            self.commands.register(ToolsCommand(self.agent_chat))
            self.commands.register(MemoryCommand(self.agent_chat, self.agent_memory))

    async def run(self) -> None:
        """Start the database check before entering the long-running OOPZ client."""
        try:
            try:
                await self.database.start()
            except DatabaseError:
                self.health.mark("database", HealthState.DEGRADED, "connection check failed")
                raise
            self.health.mark("database", HealthState.HEALTHY, "connection check passed")
            if self.settings.agent.enabled:
                await self.agent_models.reload()
                catalog = self.agent_catalog.snapshot
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
                await self.agent_runs.abandon_stale(
                    now - timedelta(seconds=self.settings.agent.stale_run_after_seconds),
                    now,
                )
            await self.bot.run()
        finally:
            await self.chat_tasks.close()
            await self.agent_summary_tasks.close()
            if self.music is not None:
                await self.music.aclose()
            await self.agent_engine.aclose()
            await self._provider.aclose()
            await self.database.close()

    async def _on_ready(self, _: EventContext) -> None:
        self.health.mark("oopz", HealthState.HEALTHY, "websocket connected")
        logger.info("Bot connected; command prefix is %r", self.settings.command_prefix)

    async def _on_message(self, message: OopzMessage, context: EventContext) -> None:
        """Route short commands inline and own slow LLM work in supervised tasks."""
        logger.debug(f"[OnMessage] Bot: new message: {message}")
        command = self.commands.parse(message.plain_text or message.text or message.content)
        if command is not None:
            if command.name == "chat":
                await self._start_chat_task(context, self.commands.dispatch(message, context))
                return
            await self.commands.dispatch(message, context)
            return
        if self._mention_handler.matches(message):
            await self._start_chat_task(context, self._mention_handler.handle(message, context))
            return
        try:
            ambient_enabled = await self._ambient_handler.matches(message, context)
        except DatabaseError:
            logger.exception("Failed to evaluate ambient chat policy")
            return
        if ambient_enabled:
            await self._start_chat_task(context, self._ambient_handler.handle(message, context))

    async def _start_chat_task(
        self,
        context: EventContext,
        operation: Coroutine[Any, Any, object],
    ) -> None:
        """Register slow work so the SDK receive loop can immediately process later events."""
        try:
            key = ConversationKey.from_oopz_context(context)
        except ValueError:
            operation.close()
            await context.reply("无法识别当前对话的位置，请稍后重试。")
            return
        if not self.chat_tasks.start(key, operation):
            await context.reply("当前对话正在生成回复；可使用 !cancel 取消后再试。")
