"""Composition root for the bot application."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from contextlib import AsyncExitStack
from datetime import UTC, datetime, timedelta
from typing import Any

from oopz_sdk import OopzBot
from oopz_sdk.events.context import EventContext
from oopz_sdk.models import Message as OopzMessage

from .commands.builtin import HelpCommand, PingCommand, StatusCommand
from .commands.execution import CommandTaskSupervisor
from .commands.parsing import CommandTextParser
from .commands.router import CommandRouter
from .composition.agent import build_agent
from .composition.music import build_music
from .composition.voice import build_voice
from .composition.web import build_web
from .core.errors import ConfigurationError, DatabaseError
from .core.health import HealthRegistry, HealthState
from .core.observability import exception_kind, opaque_ref
from .features.access.administration import RoleAdministrationService
from .features.access.commands import RoleCommand, WhoAmICommand
from .features.access.repository import SqlAlchemyRoleBindingRepository
from .features.access.service import AuthorizationService
from .features.admin.actions import DebugMessageAction, RecallMessageAction
from .features.admin.commands import (
    DebugCommand,
    InitCommand,
    RebootCommand,
    RecallCommand,
)
from .features.admin.initialization import ChannelInitializationService
from .features.admin.lifecycle import ApplicationLifecycleCoordinator
from .features.admin.models import ShutdownDisposition
from .features.admin.outbound_repository import (
    SqlAlchemyAgentDiagnosticRepository,
    SqlAlchemyOutboundMessageRepository,
)
from .features.admin.reaction_commands import (
    DebugReactionCommand,
    ReactionCommandRouter,
    RecallReactionCommand,
)
from .features.admin.recall import MessageRecallService
from .features.admin.references import ReferencedMessageResolver
from .features.admin.repository import SqlAlchemyChannelInitializationRepository
from .features.agent.commands import (
    AgentModelCommand,
    MemoryCommand,
    ProviderCommand,
    SkillsCommand,
    ToolCommand,
    ToolsCommand,
)
from .features.agent.models import ModelCapability
from .features.chat.commands import (
    CancelChatCommand,
    ChatCommand,
    ChatStatusCommand,
    ModelCommand,
    NewConversationCommand,
)
from .features.chat.openai_compatible import OpenAICompatibleChatProvider
from .features.chat.provider import ChatProvider, DisabledChatProvider
from .features.chat.repository import SqlAlchemyConversationRepository
from .features.chat.service import ChatService
from .features.chat.tasks import ChatTaskSupervisor, OutboundChatTaskCanceller
from .features.music.commands import MusicCommand
from .features.music.errors import MusicSourceUnavailableError
from .features.music.models import MusicSourceKind
from .features.voice.commands import VoiceCommand
from .features.web.errors import BrowserError
from .integrations.media.ytdlp_runner import YtDlpCapabilityProbe
from .integrations.oopz.active_presentations import ActivePresentationRegistry
from .integrations.oopz.agent_presenter import OopzAgentPresenterFactory
from .integrations.oopz.channel_catalog import OopzAreaChannelCatalog
from .integrations.oopz.chat_handlers import AmbientChatHandler, MentionChatHandler
from .integrations.oopz.chat_invocation import (
    OopzChatInvocationFactory,
    conversation_key_from_context,
)
from .integrations.oopz.command_requests import OopzCommandRequestFactory
from .integrations.oopz.editable_messages import OopzEditableMessageGateway
from .integrations.oopz.master_audio import OopzMasterPcmOutputFactory
from .integrations.oopz.message_recall import (
    OopzBotMessageRecallGateway,
    OopzRecentBotMessageLookup,
    OopzReferencedMessageParser,
)
from .integrations.oopz.message_renderer import OopzMessageRenderer
from .integrations.oopz.reaction_commands import (
    OopzReactionCommandInvocationParser,
    OopzReactionCommandResponder,
)
from .integrations.oopz.tracked_context import TrackedMessageContext
from .integrations.oopz.voice_capabilities import OopzVoiceCapabilityGate
from .integrations.oopz.voice_channel_session import OopzVoiceChannelSessionManager
from .integrations.oopz.voice_conversation import (
    OopzVoiceCommandPresenter,
)
from .integrations.oopz.voice_media import OopzVoiceMediaGateway
from .settings import (
    MUSIC_AGENT_TOOLS,
    WEB_BROWSER_INTERACTION_TOOLS,
    WEB_BROWSER_READ_TOOLS,
    WEB_SEARCH_AGENT_TOOLS,
    AppSettings,
)
from .storage.database import Database

logger = logging.getLogger(__name__)


class BotApplication:
    """Owns OOPZ integration, application resources, and feature services."""

    oopz_stop_timeout_seconds = 5.0
    oopz_run_settle_seconds = 1.0

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.health = HealthRegistry()
        self.lifecycle = ApplicationLifecycleCoordinator()
        self.database = Database(settings.database)
        self.role_bindings = SqlAlchemyRoleBindingRepository(self.database.session_factory)
        self.authorization = AuthorizationService(
            self.role_bindings,
            settings.rbac.bootstrap_owner_ids,
        )
        self.role_administration = RoleAdministrationService(
            self.role_bindings,
            self.authorization,
        )
        self.bot = OopzBot(settings.oopz)
        self.outbound_messages = SqlAlchemyOutboundMessageRepository(self.database.session_factory)
        self.agent_diagnostics = SqlAlchemyAgentDiagnosticRepository(self.database.session_factory)
        self.channel_initialization_repository = SqlAlchemyChannelInitializationRepository(
            self.database.session_factory
        )
        self.area_channel_catalog = OopzAreaChannelCatalog(self.bot)
        self.channel_initialization = ChannelInitializationService(
            self.area_channel_catalog,
            self.channel_initialization_repository,
        )
        self.voice_capability_gate = OopzVoiceCapabilityGate()
        self.master_audio = OopzMasterPcmOutputFactory.from_settings(self.bot, settings.audio)
        self.voice_channel_sessions = OopzVoiceChannelSessionManager(
            self.bot,
            allow_mixed_participants=settings.audio.enabled,
            master_factory=self.master_audio,
            master_target_buffer_ms=settings.audio.master_target_buffer_ms,
            music_queue_ms=settings.audio.music_queue_ms,
            voice_queue_ms=settings.audio.voice_queue_ms,
            mixer_levels=settings.audio.mixer_levels(),
        )
        self.voice_media = OopzVoiceMediaGateway(
            self.bot,
            settings.voice,
            settings.audio,
            master_factory=self.master_audio,
        )
        self.chat_invocations = OopzChatInvocationFactory(settings.oopz.person_uid)
        self.active_agent_presentations = ActivePresentationRegistry()
        self.editable_messages = OopzEditableMessageGateway(
            self.bot,
            self.outbound_messages,
        )
        self.agent_presenters = OopzAgentPresenterFactory(
            self.editable_messages,
            OopzMessageRenderer(),
            enabled=settings.agent.enabled and settings.agent.live_display,
            edit_interval_seconds=settings.agent.display_edit_interval_seconds,
            active_presentations=self.active_agent_presentations,
        )
        self.command_parser = CommandTextParser(settings.command_prefix)
        self.command_tasks = CommandTaskSupervisor()
        self.commands = CommandRouter(
            settings.command_prefix,
            self.authorization,
            supervisor=self.command_tasks,
        )
        self.referenced_message_parser = OopzReferencedMessageParser()
        self.command_requests = OopzCommandRequestFactory(
            self.command_parser,
            self.referenced_message_parser.parse,
        )
        enabled_agent_tools = settings.agent.enabled_tools
        music = build_music(
            settings, self.bot, self.voice_channel_sessions, self.database.session_factory
        )
        self.music = music.music
        self.music_catalog = music.music_catalog
        self.netease_music_provider = music.netease_music_provider
        self.bilibili_music_provider = music.bilibili_music_provider
        self.youtube_music_provider = music.youtube_music_provider
        self.music_playlists = music.music_playlists
        self.music_voice = music.music_voice
        self.music_ytdlp_runner = music.music_ytdlp_runner
        if not settings.music.enabled:
            enabled_agent_tools = tuple(
                name for name in enabled_agent_tools if name not in MUSIC_AGENT_TOOLS
            )
        web = build_web(settings)
        self.web_search = web.web_search
        self.browser = web.browser
        if not settings.web.search_enabled:
            enabled_agent_tools = tuple(
                name for name in enabled_agent_tools if name not in WEB_SEARCH_AGENT_TOOLS
            )
        if not settings.web.browser_enabled:
            enabled_agent_tools = tuple(
                name
                for name in enabled_agent_tools
                if name not in WEB_BROWSER_READ_TOOLS and name not in WEB_BROWSER_INTERACTION_TOOLS
            )
        if not settings.web.browser_interaction_enabled:
            enabled_agent_tools = tuple(
                name for name in enabled_agent_tools if name not in WEB_BROWSER_INTERACTION_TOOLS
            )
        agent = build_agent(
            settings,
            self.database.session_factory,
            self.bot,
            self.authorization,
            self.health,
            music.tools + web.tools,
            enabled_agent_tools,
        )
        self.agent_catalog = agent.agent_catalog
        self.agent_catalog_admin = agent.agent_catalog_admin
        self.agent_threads = agent.agent_threads
        self.agent_runs = agent.agent_runs
        self.agent_messages = agent.agent_messages
        self.agent_memory_repository = agent.agent_memory_repository
        self.agent_memory = agent.agent_memory
        self.agent_image_client = agent.agent_image_client
        self.agent_media_ingest = agent.agent_media_ingest
        self.agent_context = agent.agent_context
        self.agent_selection = agent.agent_selection
        self.agent_skill_repository = agent.agent_skill_repository
        self.agent_skill_notifier = agent.agent_skill_notifier
        self.agent_skill_library = agent.agent_skill_library
        self.agent_tool_registry = agent.agent_tool_registry
        self.agent_diagnostic_renderer = agent.agent_diagnostic_renderer
        self.agent_skill_availability = agent.agent_skill_availability
        self.agent_tool_authorization = agent.agent_tool_authorization
        self.agent_tool_policy = agent.agent_tool_policy
        self.agent_tool_availability = agent.agent_tool_availability
        self.agent_tool_executor = agent.agent_tool_executor
        self.direct_tools = agent.direct_tools
        self.agent_models = agent.agent_models
        self.agent_summary_tasks = agent.agent_summary_tasks
        self.agent_summary_service = agent.agent_summary_service
        self.agent_engine = agent.agent_engine
        self.agent_run_service = agent.agent_run_service
        self.agent_chat = agent.agent_chat
        channel_settings = agent.channel_settings
        logger.info(
            "Application configured: agent=%s tools=%s music=%s voice=%s web_search=%s browser=%s",
            settings.agent.enabled,
            len(self.agent_tool_registry.names),
            self.music is not None,
            settings.voice.enabled,
            self.web_search is not None,
            self.browser is not None,
        )
        self._provider = self._create_chat_provider()
        self.chat_tasks = ChatTaskSupervisor()
        self.recent_bot_messages = OopzRecentBotMessageLookup(self.bot)
        self.referenced_messages = ReferencedMessageResolver(
            self.outbound_messages,
            self.recent_bot_messages,
            settings.oopz.person_uid,
        )
        self.message_recall = MessageRecallService(
            self.referenced_messages,
            self.outbound_messages,
            self.active_agent_presentations,
            OutboundChatTaskCanceller(self.chat_tasks),
            OopzBotMessageRecallGateway(self.bot),
        )
        self.recall_message_action = RecallMessageAction(self.message_recall)
        self.debug_message_action = DebugMessageAction(
            self.agent_diagnostics,
            self.agent_diagnostic_renderer,
        )
        self.reaction_commands = ReactionCommandRouter(
            self.authorization,
            self.referenced_messages,
            OopzReactionCommandResponder(self.editable_messages),
        )
        self.reaction_commands.register(RecallReactionCommand(self.recall_message_action))
        self.reaction_commands.register(DebugReactionCommand(self.debug_message_action))
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
            command_prefix=settings.command_prefix,
        )
        self._ambient_handler = AmbientChatHandler(
            self.chat,
            channel_settings,
            self.agent_presenters,
            self.chat_invocations,
        )
        voice = build_voice(
            settings,
            self.database.session_factory,
            self.bot,
            self.voice_channel_sessions,
            self.voice_media,
            agent,
        )
        self.voice_configurations = voice.voice_configurations
        self.voice_sessions = voice.voice_sessions
        self.delegated_task_repository = voice.delegated_task_repository
        self.delegated_task_wakeup = voice.delegated_task_wakeup
        self.voice_task_completion_notifier = voice.voice_task_completion_notifier
        self.voice_delegated_tasks = voice.voice_delegated_tasks
        self.voice_task_mailbox = voice.voice_task_mailbox
        self.delegated_task_text_fallback = voice.delegated_task_text_fallback
        self.delegated_task_runner = voice.delegated_task_runner
        self.delegated_task_scheduler = voice.delegated_task_scheduler
        self.voice_task_tools = voice.voice_task_tools
        self.voice_runtimes = voice.voice_runtimes
        self.voice_access = voice.voice_access
        self.voice_conversations = voice.voice_conversations
        self._register_commands()
        self.bot.on_ready(self._on_ready)
        self.bot.on_message(self._on_message)
        self.bot.on("message.reaction")(self._on_message_reaction)
        self.health.mark("database", HealthState.PENDING)
        self.health.mark(
            "llm",
            HealthState.PENDING
            if settings.chat.enabled or settings.agent.enabled or settings.voice.enabled
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
            if (settings.agent.enabled or settings.voice.enabled) and settings.agent.skills_enabled
            else HealthState.DISABLED,
        )
        self.health.mark(
            "voice",
            HealthState.PENDING if settings.voice.enabled else HealthState.DISABLED,
            "experimental" if settings.voice.enabled else "feature disabled",
        )

    def _create_chat_provider(self) -> ChatProvider:
        if not self.settings.chat.enabled:
            return DisabledChatProvider()
        return OpenAICompatibleChatProvider(self.settings.chat)

    def _register_commands(self) -> None:
        self._register_builtin_commands()
        self._register_access_commands()
        self._register_admin_commands()
        self._register_chat_commands()
        self._register_music_commands()
        self._register_agent_commands()
        self._register_voice_commands()

    def _register_builtin_commands(self) -> None:
        self.commands.register_definition(PingCommand().definition())
        self.commands.register_definition(HelpCommand(self.commands).definition())
        self.commands.register_definition(StatusCommand(self.health).definition())

    def _register_access_commands(self) -> None:
        self.commands.register_definition(WhoAmICommand().definition())
        self.commands.register_definition(
            RoleCommand(
                self.authorization,
                self.role_administration,
            ).definition()
        )

    def _register_admin_commands(self) -> None:
        self.commands.register_definition(InitCommand(self.channel_initialization).definition())
        self.commands.register_definition(DebugCommand(self.debug_message_action).definition())
        self.commands.register_definition(RecallCommand(self.recall_message_action).definition())
        self.commands.register_definition(RebootCommand(self.lifecycle).definition())

    def _register_chat_commands(self) -> None:
        self.commands.register_definition(
            ChatCommand(
                self.chat,
                self.chat_tasks,
                self.agent_presenters,
                self.chat_invocations,
                prefix=self.settings.command_prefix,
            ).definition()
        )
        self.commands.register_definition(
            NewConversationCommand(self.chat, self.chat_tasks).definition()
        )
        self.commands.register_definition(
            CancelChatCommand(
                self.chat,
                self.chat_tasks,
                active_message_reports_cancel=(
                    self.settings.agent.enabled and self.settings.agent.live_display
                ),
            ).definition()
        )
        if self.settings.agent.enabled:
            self.commands.register_definition(
                AgentModelCommand(
                    self.agent_chat,
                    self.chat_tasks,
                    self.settings.command_prefix,
                ).definition()
            )
        else:
            self.commands.register_definition(
                ModelCommand(
                    self.legacy_chat,
                    self.chat_tasks,
                    self.settings.command_prefix,
                ).definition()
            )
        self.commands.register_definition(ChatStatusCommand(self.chat).definition())

    def _register_music_commands(self) -> None:
        if self.music is not None and self.music_playlists is not None:
            self.commands.register_definition(
                MusicCommand(
                    self.music,
                    self.music_playlists,
                    self.settings.command_prefix,
                ).definition()
            )

    def _register_agent_commands(self) -> None:
        if self.settings.agent.enabled:
            self.commands.register_definition(
                ProviderCommand(
                    self.agent_chat,
                    self.chat_tasks,
                    self.settings.command_prefix,
                ).definition()
            )
            self.commands.register_definition(ToolsCommand(self.agent_chat).definition())
            self.commands.register_definition(
                ToolCommand(
                    self.direct_tools,
                    self.settings.command_prefix,
                ).definition()
            )
            self.commands.register_definition(
                MemoryCommand(
                    self.agent_memory,
                    self.settings.command_prefix,
                ).definition()
            )
            if self.settings.agent.skills_enabled:
                self.commands.register_definition(
                    SkillsCommand(
                        self.agent_chat,
                        self.agent_skill_library,
                    ).definition()
                )

    def _register_voice_commands(self) -> None:
        if self.settings.voice.enabled:
            self.commands.register_definition(
                VoiceCommand(
                    self.voice_conversations,
                    self.voice_configurations,
                    OopzVoiceCommandPresenter(self.editable_messages),
                ).definition()
            )

    async def run(self) -> ShutdownDisposition:
        """Start the database check before entering the long-running OOPZ client."""
        logger.info("Application startup started")
        disposition = ShutdownDisposition.NORMAL
        if not self.settings.rbac.bootstrap_owner_ids:
            logger.warning(
                "No RBAC bootstrap owner is configured; privileged recovery requires "
                "a database role"
            )
        try:
            if self.settings.voice.enabled:
                self.voice_capability_gate.validate(self.bot.voice.capabilities)
                logger.info(
                    "OOPZ realtime voice SDK contract validated: feature_version=%s",
                    self.bot.voice.capabilities.feature_version,
                )
            if self.music_voice is not None:
                await self.music_voice.validate_capabilities()
            if self.music_ytdlp_runner is not None:
                probe = YtDlpCapabilityProbe(self.music_ytdlp_runner)
                try:
                    capabilities = await probe.validate(require_javascript=False)
                    logger.info(
                        "yt-dlp worker capability validated: version=%s",
                        capabilities.get("yt_dlp", "unknown"),
                    )
                except MusicSourceUnavailableError as exc:
                    self.health.mark("music:ytdlp", HealthState.DEGRADED, str(exc))
                    logger.warning(
                        "yt-dlp worker capability unavailable: error=%s",
                        exception_kind(exc),
                    )
                if MusicSourceKind.YOUTUBE in self.settings.music.enabled_sources:
                    try:
                        await probe.validate(require_javascript=True)
                    except MusicSourceUnavailableError as exc:
                        self.health.mark("music:youtube", HealthState.DEGRADED, str(exc))
                        logger.warning(
                            "YouTube JavaScript capability unavailable: error=%s",
                            exception_kind(exc),
                        )
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
            recovered_voice_sessions = await self.voice_sessions.recover_stale(datetime.now(UTC))
            if recovered_voice_sessions:
                logger.warning(
                    "Marked stale voice sessions interrupted after process restart: count=%s",
                    recovered_voice_sessions,
                )
            agent_runtime_enabled = self.settings.agent.enabled or self.settings.voice.enabled
            if agent_runtime_enabled:
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
                            if self.settings.agent.enabled_tools or self.settings.voice.enabled
                            else frozenset()
                        ),
                        require_user_selectable=False,
                    )
                    if default_model_id is not None
                    else None
                )
                if default_model is None:
                    raise ConfigurationError(
                        "Agent or voice mode requires an enabled application-default LLM model"
                    )
            now = datetime.now(UTC)
            abandoned = await self.agent_runs.abandon_stale(
                now - timedelta(seconds=self.settings.agent.stale_run_after_seconds),
                now,
            )
            if abandoned:
                logger.warning("Marked stale Agent runs abandoned: count=%s", abandoned)
            # Accepted voice tasks outlive the feature flag that created them.
            await self.delegated_task_scheduler.start()
            await self.delegated_task_text_fallback.start()
            logger.info("Starting OOPZ client")
            disposition = await self._run_oopz_until_lifecycle_request()
        finally:
            logger.info("Application shutdown started")
            await self._close_resources()
            logger.info("Application shutdown completed")
        return disposition

    async def _close_resources(self) -> None:
        """Try every owned resource in dependency order, even if a close fails.

        Consumers stop before their shared transports and persistence. ExitStack
        preserves cleanup exceptions while still attempting the remaining steps.
        """
        callbacks = [
            self.command_tasks.close,
            self.chat_tasks.close,
            self.agent_summary_tasks.close,
            self.voice_conversations.aclose,
            self.delegated_task_scheduler.aclose,
            self.delegated_task_text_fallback.aclose,
        ]
        if self.music is not None:
            callbacks.append(self.music.aclose)
        if self.music_ytdlp_runner is not None:
            callbacks.append(self.music_ytdlp_runner.aclose)
        callbacks.append(self.voice_channel_sessions.aclose)
        if self.browser is not None:
            callbacks.append(self.browser.aclose)
        if self.web_search is not None:
            callbacks.append(self.web_search.aclose)
        callbacks.extend(
            (
                self.agent_engine.aclose,
                self.agent_image_client.aclose,
                self._provider.aclose,
                self.database.close,
            )
        )
        async with AsyncExitStack() as resources:
            for close in reversed(callbacks):
                resources.push_async_callback(close)

    async def _run_oopz_until_lifecycle_request(self) -> ShutdownDisposition:
        bot_task = asyncio.create_task(self.bot.run(), name="oopz-bot")
        lifecycle_task = asyncio.create_task(
            self.lifecycle.wait(),
            name="application-lifecycle",
        )
        try:
            done, _ = await asyncio.wait(
                {bot_task, lifecycle_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if lifecycle_task in done:
                disposition = lifecycle_task.result()
                await self._stop_oopz_bounded()
                await self._settle_oopz_task(bot_task)
                return disposition
            await bot_task
            return ShutdownDisposition.NORMAL
        finally:
            pending = tuple(task for task in (bot_task, lifecycle_task) if not task.done())
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

    async def _stop_oopz_bounded(self) -> None:
        try:
            async with asyncio.timeout(self.oopz_stop_timeout_seconds):
                await self.bot.stop()
        except TimeoutError:
            logger.error(
                "Timed out stopping OOPZ client after %.1fs",
                self.oopz_stop_timeout_seconds,
            )
        except Exception as exc:
            logger.error("OOPZ client stop failed: error=%s", exception_kind(exc))

    async def _settle_oopz_task(self, bot_task: asyncio.Task[object]) -> None:
        if not bot_task.done():
            done, _ = await asyncio.wait(
                {bot_task},
                timeout=self.oopz_run_settle_seconds,
            )
            if not done:
                logger.warning("Cancelling OOPZ run task after bounded stop grace")
                bot_task.cancel()
                await asyncio.gather(bot_task, return_exceptions=True)
                return
        if bot_task.cancelled():
            return
        error = bot_task.exception()
        if error is not None:
            logger.warning(
                "OOPZ run task ended with an error during planned shutdown: error=%s",
                exception_kind(error),
            )

    async def _on_ready(self, _: EventContext) -> None:
        self.health.mark("oopz", HealthState.HEALTHY, "websocket connected")
        logger.info("Bot connected; command prefix is %r", self.settings.command_prefix)

    async def _on_message(self, message: OopzMessage, context: EventContext) -> None:
        """Route short commands inline and own slow LLM work in supervised tasks."""
        context = TrackedMessageContext(context, self.outbound_messages)
        logger.debug(
            "Received OOPZ message: scope=%s conversation=%s has_text=%s",
            "private" if getattr(context.event, "is_private", False) else "channel",
            self._message_reference(message, context),
            bool(message.plain_text or message.text or message.content),
        )
        try:
            command_request = self.command_requests.from_message(message, context)
        except ValueError as exc:
            logger.warning(
                "Could not project OOPZ command request: conversation=%s error=%s",
                self._message_reference(message, context),
                exception_kind(exc),
            )
            return
        if command_request is not None:
            command_text = command_request.text
            assert command_text is not None
            logger.info(
                "Dispatching command: name=%s conversation=%s",
                command_text.name,
                self._message_reference(message, context),
            )
            await self.commands.dispatch_request(command_request)
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

    async def _on_message_reaction(self, context: EventContext, event: Any) -> None:
        """Translate added reactions into curated, RBAC-protected commands."""
        invocation = OopzReactionCommandInvocationParser.parse(context, event)
        if invocation is None:
            return
        await self.reaction_commands.dispatch(invocation)

    async def _start_chat_task(
        self,
        context: EventContext,
        operation: Coroutine[Any, Any, object],
    ) -> None:
        """Register slow work so the SDK receive loop can immediately process later events."""
        try:
            key = conversation_key_from_context(context)
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
            await context.reply("当前对话正在生成回复；可使用 /cancel 取消后再试。")

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
