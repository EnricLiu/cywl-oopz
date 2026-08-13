"""Composition root for the bot application."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from datetime import UTC, datetime, timedelta
from typing import Any

from oopz_sdk import OopzBot
from oopz_sdk.events.context import EventContext
from oopz_sdk.models import Message as OopzMessage

from .commands.builtin import HelpCommand, PingCommand, StatusCommand
from .commands.execution import CommandTaskSupervisor
from .commands.parsing import CommandTextParser
from .commands.router import CommandRouter
from .core.errors import ConfigurationError, DatabaseError
from .core.health import HealthRegistry, HealthState
from .core.observability import exception_kind, opaque_ref
from .core.tasks import TaskSupervisor
from .features.access.administration import RoleAdministrationService
from .features.access.agent_tools import AgentToolAuthorizationAdapter
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
from .features.agent.delegation.mailbox import (
    DelegatedTaskTextFallbackReconciler,
    InProcessVoiceTaskCompletionNotifier,
    VoiceTaskMailboxService,
)
from .features.agent.delegation.repository import SqlAlchemyDelegatedTaskRepository
from .features.agent.delegation.runner import DelegatedAgentTaskRunner
from .features.agent.delegation.scheduler import DelegatedTaskScheduler
from .features.agent.delegation.service import (
    InProcessDelegatedTaskWakeup,
    VoiceDelegatedTaskService,
)
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
from .features.agent.run_service import AgentRunService
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
    ClearMusicQueueTool,
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
    ClearMusicPlaylistTool,
    CreateMusicPlaylistTool,
    DeleteMusicPlaylistTool,
    GetMusicPlaylistTool,
    ImportNeteasePlaylistTool,
    ListMusicPlaylistsTool,
    LoadMusicPlaylistTool,
    PreviewNeteasePlaylistTool,
    RemoveMusicPlaylistTrackTool,
    RenameMusicPlaylistTool,
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
    CancelChatCommand,
    ChatCommand,
    ChatStatusCommand,
    ModelCommand,
    NewConversationCommand,
)
from .features.chat.models import ConversationKey
from .features.chat.openai_compatible import OpenAICompatibleChatProvider
from .features.chat.provider import ChatProvider, DisabledChatProvider
from .features.chat.repository import SqlAlchemyConversationRepository
from .features.chat.service import ChatService
from .features.chat.tasks import ChatTaskSupervisor, OutboundChatTaskCanceller
from .features.music.bilibili import BilibiliMusicProvider
from .features.music.catalog import CompositeMusicCatalog, MusicProviderRegistry
from .features.music.commands import MusicCommand
from .features.music.errors import MusicSourceUnavailableError
from .features.music.models import MusicSourceKind
from .features.music.netease import NeteaseMusicProvider
from .features.music.playlist_repository import SqlAlchemyMusicPlaylistRepository
from .features.music.playlists import MusicPlaylistService
from .features.music.service import MusicRequestService
from .features.music.youtube import YouTubeMusicProvider
from .features.voice.commands import VoiceCommand
from .features.voice.repository import (
    SqlAlchemyVoiceConfigurationRepository,
    SqlAlchemyVoiceSessionRepository,
)
from .features.voice.runtime import RealtimeVoiceSessionRuntimeFactoryImpl
from .features.voice.service import VoiceConversationService
from .features.voice.task_tools import VoiceTaskControlTools
from .features.web.browser import BrowserSessionManager
from .features.web.errors import BrowserError
from .features.web.service import WebSearchService
from .integrations.media.ytdlp_runner import YtDlpCapabilityProbe, YtDlpProcessRunner
from .integrations.oopz.active_presentations import ActivePresentationRegistry
from .integrations.oopz.agent_presenter import OopzAgentPresenterFactory
from .integrations.oopz.channel_catalog import OopzAreaChannelCatalog
from .integrations.oopz.chat_handlers import AmbientChatHandler, MentionChatHandler
from .integrations.oopz.chat_invocation import OopzChatInvocationFactory
from .integrations.oopz.command_requests import OopzCommandRequestFactory
from .integrations.oopz.diagnostic_renderer import OopzAgentDiagnosticRenderer
from .integrations.oopz.editable_messages import OopzEditableMessageGateway
from .integrations.oopz.master_audio import OopzMasterPcmOutputFactory
from .integrations.oopz.message_recall import (
    OopzBotMessageRecallGateway,
    OopzRecentBotMessageLookup,
    OopzReferencedMessageParser,
)
from .integrations.oopz.message_renderer import OopzMessageRenderer
from .integrations.oopz.music import OopzMusicVoiceGateway
from .integrations.oopz.reaction_commands import (
    OopzReactionCommandInvocationParser,
    OopzReactionCommandResponder,
)
from .integrations.oopz.reactions import OopzReactionGateway
from .integrations.oopz.skill_sharing import OopzSkillShareNotifier
from .integrations.oopz.tracked_context import TrackedMessageContext
from .integrations.oopz.voice_capabilities import OopzVoiceCapabilityGate
from .integrations.oopz.voice_channel_session import OopzVoiceChannelSessionManager
from .integrations.oopz.voice_conversation import (
    OopzConversationVoiceAccess,
    OopzVoiceCommandPresenter,
)
from .integrations.oopz.voice_media import OopzVoiceMediaGateway
from .integrations.oopz.voice_task_notifications import OopzVoiceTaskTextGateway
from .integrations.voice.provider_builder import ConfiguredVoiceProviderBuilder
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
        self.music_catalog: CompositeMusicCatalog | None = None
        self.netease_music_provider: NeteaseMusicProvider | None = None
        self.bilibili_music_provider: BilibiliMusicProvider | None = None
        self.youtube_music_provider: YouTubeMusicProvider | None = None
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
        self.music_voice: OopzMusicVoiceGateway | None = None
        self.music_ytdlp_runner: YtDlpProcessRunner | None = None
        if settings.music.enabled:
            if any(
                source in {MusicSourceKind.YOUTUBE, MusicSourceKind.BILIBILI}
                for source in settings.music.enabled_sources
            ):
                self.music_ytdlp_runner = YtDlpProcessRunner(settings.music_ytdlp)
            music_providers = []
            for source in settings.music.enabled_sources:
                if source is MusicSourceKind.NETEASE:
                    self.netease_music_provider = NeteaseMusicProvider(settings.music)
                    music_providers.append(self.netease_music_provider)
                elif source is MusicSourceKind.BILIBILI:
                    assert self.music_ytdlp_runner is not None
                    self.bilibili_music_provider = BilibiliMusicProvider(
                        settings.music,
                        settings.music_ytdlp,
                        self.music_ytdlp_runner,
                    )
                    music_providers.append(self.bilibili_music_provider)
                elif source is MusicSourceKind.YOUTUBE:
                    assert self.music_ytdlp_runner is not None
                    self.youtube_music_provider = YouTubeMusicProvider(
                        settings.music,
                        settings.music_ytdlp,
                        self.music_ytdlp_runner,
                    )
                    music_providers.append(self.youtube_music_provider)
            self.music_catalog = CompositeMusicCatalog(
                MusicProviderRegistry(music_providers),
                settings.music.default_source,
            )
            self.music_voice = OopzMusicVoiceGateway(
                self.bot,
                self.voice_channel_sessions,
                settings.audio,
            )
            self.music = MusicRequestService(
                settings.music,
                self.music_catalog,
                self.music_voice,
            )
            self.music_playlists = MusicPlaylistService(
                settings.music,
                SqlAlchemyMusicPlaylistRepository(self.database.session_factory),
                self.music,
                self.netease_music_provider,
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
                    ClearMusicQueueTool(self.music, **music_tool_options),
                    SetMusicPlaybackModeTool(self.music, **music_tool_options),
                    CreateMusicPlaylistTool(self.music_playlists, **music_tool_options),
                    ListMusicPlaylistsTool(self.music_playlists, **music_tool_options),
                    GetMusicPlaylistTool(self.music_playlists, **music_tool_options),
                    AddMusicPlaylistTrackTool(self.music_playlists, **music_tool_options),
                    RemoveMusicPlaylistTrackTool(
                        self.music_playlists,
                        **music_tool_options,
                    ),
                    RenameMusicPlaylistTool(self.music_playlists, **music_tool_options),
                    DeleteMusicPlaylistTool(self.music_playlists, **music_tool_options),
                    ClearMusicPlaylistTool(self.music_playlists, **music_tool_options),
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
        self.agent_diagnostic_renderer = OopzAgentDiagnosticRenderer(
            {
                descriptor.name: descriptor.display_name
                for descriptor in self.agent_tool_registry.descriptors()
            }
        )
        self.agent_skill_availability = SkillAvailabilityService(
            max_available_skills=settings.agent.max_available_skills,
        )
        logger.info(
            "Application configured: agent=%s tools=%s music=%s voice=%s web_search=%s browser=%s",
            settings.agent.enabled,
            len(self.agent_tool_registry.names),
            self.music is not None,
            settings.voice.enabled,
            self.web_search is not None,
            self.browser is not None,
        )
        self.agent_tool_authorization = AgentToolAuthorizationAdapter(self.authorization)
        self.agent_tool_policy = ToolPolicy(self.agent_tool_authorization)
        self.agent_tool_availability = ToolAvailabilityService(
            self.agent_tool_registry,
            channel_settings,
            enabled_agent_tools,
            self.agent_tool_policy,
        )
        self.agent_tool_executor = ToolExecutor(
            self.agent_tool_registry,
            self.agent_tool_policy,
            SqlAlchemyToolExecutionRepository(self.database.session_factory),
        )
        self.direct_tools = DirectToolService(
            settings.agent,
            self.agent_tool_registry,
            self.agent_tool_availability,
            self.agent_selection,
            self.agent_tool_policy,
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
        self.agent_run_service = AgentRunService(
            self.agent_engine,
            self.agent_runs,
            self.agent_messages,
            heartbeat_interval_seconds=max(
                1.0,
                min(10.0, settings.agent.stale_run_after_seconds / 3),
            ),
            health=self.health,
        )
        self.agent_chat = AgentConversationService(
            settings.agent,
            settings.chat,
            self.agent_run_service,
            self.agent_catalog,
            self.agent_selection,
            selection_repository,
            self.agent_threads,
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
        self.voice_configurations = SqlAlchemyVoiceConfigurationRepository(
            self.database.session_factory
        )
        self.voice_sessions = SqlAlchemyVoiceSessionRepository(self.database.session_factory)
        self.delegated_task_repository = SqlAlchemyDelegatedTaskRepository(
            self.database.session_factory
        )
        self.delegated_task_wakeup = InProcessDelegatedTaskWakeup()
        self.voice_task_completion_notifier = InProcessVoiceTaskCompletionNotifier()
        self.voice_delegated_tasks = VoiceDelegatedTaskService(
            self.delegated_task_repository,
            self.delegated_task_wakeup,
            completion_notifier=self.voice_task_completion_notifier,
        )
        self.voice_task_mailbox = VoiceTaskMailboxService(
            self.delegated_task_repository,
            self.voice_task_completion_notifier,
            OopzVoiceTaskTextGateway(self.bot),
        )
        self.delegated_task_text_fallback = DelegatedTaskTextFallbackReconciler(
            self.delegated_task_repository,
            self.voice_task_completion_notifier,
            self.voice_task_mailbox,
            poll_seconds=settings.voice.mailbox_poll_seconds,
        )
        self.delegated_task_runner = DelegatedAgentTaskRunner(
            settings.agent,
            self.delegated_task_repository,
            self.delegated_task_wakeup,
            self.agent_run_service,
            self.agent_catalog,
            self.agent_threads,
            self.agent_context,
            self.agent_tool_registry,
            self.agent_skill_repository if settings.agent.skills_enabled else None,
            self.agent_skill_availability if settings.agent.skills_enabled else None,
            completion_notifier=self.voice_task_completion_notifier,
            max_task_retries=settings.agent.provider_max_retries,
            heartbeat_interval_seconds=max(
                1.0,
                min(10.0, settings.agent.stale_run_after_seconds / 3),
            ),
        )
        self.delegated_task_scheduler = DelegatedTaskScheduler(
            self.delegated_task_repository,
            self.delegated_task_wakeup,
            self.delegated_task_runner,
            completion_notifier=self.voice_task_completion_notifier,
            read_concurrency=settings.voice.read_task_concurrency,
            per_user_concurrency=settings.voice.per_user_task_concurrency,
            reconcile_seconds=settings.voice.mailbox_poll_seconds,
        )
        self.voice_task_tools = VoiceTaskControlTools(self.voice_delegated_tasks)
        self.voice_runtimes = RealtimeVoiceSessionRuntimeFactoryImpl(
            settings.voice,
            self.voice_media,
            self.voice_sessions,
            ConfiguredVoiceProviderBuilder(tool_schemas=self.voice_task_tools.schemas()),
            self.voice_task_tools,
            self.voice_task_mailbox,
        )
        self.voice_access = OopzConversationVoiceAccess(self.bot, self.voice_channel_sessions)
        self.voice_conversations = VoiceConversationService(
            settings.voice,
            self.voice_access,
            self.voice_runtimes,
            self.voice_configurations,
            self.voice_sessions,
            self.agent_memory,
            self.voice_access,
        )
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
            await self.command_tasks.close()
            await self.chat_tasks.close()
            await self.agent_summary_tasks.close()
            await self.voice_conversations.aclose()
            await self.delegated_task_scheduler.aclose()
            await self.delegated_task_text_fallback.aclose()
            if self.music is not None:
                await self.music.aclose()
            if self.music_ytdlp_runner is not None:
                await self.music_ytdlp_runner.aclose()
            await self.voice_channel_sessions.aclose()
            if self.browser is not None:
                await self.browser.aclose()
            if self.web_search is not None:
                await self.web_search.aclose()
            await self.agent_engine.aclose()
            await self._provider.aclose()
            await self.database.close()
            logger.info("Application shutdown completed")
        return disposition

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
