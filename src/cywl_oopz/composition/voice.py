"""Voice conversation and delegated-task assembly."""

from __future__ import annotations

from dataclasses import dataclass

from oopz_sdk import OopzBot
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from cywl_oopz.features.agent.delegation.mailbox import (
    DelegatedTaskTextFallbackReconciler,
    InProcessVoiceTaskCompletionNotifier,
    VoiceTaskMailboxService,
)
from cywl_oopz.features.agent.delegation.repository import SqlAlchemyDelegatedTaskRepository
from cywl_oopz.features.agent.delegation.runner import DelegatedAgentTaskRunner
from cywl_oopz.features.agent.delegation.scheduler import DelegatedTaskScheduler
from cywl_oopz.features.agent.delegation.service import (
    InProcessDelegatedTaskWakeup,
    VoiceDelegatedTaskService,
)
from cywl_oopz.features.voice.repository import (
    SqlAlchemyVoiceConfigurationRepository,
    SqlAlchemyVoiceSessionRepository,
)
from cywl_oopz.features.voice.runtime import RealtimeVoiceSessionRuntimeFactoryImpl
from cywl_oopz.features.voice.service import VoiceConversationService
from cywl_oopz.features.voice.task_tools import VoiceTaskControlTools
from cywl_oopz.integrations.oopz.voice_channel_session import OopzVoiceChannelSessionManager
from cywl_oopz.integrations.oopz.voice_conversation import (
    OopzConversationVoiceAccess,
)
from cywl_oopz.integrations.oopz.voice_media import OopzVoiceMediaGateway
from cywl_oopz.integrations.oopz.voice_task_notifications import OopzVoiceTaskTextGateway
from cywl_oopz.integrations.voice.provider_builder import ConfiguredVoiceProviderBuilder
from cywl_oopz.settings import (
    AppSettings,
)

from .agent import AgentComponents


@dataclass(slots=True)
class VoiceComponents:
    """Voice sessions and durable task recovery owned by the application."""

    voice_configurations: SqlAlchemyVoiceConfigurationRepository
    voice_sessions: SqlAlchemyVoiceSessionRepository
    delegated_task_repository: SqlAlchemyDelegatedTaskRepository
    delegated_task_wakeup: InProcessDelegatedTaskWakeup
    voice_task_completion_notifier: InProcessVoiceTaskCompletionNotifier
    voice_delegated_tasks: VoiceDelegatedTaskService
    voice_task_mailbox: VoiceTaskMailboxService
    delegated_task_text_fallback: DelegatedTaskTextFallbackReconciler
    delegated_task_runner: DelegatedAgentTaskRunner
    delegated_task_scheduler: DelegatedTaskScheduler
    voice_task_tools: VoiceTaskControlTools
    voice_runtimes: RealtimeVoiceSessionRuntimeFactoryImpl
    voice_access: OopzConversationVoiceAccess
    voice_conversations: VoiceConversationService


def build_voice(
    settings: AppSettings,
    sessions: async_sessionmaker[AsyncSession],
    bot: OopzBot,
    voice_channels: OopzVoiceChannelSessionManager,
    media: OopzVoiceMediaGateway,
    agent: AgentComponents,
) -> VoiceComponents:
    """Assemble voice control, including recovery when new sessions are disabled."""
    voice_configurations = SqlAlchemyVoiceConfigurationRepository(sessions)
    voice_sessions = SqlAlchemyVoiceSessionRepository(sessions)
    delegated_task_repository = SqlAlchemyDelegatedTaskRepository(sessions)
    delegated_task_wakeup = InProcessDelegatedTaskWakeup()
    voice_task_completion_notifier = InProcessVoiceTaskCompletionNotifier()
    voice_delegated_tasks = VoiceDelegatedTaskService(
        delegated_task_repository,
        delegated_task_wakeup,
        completion_notifier=voice_task_completion_notifier,
    )
    voice_task_mailbox = VoiceTaskMailboxService(
        delegated_task_repository,
        voice_task_completion_notifier,
        OopzVoiceTaskTextGateway(bot),
    )
    delegated_task_text_fallback = DelegatedTaskTextFallbackReconciler(
        delegated_task_repository,
        voice_task_completion_notifier,
        voice_task_mailbox,
        poll_seconds=settings.voice.mailbox_poll_seconds,
    )
    delegated_task_runner = DelegatedAgentTaskRunner(
        settings.agent,
        delegated_task_repository,
        delegated_task_wakeup,
        agent.agent_run_service,
        agent.agent_catalog,
        agent.agent_threads,
        agent.agent_context,
        agent.agent_tool_registry,
        agent.agent_skill_repository if settings.agent.skills_enabled else None,
        agent.agent_skill_availability if settings.agent.skills_enabled else None,
        completion_notifier=voice_task_completion_notifier,
        max_task_retries=settings.agent.provider_max_retries,
        heartbeat_interval_seconds=max(
            1.0,
            min(10.0, settings.agent.stale_run_after_seconds / 3),
        ),
    )
    delegated_task_scheduler = DelegatedTaskScheduler(
        delegated_task_repository,
        delegated_task_wakeup,
        delegated_task_runner,
        completion_notifier=voice_task_completion_notifier,
        read_concurrency=settings.voice.read_task_concurrency,
        per_user_concurrency=settings.voice.per_user_task_concurrency,
        reconcile_seconds=settings.voice.mailbox_poll_seconds,
    )
    voice_task_tools = VoiceTaskControlTools(voice_delegated_tasks)
    voice_runtimes = RealtimeVoiceSessionRuntimeFactoryImpl(
        settings.voice,
        media,
        voice_sessions,
        ConfiguredVoiceProviderBuilder(tool_schemas=voice_task_tools.schemas()),
        voice_task_tools,
        voice_task_mailbox,
    )
    voice_access = OopzConversationVoiceAccess(bot, voice_channels)
    voice_conversations = VoiceConversationService(
        settings.voice,
        voice_access,
        voice_runtimes,
        voice_configurations,
        voice_sessions,
        agent.agent_memory,
        voice_access,
    )
    return VoiceComponents(
        voice_configurations=voice_configurations,
        voice_sessions=voice_sessions,
        delegated_task_repository=delegated_task_repository,
        delegated_task_wakeup=delegated_task_wakeup,
        voice_task_completion_notifier=voice_task_completion_notifier,
        voice_delegated_tasks=voice_delegated_tasks,
        voice_task_mailbox=voice_task_mailbox,
        delegated_task_text_fallback=delegated_task_text_fallback,
        delegated_task_runner=delegated_task_runner,
        delegated_task_scheduler=delegated_task_scheduler,
        voice_task_tools=voice_task_tools,
        voice_runtimes=voice_runtimes,
        voice_access=voice_access,
        voice_conversations=voice_conversations,
    )
