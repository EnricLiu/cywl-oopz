"""Agent persistence, curated tools, and model runtime assembly."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

import httpx
from oopz_sdk import OopzBot
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from cywl_oopz.core.health import HealthRegistry
from cywl_oopz.core.tasks import TaskSupervisor
from cywl_oopz.features.access.agent_tools import AgentToolAuthorizationAdapter
from cywl_oopz.features.access.service import AuthorizationService
from cywl_oopz.features.agent.catalog import ProviderCatalogAdminService, ReloadableProviderCatalog
from cywl_oopz.features.agent.context import AgentContextBuilder
from cywl_oopz.features.agent.direct_tools import DirectToolService
from cywl_oopz.features.agent.media import AgentImagePolicy, AgentMediaIngestService
from cywl_oopz.features.agent.memory import MemoryService
from cywl_oopz.features.agent.memory_repository import SqlAlchemyMemoryRepository
from cywl_oopz.features.agent.pydantic_ai_engine import PydanticAiAgentEngine
from cywl_oopz.features.agent.registry import AgentModelRegistry
from cywl_oopz.features.agent.repository import (
    SqlAlchemyAgentMessageRepository,
    SqlAlchemyAgentRunRepository,
    SqlAlchemyAgentThreadRepository,
    SqlAlchemyModelSelectionRepository,
    SqlAlchemyProviderCatalogRepository,
    SqlAlchemyToolExecutionRepository,
)
from cywl_oopz.features.agent.run_service import AgentRunService
from cywl_oopz.features.agent.selection import ProviderSelectionService
from cywl_oopz.features.agent.service import AgentConversationService
from cywl_oopz.features.agent.skills.availability import SkillAvailabilityService
from cywl_oopz.features.agent.skills.library import AgentSkillLibraryService
from cywl_oopz.features.agent.skills.library_tools import (
    SKILL_LIBRARY_TOOL_NAMES,
    skill_library_tools,
)
from cywl_oopz.features.agent.skills.repository import SqlAlchemyAgentSkillRepository
from cywl_oopz.features.agent.skills.tools import LoadAgentSkillTool, ReadAgentSkillResourceTool
from cywl_oopz.features.agent.summarization import (
    PydanticAiThreadSummarizer,
    ThreadSummaryService,
)
from cywl_oopz.features.agent.tools.builtin import (
    GetAgentStatusTool,
    GetChannelSettingsTool,
    ReactToMessageTool,
)
from cywl_oopz.features.agent.tools.executor import ToolExecutor
from cywl_oopz.features.agent.tools.policy import ToolAvailabilityService, ToolPolicy
from cywl_oopz.features.agent.tools.ports import AgentTool
from cywl_oopz.features.agent.tools.registry import ToolRegistry
from cywl_oopz.integrations.oopz.diagnostic_renderer import OopzAgentDiagnosticRenderer
from cywl_oopz.integrations.oopz.image_loader import OopzImageContentLoader
from cywl_oopz.integrations.oopz.reactions import OopzReactionGateway
from cywl_oopz.integrations.oopz.skill_sharing import OopzSkillShareNotifier
from cywl_oopz.settings import (
    SKILL_AGENT_TOOLS,
    SKILL_AUTHORING_AGENT_TOOLS,
    AppSettings,
)
from cywl_oopz.storage.channel_settings import SqlAlchemyChannelSettingsRepository


@dataclass(slots=True)
class AgentComponents:
    """Agent-owned services exposed to application and voice assembly."""

    agent_catalog: ReloadableProviderCatalog
    agent_catalog_admin: ProviderCatalogAdminService
    agent_threads: SqlAlchemyAgentThreadRepository
    agent_runs: SqlAlchemyAgentRunRepository
    agent_messages: SqlAlchemyAgentMessageRepository
    agent_memory_repository: SqlAlchemyMemoryRepository
    agent_memory: MemoryService
    agent_image_client: httpx.AsyncClient
    agent_media_ingest: AgentMediaIngestService
    agent_context: AgentContextBuilder
    agent_selection: ProviderSelectionService
    agent_skill_repository: SqlAlchemyAgentSkillRepository
    agent_skill_notifier: OopzSkillShareNotifier
    agent_skill_library: AgentSkillLibraryService | None
    agent_tool_registry: ToolRegistry
    agent_diagnostic_renderer: OopzAgentDiagnosticRenderer
    agent_skill_availability: SkillAvailabilityService
    agent_tool_authorization: AgentToolAuthorizationAdapter
    agent_tool_policy: ToolPolicy
    agent_tool_availability: ToolAvailabilityService
    agent_tool_executor: ToolExecutor
    direct_tools: DirectToolService
    agent_models: AgentModelRegistry
    agent_summary_tasks: TaskSupervisor[UUID]
    agent_summary_service: ThreadSummaryService
    agent_engine: PydanticAiAgentEngine
    agent_run_service: AgentRunService
    agent_chat: AgentConversationService
    channel_settings: SqlAlchemyChannelSettingsRepository


def build_agent(
    settings: AppSettings,
    sessions: async_sessionmaker[AsyncSession],
    bot: OopzBot,
    authorization: AuthorizationService,
    health: HealthRegistry,
    extra_tools: tuple[AgentTool, ...],
    enabled_agent_tools: tuple[str, ...],
) -> AgentComponents:
    """Assemble persistence, tools, and one reusable Agent runtime."""
    catalog_repository = SqlAlchemyProviderCatalogRepository(sessions)
    agent_catalog = ReloadableProviderCatalog(catalog_repository)
    agent_catalog_admin = ProviderCatalogAdminService(
        catalog_repository,
        agent_catalog,
    )
    agent_threads = SqlAlchemyAgentThreadRepository(sessions)
    agent_runs = SqlAlchemyAgentRunRepository(sessions)
    agent_messages = SqlAlchemyAgentMessageRepository(sessions)
    agent_memory_repository = SqlAlchemyMemoryRepository(sessions)
    agent_memory = MemoryService(settings.agent, agent_memory_repository)
    agent_image_client = httpx.AsyncClient(
        timeout=httpx.Timeout(10.0, connect=3.0),
        follow_redirects=False,
    )
    agent_media_ingest = AgentMediaIngestService(
        OopzImageContentLoader(agent_image_client),
        AgentImagePolicy(
            max_images=settings.agent.max_input_images,
            max_image_bytes=settings.agent.max_input_image_bytes,
            max_total_bytes=settings.agent.max_input_image_total_bytes,
            max_pixels=settings.agent.max_input_image_pixels,
            max_parallel_downloads=settings.agent.max_input_image_downloads,
        ),
    )
    agent_context = AgentContextBuilder(
        settings.agent,
        agent_messages,
        agent_memory,
    )
    selection_repository = SqlAlchemyModelSelectionRepository(sessions)
    agent_selection = ProviderSelectionService(
        agent_catalog,
        selection_repository,
    )
    channel_settings = SqlAlchemyChannelSettingsRepository(sessions)
    agent_skill_repository = SqlAlchemyAgentSkillRepository(sessions)
    agent_skill_notifier = OopzSkillShareNotifier(bot)
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
            OopzReactionGateway(bot),
            timeout_seconds=settings.agent.tool_timeout_seconds,
            max_output_characters=settings.agent.max_tool_result_characters,
        ),
    ]
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
    agent_tools.extend(extra_tools)
    agent_skill_library: AgentSkillLibraryService | None = None
    if settings.agent.skills_enabled and settings.agent.skill_authoring_enabled:
        registered_skill_tools = (
            frozenset(tool.descriptor.name for tool in agent_tools) | SKILL_LIBRARY_TOOL_NAMES
        )
        agent_skill_library = AgentSkillLibraryService(
            agent_skill_repository,
            registered_tools=registered_skill_tools,
            max_personal_skills=settings.agent.max_personal_skills,
            max_available_skills=settings.agent.max_available_skills,
            max_resources_per_skill=settings.agent.max_resources_per_skill,
            max_instruction_characters=(settings.agent.max_skill_instruction_characters),
            max_resource_characters=settings.agent.max_skill_resource_characters,
            max_accepted_shared_skills=settings.agent.max_accepted_shared_skills,
            max_share_recipients_per_call=(settings.agent.max_skill_share_recipients_per_call),
            notifier=agent_skill_notifier,
        )
        agent_tools.extend(skill_library_tools(agent_skill_library))
    else:
        enabled_agent_tools = tuple(
            name for name in enabled_agent_tools if name not in SKILL_AUTHORING_AGENT_TOOLS
        )
    agent_tool_registry = ToolRegistry(agent_tools)
    agent_diagnostic_renderer = OopzAgentDiagnosticRenderer(
        {
            descriptor.name: descriptor.display_name
            for descriptor in agent_tool_registry.descriptors()
        }
    )
    agent_skill_availability = SkillAvailabilityService(
        max_available_skills=settings.agent.max_available_skills,
    )
    agent_tool_authorization = AgentToolAuthorizationAdapter(authorization)
    agent_tool_policy = ToolPolicy(agent_tool_authorization)
    agent_tool_availability = ToolAvailabilityService(
        agent_tool_registry,
        channel_settings,
        enabled_agent_tools,
        agent_tool_policy,
    )
    agent_tool_executor = ToolExecutor(
        agent_tool_registry,
        agent_tool_policy,
        SqlAlchemyToolExecutionRepository(sessions),
    )
    direct_tools = DirectToolService(
        settings.agent,
        agent_tool_registry,
        agent_tool_availability,
        agent_selection,
        agent_tool_policy,
    )
    agent_models = AgentModelRegistry(
        agent_catalog,
        default_max_retries=settings.agent.provider_max_retries,
    )
    agent_summary_tasks = TaskSupervisor(lambda thread_id: f"agent-summary:{thread_id}")
    agent_summary_service = ThreadSummaryService(
        settings.agent,
        PydanticAiThreadSummarizer(agent_models, settings.agent),
        agent_threads,
        agent_messages,
    )
    agent_engine = PydanticAiAgentEngine(
        agent_models,
        agent_tool_executor,
    )
    agent_run_service = AgentRunService(
        agent_engine,
        agent_runs,
        agent_messages,
        heartbeat_interval_seconds=max(
            1.0,
            min(10.0, settings.agent.stale_run_after_seconds / 3),
        ),
        health=health,
    )
    agent_chat = AgentConversationService(
        settings.agent,
        settings.chat,
        agent_run_service,
        agent_catalog,
        agent_selection,
        selection_repository,
        agent_threads,
        agent_messages,
        agent_tool_availability,
        agent_skill_repository if settings.agent.skills_enabled else None,
        agent_skill_availability if settings.agent.skills_enabled else None,
        context_builder=agent_context,
        summary_service=agent_summary_service,
        summary_tasks=agent_summary_tasks,
        media_ingest=agent_media_ingest,
        health=health,
    )
    return AgentComponents(
        agent_catalog=agent_catalog,
        agent_catalog_admin=agent_catalog_admin,
        agent_threads=agent_threads,
        agent_runs=agent_runs,
        agent_messages=agent_messages,
        agent_memory_repository=agent_memory_repository,
        agent_memory=agent_memory,
        agent_image_client=agent_image_client,
        agent_media_ingest=agent_media_ingest,
        agent_context=agent_context,
        agent_selection=agent_selection,
        agent_skill_repository=agent_skill_repository,
        agent_skill_notifier=agent_skill_notifier,
        agent_skill_library=agent_skill_library,
        agent_tool_registry=agent_tool_registry,
        agent_diagnostic_renderer=agent_diagnostic_renderer,
        agent_skill_availability=agent_skill_availability,
        agent_tool_authorization=agent_tool_authorization,
        agent_tool_policy=agent_tool_policy,
        agent_tool_availability=agent_tool_availability,
        agent_tool_executor=agent_tool_executor,
        direct_tools=direct_tools,
        agent_models=agent_models,
        agent_summary_tasks=agent_summary_tasks,
        agent_summary_service=agent_summary_service,
        agent_engine=agent_engine,
        agent_run_service=agent_run_service,
        agent_chat=agent_chat,
        channel_settings=channel_settings,
    )
