"""User-facing orchestration for bounded Agent conversations."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from cywl_oopz.core.errors import ProviderSelectionError
from cywl_oopz.core.health import HealthRegistry, HealthState
from cywl_oopz.core.observability import exception_kind, opaque_ref
from cywl_oopz.core.tasks import TaskSupervisor
from cywl_oopz.features.chat.history import HistoryTrimmer
from cywl_oopz.features.chat.locks import ConversationLockPool
from cywl_oopz.features.chat.models import (
    ChatInvocation,
    ChatResponse,
    ChatStatus,
    ConversationKey,
)
from cywl_oopz.features.chat.progress import (
    ConversationProgressEvent,
    ProgressKind,
    ProgressSink,
    emit_progress,
)
from cywl_oopz.features.chat.rate_limit import RateLimitService
from cywl_oopz.settings import AgentSettings, ChatSettings

from .catalog import ReloadableProviderCatalog
from .context import AgentContextBuilder
from .models import (
    AgentIdentity,
    AgentRunLimits,
    AgentThread,
    ModelCapability,
    ModelCatalogView,
    ModelSelection,
    SelectableModel,
)
from .ports import (
    AgentMessageRepository,
    AgentThreadRepository,
    ModelSelectionRepository,
)
from .run_service import AgentRunService, AgentRunSpec
from .selection import ProviderSelectionService
from .skills.availability import SkillAvailabilityService
from .skills.models import AgentSkillDiscovery
from .skills.ports import AgentSkillReadRepository
from .skills.scope import AgentSkillRunScope
from .skills.tools import SKILL_TOOL_NAMES
from .summarization import ThreadSummaryService
from .tools.policy import AvailableTool, ToolAvailabilityService

logger = logging.getLogger(__name__)


class AgentConversationService:
    """Coordinate short DB transactions around model I/O without exposing OOPZ."""

    def __init__(
        self,
        settings: AgentSettings,
        chat_settings: ChatSettings,
        run_service: AgentRunService,
        catalog: ReloadableProviderCatalog,
        selection: ProviderSelectionService,
        selection_repository: ModelSelectionRepository,
        threads: AgentThreadRepository,
        messages: AgentMessageRepository,
        tool_availability: ToolAvailabilityService | None = None,
        skill_repository: AgentSkillReadRepository | None = None,
        skill_availability: SkillAvailabilityService | None = None,
        context_builder: AgentContextBuilder | None = None,
        summary_service: ThreadSummaryService | None = None,
        summary_tasks: TaskSupervisor[UUID] | None = None,
        *,
        rate_limits: RateLimitService | None = None,
        locks: ConversationLockPool | None = None,
        health: HealthRegistry | None = None,
    ) -> None:
        self._settings = settings
        self._run_service = run_service
        self._catalog = catalog
        self._selection = selection
        self._selection_repository = selection_repository
        self._threads = threads
        self._messages = messages
        self._tool_availability = tool_availability
        self._skill_repository = skill_repository
        self._skill_availability = skill_availability
        self._context_builder = context_builder or AgentContextBuilder(settings, messages)
        self._summary_service = summary_service
        self._summary_tasks = summary_tasks
        self._required_capabilities = (
            frozenset({ModelCapability.TOOL_CALLING}) if settings.enabled_tools else frozenset()
        )
        self._rate_limits = rate_limits or RateLimitService(chat_settings)
        self._locks = locks or ConversationLockPool()
        self._input_validator = HistoryTrimmer(
            max_messages=settings.max_history_messages,
            max_characters=settings.max_history_characters,
        )
        self._health = health

    @property
    def enabled(self) -> bool:
        """Agent mode is itself the text-chat feature flag."""
        return self._settings.enabled

    async def ask(
        self,
        key: ConversationKey,
        prompt: str,
        *,
        invocation: ChatInvocation | None = None,
        progress: ProgressSink | None = None,
    ) -> ChatResponse:
        """Run one bounded Agent turn with durable run, message, and tool records."""
        started_at = time.perf_counter()
        content = prompt.strip()
        if not content:
            raise ValueError("Agent prompt must not be empty")
        self._input_validator.validate_input(content)
        conversation = self._conversation_ref(key)
        logger.info(
            "Agent request received: conversation=%s prompt_characters=%s",
            conversation,
            len(content),
        )
        await emit_progress(progress, ConversationProgressEvent(ProgressKind.ACCEPTED))

        async with self._locks.hold(key):
            async with await self._rate_limits.acquire(key):
                now = datetime.now(UTC)
                thread = await self._get_or_create_thread(key, now)
                selection = await self._selection.resolve(
                    key,
                    required_capabilities=self._required_capabilities,
                )
                identity = AgentIdentity(
                    key.person_id,
                    key,
                    source_message_id=(
                        invocation.source_message_id if invocation is not None else ""
                    ),
                    transport_channel_id=(
                        invocation.transport_channel_id if invocation is not None else ""
                    ),
                    mentioned_person_ids=(
                        invocation.mentioned_person_ids if invocation is not None else ()
                    ),
                )
                enabled_tools = (
                    await self._tool_availability.names(identity, selection.model)
                    if self._tool_availability is not None
                    else ()
                )
                available_skills: tuple[AgentSkillDiscovery, ...] = ()
                skill_scope: AgentSkillRunScope | None = None
                if self._skill_repository is not None and self._skill_availability is not None:
                    discoveries = await self._discover_skills(identity.person_id)
                    available_skills = self._skill_availability.resolve(
                        discoveries,
                        enabled_tools,
                    )
                    if available_skills:
                        skill_scope = AgentSkillRunScope(
                            self._skill_repository,
                            identity.person_id,
                            available_skills,
                            max_activations=self._settings.max_skill_activations,
                            max_resources=self._settings.max_skill_resources,
                            max_instruction_characters=(
                                self._settings.max_skill_instruction_characters
                            ),
                            max_resource_characters=(self._settings.max_skill_resource_characters),
                            max_context_characters=self._settings.max_skill_context_characters,
                        )
                    else:
                        enabled_tools = tuple(
                            name for name in enabled_tools if name not in SKILL_TOOL_NAMES
                        )
                context = await self._context_builder.build(
                    thread,
                    identity,
                    available_skills=available_skills,
                )
                limits = self._run_limits()
                outcome = await self._run_service.run(
                    AgentRunSpec(
                        thread=thread,
                        identity=identity,
                        prompt=content,
                        model=selection.model,
                        selection_source=selection.source,
                        enabled_tools=enabled_tools,
                        limits=limits,
                        context=context,
                        skill_scope=skill_scope,
                    ),
                    progress,
                )
                result = outcome.result
                finished_at = datetime.now(UTC)
                try:
                    await self._threads.refresh_expiry(
                        thread.id,
                        finished_at + timedelta(seconds=self._settings.session_ttl_seconds),
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning(
                        "Could not refresh Agent thread expiry: run=%s conversation=%s "
                        "phase=thread_expiry error=%s",
                        opaque_ref(str(outcome.run_id)),
                        conversation,
                        exception_kind(exc),
                        exc_info=True,
                    )
                if (
                    not outcome.persistence_degraded
                    and self._summary_service is not None
                    and self._summary_tasks is not None
                ):
                    summary_operation = self._summary_service.maybe_summarize(
                        thread,
                        selection.model,
                    )
                    try:
                        self._summary_tasks.start(thread.id, summary_operation)
                    except Exception as exc:
                        summary_operation.close()
                        logger.warning(
                            "Could not schedule Agent thread summary: run=%s "
                            "conversation=%s phase=summary_schedule error=%s",
                            opaque_ref(str(outcome.run_id)),
                            conversation,
                            exception_kind(exc),
                            exc_info=True,
                        )
                logger.debug(
                    "Agent conversation turn completed: run=%s conversation=%s skills=%s",
                    opaque_ref(str(outcome.run_id)),
                    conversation,
                    len(available_skills),
                )
                return ChatResponse(
                    content=result.output,
                    model=f"{selection.model.provider_alias}/{selection.model.model_alias}",
                    finish_reason=result.stop_reason.value,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    elapsed_seconds=time.perf_counter() - started_at,
                    model_requests=result.model_requests,
                    tool_calls=result.tool_calls,
                )

    async def clear(self, key: ConversationKey) -> None:
        """Delete this scoped Agent thread and all cascading records."""
        async with self._locks.hold(key):
            await self._threads.delete(key)
        logger.info("Agent conversation cleared: conversation=%s", self._conversation_ref(key))

    async def select_model(self, key: ConversationKey, model: str) -> str:
        """Select a qualified model or a model alias within the current Provider."""
        choice = model.strip()
        if not choice:
            raise ValueError("Model choice must not be empty")
        await self._catalog.refresh()
        snapshot = self._catalog.snapshot
        if "/" in choice:
            provider_alias, model_alias = choice.split("/", 1)
            reference = snapshot.find_selectable(
                provider_alias,
                model_alias,
                required_capabilities=self._required_capabilities,
            )
        else:
            selectable = snapshot.selectable_models(
                required_capabilities=self._required_capabilities,
            )
            current = await self._selection.resolve(
                key,
                required_capabilities=self._required_capabilities,
                refresh_catalog=False,
            )
            provider_matches = tuple(
                item
                for item in selectable
                if item.provider_id == current.model.provider_id
                and item.model_alias.casefold() == choice.casefold()
            )
            if len(provider_matches) == 1:
                reference = provider_matches[0]
            else:
                global_matches = tuple(
                    item for item in selectable if item.model_alias.casefold() == choice.casefold()
                )
                reference = global_matches[0] if len(global_matches) == 1 else None
        if reference is None:
            raise ValueError("The requested Agent model is not available")
        await self._select_thread_model(key, reference.model_id)
        logger.info(
            "Agent model selected: conversation=%s model=%s/%s",
            self._conversation_ref(key),
            reference.provider_alias,
            reference.model_alias,
        )
        return f"{reference.provider_alias}/{reference.model_alias}"

    async def select_provider(
        self,
        key: ConversationKey,
        provider_alias: str,
        model_alias: str | None,
        *,
        user_default: bool,
    ) -> str:
        """Select a Provider default or named model for a thread or user."""
        await self._catalog.refresh()
        reference = self._catalog.snapshot.find_selectable(
            provider_alias,
            model_alias,
            required_capabilities=self._required_capabilities,
        )
        if reference is None:
            raise ValueError("The requested Provider/model is not available")
        if user_default:
            await self._selection_repository.set_user_model(key.person_id, reference.model_id)
        else:
            await self._select_thread_model(key, reference.model_id)
        logger.info(
            "Agent provider selected: scope=%s conversation=%s model=%s/%s",
            "user" if user_default else "thread",
            self._conversation_ref(key),
            reference.provider_alias,
            reference.model_alias,
        )
        return f"{reference.provider_alias}/{reference.model_alias}"

    async def model_catalog_view(self, key: ConversationKey) -> ModelCatalogView:
        """Read PostgreSQL once for an interactive selection and choice view."""
        await self._catalog.refresh()
        selection = await self._selection.resolve(
            key,
            required_capabilities=self._required_capabilities,
            refresh_catalog=False,
        )
        return ModelCatalogView(selection, self._model_choices())

    async def list_model_choices(self) -> tuple[SelectableModel, ...]:
        """Read PostgreSQL and list safe display metadata without credentials."""
        await self._catalog.refresh()
        return self._model_choices()

    def _model_choices(self) -> tuple[SelectableModel, ...]:
        """Project the operation snapshot into safe user-facing choices."""
        snapshot = self._catalog.snapshot
        return tuple(
            SelectableModel(
                provider_alias=model.provider_alias,
                provider_display_name=snapshot.providers[model.provider_id].display_name,
                model_alias=model.model_alias,
                model_display_name=snapshot.models[model.model_id].display_name,
                is_provider_default=snapshot.models[model.model_id].is_provider_default,
            )
            for model in snapshot.selectable_models(
                required_capabilities=self._required_capabilities,
            )
        )

    async def list_models(self) -> tuple[str, ...]:
        """Read PostgreSQL and return the former safe alias-only representation."""
        return tuple(choice.qualified_alias for choice in await self.list_model_choices())

    async def current_selection(self, key: ConversationKey) -> ModelSelection:
        """Resolve current selection for status and Provider commands."""
        return await self._selection.resolve(
            key,
            required_capabilities=self._required_capabilities,
        )

    async def available_tools(self, key: ConversationKey) -> tuple[AvailableTool, ...]:
        """Return the exact safe tool set visible to a run in this conversation."""
        if self._tool_availability is None:
            return ()
        selection = await self.current_selection(key)
        return await self._tool_availability.resolve(
            AgentIdentity(key.person_id, key),
            selection.model,
        )

    async def available_skills(
        self,
        key: ConversationKey,
    ) -> tuple[AgentSkillDiscovery, ...]:
        """Return Skills visible after applying this conversation's actual tool set."""
        if (
            self._tool_availability is None
            or self._skill_repository is None
            or self._skill_availability is None
        ):
            return ()
        selection = await self.current_selection(key)
        identity = AgentIdentity(key.person_id, key)
        enabled_tools = await self._tool_availability.names(identity, selection.model)
        discoveries = await self._discover_skills(identity.person_id)
        return self._skill_availability.resolve(
            discoveries,
            enabled_tools,
        )

    async def _discover_skills(
        self,
        person_id: str,
    ) -> tuple[AgentSkillDiscovery, ...]:
        if self._skill_repository is None:
            return ()
        try:
            discoveries = await self._skill_repository.list_accessible(person_id)
        except Exception:
            if self._health is not None:
                self._health.mark("skills", HealthState.DEGRADED, "library query failed")
            raise
        if self._health is not None:
            self._health.mark("skills", HealthState.HEALTHY, "library queried")
        return discoveries

    async def status(self, key: ConversationKey) -> ChatStatus:
        """Return safe thread metadata compatible with existing chat controllers."""
        now = datetime.now(UTC)
        thread = await self._load_active_thread(key, now)
        try:
            selection = await self._selection.resolve(
                key,
                required_capabilities=self._required_capabilities,
            )
            model_name = f"{selection.model.provider_alias}/{selection.model.model_alias}"
        except ProviderSelectionError:
            model_name = ""
        return ChatStatus(
            enabled=True,
            active=thread is not None,
            model=model_name,
            history_message_count=(
                await self._messages.count(thread.id) if thread is not None else 0
            ),
            expires_at=thread.expires_at if thread is not None else None,
            cooldown_seconds=await self._rate_limits.cooldown_remaining(key),
        )

    async def _select_thread_model(self, key: ConversationKey, model_id: UUID) -> None:
        async with self._locks.hold(key):
            await self._get_or_create_thread(key, datetime.now(UTC))
            await self._threads.set_selected_model(key, model_id)

    async def _get_or_create_thread(
        self,
        key: ConversationKey,
        now: datetime,
    ) -> AgentThread:
        thread = await self._load_active_thread(key, now)
        if thread is not None:
            return thread
        thread = AgentThread(
            id=uuid4(),
            key=key,
            selected_model_id=None,
            expires_at=now + timedelta(seconds=self._settings.session_ttl_seconds),
        )
        await self._threads.add(thread)
        logger.debug(
            "Created Agent thread: thread=%s conversation=%s",
            thread.id,
            self._conversation_ref(key),
        )
        return thread

    async def _load_active_thread(
        self,
        key: ConversationKey,
        now: datetime,
    ) -> AgentThread | None:
        thread = await self._threads.get(key)
        if thread is not None and thread.is_expired(now):
            await self._threads.delete(key)
            logger.info(
                "Expired Agent thread removed: thread=%s conversation=%s",
                thread.id,
                self._conversation_ref(key),
            )
            return None
        return thread

    def _run_limits(self) -> AgentRunLimits:
        return AgentRunLimits(
            timeout_seconds=self._settings.timeout_seconds,
            max_model_requests=self._settings.max_model_requests,
            max_tool_calls=self._settings.max_tool_calls,
            max_total_tokens=self._settings.max_total_tokens,
            max_parallel_tools=self._settings.max_parallel_tools,
        )

    @staticmethod
    def _conversation_ref(key: ConversationKey) -> str:
        return opaque_ref(key.scope, key.area_id, key.channel_id, key.person_id)
