"""User-facing orchestration for bounded Agent conversations."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from cywl_oopz.core.errors import (
    DatabaseError,
    ProviderError,
    ProviderSelectionError,
    ProviderTimeoutError,
)
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
    AgentMessage,
    AgentRun,
    AgentRunLimits,
    AgentRunRequest,
    AgentRunResult,
    AgentRunState,
    AgentStopReason,
    AgentThread,
    ModelCapability,
    ModelSelection,
)
from .ports import (
    AgentEngine,
    AgentMessageRepository,
    AgentRunRepository,
    AgentThreadRepository,
    ModelSelectionRepository,
)
from .selection import ProviderSelectionService
from .summarization import ThreadSummaryService
from .tools.policy import AvailableTool, ToolAvailabilityService

logger = logging.getLogger(__name__)


class AgentConversationService:
    """Coordinate short DB transactions around model I/O without exposing OOPZ."""

    def __init__(
        self,
        settings: AgentSettings,
        chat_settings: ChatSettings,
        engine: AgentEngine,
        catalog: ReloadableProviderCatalog,
        selection: ProviderSelectionService,
        selection_repository: ModelSelectionRepository,
        threads: AgentThreadRepository,
        runs: AgentRunRepository,
        messages: AgentMessageRepository,
        tool_availability: ToolAvailabilityService | None = None,
        context_builder: AgentContextBuilder | None = None,
        summary_service: ThreadSummaryService | None = None,
        summary_tasks: TaskSupervisor[UUID] | None = None,
        *,
        rate_limits: RateLimitService | None = None,
        locks: ConversationLockPool | None = None,
        health: HealthRegistry | None = None,
    ) -> None:
        self._settings = settings
        self._engine = engine
        self._catalog = catalog
        self._selection = selection
        self._selection_repository = selection_repository
        self._threads = threads
        self._runs = runs
        self._messages = messages
        self._tool_availability = tool_availability
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
                )
                enabled_tools = (
                    await self._tool_availability.names(identity, selection.model)
                    if self._tool_availability is not None
                    else ()
                )
                context = await self._context_builder.build(thread, identity)
                run_id = uuid4()
                state = AgentRunState(run_id).start(now)
                limits = self._run_limits()
                await self._runs.add(
                    AgentRun(
                        id=run_id,
                        thread_id=thread.id,
                        provider_id=selection.model.provider_id,
                        model_id=selection.model.model_id,
                        selection_source=selection.source,
                        limits=limits,
                        state=state,
                        heartbeat_at=now,
                    )
                )
                await self._messages.append(
                    thread.id,
                    run_id,
                    (AgentMessage("user", "text", {"text": content}),),
                )
                request = AgentRunRequest(
                    run_id=run_id,
                    thread_id=thread.id,
                    identity=identity,
                    model=selection.model,
                    prompt=content,
                    context=context,
                    enabled_tools=enabled_tools,
                    limits=limits,
                )
                logger.info(
                    "Agent run started: run=%s conversation=%s model=%s/%s "
                    "context_messages=%s tools=%s",
                    run_id,
                    conversation,
                    selection.model.provider_alias,
                    selection.model.model_alias,
                    len(context),
                    len(enabled_tools),
                )
                try:
                    result = await self._engine.run(request, progress)
                except asyncio.CancelledError:
                    logger.info("Agent run cancelled: run=%s conversation=%s", run_id, conversation)
                    await self._finish_after_interrupt(
                        state,
                        AgentStopReason.CANCELLED,
                        "cancelled",
                    )
                    raise
                except ProviderTimeoutError as exc:
                    logger.warning(
                        "Agent run timed out: run=%s conversation=%s error=%s",
                        run_id,
                        conversation,
                        exception_kind(exc),
                    )
                    await self._finish_after_interrupt(
                        state,
                        AgentStopReason.TIMEOUT,
                        "provider_timeout",
                    )
                    self._mark_health(HealthState.DEGRADED, "request timed out")
                    raise
                except ProviderError as exc:
                    logger.warning(
                        "Agent provider failed: run=%s conversation=%s error=%s",
                        run_id,
                        conversation,
                        exception_kind(exc),
                    )
                    await self._finish_after_interrupt(
                        state,
                        AgentStopReason.PROVIDER_ERROR,
                        "provider_error",
                    )
                    self._mark_health(HealthState.DEGRADED, "request failed")
                    raise
                except Exception as exc:
                    logger.error(
                        "Agent run failed unexpectedly: run=%s conversation=%s error=%s",
                        run_id,
                        conversation,
                        exception_kind(exc),
                    )
                    await self._finish_after_interrupt(
                        state,
                        AgentStopReason.INVALID_OUTPUT,
                        "agent_error",
                    )
                    self._mark_health(HealthState.DEGRADED, "Agent run failed")
                    raise

                finished_at = datetime.now(UTC)
                await self._messages.append(
                    thread.id,
                    run_id,
                    result.intermediate_messages
                    + (
                        AgentMessage(
                            "assistant",
                            "text",
                            {"text": result.output},
                            input_tokens=result.input_tokens,
                            output_tokens=result.output_tokens,
                        ),
                    ),
                )
                await self._runs.finish(
                    state.finish(result.stop_reason, finished_at),
                    usage=self._usage(result),
                )
                await self._threads.refresh_expiry(
                    thread.id,
                    finished_at + timedelta(seconds=self._settings.session_ttl_seconds),
                )
                if self._summary_service is not None and self._summary_tasks is not None:
                    self._summary_tasks.start(
                        thread.id,
                        self._summary_service.maybe_summarize(
                            thread,
                            selection.model,
                        ),
                    )
                self._mark_health(HealthState.HEALTHY, "last Agent run succeeded")
                logger.info(
                    "Agent run completed: run=%s conversation=%s reason=%s elapsed_seconds=%.3f "
                    "model_requests=%s tool_calls=%s input_tokens=%s output_tokens=%s",
                    run_id,
                    conversation,
                    result.stop_reason.value,
                    time.perf_counter() - started_at,
                    result.model_requests,
                    result.tool_calls,
                    result.input_tokens,
                    result.output_tokens,
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
        """Compatibility path for `!model`, accepting `provider/model` or a unique alias."""
        choice = model.strip()
        if not choice:
            raise ValueError("Model choice must not be empty")
        snapshot = self._catalog.snapshot
        if "/" in choice:
            provider_alias, model_alias = choice.split("/", 1)
            reference = snapshot.find_selectable(
                provider_alias,
                model_alias,
                required_capabilities=self._required_capabilities,
            )
        else:
            matches = tuple(
                item
                for item in snapshot.selectable_models(
                    required_capabilities=self._required_capabilities,
                )
                if item.model_alias.casefold() == choice.casefold()
            )
            reference = matches[0] if len(matches) == 1 else None
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

    def list_models(self) -> tuple[str, ...]:
        """List safe aliases only; endpoints and API keys never enter chat replies."""
        return tuple(
            f"{model.provider_alias}/{model.model_alias}"
            for model in self._catalog.snapshot.selectable_models(
                required_capabilities=self._required_capabilities,
            )
        )

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

    async def _finish_after_interrupt(
        self,
        state: AgentRunState,
        reason: AgentStopReason,
        error_code: str,
    ) -> None:
        try:
            await self._runs.finish(
                state.finish(reason, datetime.now(UTC)),
                usage={},
                error_code=error_code,
            )
        except DatabaseError as exc:
            logger.warning(
                "Could not persist interrupted Agent run: run=%s reason=%s error=%s",
                state.run_id,
                reason.value,
                exception_kind(exc),
            )

    @staticmethod
    def _usage(result: AgentRunResult) -> dict[str, object]:
        return {
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "model_requests": result.model_requests,
            "tool_calls": result.tool_calls,
        }

    def _mark_health(self, state: HealthState, detail: str) -> None:
        if self._health is not None:
            self._health.mark("llm", state, detail)

    @staticmethod
    def _conversation_ref(key: ConversationKey) -> str:
        return opaque_ref(key.scope, key.area_id, key.channel_id, key.person_id)
