"""Reusable primitive for one durable, bounded Agent engine run."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID, uuid4

from cywl_oopz.core.errors import AgentInternalError, ProviderError, ProviderTimeoutError
from cywl_oopz.core.health import HealthRegistry, HealthState
from cywl_oopz.core.observability import exception_kind, opaque_ref
from cywl_oopz.features.chat.progress import ProgressSink, RunTraceSink

from .input import AgentUserInput
from .models import (
    AgentIdentity,
    AgentMessage,
    AgentModelRef,
    AgentRun,
    AgentRunLimits,
    AgentRunRequest,
    AgentRunResult,
    AgentRunState,
    AgentStopReason,
    AgentThread,
    ModelSelectionSource,
)
from .ports import AgentEngine, AgentMessageRepository, AgentRunRepository
from .skills.scope import AgentSkillRunScope

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AgentRunSpec:
    """All trusted, already-resolved inputs for one isolated Agent run."""

    thread: AgentThread
    identity: AgentIdentity
    prompt: str
    model: AgentModelRef
    selection_source: ModelSelectionSource
    enabled_tools: tuple[str, ...]
    limits: AgentRunLimits
    context: tuple[AgentMessage, ...]
    skill_scope: AgentSkillRunScope | None = field(default=None, compare=False, repr=False)
    user_input: AgentUserInput | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ValueError("Agent run prompt must not be empty")


@dataclass(frozen=True, slots=True)
class AgentRunOutcome:
    run_id: UUID
    result: AgentRunResult
    elapsed_seconds: float
    persistence_degraded: bool = False


class AgentRunService:
    """Persist and execute one run without conversation or transport policy."""

    def __init__(
        self,
        engine: AgentEngine,
        runs: AgentRunRepository,
        messages: AgentMessageRepository,
        *,
        heartbeat_interval_seconds: float = 10.0,
        health: HealthRegistry | None = None,
    ) -> None:
        if heartbeat_interval_seconds <= 0:
            raise ValueError("Agent heartbeat interval must be positive")
        self._engine = engine
        self._runs = runs
        self._messages = messages
        self._heartbeat_interval_seconds = heartbeat_interval_seconds
        self._health = health

    async def run(
        self,
        spec: AgentRunSpec,
        progress: ProgressSink | None = None,
    ) -> AgentRunOutcome:
        started_monotonic = time.perf_counter()
        started_at = datetime.now(UTC)
        run_id = uuid4()
        state = AgentRunState(run_id).start(started_at)
        request = AgentRunRequest(
            run_id=run_id,
            thread_id=spec.thread.id,
            identity=spec.identity,
            model=spec.model,
            prompt=spec.prompt,
            context=spec.context,
            enabled_tools=spec.enabled_tools,
            limits=spec.limits,
            skill_scope=spec.skill_scope,
            user_input=spec.user_input,
        )
        run_ref = opaque_ref(str(run_id))
        conversation_ref = opaque_ref(
            spec.identity.conversation.scope,
            spec.identity.conversation.area_id,
            spec.identity.conversation.channel_id,
            spec.identity.person_id,
        )
        await self._runs.add(
            AgentRun(
                id=run_id,
                thread_id=spec.thread.id,
                provider_id=spec.model.provider_id,
                model_id=spec.model.model_id,
                selection_source=spec.selection_source,
                limits=spec.limits,
                state=state,
                heartbeat_at=started_at,
            )
        )
        if isinstance(progress, RunTraceSink):
            try:
                await progress.bind_run(run_id)
            except Exception as exc:
                logger.warning(
                    "Could not bind Agent run to progress session: run=%s error=%s",
                    opaque_ref(str(run_id)),
                    exception_kind(exc),
                )
        try:
            append_user_input = getattr(self._messages, "append_user_input", None)
            if (
                spec.user_input is not None
                and spec.user_input.has_images
                and callable(append_user_input)
            ):
                await append_user_input(spec.thread.id, run_id, spec.user_input)
            else:
                await self._messages.append(
                    spec.thread.id,
                    run_id,
                    (AgentMessage("user", "text", {"text": spec.prompt}),),
                )
        except asyncio.CancelledError:
            await self._finish_after_interrupt(
                state,
                AgentStopReason.CANCELLED,
                "cancelled",
            )
            raise
        except Exception as exc:
            logger.exception(
                "Could not persist Agent user message: run=%s conversation=%s "
                "phase=user_persist error=%s",
                run_ref,
                conversation_ref,
                exception_kind(exc),
            )
            await self._finish_after_interrupt(
                state,
                AgentStopReason.INVALID_OUTPUT,
                "user_message_persistence_error",
            )
            self._mark_health(HealthState.DEGRADED, "Agent message persistence failed")
            raise
        logger.info(
            "Agent run started: run=%s conversation=%s model=%s/%s context_messages=%s tools=%s",
            run_ref,
            conversation_ref,
            spec.model.provider_alias,
            spec.model.model_alias,
            len(spec.context),
            len(spec.enabled_tools),
        )
        heartbeat = asyncio.create_task(
            self._heartbeat(run_id),
            name=f"agent-run-heartbeat:{run_id}",
        )
        try:
            try:
                result = await self._engine.run(request, progress)
            except asyncio.CancelledError:
                logger.info(
                    "Agent run cancelled: run=%s conversation=%s", run_ref, conversation_ref
                )
                await self._finish_after_interrupt(
                    state,
                    AgentStopReason.CANCELLED,
                    "cancelled",
                )
                raise
            except ProviderTimeoutError as exc:
                logger.warning(
                    "Agent run timed out: run=%s conversation=%s error=%s",
                    run_ref,
                    conversation_ref,
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
                    run_ref,
                    conversation_ref,
                    exception_kind(exc),
                )
                await self._finish_after_interrupt(
                    state,
                    AgentStopReason.PROVIDER_ERROR,
                    "provider_error",
                )
                self._mark_health(HealthState.DEGRADED, "request failed")
                raise
            except AgentInternalError as exc:
                logger.error(
                    "Agent adapter failed internally: run=%s conversation=%s phase=engine error=%s",
                    run_ref,
                    conversation_ref,
                    exception_kind(exc),
                    exc_info=True,
                )
                await self._finish_after_interrupt(
                    state,
                    AgentStopReason.INVALID_OUTPUT,
                    "agent_internal",
                )
                self._mark_health(HealthState.DEGRADED, "Agent adapter failed")
                raise
            except Exception as exc:
                logger.error(
                    "Agent run failed unexpectedly: run=%s conversation=%s error=%s",
                    run_ref,
                    conversation_ref,
                    exception_kind(exc),
                    exc_info=True,
                )
                await self._finish_after_interrupt(
                    state,
                    AgentStopReason.INVALID_OUTPUT,
                    "agent_error",
                )
                self._mark_health(HealthState.DEGRADED, "Agent run failed")
                raise
        finally:
            heartbeat.cancel()
            try:
                await heartbeat
            except asyncio.CancelledError:
                pass

        persistence_degraded = not await self._persist_result(
            spec,
            state,
            result,
            run_ref=run_ref,
            conversation_ref=conversation_ref,
        )
        elapsed_seconds = time.perf_counter() - started_monotonic
        if persistence_degraded:
            self._mark_health(HealthState.DEGRADED, "Agent result persistence failed")
        else:
            self._mark_health(HealthState.HEALTHY, "last Agent run succeeded")
        logger.info(
            "Agent run completed: run=%s conversation=%s reason=%s elapsed_seconds=%.3f "
            "model_requests=%s tool_calls=%s input_tokens=%s output_tokens=%s "
            "persistence_degraded=%s",
            run_ref,
            conversation_ref,
            result.stop_reason.value,
            elapsed_seconds,
            result.model_requests,
            result.tool_calls,
            result.input_tokens,
            result.output_tokens,
            persistence_degraded,
        )
        return AgentRunOutcome(run_id, result, elapsed_seconds, persistence_degraded)

    async def _persist_result(
        self,
        spec: AgentRunSpec,
        state: AgentRunState,
        result: AgentRunResult,
        *,
        run_ref: str,
        conversation_ref: str,
    ) -> bool:
        try:
            messages = result.intermediate_messages + (
                AgentMessage(
                    "assistant",
                    "text",
                    {"text": result.output},
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                ),
            )
            await self._messages.append(
                spec.thread.id,
                state.run_id,
                messages,
            )
        except asyncio.CancelledError:
            await self._finish_after_interrupt(
                state,
                AgentStopReason.CANCELLED,
                "cancelled",
            )
            raise
        except Exception as exc:
            logger.exception(
                "Could not persist Agent result messages: run=%s conversation=%s "
                "phase=result_persist error=%s",
                run_ref,
                conversation_ref,
                exception_kind(exc),
            )
            await self._finish_after_interrupt(
                state,
                AgentStopReason.INVALID_OUTPUT,
                "result_message_persistence_error",
            )
            return False

        try:
            await self._runs.finish(
                state.finish(result.stop_reason, datetime.now(UTC)),
                usage=self._usage(result),
            )
        except asyncio.CancelledError:
            await self._finish_after_interrupt(
                state,
                AgentStopReason.CANCELLED,
                "cancelled",
            )
            raise
        except Exception as exc:
            logger.exception(
                "Could not persist successful Agent run: run=%s conversation=%s "
                "phase=run_finish error=%s",
                run_ref,
                conversation_ref,
                exception_kind(exc),
            )
            return False
        return True

    async def _heartbeat(self, run_id: UUID) -> None:
        while True:
            await asyncio.sleep(self._heartbeat_interval_seconds)
            try:
                alive = await self._runs.heartbeat(run_id, datetime.now(UTC))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Agent run heartbeat failed: run=%s error=%s",
                    opaque_ref(str(run_id)),
                    exception_kind(exc),
                )
                continue
            if not alive:
                logger.warning(
                    "Agent run heartbeat lost ownership: run=%s", opaque_ref(str(run_id))
                )
                return

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
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                "Could not persist interrupted Agent run: run=%s reason=%s "
                "phase=run_finish error=%s",
                opaque_ref(str(state.run_id)),
                reason.value,
                exception_kind(exc),
                exc_info=True,
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
