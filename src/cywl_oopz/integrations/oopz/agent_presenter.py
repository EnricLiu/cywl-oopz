"""Single-message OOPZ presenter for one conversational Agent loop."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import UUID

from oopz_sdk.exceptions import OopzConnectionError, OopzRateLimitError

from cywl_oopz.commands.models import CommandRequest
from cywl_oopz.core.observability import opaque_ref
from cywl_oopz.features.admin.models import OutboundMessageKind, OutboundMessageState
from cywl_oopz.features.agent.display import (
    AgentLoopReducer,
    AgentLoopViewState,
    DisplayPhase,
    ToolStepStatus,
)
from cywl_oopz.features.chat.models import ChatResponse
from cywl_oopz.features.chat.progress import (
    ConversationProgressEvent,
    ConversationProgressSession,
    NoopProgressSession,
    ProgressKind,
)

from .active_presentations import ActivePresentationRegistry
from .editable_messages import (
    EditableMessageRef,
    MessageAddress,
    OopzEditableMessageGateway,
)
from .message_renderer import OopzMessageRenderer, OopzRenderContext

logger = logging.getLogger(__name__)

_SEMANTIC_KINDS = frozenset(
    {
        ProgressKind.ACCEPTED,
        ProgressKind.THINKING,
        ProgressKind.MODEL_RETRY,
        ProgressKind.TEXT_RESET,
        ProgressKind.TOOL_STARTED,
        ProgressKind.TOOL_UPDATED,
        ProgressKind.TOOL_SUCCEEDED,
        ProgressKind.TOOL_FAILED,
        ProgressKind.COMPLETED,
        ProgressKind.FAILED,
        ProgressKind.CANCELLED,
    }
)


def _diagnostic_snapshot(
    state: AgentLoopViewState,
    provider_retries: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "version": 1,
        "phase": state.phase.value,
        "final_text": state.final_text,
        "terminal_message": state.terminal_message,
        "elapsed_seconds": state.elapsed_seconds,
        "input_tokens": state.input_tokens,
        "output_tokens": state.output_tokens,
        "model_requests": state.model_requests,
        "tool_calls": state.tool_calls,
        "provider_retry_count": state.provider_retry_count,
        "provider_retries": provider_retries[-8:],
        "steps": [
            {
                "call_id": step.call_id,
                "tool_name": step.tool_name,
                "display_name": step.display_name,
                "status": step.status.value,
                "subject": step.subject,
                "summary": step.summary,
                "items": list(step.items),
                "preview_lines": list(step.preview_lines),
            }
            for step in state.steps
        ],
    }


class OopzPassiveAgentTraceSession:
    """Collect trace state when the final Agent answer uses a normal reply."""

    def __init__(self, gateway: OopzEditableMessageGateway, address: MessageAddress) -> None:
        self._gateway = gateway
        self._address = address
        self._reducer = AgentLoopReducer()
        self._state = AgentLoopViewState()
        self._run_id: UUID | None = None
        self._provider_retries: list[dict[str, object]] = []

    @property
    def owns_message(self) -> bool:
        return False

    async def bind_run(self, run_id: UUID) -> None:
        self._run_id = run_id

    async def emit(self, event: ConversationProgressEvent) -> None:
        if event.kind is ProgressKind.MODEL_RETRY:
            self._provider_retries.append(
                {
                    "attempt": event.retry_attempt,
                    "max_attempts": event.retry_max_attempts,
                    "delay_seconds": event.retry_delay_seconds,
                    "reason": event.retry_reason,
                }
            )
        self._state = self._reducer.apply(self._state, event)

    async def record_delivery(
        self,
        message: Any,
        *,
        response: ChatResponse | None = None,
        failure_message: str = "",
        cancelled: bool = False,
    ) -> None:
        if response is not None:
            await self.emit(
                ConversationProgressEvent(
                    ProgressKind.COMPLETED,
                    text=response.content,
                    elapsed_seconds=response.elapsed_seconds,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    model_requests=response.model_requests,
                    tool_calls=response.tool_calls,
                )
            )
        elif cancelled:
            await self.emit(ConversationProgressEvent(ProgressKind.CANCELLED))
        else:
            await self.emit(
                ConversationProgressEvent(
                    ProgressKind.FAILED,
                    text=failure_message or "处理请求时出现了问题，请稍后重试。",
                )
            )
        reference = EditableMessageRef(
            message_id=str(getattr(message, "message_id", "")),
            timestamp=str(getattr(message, "timestamp", "")),
            scope=self._address.scope,
            area_id=self._address.area_id,
            channel_id=self._address.channel_id,
            target_person_id=self._address.target_person_id,
            reference_message_id=self._address.reference_message_id,
        )
        try:
            await self._gateway.track_created(
                reference,
                kind=OutboundMessageKind.AGENT_RESPONSE,
                state=OutboundMessageState.FINAL,
                owner_person_id=self._address.owner_person_id,
            )
            await self._gateway.promote_agent_response(
                reference,
                self._run_id,
                _diagnostic_snapshot(self._state, self._provider_retries),
            )
        except Exception as exc:
            logger.warning(
                "Passive Agent response tracking degraded: message=%s error=%s",
                opaque_ref(reference.message_id),
                type(exc).__name__,
            )

    async def complete(self, response: ChatResponse) -> None:
        del response

    async def fail(self, message: str) -> None:
        del message

    async def cancel(self) -> None:
        return None

    async def aclose(self) -> None:
        return None


class OopzAgentLoopMessage:
    """Reduce fast events and serialize bounded edits through one worker."""

    min_text_delta_characters = 24
    max_refresh_seconds = 1.5
    max_retryable_failures = 3

    def __init__(
        self,
        gateway: OopzEditableMessageGateway,
        address: MessageAddress,
        renderer: OopzMessageRenderer,
        *,
        edit_interval_seconds: float,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        heartbeat_interval_seconds: float | None = None,
        active_presentations: ActivePresentationRegistry | None = None,
    ) -> None:
        if edit_interval_seconds <= 0:
            raise ValueError("Edit interval must be positive")
        heartbeat_interval_seconds = (
            self.max_refresh_seconds
            if heartbeat_interval_seconds is None
            else heartbeat_interval_seconds
        )
        if heartbeat_interval_seconds <= 0:
            raise ValueError("Heartbeat interval must be positive")
        self._gateway = gateway
        self._address = address
        self._renderer = renderer
        self._edit_interval = edit_interval_seconds
        self._clock = clock
        self._sleep = sleep
        self._heartbeat_interval = max(
            heartbeat_interval_seconds,
            edit_interval_seconds,
        )
        self._reducer = AgentLoopReducer()
        self._state = AgentLoopViewState()
        self._state_lock = asyncio.Lock()
        self._message: EditableMessageRef | None = None
        self._worker: asyncio.Task[None] | None = None
        self._wake = asyncio.Event()
        self._terminal_done = asyncio.Event()
        self._closing = False
        self._disabled = False
        self._terminal_delivery_failed = False
        self._fallback_attempted = False
        self._dirty_revision = 0
        self._flushed_revision = 0
        self._pending_delta_characters = 0
        self._last_snapshot = ""
        self._next_edit_at = 0.0
        self._retryable_failures = 0
        self._tool_started_at: dict[str, float] = {}
        self._retry_started_at: float | None = None
        self._activity_frame = 0
        self._run_id: UUID | None = None
        self._provider_retries: list[dict[str, object]] = []
        self._active_presentations = active_presentations
        self._dismissed = False

    @property
    def owns_message(self) -> bool:
        return self._message is not None

    @property
    def state(self) -> AgentLoopViewState:
        """Expose immutable state for diagnostics and focused tests."""
        return self._state

    async def open(self) -> None:
        """Create the placeholder before model work begins."""
        snapshot = self._renderer.render(self._state)
        logger.debug("Opening OOPZ Agent display: address=%s", self._address_ref())
        self._message = await self._gateway.create_reply(self._address, snapshot)
        await self._track_created(self._message, OutboundMessageState.ACTIVE)
        self._last_snapshot = snapshot
        self._next_edit_at = self._now() + self._edit_interval
        self._worker = asyncio.create_task(
            self._run_worker(),
            name="oopz-agent-loop-message",
        )
        if self._active_presentations is not None:
            await self._active_presentations.register(self._message.message_id, self)
        logger.info("Opened OOPZ Agent display: address=%s", self._address_ref())

    async def bind_run(self, run_id: UUID) -> None:
        """Link the durable run without exposing it as a progress event."""
        self._run_id = run_id
        message = self._message
        if message is not None:
            await self._bind_gateway_run(message, run_id)

    async def emit(self, event: ConversationProgressEvent) -> None:
        async with self._state_lock:
            if self._dismissed:
                return
            previous_phase = self._state.phase
            if event.kind is ProgressKind.TOOL_STARTED:
                self._tool_started_at.setdefault(event.call_id, self._now())
            elif event.kind is ProgressKind.MODEL_RETRY:
                self._retry_started_at = self._now()
                self._activity_frame = 0
                self._next_edit_at = 0.0
                self._provider_retries.append(
                    {
                        "attempt": event.retry_attempt,
                        "max_attempts": event.retry_max_attempts,
                        "delay_seconds": event.retry_delay_seconds,
                        "reason": event.retry_reason,
                    }
                )
            elif event.kind is ProgressKind.THINKING and previous_phase is DisplayPhase.RETRYING:
                self._retry_started_at = None
                self._next_edit_at = 0.0
            elif event.kind in {
                ProgressKind.TOOL_SUCCEEDED,
                ProgressKind.TOOL_FAILED,
            }:
                self._tool_started_at.pop(event.call_id, None)
            elif event.kind in {
                ProgressKind.COMPLETED,
                ProgressKind.FAILED,
                ProgressKind.CANCELLED,
            }:
                self._tool_started_at.clear()
                self._retry_started_at = None
            previous_revision = self._state.revision
            self._state = self._reducer.apply(self._state, event)
            if self._state.revision == previous_revision:
                return
            self._dirty_revision = self._state.revision
            if event.kind is ProgressKind.TEXT_DELTA:
                self._pending_delta_characters += len(event.text)
            should_wake = (
                event.kind in _SEMANTIC_KINDS
                or self._pending_delta_characters >= self.min_text_delta_characters
                or self._state.terminal
            )
            if self._disabled and self._state.terminal:
                self._terminal_delivery_failed = True
                self._terminal_done.set()
        if should_wake and not self._disabled:
            self._wake.set()

    async def complete(self, response: ChatResponse) -> None:
        await self._finish(
            ConversationProgressEvent(
                ProgressKind.COMPLETED,
                text=response.content,
                elapsed_seconds=response.elapsed_seconds,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                model_requests=response.model_requests,
                tool_calls=response.tool_calls,
            )
        )

    async def fail(self, message: str) -> None:
        await self._finish(
            ConversationProgressEvent(
                ProgressKind.FAILED,
                text=message,
            )
        )

    async def cancel(self) -> None:
        await self._finish(ConversationProgressEvent(ProgressKind.CANCELLED))

    async def aclose(self) -> None:
        self._closing = True
        self._wake.set()
        worker = self._worker
        try:
            if worker is not None:
                await worker
                self._worker = None
        finally:
            if self._active_presentations is not None and self._message is not None:
                await self._active_presentations.discard(self._message.message_id, self)

    async def dismiss(self) -> None:
        """Permanently stop future edits/fallback before OOPZ recall."""
        async with self._state_lock:
            self._dismissed = True
            self._disabled = True
            self._closing = True
            self._terminal_delivery_failed = False
            self._terminal_done.set()
        self._wake.set()
        worker = self._worker
        if worker is not None:
            await worker
            self._worker = None
        logger.info("Dismissed OOPZ Agent display: address=%s", self._address_ref())

    async def _finish(self, event: ConversationProgressEvent) -> None:
        if not self.owns_message or self._dismissed:
            return
        await self.emit(event)
        self._wake.set()
        await self._terminal_done.wait()
        if self._dismissed:
            return
        if self._terminal_delivery_failed and not self._fallback_attempted:
            self._fallback_attempted = True
            fallback_delivered = False
            logger.warning(
                "Attempting OOPZ Agent terminal fallback: address=%s",
                self._address_ref(),
            )
            try:
                fallback = await self._gateway.create_reply(
                    self._address,
                    self._renderer.render(self._state),
                )
                await self._track_created(fallback, OutboundMessageState.FINAL)
                if self._run_id is not None:
                    await self._bind_gateway_run(fallback, self._run_id)
                await self._finalize(fallback)
                fallback_delivered = True
            except Exception as exc:
                logger.warning(
                    "OOPZ Agent terminal fallback failed: %s",
                    type(exc).__name__,
                )
            else:
                logger.info(
                    "Delivered OOPZ Agent terminal fallback: address=%s",
                    self._address_ref(),
                )
            if self._message is not None:
                if fallback_delivered:
                    await self._supersede(self._message)
                else:
                    await self._finalize(self._message)
        elif self._message is not None:
            await self._finalize(self._message)

    async def _run_worker(self) -> None:
        while True:
            if self._dismissed:
                return
            if self._closing and self._dirty_revision <= self._flushed_revision:
                return
            heartbeat_due = False
            try:
                await asyncio.wait_for(
                    self._wake.wait(),
                    timeout=self._heartbeat_interval,
                )
            except TimeoutError:
                heartbeat_due = True
            self._wake.clear()
            if self._dismissed:
                return
            has_live_activity = self._has_live_activity()
            if self._dirty_revision <= self._flushed_revision and not (
                heartbeat_due and has_live_activity
            ):
                if self._closing:
                    return
                continue

            terminal = self._state.terminal
            if not terminal:
                delay = self._next_edit_at - self._now()
                if delay > 0:
                    await self._sleep(delay)

            if self._dismissed:
                return

            async with self._state_lock:
                state = self._state
                revision = self._dirty_revision
                if heartbeat_due:
                    self._activity_frame += 1
                snapshot = self._renderer.render(
                    state,
                    self._render_context(state),
                )

            if snapshot == self._last_snapshot:
                self._mark_flushed(revision, state.terminal)
                if state.terminal:
                    return
                continue

            message = self._message
            if message is None:
                self._disable(state.terminal)
                return
            try:
                await self._gateway.replace(message, snapshot)
            except (OopzConnectionError, OopzRateLimitError) as exc:
                self._retryable_failures += 1
                logger.warning(
                    "Retryable OOPZ Agent edit failure %s/%s: %s",
                    self._retryable_failures,
                    self.max_retryable_failures,
                    type(exc).__name__,
                )
                if self._retryable_failures >= self.max_retryable_failures:
                    self._disable(state.terminal)
                    return
                await self._sleep(self._edit_interval)
                self._wake.set()
                continue
            except Exception as exc:
                logger.warning(
                    "OOPZ Agent edit disabled after deterministic failure: %s",
                    type(exc).__name__,
                )
                self._disable(state.terminal)
                return

            self._retryable_failures = 0
            self._last_snapshot = snapshot
            self._next_edit_at = self._now() + self._edit_interval
            self._pending_delta_characters = 0
            self._mark_flushed(revision, state.terminal)
            if state.terminal:
                logger.info(
                    "Delivered terminal OOPZ Agent display: address=%s",
                    self._address_ref(),
                )
                return
            if self._dirty_revision > revision:
                self._wake.set()

    def _has_live_activity(self) -> bool:
        return self._state.phase is DisplayPhase.RETRYING or any(
            step.status is ToolStepStatus.RUNNING for step in self._state.steps
        )

    def _render_context(self, state: AgentLoopViewState) -> OopzRenderContext:
        now = self._now()
        elapsed = tuple(
            (
                step.call_id,
                max(now - started_at, 0),
            )
            for step in state.steps
            if (started_at := self._tool_started_at.get(step.call_id)) is not None
        )
        retry_remaining = None
        if (
            state.phase is DisplayPhase.RETRYING
            and state.retry_delay_seconds is not None
            and self._retry_started_at is not None
        ):
            retry_remaining = max(
                state.retry_delay_seconds - (now - self._retry_started_at),
                0.0,
            )
        return OopzRenderContext(
            running_elapsed_seconds=elapsed,
            retry_remaining_seconds=retry_remaining,
            activity_frame=self._activity_frame,
        )

    def _mark_flushed(self, revision: int, terminal: bool) -> None:
        self._flushed_revision = max(self._flushed_revision, revision)
        if terminal:
            self._terminal_done.set()

    def _disable(self, terminal: bool) -> None:
        self._disabled = True
        self._terminal_delivery_failed = terminal
        logger.warning(
            "Disabled OOPZ Agent display: address=%s terminal=%s",
            self._address_ref(),
            terminal,
        )
        if terminal:
            self._terminal_done.set()

    def _address_ref(self) -> str:
        return opaque_ref(
            self._address.scope,
            self._address.area_id,
            self._address.channel_id,
            self._address.target_person_id,
            self._address.reference_message_id,
        )

    def _now(self) -> float:
        if self._clock is not None:
            return self._clock()
        return asyncio.get_running_loop().time()

    async def _track_created(
        self,
        message: EditableMessageRef,
        state: OutboundMessageState,
    ) -> None:
        tracker = getattr(self._gateway, "track_created", None)
        if tracker is None:
            return
        try:
            await tracker(
                message,
                kind=OutboundMessageKind.AGENT_RESPONSE,
                state=state,
                owner_person_id=self._address.owner_person_id,
            )
        except Exception as exc:
            logger.warning(
                "Agent display tracking degraded: message=%s error=%s",
                opaque_ref(message.message_id),
                type(exc).__name__,
            )

    async def _bind_gateway_run(self, message: EditableMessageRef, run_id: UUID) -> None:
        binder = getattr(self._gateway, "bind_agent_run", None)
        if binder is not None:
            try:
                await binder(message, run_id)
            except Exception as exc:
                logger.warning(
                    "Agent display run binding degraded: message=%s error=%s",
                    opaque_ref(message.message_id),
                    type(exc).__name__,
                )

    async def _finalize(self, message: EditableMessageRef) -> None:
        finalizer = getattr(self._gateway, "finalize", None)
        if finalizer is not None:
            try:
                await finalizer(message, self._diagnostic_snapshot())
            except Exception as exc:
                logger.warning(
                    "Agent display finalization degraded: message=%s error=%s",
                    opaque_ref(message.message_id),
                    type(exc).__name__,
                )

    async def _supersede(self, message: EditableMessageRef) -> None:
        updater = getattr(self._gateway, "supersede", None)
        if updater is not None:
            try:
                await updater(message, self._diagnostic_snapshot())
            except Exception as exc:
                logger.warning(
                    "Agent display supersede degraded: message=%s error=%s",
                    opaque_ref(message.message_id),
                    type(exc).__name__,
                )

    def _diagnostic_snapshot(self) -> dict[str, object]:
        return _diagnostic_snapshot(self._state, self._provider_retries)


class OopzAgentPresenterFactory:
    """Open live sessions without making display availability a chat dependency."""

    def __init__(
        self,
        gateway: OopzEditableMessageGateway,
        renderer: OopzMessageRenderer,
        *,
        enabled: bool,
        edit_interval_seconds: float,
        active_presentations: ActivePresentationRegistry | None = None,
    ) -> None:
        self._gateway = gateway
        self._renderer = renderer
        self._enabled = enabled
        self._edit_interval_seconds = edit_interval_seconds
        self._active_presentations = active_presentations

    async def open(self, context: Any) -> ConversationProgressSession:
        try:
            address = (
                MessageAddress.from_command_request(context)
                if isinstance(context, CommandRequest)
                else MessageAddress.from_oopz_context(context)
            )
        except Exception as exc:
            logger.warning(
                "Could not resolve OOPZ Agent response address: %s",
                type(exc).__name__,
            )
            return NoopProgressSession()
        passive = OopzPassiveAgentTraceSession(self._gateway, address)
        if not self._enabled:
            return passive
        try:
            session = OopzAgentLoopMessage(
                self._gateway,
                address,
                self._renderer,
                edit_interval_seconds=self._edit_interval_seconds,
                active_presentations=self._active_presentations,
            )
            await session.open()
            return session
        except Exception as exc:
            logger.warning(
                "Could not create OOPZ Agent display message: %s",
                type(exc).__name__,
            )
            return passive
