"""Single-message OOPZ presenter for one conversational Agent loop."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from oopz_sdk.exceptions import OopzConnectionError, OopzRateLimitError

from cywl_oopz.features.agent.display import AgentLoopReducer, AgentLoopViewState
from cywl_oopz.features.chat.models import ChatResponse
from cywl_oopz.features.chat.progress import (
    ConversationProgressEvent,
    ConversationProgressSession,
    NoopProgressSession,
    ProgressKind,
)

from .editable_messages import (
    EditableMessageRef,
    MessageAddress,
    OopzEditableMessageGateway,
)
from .message_renderer import OopzMessageRenderer

logger = logging.getLogger(__name__)

_SEMANTIC_KINDS = frozenset(
    {
        ProgressKind.ACCEPTED,
        ProgressKind.THINKING,
        ProgressKind.TEXT_RESET,
        ProgressKind.TOOL_STARTED,
        ProgressKind.TOOL_SUCCEEDED,
        ProgressKind.TOOL_FAILED,
        ProgressKind.COMPLETED,
        ProgressKind.FAILED,
        ProgressKind.CANCELLED,
    }
)


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
    ) -> None:
        if edit_interval_seconds <= 0:
            raise ValueError("Edit interval must be positive")
        self._gateway = gateway
        self._address = address
        self._renderer = renderer
        self._edit_interval = edit_interval_seconds
        self._clock = clock
        self._sleep = sleep
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
        self._message = await self._gateway.create_reply(self._address, snapshot)
        self._last_snapshot = snapshot
        self._next_edit_at = self._now() + self._edit_interval
        self._worker = asyncio.create_task(
            self._run_worker(),
            name="oopz-agent-loop-message",
        )

    async def emit(self, event: ConversationProgressEvent) -> None:
        async with self._state_lock:
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
        if worker is not None:
            await worker
            self._worker = None

    async def _finish(self, event: ConversationProgressEvent) -> None:
        if not self.owns_message:
            return
        await self.emit(event)
        self._wake.set()
        await self._terminal_done.wait()
        if self._terminal_delivery_failed and not self._fallback_attempted:
            self._fallback_attempted = True
            try:
                await self._gateway.create_reply(
                    self._address,
                    self._renderer.render(self._state),
                )
            except Exception as exc:
                logger.warning(
                    "OOPZ Agent terminal fallback failed: %s",
                    type(exc).__name__,
                )

    async def _run_worker(self) -> None:
        while True:
            if self._closing and self._dirty_revision <= self._flushed_revision:
                return
            try:
                await asyncio.wait_for(
                    self._wake.wait(),
                    timeout=self.max_refresh_seconds,
                )
            except TimeoutError:
                pass
            self._wake.clear()
            if self._dirty_revision <= self._flushed_revision:
                if self._closing:
                    return
                continue

            terminal = self._state.terminal
            if not terminal:
                delay = self._next_edit_at - self._now()
                if delay > 0:
                    await self._sleep(delay)

            async with self._state_lock:
                state = self._state
                revision = self._dirty_revision
                snapshot = self._renderer.render(state)

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
                return
            if self._dirty_revision > revision:
                self._wake.set()

    def _mark_flushed(self, revision: int, terminal: bool) -> None:
        self._flushed_revision = max(self._flushed_revision, revision)
        if terminal:
            self._terminal_done.set()

    def _disable(self, terminal: bool) -> None:
        self._disabled = True
        self._terminal_delivery_failed = terminal
        if terminal:
            self._terminal_done.set()

    def _now(self) -> float:
        if self._clock is not None:
            return self._clock()
        return asyncio.get_running_loop().time()


class OopzAgentPresenterFactory:
    """Open live sessions without making display availability a chat dependency."""

    def __init__(
        self,
        gateway: OopzEditableMessageGateway,
        renderer: OopzMessageRenderer,
        *,
        enabled: bool,
        edit_interval_seconds: float,
    ) -> None:
        self._gateway = gateway
        self._renderer = renderer
        self._enabled = enabled
        self._edit_interval_seconds = edit_interval_seconds

    async def open(self, context: Any) -> ConversationProgressSession:
        if not self._enabled:
            return NoopProgressSession()
        try:
            address = MessageAddress.from_oopz_context(context)
            session = OopzAgentLoopMessage(
                self._gateway,
                address,
                self._renderer,
                edit_interval_seconds=self._edit_interval_seconds,
            )
            await session.open()
            return session
        except Exception as exc:
            logger.warning(
                "Could not create OOPZ Agent display message: %s",
                type(exc).__name__,
            )
            return NoopProgressSession()
