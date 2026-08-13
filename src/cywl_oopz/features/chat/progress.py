"""Provider-neutral progress events for one conversational request."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from .models import ChatResponse

logger = logging.getLogger(__name__)


class ProgressKind(StrEnum):
    """Stable lifecycle events understood by presentation adapters."""

    ACCEPTED = "accepted"
    THINKING = "thinking"
    MODEL_RETRY = "model_retry"
    TEXT_RESET = "text_reset"
    TEXT_DELTA = "text_delta"
    TOOL_STARTED = "tool_started"
    TOOL_UPDATED = "tool_updated"
    TOOL_SUCCEEDED = "tool_succeeded"
    TOOL_FAILED = "tool_failed"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


_TOOL_KINDS = frozenset(
    {
        ProgressKind.TOOL_STARTED,
        ProgressKind.TOOL_UPDATED,
        ProgressKind.TOOL_SUCCEEDED,
        ProgressKind.TOOL_FAILED,
    }
)
_TEXT_KINDS = frozenset(
    {
        ProgressKind.TEXT_DELTA,
        ProgressKind.COMPLETED,
        ProgressKind.FAILED,
    }
)


@dataclass(frozen=True, slots=True)
class ConversationProgressEvent:
    """A deliberately bounded event with display-safe tool and run summaries."""

    kind: ProgressKind
    event_id: str = ""
    call_id: str = ""
    tool_name: str = ""
    tool_display_name: str = ""
    tool_subject: str = ""
    tool_summary: str = ""
    tool_items: tuple[str, ...] = ()
    tool_preview_lines: tuple[str, ...] = ()
    text: str = ""
    elapsed_seconds: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    model_requests: int | None = None
    tool_calls: int | None = None
    retry_attempt: int | None = None
    retry_max_attempts: int | None = None
    retry_delay_seconds: float | None = None
    retry_reason: str = ""

    def __post_init__(self) -> None:
        if self.kind in _TOOL_KINDS:
            if not self.call_id.strip() or not self.tool_name.strip():
                raise ValueError("Tool progress requires call_id and tool_name")
            if not self.tool_display_name.strip():
                raise ValueError("Tool progress requires a display name")
            if (
                len(self.tool_display_name) > 48
                or "\n" in self.tool_display_name
                or "\r" in self.tool_display_name
            ):
                raise ValueError("Tool progress display name must be one short line")
            self._validate_tool_text(self.tool_subject, "subject", 80)
            self._validate_tool_text(self.tool_summary, "summary", 100)
            self._validate_tool_lines(self.tool_items, "items", maximum=3, line_limit=180)
            self._validate_tool_lines(
                self.tool_preview_lines,
                "preview lines",
                maximum=3,
                line_limit=120,
            )
        elif (
            self.call_id
            or self.tool_name
            or self.tool_display_name
            or self.tool_subject
            or self.tool_summary
            or self.tool_items
            or self.tool_preview_lines
        ):
            raise ValueError("Only tool progress may carry tool identity")
        retry_values = (
            self.retry_attempt,
            self.retry_max_attempts,
            self.retry_delay_seconds,
        )
        if self.kind is ProgressKind.MODEL_RETRY:
            if (
                self.retry_attempt is None
                or self.retry_max_attempts is None
                or self.retry_delay_seconds is None
                or self.retry_attempt <= 0
                or self.retry_max_attempts <= 0
                or self.retry_attempt > self.retry_max_attempts
                or self.retry_delay_seconds < 0
            ):
                raise ValueError("Model retry progress requires valid attempt and delay values")
            if (
                not self.retry_reason.strip()
                or len(self.retry_reason) > 80
                or "\n" in self.retry_reason
                or "\r" in self.retry_reason
            ):
                raise ValueError("Model retry progress requires one bounded reason")
        elif any(value is not None for value in retry_values) or self.retry_reason:
            raise ValueError("Only model retry progress may carry retry details")
        if self.kind in _TEXT_KINDS and not self.text:
            raise ValueError(f"{self.kind.value} progress requires text")
        if self.kind not in _TEXT_KINDS and self.text:
            raise ValueError(f"{self.kind.value} progress must not carry text")
        statistics = (
            self.elapsed_seconds,
            self.input_tokens,
            self.output_tokens,
            self.model_requests,
            self.tool_calls,
        )
        if self.kind is not ProgressKind.COMPLETED and any(
            value is not None for value in statistics
        ):
            raise ValueError("Only completed progress may carry run statistics")
        if any(value is not None and value < 0 for value in statistics):
            raise ValueError("Progress statistics must not be negative")

    @staticmethod
    def _validate_tool_text(value: str, label: str, limit: int) -> None:
        if len(value) > limit or "\n" in value or "\r" in value:
            raise ValueError(f"Tool progress {label} must be one bounded line")

    @classmethod
    def _validate_tool_lines(
        cls,
        values: tuple[str, ...],
        label: str,
        *,
        maximum: int,
        line_limit: int,
    ) -> None:
        if not isinstance(values, tuple) or len(values) > maximum:
            raise ValueError(f"Tool progress {label} must be a bounded tuple")
        for value in values:
            cls._validate_tool_text(value, label, line_limit)


class ProgressSink(Protocol):
    """Async target for best-effort conversation progress."""

    async def emit(self, event: ConversationProgressEvent) -> None:
        """Observe one lifecycle event without controlling Agent success."""


@runtime_checkable
class RunTraceSink(Protocol):
    """Optional durable run linkage implemented by tracked presentations."""

    async def bind_run(self, run_id: UUID) -> None:
        """Link a run as soon as its running record exists."""


@runtime_checkable
class DirectResponseTraceSink(Protocol):
    """Optional trace hook for non-live replies sent by the controller."""

    async def record_delivery(
        self,
        message: Any,
        *,
        response: ChatResponse | None = None,
        failure_message: str = "",
        cancelled: bool = False,
    ) -> None:
        """Bind one already-sent direct reply to the current run and snapshot."""


class ConversationProgressSession(ProgressSink, Protocol):
    """Lifecycle owned by one user-visible conversational request."""

    @property
    def owns_message(self) -> bool:
        """Whether this session created and now owns the response message."""

    async def complete(self, response: ChatResponse) -> None:
        """Flush a successful terminal response."""

    async def fail(self, message: str) -> None:
        """Flush a safe user-visible failure."""

    async def cancel(self) -> None:
        """Flush cancellation into the owned message."""

    async def aclose(self) -> None:
        """Release any session-owned background work."""


class ConversationPresenterFactory(Protocol):
    """Open one presentation session at an integration-owned context boundary."""

    async def open(self, context: Any) -> ConversationProgressSession:
        """Create a display message or return an unavailable no-op session."""


class NoopProgressSession:
    """Unavailable session used when live presentation is disabled or failed to open."""

    @property
    def owns_message(self) -> bool:
        return False

    async def emit(self, event: ConversationProgressEvent) -> None:
        del event

    async def bind_run(self, run_id: UUID) -> None:
        del run_id

    async def complete(self, response: ChatResponse) -> None:
        del response

    async def fail(self, message: str) -> None:
        del message

    async def cancel(self) -> None:
        return None

    async def aclose(self) -> None:
        return None


class NoopPresenterFactory:
    """Factory preserving the pre-live-display controller path."""

    async def open(self, context: Any) -> ConversationProgressSession:
        del context
        return NoopProgressSession()


async def emit_progress(
    progress: ProgressSink | None,
    event: ConversationProgressEvent,
) -> None:
    """Best-effort delivery that never controls conversation success."""
    if progress is None:
        return
    try:
        await progress.emit(event)
    except Exception as exc:
        logger.warning(
            "Conversation progress sink failed for %s: %s",
            event.kind.value,
            type(exc).__name__,
        )
