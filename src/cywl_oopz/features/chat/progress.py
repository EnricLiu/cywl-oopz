"""Provider-neutral progress events for one conversational request."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from .models import ChatResponse

logger = logging.getLogger(__name__)


class ProgressKind(StrEnum):
    """Stable lifecycle events understood by presentation adapters."""

    ACCEPTED = "accepted"
    THINKING = "thinking"
    TEXT_RESET = "text_reset"
    TEXT_DELTA = "text_delta"
    TOOL_STARTED = "tool_started"
    TOOL_SUCCEEDED = "tool_succeeded"
    TOOL_FAILED = "tool_failed"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


_TOOL_KINDS = frozenset(
    {
        ProgressKind.TOOL_STARTED,
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
    """A deliberately safe event with no raw tool arguments or outputs."""

    kind: ProgressKind
    event_id: str = ""
    call_id: str = ""
    tool_name: str = ""
    tool_display_name: str = ""
    text: str = ""

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
        elif self.call_id or self.tool_name or self.tool_display_name:
            raise ValueError("Only tool progress may carry tool identity")
        if self.kind in _TEXT_KINDS and not self.text:
            raise ValueError(f"{self.kind.value} progress requires text")
        if self.kind not in _TEXT_KINDS and self.text:
            raise ValueError(f"{self.kind.value} progress must not carry text")


class ProgressSink(Protocol):
    """Async target for best-effort conversation progress."""

    async def emit(self, event: ConversationProgressEvent) -> None:
        """Observe one lifecycle event without controlling Agent success."""


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
