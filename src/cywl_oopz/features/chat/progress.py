"""Provider-neutral progress events for one conversational request."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


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


class NoopProgressSink:
    """Default sink used when live presentation is unavailable or disabled."""

    async def emit(self, event: ConversationProgressEvent) -> None:
        del event
