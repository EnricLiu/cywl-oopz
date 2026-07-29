"""Run-scoped bridge from provider transport retries to conversation progress."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from cywl_oopz.features.chat.progress import (
    ConversationProgressEvent,
    ProgressKind,
    ProgressSink,
    emit_progress,
)


@dataclass(slots=True)
class ProviderRetryProgress:
    """Emit bounded retry lifecycle events for the current Agent run."""

    sink: ProgressSink | None
    sequence: int = 0

    async def waiting(
        self,
        *,
        attempt: int,
        max_attempts: int,
        delay_seconds: float,
        reason: str,
    ) -> None:
        """Show one scheduled retry without exposing endpoint or exception text."""
        self.sequence += 1
        await emit_progress(
            self.sink,
            ConversationProgressEvent(
                ProgressKind.MODEL_RETRY,
                event_id=f"provider-retry-{self.sequence}",
                retry_attempt=attempt,
                retry_max_attempts=max_attempts,
                retry_delay_seconds=delay_seconds,
                retry_reason=reason,
            ),
        )

    async def resumed(self) -> None:
        """Return the display to thinking when the next attempt starts."""
        self.sequence += 1
        await emit_progress(
            self.sink,
            ConversationProgressEvent(
                ProgressKind.THINKING,
                event_id=f"provider-retry-{self.sequence}",
            ),
        )


_CURRENT_RETRY_PROGRESS: ContextVar[ProviderRetryProgress | None] = ContextVar(
    "cywl_provider_retry_progress",
    default=None,
)


@contextmanager
def bind_provider_retry_progress(
    sink: ProgressSink | None,
) -> Iterator[ProviderRetryProgress]:
    """Bind one reporter to async work spawned within the current run context."""
    reporter = ProviderRetryProgress(sink)
    token = _CURRENT_RETRY_PROGRESS.set(reporter)
    try:
        yield reporter
    finally:
        _CURRENT_RETRY_PROGRESS.reset(token)


def current_provider_retry_progress() -> ProviderRetryProgress | None:
    """Return the reporter inherited by the current async task."""
    return _CURRENT_RETRY_PROGRESS.get()
