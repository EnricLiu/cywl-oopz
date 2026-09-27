"""Internal events exchanged by voice runtime workers and the control loop."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field

from .events import (
    VoiceAssistantAudio,
    VoiceModelEvent,
    VoiceTranscriptFinal,
)
from .models import (
    PlaybackCursor,
    VoiceMediaEndReason,
    VoiceStopReason,
    VoiceTaskNotification,
)
from .notifications import (
    VoiceTaskNotificationStrategy,
)
from .ports import (
    RealtimeVoiceSession,
    VoiceMediaSession,
)


@dataclass(frozen=True, slots=True)
class _ProviderEvent:
    session: RealtimeVoiceSession
    event: VoiceModelEvent


@dataclass(frozen=True, slots=True)
class _StopRequested:
    reason: VoiceStopReason


@dataclass(frozen=True, slots=True)
class _MediaEnded:
    generation: int
    reason: VoiceMediaEndReason
    error_kind: str | None


@dataclass(frozen=True, slots=True)
class _MediaRecovered:
    generation: int
    media: VoiceMediaSession


@dataclass(frozen=True, slots=True)
class _MediaRecoveryFailed:
    generation: int
    reason: VoiceMediaEndReason
    error_kind: str


@dataclass(frozen=True, slots=True)
class _PumpFailed:
    pump: str
    error_kind: str
    retryable_provider: bool = False
    media_generation: int | None = None


@dataclass(frozen=True, slots=True)
class _WatchdogExpired:
    reason: VoiceStopReason


@dataclass(frozen=True, slots=True)
class _ResponseDrained:
    response_id: str
    generation: int
    cursor: PlaybackCursor


@dataclass(frozen=True, slots=True)
class _ToolCallFinished:
    session: RealtimeVoiceSession
    call_id: str
    name: str
    output: Mapping[str, object]
    elapsed_ms: float


@dataclass(frozen=True, slots=True)
class _AudioBarrier:
    response_id: str
    generation: int
    routed: asyncio.Future[None]


@dataclass(frozen=True, slots=True)
class _MailboxAvailable:
    """A lossy completion signal or periodic reconciliation tick."""


@dataclass(frozen=True, slots=True)
class _MailboxClaimed:
    notices: tuple[VoiceTaskNotification, ...]
    error_kind: str = ""


@dataclass(frozen=True, slots=True)
class _MailboxPresented:
    notices: tuple[VoiceTaskNotification, ...]
    strategy: VoiceTaskNotificationStrategy
    succeeded: bool

    @property
    def count(self) -> int:
        return len(self.notices)


@dataclass(slots=True)
class _ActiveResponse:
    response_id: str
    generation: int
    provider_item_id: str = ""
    provider_done: bool = False
    pending_transcript: VoiceTranscriptFinal | None = None
    usage: dict[str, int | float] = field(default_factory=dict)


_AudioEvent = VoiceAssistantAudio | _AudioBarrier
_ControlEvent = (
    _ProviderEvent
    | _StopRequested
    | _MediaEnded
    | _MediaRecovered
    | _MediaRecoveryFailed
    | _PumpFailed
    | _WatchdogExpired
    | _ResponseDrained
    | _ToolCallFinished
    | _MailboxAvailable
    | _MailboxClaimed
    | _MailboxPresented
)
