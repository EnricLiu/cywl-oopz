"""Project-owned ports for voice access, media, Providers, and session runtimes."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol
from uuid import UUID

from .events import VoiceModelEvent
from .models import (
    PcmChunk,
    PlaybackCursor,
    RemoteAudioFrame,
    VoiceChannelKey,
    VoiceInternalContextItem,
    VoiceMediaTerminal,
    VoiceProviderCapabilities,
    VoiceRecoveryContext,
    VoiceRuntimeResult,
    VoiceRuntimeStats,
    VoiceRuntimeStatus,
    VoiceSessionDescriptor,
    VoiceSessionState,
    VoiceSessionStatus,
    VoiceStopReason,
    VoiceTaskNotification,
)
from .settings import (
    PersistedVoiceSessionStatus,
    SelectableVoiceModel,
    VoiceStartConfiguration,
    VoiceTurnRole,
    VoiceUserSelection,
)


class VoiceLease(Protocol):
    """Exclusive backend ownership token with idempotent release."""

    @property
    def released(self) -> bool: ...

    async def release(self) -> bool: ...


class VoiceAccessGateway(Protocol):
    """Resolve the owner channel and arbitrate the single voice backend."""

    async def voice_channel_for_user(self, area_id: str, person_id: str) -> str | None: ...

    async def try_acquire(
        self,
        channel: VoiceChannelKey,
        owner_key: str,
    ) -> VoiceLease | None: ...


class VoiceMediaSession(Protocol):
    """Streaming media boundary owned by one conversation."""

    def input_frames(self) -> AsyncIterator[RemoteAudioFrame]: ...

    async def wait_input_closed(self) -> VoiceMediaTerminal: ...

    async def write_output(self, chunk: PcmChunk) -> PlaybackCursor: ...

    async def flush_output(self) -> PlaybackCursor: ...

    async def drain_output(self) -> PlaybackCursor: ...

    async def current_cursor(self) -> PlaybackCursor: ...

    async def aclose(self) -> None: ...


class VoiceMediaGateway(Protocol):
    """Open owner-only input and one incremental output under an acquired lease."""

    async def open(
        self,
        descriptor: VoiceSessionDescriptor,
        lease: VoiceLease,
    ) -> VoiceMediaSession: ...


class RealtimeVoiceSession(Protocol):
    """Provider-neutral realtime WebSocket session."""

    async def send_audio(self, chunk: PcmChunk) -> None: ...

    async def interrupt(self, cursor: PlaybackCursor) -> None: ...

    async def complete_tool_call(
        self,
        call_id: str,
        output: Mapping[str, object],
    ) -> None: ...

    async def request_proactive_response(self, item: VoiceInternalContextItem) -> None: ...

    def events(self) -> AsyncIterator[VoiceModelEvent]: ...

    async def finish(self) -> None: ...

    async def aclose(self) -> None: ...


class RealtimeVoiceProvider(Protocol):
    """Create independently replaceable realtime Provider sessions."""

    @property
    def capabilities(self) -> VoiceProviderCapabilities: ...

    async def connect(self, descriptor: VoiceSessionDescriptor) -> RealtimeVoiceSession: ...

    async def aclose(self) -> None: ...


class VoiceRuntimeStatusSink(Protocol):
    """Receive non-blocking runtime snapshots on the owning event loop."""

    def emit(self, status: VoiceRuntimeStatus) -> None: ...


class VoiceSessionStatusSink(Protocol):
    """Own one optional user-facing session status surface."""

    @property
    def owns_message(self) -> bool: ...

    def emit(self, status: VoiceSessionStatus) -> None: ...

    async def aclose(self) -> None: ...


class VoiceMemoryContextSource(Protocol):
    """Load a bounded long-term memory projection for one voice session."""

    async def context_text(self, person_id: str) -> str: ...


@dataclass(frozen=True, slots=True)
class VoiceSessionRuntimeContext:
    """Resources pinned for one runtime generation."""

    descriptor: VoiceSessionDescriptor
    lease: VoiceLease
    configuration: VoiceStartConfiguration
    status_sink: VoiceRuntimeStatusSink | None = None
    memory_context: str = ""
    recovery_context: VoiceRecoveryContext = field(default_factory=VoiceRecoveryContext)

    def __post_init__(self) -> None:
        if len(self.memory_context) > 1500:
            raise ValueError("Voice memory context exceeds 1500 characters")


class VoiceSessionRuntime(Protocol):
    """Coordinator runtime controlled by the single-active-session facade."""

    @property
    def state(self) -> VoiceSessionState: ...

    @property
    def stats(self) -> VoiceRuntimeStats: ...

    async def start(self) -> None: ...

    async def wait_finished(self) -> VoiceRuntimeResult: ...

    async def request_stop(self, reason: VoiceStopReason) -> None: ...

    async def aclose(self) -> None: ...


class VoiceSessionRuntimeFactory(Protocol):
    """Build a session runtime from pinned identity and lease resources."""

    async def create(self, context: VoiceSessionRuntimeContext) -> VoiceSessionRuntime: ...


class VoiceTaskControlHandler(Protocol):
    """Execute only bounded realtime task-control operations."""

    async def execute(
        self,
        descriptor: VoiceSessionDescriptor,
        call_id: str,
        name: str,
        arguments: Mapping[str, object],
    ) -> Mapping[str, object]: ...


class VoiceTaskMailbox(Protocol):
    """Claim and present terminal delegated tasks for one active voice session."""

    async def wait(self, owner_person_id: str, timeout_seconds: float) -> bool: ...

    async def claim(
        self,
        session_id: UUID,
        limit: int,
    ) -> tuple[VoiceTaskNotification, ...]: ...

    async def present_text(self, notices: tuple[VoiceTaskNotification, ...]) -> bool: ...

    async def mark_presented(self, task_ids: tuple[UUID, ...]) -> None: ...

    async def defer(self, task_ids: tuple[UUID, ...]) -> None: ...


class VoiceConfigurationRepository(Protocol):
    """Fresh-read voice catalog and user preference operations."""

    async def resolve_start_configuration(
        self,
        owner_person_id: str,
        channel: VoiceChannelKey,
    ) -> VoiceStartConfiguration: ...

    async def list_selectable_models(
        self,
        owner_person_id: str,
    ) -> tuple[SelectableVoiceModel, ...]: ...

    async def user_selection(self, owner_person_id: str) -> VoiceUserSelection: ...

    async def set_user_model(self, owner_person_id: str, selector: str) -> SelectableVoiceModel: ...

    async def set_user_voice(self, owner_person_id: str, voice_id: str) -> None: ...


class VoiceSessionRepository(Protocol):
    """Persist session lifecycle and final transcript turns."""

    async def recover_stale(self, now: datetime) -> int:
        """Terminate sessions left non-terminal by an earlier process generation."""
        ...

    async def create(
        self,
        descriptor: VoiceSessionDescriptor,
        configuration: VoiceStartConfiguration,
    ) -> None: ...

    async def mark_active(self, session_id: UUID) -> None: ...

    async def mark_recovering(self, session_id: UUID) -> None: ...

    async def finish(
        self,
        session_id: UUID,
        status: PersistedVoiceSessionStatus,
        stop_reason: str,
        *,
        usage: dict[str, Any] | None = None,
        summary: str = "",
    ) -> None: ...

    async def append_final_turn(
        self,
        session_id: UUID,
        sequence: int,
        role: VoiceTurnRole,
        transcript: str,
        *,
        provider_item_id: str = "",
        usage: dict[str, Any] | None = None,
    ) -> None: ...
