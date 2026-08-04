"""Project-owned ports for voice access, media, Providers, and session runtimes."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import UUID

from .events import VoiceModelEvent
from .models import (
    PcmChunk,
    PlaybackCursor,
    RemoteAudioFrame,
    VoiceChannelKey,
    VoiceMediaTerminal,
    VoiceProviderCapabilities,
    VoiceRuntimeResult,
    VoiceSessionDescriptor,
    VoiceStopReason,
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

    def events(self) -> AsyncIterator[VoiceModelEvent]: ...

    async def finish(self) -> None: ...

    async def aclose(self) -> None: ...


class RealtimeVoiceProvider(Protocol):
    """Create independently replaceable realtime Provider sessions."""

    @property
    def capabilities(self) -> VoiceProviderCapabilities: ...

    async def connect(self, descriptor: VoiceSessionDescriptor) -> RealtimeVoiceSession: ...

    async def aclose(self) -> None: ...


@dataclass(frozen=True, slots=True)
class VoiceSessionRuntimeContext:
    """Resources pinned for one runtime generation."""

    descriptor: VoiceSessionDescriptor
    lease: VoiceLease
    configuration: VoiceStartConfiguration


class VoiceSessionRuntime(Protocol):
    """Coordinator runtime controlled by the single-active-session facade."""

    async def start(self) -> None: ...

    async def wait_finished(self) -> VoiceRuntimeResult: ...

    async def request_stop(self, reason: VoiceStopReason) -> None: ...

    async def aclose(self) -> None: ...


class VoiceSessionRuntimeFactory(Protocol):
    """Build a session runtime from pinned identity and lease resources."""

    async def create(self, context: VoiceSessionRuntimeContext) -> VoiceSessionRuntime: ...


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

    async def create(
        self,
        descriptor: VoiceSessionDescriptor,
        configuration: VoiceStartConfiguration,
    ) -> None: ...

    async def mark_active(self, session_id: UUID) -> None: ...

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
