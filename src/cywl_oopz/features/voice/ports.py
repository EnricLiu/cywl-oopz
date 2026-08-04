"""Project-owned ports for voice access, media, Providers, and session runtimes."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

from .events import VoiceModelEvent
from .models import (
    PcmChunk,
    PlaybackCursor,
    RemoteAudioFrame,
    VoiceChannelKey,
    VoiceProviderCapabilities,
    VoiceRuntimeResult,
    VoiceSessionDescriptor,
    VoiceStopReason,
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

    async def write_output(self, chunk: PcmChunk) -> PlaybackCursor: ...

    async def flush_output(self) -> PlaybackCursor: ...

    async def drain_output(self) -> PlaybackCursor: ...

    async def current_cursor(self) -> PlaybackCursor: ...

    async def aclose(self) -> None: ...


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


class VoiceSessionRuntime(Protocol):
    """Coordinator runtime controlled by the single-active-session facade."""

    async def start(self) -> None: ...

    async def wait_finished(self) -> VoiceRuntimeResult: ...

    async def request_stop(self, reason: VoiceStopReason) -> None: ...

    async def aclose(self) -> None: ...


class VoiceSessionRuntimeFactory(Protocol):
    """Build a session runtime from pinned identity and lease resources."""

    async def create(self, context: VoiceSessionRuntimeContext) -> VoiceSessionRuntime: ...
