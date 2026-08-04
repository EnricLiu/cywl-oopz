"""Deterministic in-memory voice adapters used by service and contract tests."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from cywl_oopz.features.voice.events import VoiceModelEvent
from cywl_oopz.features.voice.models import (
    PcmChunk,
    PlaybackCursor,
    RemoteAudioFrame,
    VoiceChannelKey,
    VoiceMediaEndReason,
    VoiceMediaTerminal,
    VoiceProviderCapabilities,
    VoiceRuntimeResult,
    VoiceSessionDescriptor,
    VoiceStopReason,
)
from cywl_oopz.features.voice.ports import VoiceSessionRuntimeContext


class FakeVoiceLease:
    """Idempotent in-memory lease token."""

    def __init__(self, gateway: FakeVoiceAccessGateway) -> None:
        self._gateway = gateway
        self._released = False

    @property
    def released(self) -> bool:
        return self._released

    async def release(self) -> bool:
        if self._released:
            return False
        self._released = True
        if self._gateway.active_lease is self:
            self._gateway.active_lease = None
            self._gateway.release_count += 1
            return True
        return False


class FakeVoiceAccessGateway:
    """Configurable channel resolver and exclusive lease owner."""

    def __init__(self) -> None:
        self.channels: dict[tuple[str, str], str] = {}
        self.active_lease: FakeVoiceLease | None = None
        self.acquisitions: list[tuple[VoiceChannelKey, str]] = []
        self.release_count = 0
        self.force_busy = False

    async def voice_channel_for_user(self, area_id: str, person_id: str) -> str | None:
        return self.channels.get((area_id, person_id))

    async def try_acquire(
        self,
        channel: VoiceChannelKey,
        owner_key: str,
    ) -> FakeVoiceLease | None:
        self.acquisitions.append((channel, owner_key))
        if self.force_busy or self.active_lease is not None:
            return None
        lease = FakeVoiceLease(self)
        self.active_lease = lease
        return lease


class FakeVoiceMediaSession:
    """Queue-backed media stream with an immediately rendered output cursor."""

    def __init__(self, *, sample_rate: int = 24_000) -> None:
        self._inputs: asyncio.Queue[RemoteAudioFrame | None] = asyncio.Queue()
        self._input_closed = asyncio.Event()
        self._input_terminal = VoiceMediaTerminal(VoiceMediaEndReason.CLOSED_BY_CALLER)
        self._cursor = PlaybackCursor(0, 0, 0, 0, sample_rate)
        self.outputs: list[PcmChunk] = []
        self.closed = False

    async def push_input(self, frame: RemoteAudioFrame) -> None:
        await self._inputs.put(frame)

    async def end_input(
        self,
        reason: VoiceMediaEndReason = VoiceMediaEndReason.OWNER_LEFT,
    ) -> None:
        self._input_terminal = VoiceMediaTerminal(reason)
        self._input_closed.set()
        await self._inputs.put(None)

    async def input_frames(self) -> AsyncIterator[RemoteAudioFrame]:
        while True:
            frame = await self._inputs.get()
            if frame is None:
                return
            yield frame

    async def wait_input_closed(self) -> VoiceMediaTerminal:
        await self._input_closed.wait()
        return self._input_terminal

    async def write_output(self, chunk: PcmChunk) -> PlaybackCursor:
        if self.closed:
            raise RuntimeError("Fake voice media session is closed")
        self.outputs.append(chunk)
        samples = len(chunk.pcm) // chunk.format.frame_width_bytes
        accepted = self._cursor.accepted_samples + samples
        self._cursor = PlaybackCursor(
            chunk.generation,
            accepted,
            accepted,
            0,
            chunk.format.sample_rate,
        )
        return self._cursor

    async def flush_output(self) -> PlaybackCursor:
        self._cursor = PlaybackCursor(
            self._cursor.generation + 1,
            self._cursor.accepted_samples,
            self._cursor.rendered_samples,
            0,
            self._cursor.sample_rate,
        )
        return self._cursor

    async def drain_output(self) -> PlaybackCursor:
        return self._cursor

    async def current_cursor(self) -> PlaybackCursor:
        return self._cursor

    async def aclose(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._input_terminal = VoiceMediaTerminal(VoiceMediaEndReason.CLOSED_BY_CALLER)
        self._input_closed.set()
        await self._inputs.put(None)


class FakeRealtimeVoiceSession:
    """Record Provider input and yield explicitly emitted model events."""

    def __init__(self) -> None:
        self.sent_audio: list[PcmChunk] = []
        self._events: asyncio.Queue[VoiceModelEvent | None] = asyncio.Queue()
        self.finished = False
        self.closed = False

    async def send_audio(self, chunk: PcmChunk) -> None:
        if self.closed:
            raise RuntimeError("Fake realtime voice session is closed")
        self.sent_audio.append(chunk)

    async def emit(self, event: VoiceModelEvent) -> None:
        await self._events.put(event)

    async def events(self) -> AsyncIterator[VoiceModelEvent]:
        while True:
            event = await self._events.get()
            if event is None:
                return
            yield event

    async def finish(self) -> None:
        self.finished = True

    async def aclose(self) -> None:
        if self.closed:
            return
        self.closed = True
        await self._events.put(None)


class FakeRealtimeVoiceProvider:
    """Create inspectable fake Provider sessions without network access."""

    def __init__(self, capabilities: VoiceProviderCapabilities | None = None) -> None:
        self._capabilities = capabilities or VoiceProviderCapabilities()
        self.descriptors: list[VoiceSessionDescriptor] = []
        self.sessions: list[FakeRealtimeVoiceSession] = []
        self.closed = False

    @property
    def capabilities(self) -> VoiceProviderCapabilities:
        return self._capabilities

    async def connect(self, descriptor: VoiceSessionDescriptor) -> FakeRealtimeVoiceSession:
        session = FakeRealtimeVoiceSession()
        self.descriptors.append(descriptor)
        self.sessions.append(session)
        return session

    async def aclose(self) -> None:
        if self.closed:
            return
        self.closed = True
        await asyncio.gather(*(session.aclose() for session in self.sessions))


class FakeVoiceSessionRuntime:
    """Minimal controllable runtime for lifecycle tests."""

    def __init__(self) -> None:
        self.started = False
        self.closed = False
        self.stop_requests: list[VoiceStopReason] = []
        self._finished = asyncio.Event()
        self._result = VoiceRuntimeResult(VoiceStopReason.RUNTIME_ENDED)

    async def start(self) -> None:
        self.started = True

    async def wait_finished(self) -> VoiceRuntimeResult:
        await self._finished.wait()
        return self._result

    async def request_stop(self, reason: VoiceStopReason) -> None:
        self.stop_requests.append(reason)
        self._result = VoiceRuntimeResult(reason)
        self._finished.set()

    async def finish(self, reason: VoiceStopReason = VoiceStopReason.RUNTIME_ENDED) -> None:
        self._result = VoiceRuntimeResult(reason)
        self._finished.set()

    async def aclose(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._finished.set()


class FakeVoiceSessionRuntimeFactory:
    """Create and retain deterministic runtimes for service assertions."""

    def __init__(self) -> None:
        self.contexts: list[VoiceSessionRuntimeContext] = []
        self.runtimes: list[FakeVoiceSessionRuntime] = []

    async def create(self, context: VoiceSessionRuntimeContext) -> FakeVoiceSessionRuntime:
        runtime = FakeVoiceSessionRuntime()
        self.contexts.append(context)
        self.runtimes.append(runtime)
        return runtime
