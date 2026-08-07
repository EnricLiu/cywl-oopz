"""Deterministic in-memory voice adapters used by service and contract tests."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from uuid import UUID

from cywl_oopz.features.voice.errors import (
    VoiceModelSelectionError,
    VoiceSpeakerSelectionError,
)
from cywl_oopz.features.voice.events import VoiceModelEvent
from cywl_oopz.features.voice.models import (
    PcmChunk,
    PlaybackCursor,
    RemoteAudioFrame,
    VoiceChannelKey,
    VoiceInternalContextItem,
    VoiceMediaEndReason,
    VoiceMediaTerminal,
    VoiceProviderCapabilities,
    VoiceRuntimeResult,
    VoiceRuntimeStats,
    VoiceRuntimeStatus,
    VoiceSessionDescriptor,
    VoiceSessionState,
    VoiceStopReason,
)
from cywl_oopz.features.voice.ports import VoiceSessionRuntimeContext
from cywl_oopz.features.voice.settings import (
    PersistedVoiceSessionStatus,
    SelectableVoiceModel,
    VoiceChannelConfiguration,
    VoiceModelConfiguration,
    VoiceModelMode,
    VoiceProviderConfiguration,
    VoiceProviderProtocol,
    VoiceStartConfiguration,
    VoiceTurnRole,
    VoiceUserSelection,
)

_FAKE_PROVIDER_ID = UUID("10000000-0000-0000-0000-000000000001")
_FAKE_MODEL_ID = UUID("20000000-0000-0000-0000-000000000001")


class FakeVoiceConfigurationRepository:
    """Fresh-read compatible catalog fake with mutable test configuration."""

    def __init__(self) -> None:
        self.resolve_calls: list[tuple[str, VoiceChannelKey]] = []
        self.selections: dict[str, VoiceUserSelection] = {}
        self.model = SelectableVoiceModel(
            _FAKE_MODEL_ID,
            "fake",
            "realtime",
            "Fake realtime",
            VoiceModelMode.NATIVE_REALTIME,
        )

    async def resolve_start_configuration(
        self,
        owner_person_id: str,
        channel: VoiceChannelKey,
    ) -> VoiceStartConfiguration:
        self.resolve_calls.append((owner_person_id, channel))
        selection = self.selections.get(owner_person_id, VoiceUserSelection())
        return VoiceStartConfiguration(
            provider=VoiceProviderConfiguration(
                _FAKE_PROVIDER_ID,
                "fake",
                "Fake",
                VoiceProviderProtocol.QWEN_OMNI_REALTIME_WS,
                "wss://voice.invalid/realtime",
                {"api_key": "fake"},
                {},
            ),
            model=VoiceModelConfiguration(
                _FAKE_MODEL_ID,
                _FAKE_PROVIDER_ID,
                "realtime",
                "fake-realtime",
                "Fake realtime",
                VoiceModelMode.NATIVE_REALTIME,
                {},
                {},
                {},
                {},
            ),
            channel=VoiceChannelConfiguration(channel, "voice_readonly_v1", 300),
            voice_id=selection.voice_id,
            duplex_mode=selection.duplex_mode,
            delegated_agent_model_id=selection.delegated_agent_model_id,
        )

    async def list_selectable_models(
        self,
        owner_person_id: str,
    ) -> tuple[SelectableVoiceModel, ...]:
        selected = self.selections.get(owner_person_id, VoiceUserSelection())
        return (
            SelectableVoiceModel(
                self.model.id,
                self.model.provider_alias,
                self.model.model_alias,
                self.model.display_name,
                self.model.mode,
                selected.preferred_model_id == self.model.id,
            ),
        )

    async def user_selection(self, owner_person_id: str) -> VoiceUserSelection:
        return self.selections.get(owner_person_id, VoiceUserSelection())

    async def set_user_model(
        self,
        owner_person_id: str,
        selector: str,
    ) -> SelectableVoiceModel:
        if selector != self.model.selector:
            raise VoiceModelSelectionError
        current = await self.user_selection(owner_person_id)
        self.selections[owner_person_id] = VoiceUserSelection(
            self.model.id,
            current.voice_id,
            current.duplex_mode,
            current.delegated_agent_model_id,
        )
        return SelectableVoiceModel(
            self.model.id,
            self.model.provider_alias,
            self.model.model_alias,
            self.model.display_name,
            self.model.mode,
            True,
        )

    async def set_user_voice(self, owner_person_id: str, voice_id: str) -> None:
        if not voice_id.strip() or len(voice_id.strip()) > 128:
            raise VoiceSpeakerSelectionError
        current = await self.user_selection(owner_person_id)
        self.selections[owner_person_id] = VoiceUserSelection(
            current.preferred_model_id,
            voice_id,
            current.duplex_mode,
            current.delegated_agent_model_id,
        )


class FakeVoiceSessionRepository:
    """Capture durable lifecycle calls without a database."""

    def __init__(self) -> None:
        self.created: list[tuple[VoiceSessionDescriptor, VoiceStartConfiguration]] = []
        self.active: list[UUID] = []
        self.recovering: list[UUID] = []
        self.finished: list[tuple[UUID, PersistedVoiceSessionStatus, str]] = []
        self.finished_usage: list[dict[str, object]] = []
        self.turns: list[tuple[UUID, int, VoiceTurnRole, str]] = []
        self.turn_usage: list[dict[str, object]] = []
        self.stale_recovery_count = 0
        self.recovery_calls = 0

    async def recover_stale(self, now) -> int:
        del now
        self.recovery_calls += 1
        return self.stale_recovery_count

    async def create(self, descriptor, configuration) -> None:
        self.created.append((descriptor, configuration))

    async def mark_active(self, session_id: UUID) -> None:
        self.active.append(session_id)

    async def mark_recovering(self, session_id: UUID) -> None:
        self.recovering.append(session_id)

    async def finish(self, session_id, status, stop_reason, *, usage=None, summary="") -> None:
        del summary
        self.finished.append((session_id, status, stop_reason))
        self.finished_usage.append(dict(usage or {}))

    async def append_final_turn(
        self,
        session_id,
        sequence,
        role,
        transcript,
        *,
        provider_item_id="",
        usage=None,
    ) -> None:
        del provider_item_id
        self.turns.append((session_id, sequence, role, transcript))
        self.turn_usage.append(dict(usage or {}))


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
        self.flushes: list[PlaybackCursor] = []
        self.drain_count = 0
        self.drain_gate: asyncio.Event | None = None
        self.user_speaking_changes: list[bool] = []
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
        self.flushes.append(self._cursor)
        return self._cursor

    async def drain_output(self) -> PlaybackCursor:
        self.drain_count += 1
        if self.drain_gate is not None:
            await self.drain_gate.wait()
        return self._cursor

    async def current_cursor(self) -> PlaybackCursor:
        return self._cursor

    async def set_user_speaking(self, speaking: bool) -> None:
        self.user_speaking_changes.append(speaking)

    async def aclose(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._input_terminal = VoiceMediaTerminal(VoiceMediaEndReason.CLOSED_BY_CALLER)
        self._input_closed.set()
        await self._inputs.put(None)


class FakeVoiceMediaGateway:
    """Open a fresh inspectable media session for each runtime."""

    def __init__(self) -> None:
        self.opens: list[tuple[VoiceSessionDescriptor, object]] = []
        self.sessions: list[FakeVoiceMediaSession] = []

    async def open(self, descriptor, lease) -> FakeVoiceMediaSession:
        session = FakeVoiceMediaSession()
        self.opens.append((descriptor, lease))
        self.sessions.append(session)
        return session


class FakeRealtimeVoiceSession:
    """Record Provider input and yield explicitly emitted model events."""

    def __init__(self) -> None:
        self.sent_audio: list[PcmChunk] = []
        self.interruptions: list[PlaybackCursor] = []
        self.interrupt_error: Exception | None = None
        self.tool_outputs: list[tuple[str, dict[str, object]]] = []
        self.proactive_items: list[VoiceInternalContextItem] = []
        self.proactive_error: Exception | None = None
        self._events: asyncio.Queue[VoiceModelEvent | None] = asyncio.Queue()
        self.finished = False
        self.closed = False

    async def send_audio(self, chunk: PcmChunk) -> None:
        if self.closed:
            raise RuntimeError("Fake realtime voice session is closed")
        self.sent_audio.append(chunk)

    async def emit(self, event: VoiceModelEvent) -> None:
        await self._events.put(event)

    async def interrupt(self, cursor: PlaybackCursor) -> None:
        if self.interrupt_error is not None:
            raise self.interrupt_error
        self.interruptions.append(cursor)

    async def complete_tool_call(
        self,
        call_id: str,
        output: Mapping[str, object],
    ) -> None:
        self.tool_outputs.append((call_id, dict(output)))

    async def request_proactive_response(self, item: VoiceInternalContextItem) -> None:
        if self.proactive_error is not None:
            raise self.proactive_error
        self.proactive_items.append(item)

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

    def __init__(self, context: VoiceSessionRuntimeContext | None = None) -> None:
        self._context = context
        self.started = False
        self.closed = False
        self.stop_requests: list[VoiceStopReason] = []
        self._finished = asyncio.Event()
        self._result = VoiceRuntimeResult(VoiceStopReason.RUNTIME_ENDED)
        self.usage: dict[str, object] = {}
        self._state = VoiceSessionState.STARTING
        self._stats = VoiceRuntimeStats()

    @property
    def state(self) -> VoiceSessionState:
        return self._state

    @property
    def stats(self) -> VoiceRuntimeStats:
        return self._stats

    async def start(self) -> None:
        self.started = True
        self.set_state(VoiceSessionState.LISTENING)

    async def wait_finished(self) -> VoiceRuntimeResult:
        await self._finished.wait()
        return self._result

    async def request_stop(self, reason: VoiceStopReason) -> None:
        self.stop_requests.append(reason)
        self.set_state(VoiceSessionState.CLOSING)
        self._result = VoiceRuntimeResult(reason, dict(self.usage))
        self._finished.set()

    async def finish(self, reason: VoiceStopReason = VoiceStopReason.RUNTIME_ENDED) -> None:
        self._result = VoiceRuntimeResult(reason, dict(self.usage))
        self.set_state(VoiceSessionState.CLOSED)
        self._finished.set()

    async def aclose(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.set_state(VoiceSessionState.CLOSED)
        self._finished.set()

    def set_state(
        self,
        state: VoiceSessionState,
        stats: VoiceRuntimeStats | None = None,
    ) -> None:
        self._state = state
        if stats is not None:
            self._stats = stats
        sink = self._context.status_sink if self._context is not None else None
        if sink is not None:
            sink.emit(VoiceRuntimeStatus(self._state, self._stats))


class FakeVoiceSessionRuntimeFactory:
    """Create and retain deterministic runtimes for service assertions."""

    def __init__(self) -> None:
        self.contexts: list[VoiceSessionRuntimeContext] = []
        self.runtimes: list[FakeVoiceSessionRuntime] = []

    async def create(self, context: VoiceSessionRuntimeContext) -> FakeVoiceSessionRuntime:
        runtime = FakeVoiceSessionRuntime(context)
        self.contexts.append(context)
        self.runtimes.append(runtime)
        return runtime
