"""Provider-neutral realtime session coordinator with bounded async pumps."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections import deque
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, replace

from cywl_oopz.core.observability import exception_kind, opaque_ref
from cywl_oopz.settings import VoiceSettings

from .audio import VoiceAudioIngress, VoiceInputQueue, VoiceOutputTransitQueue
from .errors import (
    VoiceAudioQueueClosedError,
    VoiceProviderAuthenticationError,
    VoiceProviderConfigurationError,
    VoiceProviderDisconnectedError,
)
from .events import (
    VoiceAssistantAudio,
    VoiceModelEvent,
    VoiceProviderFailed,
    VoiceResponseCancelled,
    VoiceResponseCompleted,
    VoiceResponseStarted,
    VoiceSessionFinished,
    VoiceSessionReady,
    VoiceToolCall,
    VoiceTranscriptFinal,
    VoiceUserSpeechStarted,
    VoiceUserSpeechStopped,
)
from .models import (
    PcmChunk,
    PlaybackCursor,
    VoiceMediaEndReason,
    VoiceRuntimeResult,
    VoiceRuntimeStats,
    VoiceSessionState,
    VoiceStopReason,
)
from .ports import (
    RealtimeVoiceProvider,
    RealtimeVoiceSession,
    VoiceMediaGateway,
    VoiceMediaSession,
    VoiceSessionRepository,
    VoiceSessionRuntime,
    VoiceSessionRuntimeContext,
    VoiceSessionRuntimeFactory,
    VoiceTaskControlHandler,
)
from .settings import VoiceTurnRole

logger = logging.getLogger(__name__)

ProviderBuilder = Callable[[VoiceSessionRuntimeContext], RealtimeVoiceProvider]
_AUDIO_STAGING_CHUNKS = 8
_CANCELLED_RESPONSE_HISTORY = 64
_COMPLETED_TOOL_CALL_HISTORY = 128
_INTERRUPT_TIMEOUT_SECONDS = 1.5
_TASK_SUBMIT_TIMEOUT_SECONDS = 0.25
_TASK_READ_TIMEOUT_SECONDS = 0.15


@dataclass(frozen=True, slots=True)
class _ProviderEvent:
    session: RealtimeVoiceSession
    event: VoiceModelEvent


@dataclass(frozen=True, slots=True)
class _StopRequested:
    reason: VoiceStopReason


@dataclass(frozen=True, slots=True)
class _MediaEnded:
    reason: VoiceMediaEndReason
    error_kind: str | None


@dataclass(frozen=True, slots=True)
class _PumpFailed:
    pump: str
    error_kind: str
    retryable_provider: bool = False


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


@dataclass(slots=True)
class _ActiveResponse:
    response_id: str
    generation: int
    provider_item_id: str = ""
    provider_done: bool = False
    pending_transcript: VoiceTranscriptFinal | None = None


_AudioEvent = VoiceAssistantAudio | _AudioBarrier
_ControlEvent = (
    _ProviderEvent
    | _StopRequested
    | _MediaEnded
    | _PumpFailed
    | _WatchdogExpired
    | _ResponseDrained
    | _ToolCallFinished
)


class RealtimeVoiceSessionRuntimeImpl(VoiceSessionRuntime):
    """Coordinate media and Provider tasks; only the control loop mutates session state."""

    def __init__(
        self,
        context: VoiceSessionRuntimeContext,
        settings: VoiceSettings,
        media_gateway: VoiceMediaGateway,
        sessions: VoiceSessionRepository,
        provider_builder: ProviderBuilder,
        task_controls: VoiceTaskControlHandler | None = None,
    ) -> None:
        self._context = context
        self._settings = settings
        self._media_gateway = media_gateway
        self._sessions = sessions
        self._provider_builder = provider_builder
        self._task_controls = task_controls
        self._control: asyncio.Queue[_ControlEvent] = asyncio.Queue(settings.event_queue_size)
        self._audio_events: asyncio.Queue[_AudioEvent] = asyncio.Queue(_AUDIO_STAGING_CHUNKS)
        self._input = VoiceInputQueue(settings.input_queue_ms)
        self._output = VoiceOutputTransitQueue(settings.output_queue_ms)
        self._state = VoiceSessionState.STARTING
        self._provider_ready = asyncio.Event()
        self._result: asyncio.Future[VoiceRuntimeResult] | None = None
        self._media: VoiceMediaSession | None = None
        self._provider: RealtimeVoiceProvider | None = None
        self._provider_session: RealtimeVoiceSession | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._drain_task: asyncio.Task[None] | None = None
        self._active_response: _ActiveResponse | None = None
        self._cancelled_response_ids: deque[str] = deque()
        self._cancelled_response_set: set[str] = set()
        self._tool_calls_in_flight: set[str] = set()
        self._completed_tool_call_ids: deque[str] = deque()
        self._completed_tool_call_set: set[str] = set()
        self._user_speaking = False
        self._turn_sequence = 0
        self._usage: dict[str, int | float] = {}
        self._stats = VoiceRuntimeStats()
        self._last_activity = time.monotonic()
        self._started_at = self._last_activity
        self._start_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._started = False
        self._closed = False

    @property
    def state(self) -> VoiceSessionState:
        return self._state

    @property
    def stats(self) -> VoiceRuntimeStats:
        return self._stats

    async def start(self) -> None:
        async with self._start_lock:
            if self._started:
                return
            if self._closed:
                raise RuntimeError("Voice runtime is closed")
            self._result = asyncio.get_running_loop().create_future()
            self._media = await self._media_gateway.open(
                self._context.descriptor,
                self._context.lease,
            )
            await self._connect_provider()
            self._spawn(self._control_loop(), "voice-control")
            self._spawn(self._oopz_input_pump(), "voice-oopz-input")
            self._spawn(self._provider_input_sender(), "voice-provider-input")
            self._spawn(self._provider_audio_router(), "voice-provider-audio")
            self._spawn(self._oopz_output_pump(), "voice-oopz-output")
            self._spawn(self._media_terminal_watcher(), "voice-media-terminal")
            self._spawn(self._watchdog(), "voice-watchdog")
            session = self._provider_session
            if session is None:  # pragma: no cover - guarded by _connect_provider
                raise RuntimeError("Voice Provider session was not connected")
            self._spawn(self._provider_event_pump(session), "voice-provider-events")
            try:
                async with asyncio.timeout(self._settings.start_timeout_seconds):
                    ready = asyncio.create_task(self._provider_ready.wait())
                    try:
                        done, _ = await asyncio.wait(
                            {ready, self._result},
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        if self._result in done or ready not in done:
                            raise VoiceProviderDisconnectedError(
                                "Provider failed before session configuration was ready"
                            )
                    finally:
                        ready.cancel()
                        with suppress(asyncio.CancelledError):
                            await ready
            except TimeoutError as exc:
                raise VoiceProviderDisconnectedError(
                    "Provider did not acknowledge session configuration"
                ) from exc
            self._started = True

    async def wait_finished(self) -> VoiceRuntimeResult:
        if self._result is None:
            raise RuntimeError("Voice runtime has not started")
        return await self._result

    async def request_stop(self, reason: VoiceStopReason) -> None:
        if self._closed:
            return
        await self._control.put(_StopRequested(reason))

    async def aclose(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            self._closed = True
            self._provider_ready.clear()
            await self._input.aclose()
            await self._output.aclose()
            current = asyncio.current_task()
            tasks = tuple(task for task in self._tasks if task is not current)
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            self._tasks.clear()
            provider_session = self._provider_session
            if provider_session is not None:
                with suppress(Exception):
                    async with asyncio.timeout(1.0):
                        await provider_session.finish()
            provider = self._provider
            if provider is not None:
                with suppress(Exception):
                    await provider.aclose()
            media = self._media
            if media is not None:
                with suppress(Exception):
                    await media.flush_output()
                await media.aclose()
            self._state = VoiceSessionState.CLOSED

    async def _connect_provider(self) -> None:
        last_error: Exception | None = None
        for attempt in range(1, self._settings.provider_connect_attempts + 1):
            provider = self._provider_builder(self._context)
            try:
                session = await provider.connect(self._context.descriptor)
            except asyncio.CancelledError:
                with suppress(Exception):
                    await provider.aclose()
                raise
            except Exception as exc:
                last_error = exc
                with suppress(Exception):
                    await provider.aclose()
                if (
                    isinstance(
                        exc,
                        (VoiceProviderAuthenticationError, VoiceProviderConfigurationError),
                    )
                    or attempt >= self._settings.provider_connect_attempts
                ):
                    break
                delay = min(1.5, 0.2 * (2 ** (attempt - 1)))
                delay *= random.uniform(0.8, 1.2)
                logger.warning(
                    "Voice Provider connect retry: session=%s attempt=%d error=%s",
                    opaque_ref(str(self._context.descriptor.session_id)),
                    attempt,
                    exception_kind(exc),
                )
                await asyncio.sleep(delay)
            else:
                self._provider = provider
                self._provider_session = session
                return
        raise VoiceProviderDisconnectedError(
            "Voice Provider connection attempts exhausted"
        ) from last_error

    async def _recover_provider(self, *, playout_flushed: bool = False) -> bool:
        self._state = VoiceSessionState.RECOVERING
        self._provider_ready.clear()
        self._discard_active_response()
        await self._cancel_response_drain()
        try:
            await self._sessions.mark_recovering(self._context.descriptor.session_id)
        except Exception as exc:
            logger.warning(
                "Could not persist voice recovery state: session=%s error=%s",
                opaque_ref(str(self._context.descriptor.session_id)),
                exception_kind(exc),
            )
        if not playout_flushed:
            await self._output.flush()
            media = self._media
            if media is not None:
                with suppress(Exception):
                    await media.flush_output()
        old_provider = self._provider
        self._provider = None
        self._provider_session = None
        if old_provider is not None:
            with suppress(Exception):
                await old_provider.aclose()
        try:
            await self._connect_provider()
        except Exception as exc:
            logger.warning(
                "Voice Provider recovery exhausted: session=%s error=%s",
                opaque_ref(str(self._context.descriptor.session_id)),
                exception_kind(exc),
            )
            return False
        session = self._provider_session
        if session is None:  # pragma: no cover - guarded by _connect_provider
            return False
        self._spawn(self._provider_event_pump(session), "voice-provider-events-recovered")
        return True

    async def _control_loop(self) -> None:
        while not self._closed:
            event = await self._control.get()
            if isinstance(event, _StopRequested):
                self._state = VoiceSessionState.CLOSING
                self._complete(event.reason)
                return
            if isinstance(event, _WatchdogExpired):
                self._state = VoiceSessionState.CLOSING
                self._complete(event.reason)
                return
            if isinstance(event, _MediaEnded):
                self._state = VoiceSessionState.CLOSING
                reason = (
                    VoiceStopReason.OWNER_LEFT
                    if event.reason
                    in {
                        VoiceMediaEndReason.OWNER_LEFT,
                        VoiceMediaEndReason.OWNER_UNPUBLISHED,
                    }
                    else VoiceStopReason.MEDIA_ENDED
                )
                self._complete(reason)
                return
            if isinstance(event, _PumpFailed):
                if event.retryable_provider and await self._recover_provider():
                    if event.pump == "provider_input":
                        self._spawn(
                            self._provider_input_sender(),
                            "voice-provider-input-recovered",
                        )
                    elif event.pump == "provider_output_backpressure":
                        self._spawn(
                            self._provider_audio_router(),
                            "voice-provider-audio-recovered",
                        )
                    continue
                self._state = VoiceSessionState.FAILED
                reason = (
                    VoiceStopReason.PROVIDER_FAILED
                    if event.retryable_provider or event.pump.startswith("provider")
                    else VoiceStopReason.MEDIA_ENDED
                )
                self._complete(reason)
                return
            if isinstance(event, _ResponseDrained):
                await self._handle_response_drained(event)
                continue
            if isinstance(event, _ToolCallFinished):
                await self._handle_tool_call_finished(event)
                continue
            if event.session is not self._provider_session:
                continue
            model_event = event.event
            self._last_activity = time.monotonic()
            if isinstance(model_event, VoiceSessionReady):
                self._state = (
                    VoiceSessionState.USER_SPEAKING
                    if self._user_speaking
                    else VoiceSessionState.LISTENING
                )
                self._provider_ready.set()
                if self._started:
                    try:
                        await self._sessions.mark_active(self._context.descriptor.session_id)
                    except Exception as exc:
                        logger.warning(
                            "Could not persist recovered voice active state: session=%s error=%s",
                            opaque_ref(str(self._context.descriptor.session_id)),
                            exception_kind(exc),
                        )
            elif isinstance(model_event, VoiceUserSpeechStarted):
                if self._user_speaking:
                    self._stats = replace(
                        self._stats,
                        duplicate_speech_started=self._stats.duplicate_speech_started + 1,
                    )
                    continue
                self._user_speaking = True
                if self._active_response is None:
                    self._state = VoiceSessionState.USER_SPEAKING
                    continue
                await self._interrupt_active_response(
                    notify_provider=True,
                    target_state=VoiceSessionState.USER_SPEAKING,
                    count_barge_in=True,
                )
            elif isinstance(model_event, VoiceUserSpeechStopped):
                self._user_speaking = False
                self._state = VoiceSessionState.THINKING
            elif isinstance(model_event, VoiceResponseStarted):
                await self._start_response(model_event.response_id)
            elif isinstance(model_event, VoiceAssistantAudio):
                await self._stage_assistant_audio(model_event)
            elif isinstance(model_event, VoiceTranscriptFinal):
                await self._handle_final_transcript(model_event)
            elif isinstance(model_event, VoiceResponseCompleted):
                self._accumulate_usage(model_event.usage)
                await self._complete_response_playout(model_event.response_id)
            elif isinstance(model_event, VoiceResponseCancelled):
                self._accumulate_usage(model_event.usage)
                if model_event.response_id in self._cancelled_response_set:
                    continue
                active = self._active_response
                if active is None or active.response_id != model_event.response_id:
                    continue
                await self._interrupt_active_response(
                    notify_provider=False,
                    target_state=(
                        VoiceSessionState.USER_SPEAKING
                        if self._user_speaking
                        else VoiceSessionState.LISTENING
                    ),
                    count_barge_in=False,
                )
            elif isinstance(model_event, VoiceToolCall):
                self._start_tool_call(event.session, model_event)
            elif isinstance(model_event, VoiceProviderFailed):
                if model_event.retryable and await self._recover_provider():
                    continue
                self._state = VoiceSessionState.FAILED
                self._complete(VoiceStopReason.PROVIDER_FAILED)
                return
            elif isinstance(model_event, VoiceSessionFinished):
                self._state = VoiceSessionState.CLOSING
                self._complete(VoiceStopReason.RUNTIME_ENDED)
                return

    def _start_tool_call(
        self,
        session: RealtimeVoiceSession,
        event: VoiceToolCall,
    ) -> None:
        if (
            event.call_id in self._tool_calls_in_flight
            or event.call_id in self._completed_tool_call_set
        ):
            return
        self._tool_calls_in_flight.add(event.call_id)
        self._stats = replace(
            self._stats,
            task_control_calls=self._stats.task_control_calls + 1,
        )
        self._spawn(
            self._execute_tool_call(session, event),
            "voice-task-control",
        )

    async def _execute_tool_call(
        self,
        session: RealtimeVoiceSession,
        event: VoiceToolCall,
    ) -> None:
        controls = self._task_controls
        started_at = time.monotonic()
        if controls is None:
            output: Mapping[str, object] = {"ok": False, "code": "tool_not_available"}
        else:
            try:
                timeout = (
                    _TASK_SUBMIT_TIMEOUT_SECONDS
                    if event.name == "delegate_agent_task"
                    else _TASK_READ_TIMEOUT_SECONDS
                )
                async with asyncio.timeout(timeout):
                    output = await controls.execute(
                        self._context.descriptor,
                        event.call_id,
                        event.name,
                        event.arguments,
                    )
            except asyncio.CancelledError:
                raise
            except TimeoutError:
                output = {"ok": False, "code": "temporarily_unavailable"}
            except Exception as exc:
                logger.exception(
                    "Voice task control handler failed: session=%s tool=%s error=%s",
                    opaque_ref(str(self._context.descriptor.session_id)),
                    event.name,
                    exception_kind(exc),
                )
                output = {"ok": False, "code": "internal_error"}
        await self._control.put(
            _ToolCallFinished(
                session,
                event.call_id,
                event.name,
                dict(output),
                (time.monotonic() - started_at) * 1000,
            )
        )

    async def _handle_tool_call_finished(self, event: _ToolCallFinished) -> None:
        self._tool_calls_in_flight.discard(event.call_id)
        failed = event.output.get("ok") is not True
        self._stats = replace(
            self._stats,
            task_control_failures=(
                self._stats.task_control_failures + 1
                if failed
                else self._stats.task_control_failures
            ),
            last_task_control_ms=event.elapsed_ms,
            max_task_control_ms=max(self._stats.max_task_control_ms, event.elapsed_ms),
        )
        if event.session is not self._provider_session:
            return
        try:
            async with asyncio.timeout(_INTERRUPT_TIMEOUT_SECONDS):
                await event.session.complete_tool_call(event.call_id, event.output)
        except Exception as exc:
            logger.warning(
                "Voice tool result delivery failed: session=%s tool=%s error=%s",
                opaque_ref(str(self._context.descriptor.session_id)),
                event.name,
                exception_kind(exc),
            )
            if not await self._recover_provider():
                self._state = VoiceSessionState.FAILED
                self._complete(VoiceStopReason.PROVIDER_FAILED)
            return
        self._remember_completed_tool_call(event.call_id)
        logger.info(
            "Voice task control completed: session=%s tool=%s call=%s elapsed_ms=%.1f ok=%s",
            opaque_ref(str(self._context.descriptor.session_id)),
            event.name,
            opaque_ref(event.call_id),
            event.elapsed_ms,
            not failed,
        )

    def _remember_completed_tool_call(self, call_id: str) -> None:
        if len(self._completed_tool_call_ids) >= _COMPLETED_TOOL_CALL_HISTORY:
            expired = self._completed_tool_call_ids.popleft()
            self._completed_tool_call_set.discard(expired)
        self._completed_tool_call_ids.append(call_id)
        self._completed_tool_call_set.add(call_id)

    async def _start_response(self, response_id: str) -> None:
        if response_id in self._cancelled_response_set:
            return
        active = self._active_response
        if active is not None:
            if active.response_id == response_id:
                return
            logger.warning(
                "Voice Provider started overlapping response; discarding old playout: "
                "session=%s old=%s new=%s",
                opaque_ref(str(self._context.descriptor.session_id)),
                opaque_ref(active.response_id),
                opaque_ref(response_id),
            )
            if not await self._interrupt_active_response(
                notify_provider=False,
                target_state=VoiceSessionState.THINKING,
                count_barge_in=False,
            ):
                return
        generation = await self._output.start_generation()
        self._active_response = _ActiveResponse(response_id, generation)
        self._stats = replace(
            self._stats,
            responses_started=self._stats.responses_started + 1,
        )
        self._state = VoiceSessionState.THINKING

    async def _stage_assistant_audio(self, event: VoiceAssistantAudio) -> None:
        active = self._active_response
        if (
            active is None
            or event.response_id in self._cancelled_response_set
            or event.response_id != active.response_id
        ):
            self._stats = replace(
                self._stats,
                late_audio_dropped=self._stats.late_audio_dropped + 1,
            )
            return
        if event.provider_item_id:
            active.provider_item_id = event.provider_item_id
        self._state = VoiceSessionState.SPEAKING
        try:
            self._audio_events.put_nowait(event)
        except asyncio.QueueFull:
            self._stats = replace(
                self._stats,
                output_overflows=self._stats.output_overflows + 1,
            )
            logger.warning(
                "Voice output staging overflow; cancelling response: session=%s response=%s",
                opaque_ref(str(self._context.descriptor.session_id)),
                opaque_ref(active.response_id),
            )
            await self._interrupt_active_response(
                notify_provider=True,
                target_state=(
                    VoiceSessionState.USER_SPEAKING
                    if self._user_speaking
                    else VoiceSessionState.LISTENING
                ),
                count_barge_in=False,
            )

    async def _handle_final_transcript(self, event: VoiceTranscriptFinal) -> None:
        if event.role == "user":
            await self._persist_transcript(event)
            return
        response_id = event.response_id
        active = self._active_response
        if not response_id and active is not None:
            response_id = active.response_id
        if response_id in self._cancelled_response_set:
            self._stats = replace(
                self._stats,
                interrupted_transcripts_dropped=(self._stats.interrupted_transcripts_dropped + 1),
            )
            return
        if active is None or response_id != active.response_id:
            logger.debug(
                "Ignoring assistant transcript without active response: session=%s response=%s",
                opaque_ref(str(self._context.descriptor.session_id)),
                opaque_ref(response_id),
            )
            return
        active.pending_transcript = event

    async def _persist_transcript(self, event: VoiceTranscriptFinal) -> None:
        self._turn_sequence += 1
        try:
            await self._sessions.append_final_turn(
                self._context.descriptor.session_id,
                self._turn_sequence,
                VoiceTurnRole(event.role),
                event.text,
                provider_item_id=event.provider_item_id,
            )
        except Exception as exc:
            logger.warning(
                "Could not persist final voice transcript: session=%s sequence=%d error=%s",
                opaque_ref(str(self._context.descriptor.session_id)),
                self._turn_sequence,
                exception_kind(exc),
            )

    async def _complete_response_playout(self, response_id: str) -> None:
        if response_id in self._cancelled_response_set:
            return
        active = self._active_response
        if active is None or active.response_id != response_id:
            return
        if active.provider_done:
            return
        active.provider_done = True
        await self._cancel_response_drain()
        task = self._spawn(
            self._drain_response(response_id, active.generation),
            "voice-response-drain",
        )
        self._drain_task = task

    async def _drain_response(
        self,
        response_id: str,
        generation: int,
    ) -> None:
        try:
            routed = asyncio.get_running_loop().create_future()
            await self._audio_events.put(_AudioBarrier(response_id, generation, routed))
            await routed
            await self._output.wait_empty(generation)
            cursor = await self._require_media().drain_output()
            await self._control.put(_ResponseDrained(response_id, generation, cursor))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._control.put(_PumpFailed("oopz_output_drain", exception_kind(exc)))

    async def _handle_response_drained(self, event: _ResponseDrained) -> None:
        active = self._active_response
        if (
            active is None
            or active.response_id != event.response_id
            or active.generation != event.generation
        ):
            return
        transcript = active.pending_transcript
        self._active_response = None
        self._drain_task = None
        if transcript is not None:
            await self._persist_transcript(transcript)
        self._stats = replace(
            self._stats,
            responses_drained=self._stats.responses_drained + 1,
        )
        self._state = (
            VoiceSessionState.USER_SPEAKING if self._user_speaking else VoiceSessionState.LISTENING
        )
        logger.debug(
            "Voice response playout drained: session=%s response=%s generation=%d rendered_ms=%d",
            opaque_ref(str(self._context.descriptor.session_id)),
            opaque_ref(event.response_id),
            event.generation,
            event.cursor.rendered_ms,
        )

    async def _interrupt_active_response(
        self,
        *,
        notify_provider: bool,
        target_state: VoiceSessionState,
        count_barge_in: bool,
    ) -> bool:
        active = self._active_response
        if active is None:
            self._state = target_state
            return True
        started_at = time.monotonic()
        self._state = VoiceSessionState.INTERRUPTING
        if count_barge_in:
            self._stats = replace(
                self._stats,
                barge_in_count=self._stats.barge_in_count + 1,
            )
        self._discard_active_response()
        await self._cancel_response_drain()
        try:
            await self._output.flush()
            cursor = await self._require_media().flush_output()
        except Exception as exc:
            logger.error(
                "Voice local playout flush failed: session=%s error=%s",
                opaque_ref(str(self._context.descriptor.session_id)),
                exception_kind(exc),
            )
            self._state = VoiceSessionState.FAILED
            self._complete(VoiceStopReason.MEDIA_ENDED)
            return False
        flush_ms = (time.monotonic() - started_at) * 1000
        if count_barge_in:
            self._stats = replace(
                self._stats,
                last_barge_in_flush_ms=flush_ms,
                max_barge_in_flush_ms=max(self._stats.max_barge_in_flush_ms, flush_ms),
            )
        if notify_provider and not active.provider_done:
            session = self._provider_session
            if session is not None:
                try:
                    async with asyncio.timeout(_INTERRUPT_TIMEOUT_SECONDS):
                        await session.interrupt(cursor)
                except Exception as exc:
                    logger.warning(
                        "Voice Provider interrupt failed after local flush: "
                        "session=%s response=%s error=%s",
                        opaque_ref(str(self._context.descriptor.session_id)),
                        opaque_ref(active.response_id),
                        exception_kind(exc),
                    )
                    if await self._recover_provider(playout_flushed=True):
                        return True
                    self._state = VoiceSessionState.FAILED
                    self._complete(VoiceStopReason.PROVIDER_FAILED)
                    return False
        self._state = target_state
        logger.info(
            "Voice response interrupted: session=%s response=%s generation=%d "
            "rendered_ms=%d flush_ms=%.1f provider_cancel=%s",
            opaque_ref(str(self._context.descriptor.session_id)),
            opaque_ref(active.response_id),
            active.generation,
            cursor.rendered_ms,
            flush_ms,
            notify_provider and not active.provider_done,
        )
        return True

    async def _cancel_response_drain(self) -> None:
        task = self._drain_task
        self._drain_task = None
        if task is None or task.done() or task is asyncio.current_task():
            return
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    def _discard_active_response(self) -> None:
        active = self._active_response
        if active is None:
            return
        self._remember_cancelled_response(active.response_id)
        if active.pending_transcript is not None:
            self._stats = replace(
                self._stats,
                interrupted_transcripts_dropped=(self._stats.interrupted_transcripts_dropped + 1),
            )
        self._active_response = None

    def _remember_cancelled_response(self, response_id: str) -> None:
        if response_id in self._cancelled_response_set:
            return
        if len(self._cancelled_response_ids) >= _CANCELLED_RESPONSE_HISTORY:
            expired = self._cancelled_response_ids.popleft()
            self._cancelled_response_set.discard(expired)
        self._cancelled_response_ids.append(response_id)
        self._cancelled_response_set.add(response_id)

    def _accumulate_usage(self, usage) -> None:
        for key, value in usage.items():
            self._usage[key] = self._usage.get(key, 0) + value

    async def _oopz_input_pump(self) -> None:
        media = self._require_media()
        ingress = VoiceAudioIngress()
        try:
            async for frame in media.input_frames():
                self._last_activity = time.monotonic()
                for packet in ingress.push(frame):
                    dropped = self._input.put(packet)
                    if dropped:
                        logger.warning(
                            "Voice Provider input queue dropped oldest packet: session=%s depth=%d",
                            opaque_ref(str(self._context.descriptor.session_id)),
                            self._input.qsize,
                        )
            for packet in ingress.flush():
                self._input.put(packet)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._control.put(_PumpFailed("oopz_input", exception_kind(exc)))

    async def _provider_input_sender(self) -> None:
        try:
            while True:
                packet = await self._input.get()
                await self._provider_ready.wait()
                session = self._provider_session
                if session is None:
                    continue
                await session.send_audio(packet)
        except VoiceAudioQueueClosedError:
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._control.put(
                _PumpFailed("provider_input", exception_kind(exc), retryable_provider=True)
            )

    async def _provider_audio_router(self) -> None:
        try:
            while True:
                event = await self._audio_events.get()
                try:
                    if isinstance(event, _AudioBarrier):
                        if not event.routed.done():
                            event.routed.set_result(None)
                        continue
                    active = self._active_response
                    if (
                        active is None
                        or event.response_id != active.response_id
                        or event.response_id in self._cancelled_response_set
                    ):
                        self._stats = replace(
                            self._stats,
                            late_audio_dropped=self._stats.late_audio_dropped + 1,
                        )
                        continue
                    chunk = event.chunk
                    accepted = await self._output.put(
                        PcmChunk(
                            chunk.pcm,
                            chunk.format,
                            chunk.duration_ms,
                            active.generation,
                        )
                    )
                    if not accepted:
                        self._stats = replace(
                            self._stats,
                            late_audio_dropped=self._stats.late_audio_dropped + 1,
                        )
                finally:
                    self._audio_events.task_done()
        except VoiceAudioQueueClosedError:
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._control.put(
                _PumpFailed(
                    "provider_output_backpressure",
                    exception_kind(exc),
                    retryable_provider=True,
                )
            )

    async def _provider_event_pump(self, session: RealtimeVoiceSession) -> None:
        terminal = False
        try:
            async for event in session.events():
                terminal = isinstance(event, VoiceProviderFailed | VoiceSessionFinished)
                await self._control.put(_ProviderEvent(session, event))
            if not terminal and not self._closed and session is self._provider_session:
                await self._control.put(_PumpFailed("provider_events_eof", "premature_eof", True))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._control.put(
                _PumpFailed("provider_events", exception_kind(exc), retryable_provider=True)
            )

    async def _oopz_output_pump(self) -> None:
        media = self._require_media()
        try:
            while True:
                chunk = await self._output.get()
                if chunk.generation != self._output.generation:
                    continue
                try:
                    await media.write_output(chunk)
                except Exception:
                    if chunk.generation != self._output.generation:
                        logger.debug(
                            "Ignoring output write interrupted by generation flush: "
                            "session=%s generation=%d",
                            opaque_ref(str(self._context.descriptor.session_id)),
                            chunk.generation,
                        )
                        continue
                    raise
        except VoiceAudioQueueClosedError:
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._control.put(_PumpFailed("oopz_output", exception_kind(exc)))

    async def _media_terminal_watcher(self) -> None:
        try:
            terminal = await self._require_media().wait_input_closed()
            await self._control.put(_MediaEnded(terminal.reason, terminal.error_kind))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._control.put(
                _MediaEnded(VoiceMediaEndReason.TRANSPORT_LOST, exception_kind(exc))
            )

    async def _watchdog(self) -> None:
        idle_seconds = self._context.configuration.channel.idle_timeout_seconds
        while True:
            await asyncio.sleep(min(1.0, idle_seconds / 4))
            now = time.monotonic()
            if now - self._started_at >= self._settings.max_session_seconds:
                await self._control.put(_WatchdogExpired(VoiceStopReason.MAX_DURATION))
                return
            if now - self._last_activity >= idle_seconds:
                await self._control.put(_WatchdogExpired(VoiceStopReason.IDLE_TIMEOUT))
                return

    def _spawn(self, coroutine, label: str) -> asyncio.Task[None]:
        task = asyncio.create_task(
            coroutine,
            name=f"{label}:{self._context.descriptor.session_id}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._task_done)
        return task

    def _task_done(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is None:
            return
        logger.error(
            "Voice runtime task failed unexpectedly: session=%s task=%s error=%s",
            opaque_ref(str(self._context.descriptor.session_id)),
            task.get_name().split(":", 1)[0],
            exception_kind(error),
        )
        self._state = VoiceSessionState.FAILED
        self._complete(VoiceStopReason.PROVIDER_FAILED)

    def _complete(self, reason: VoiceStopReason) -> None:
        if self._result is not None and not self._result.done():
            self._result.set_result(
                VoiceRuntimeResult(
                    reason,
                    dict(self._usage),
                    self._stats.as_metrics(),
                )
            )

    def _require_media(self) -> VoiceMediaSession:
        if self._media is None:  # pragma: no cover - start ordering invariant
            raise RuntimeError("Voice media is not open")
        return self._media


class RealtimeVoiceSessionRuntimeFactoryImpl(VoiceSessionRuntimeFactory):
    """Build one coordinator while keeping all dependencies in BotApplication."""

    def __init__(
        self,
        settings: VoiceSettings,
        media_gateway: VoiceMediaGateway,
        sessions: VoiceSessionRepository,
        provider_builder: ProviderBuilder,
        task_controls: VoiceTaskControlHandler | None = None,
    ) -> None:
        self._settings = settings
        self._media_gateway = media_gateway
        self._sessions = sessions
        self._provider_builder = provider_builder
        self._task_controls = task_controls

    async def create(self, context: VoiceSessionRuntimeContext) -> VoiceSessionRuntime:
        return RealtimeVoiceSessionRuntimeImpl(
            context,
            self._settings,
            self._media_gateway,
            self._sessions,
            self._provider_builder,
            self._task_controls,
        )
