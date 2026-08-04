"""Provider-neutral realtime session coordinator with bounded async pumps."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass

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
    VoiceResponseCompleted,
    VoiceSessionFinished,
    VoiceSessionReady,
    VoiceTranscriptFinal,
    VoiceUserSpeechStarted,
    VoiceUserSpeechStopped,
)
from .models import (
    PcmChunk,
    VoiceMediaEndReason,
    VoiceRuntimeResult,
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
)
from .settings import VoiceTurnRole

logger = logging.getLogger(__name__)

ProviderBuilder = Callable[[VoiceSessionRuntimeContext], RealtimeVoiceProvider]


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


_ControlEvent = _ProviderEvent | _StopRequested | _MediaEnded | _PumpFailed | _WatchdogExpired


class RealtimeVoiceSessionRuntimeImpl(VoiceSessionRuntime):
    """Coordinate media and Provider tasks; only the control loop mutates session state."""

    def __init__(
        self,
        context: VoiceSessionRuntimeContext,
        settings: VoiceSettings,
        media_gateway: VoiceMediaGateway,
        sessions: VoiceSessionRepository,
        provider_builder: ProviderBuilder,
    ) -> None:
        self._context = context
        self._settings = settings
        self._media_gateway = media_gateway
        self._sessions = sessions
        self._provider_builder = provider_builder
        self._control: asyncio.Queue[_ControlEvent] = asyncio.Queue(settings.event_queue_size)
        self._input = VoiceInputQueue(settings.input_queue_ms)
        self._output = VoiceOutputTransitQueue(settings.output_queue_ms)
        self._state = VoiceSessionState.STARTING
        self._provider_ready = asyncio.Event()
        self._result: asyncio.Future[VoiceRuntimeResult] | None = None
        self._media: VoiceMediaSession | None = None
        self._provider: RealtimeVoiceProvider | None = None
        self._provider_session: RealtimeVoiceSession | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._turn_sequence = 0
        self._usage: dict[str, int | float] = {}
        self._last_activity = time.monotonic()
        self._started_at = self._last_activity
        self._start_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._started = False
        self._closed = False

    @property
    def state(self) -> VoiceSessionState:
        return self._state

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

    async def _recover_provider(self) -> bool:
        self._state = VoiceSessionState.RECOVERING
        self._provider_ready.clear()
        try:
            await self._sessions.mark_recovering(self._context.descriptor.session_id)
        except Exception as exc:
            logger.warning(
                "Could not persist voice recovery state: session=%s error=%s",
                opaque_ref(str(self._context.descriptor.session_id)),
                exception_kind(exc),
            )
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
                    continue
                self._state = VoiceSessionState.FAILED
                reason = (
                    VoiceStopReason.PROVIDER_FAILED
                    if event.retryable_provider or event.pump.startswith("provider")
                    else VoiceStopReason.MEDIA_ENDED
                )
                self._complete(reason)
                return
            if event.session is not self._provider_session:
                continue
            model_event = event.event
            self._last_activity = time.monotonic()
            if isinstance(model_event, VoiceSessionReady):
                self._state = VoiceSessionState.LISTENING
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
                self._state = VoiceSessionState.USER_SPEAKING
            elif isinstance(model_event, VoiceUserSpeechStopped):
                self._state = VoiceSessionState.THINKING
            elif isinstance(model_event, VoiceAssistantAudio):
                self._state = VoiceSessionState.SPEAKING
                chunk = model_event.chunk
                await self._output.put(
                    PcmChunk(
                        chunk.pcm,
                        chunk.format,
                        chunk.duration_ms,
                        self._output.generation,
                    )
                )
            elif isinstance(model_event, VoiceTranscriptFinal):
                self._turn_sequence += 1
                try:
                    await self._sessions.append_final_turn(
                        self._context.descriptor.session_id,
                        self._turn_sequence,
                        VoiceTurnRole(model_event.role),
                        model_event.text,
                        provider_item_id=model_event.provider_item_id,
                    )
                except Exception as exc:
                    logger.warning(
                        "Could not persist final voice transcript: session=%s sequence=%d error=%s",
                        opaque_ref(str(self._context.descriptor.session_id)),
                        self._turn_sequence,
                        exception_kind(exc),
                    )
            elif isinstance(model_event, VoiceResponseCompleted):
                for key, value in model_event.usage.items():
                    self._usage[key] = self._usage.get(key, 0) + value
                self._state = VoiceSessionState.LISTENING
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
                await media.write_output(await self._output.get())
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

    def _spawn(self, coroutine, label: str) -> None:
        task = asyncio.create_task(
            coroutine,
            name=f"{label}:{self._context.descriptor.session_id}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._task_done)

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
            self._result.set_result(VoiceRuntimeResult(reason, dict(self._usage)))

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
    ) -> None:
        self._settings = settings
        self._media_gateway = media_gateway
        self._sessions = sessions
        self._provider_builder = provider_builder

    async def create(self, context: VoiceSessionRuntimeContext) -> VoiceSessionRuntime:
        return RealtimeVoiceSessionRuntimeImpl(
            context,
            self._settings,
            self._media_gateway,
            self._sessions,
            self._provider_builder,
        )
