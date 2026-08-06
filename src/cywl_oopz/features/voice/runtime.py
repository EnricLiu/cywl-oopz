"""Provider-neutral realtime session coordinator with bounded async pumps."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections import deque
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field, replace
from uuid import UUID

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
    VoiceProviderErrorEvent,
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
    VoiceProviderCapabilities,
    VoiceRecoveryContext,
    VoiceRecoveryTask,
    VoiceRecoveryTurn,
    VoiceRuntimeResult,
    VoiceRuntimeStats,
    VoiceRuntimeStatus,
    VoiceSessionState,
    VoiceStopReason,
    VoiceTaskNotification,
)
from .notifications import (
    VoiceTaskNotificationStrategy,
    compile_internal_task_context,
    select_task_notification_strategy,
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
    VoiceTaskMailbox,
)
from .settings import VoiceTurnRole

logger = logging.getLogger(__name__)

ProviderBuilder = Callable[[VoiceSessionRuntimeContext], RealtimeVoiceProvider]
_AUDIO_STAGING_CHUNKS = 128
_CANCELLED_RESPONSE_HISTORY = 64
_RESPONSE_USAGE_HISTORY = 64
_COMPLETED_TOOL_CALL_HISTORY = 128
_INTERRUPT_TIMEOUT_SECONDS = 1.5
_TASK_SUBMIT_TIMEOUT_SECONDS = 0.25
_TASK_READ_TIMEOUT_SECONDS = 0.15
_NOTIFICATION_SILENCE_SECONDS = 0.7
_NOTIFICATION_COALESCE_SECONDS = 0.4
_NOTIFICATION_COOLDOWN_SECONDS = 5.0
_NOTIFICATION_BATCH_LIMIT = 3
_NOTIFICATION_PERSIST_ATTEMPTS = 3
_NOTIFICATION_PERSIST_RETRY_SECONDS = 0.05


def _split_pcm_chunk(chunk: PcmChunk, maximum_duration_ms: int) -> tuple[PcmChunk, ...]:
    """Split a Provider delta so no piece can exceed the transit queue bound."""
    if chunk.duration_ms <= maximum_duration_ms:
        return (chunk,)
    maximum_samples = max(1, chunk.format.sample_rate * maximum_duration_ms // 1_000)
    frame_width_bytes = chunk.format.frame_width_bytes
    chunks = []
    for start_sample in range(0, len(chunk.pcm) // frame_width_bytes, maximum_samples):
        pcm = chunk.pcm[
            start_sample * frame_width_bytes : (start_sample + maximum_samples) * frame_width_bytes
        ]
        duration_ms = round(
            (len(pcm) // frame_width_bytes) * 1_000 / chunk.format.sample_rate
        )
        chunks.append(PcmChunk(pcm, chunk.format, duration_ms, chunk.generation))
    return tuple(chunks)


def _compact_recovery_text(text: str, limit: int) -> str:
    """Normalize Provider/user text before retaining a small reconnect snapshot."""
    normalized = " ".join(text.split())
    return normalized[:limit] or "（无可用文本）"


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
        task_mailbox: VoiceTaskMailbox | None = None,
    ) -> None:
        self._context = context
        self._settings = settings
        self._media_gateway = media_gateway
        self._sessions = sessions
        self._provider_builder = provider_builder
        self._task_controls = task_controls
        self._task_mailbox = task_mailbox
        self._control: asyncio.Queue[_ControlEvent] = asyncio.Queue(settings.event_queue_size)
        self._audio_events: asyncio.Queue[_AudioEvent] = asyncio.Queue(_AUDIO_STAGING_CHUNKS)
        self._input = VoiceInputQueue(settings.input_queue_ms)
        self._output = VoiceOutputTransitQueue(settings.output_queue_ms)
        self._state = VoiceSessionState.STARTING
        self._provider_ready = asyncio.Event()
        self._result: asyncio.Future[VoiceRuntimeResult] | None = None
        self._stop_signal: asyncio.Future[VoiceStopReason] | None = None
        self._media: VoiceMediaSession | None = None
        self._pending_media: VoiceMediaSession | None = None
        self._media_generation = 0
        self._media_tasks: set[asyncio.Task[None]] = set()
        self._media_recovery_task: asyncio.Task[None] | None = None
        self._media_recovery_started_at: float | None = None
        self._provider: RealtimeVoiceProvider | None = None
        self._provider_session: RealtimeVoiceSession | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._drain_task: asyncio.Task[None] | None = None
        self._active_response: _ActiveResponse | None = None
        self._cancelled_response_ids: deque[str] = deque()
        self._cancelled_response_set: set[str] = set()
        self._usage_response_ids: deque[str] = deque()
        self._usage_response_set: set[str] = set()
        self._tool_calls_in_flight: set[str] = set()
        self._completed_tool_call_ids: deque[str] = deque()
        self._completed_tool_call_set: set[str] = set()
        self._user_speaking = False
        self._turn_sequence = 0
        self._usage: dict[str, int | float] = {}
        self._stats = VoiceRuntimeStats()
        self._last_activity = time.monotonic()
        self._started_at = self._last_activity
        self._provider_connect_started_at = self._last_activity
        self._recovery_started_at: float | None = None
        self._last_user_speech_stopped = self._last_activity - _NOTIFICATION_SILENCE_SECONDS
        self._last_notification_attempt = self._last_activity - _NOTIFICATION_COOLDOWN_SECONDS
        self._mailbox_available_pending = False
        self._mailbox_claim_in_flight = False
        self._notification_in_flight = False
        self._proactive_task_notices: tuple[VoiceTaskNotification, ...] = ()
        self._recovery_turns: deque[VoiceRecoveryTurn] = deque(
            context.recovery_context.turns,
            maxlen=8,
        )
        self._recovery_tasks: deque[VoiceRecoveryTask] = deque(
            context.recovery_context.tasks,
            maxlen=3,
        )
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
            loop = asyncio.get_running_loop()
            self._result = loop.create_future()
            self._stop_signal = loop.create_future()
            self._media = await self._media_gateway.open(
                self._context.descriptor,
                self._context.lease,
            )
            await self._connect_provider()
            self._spawn(self._control_loop(), "voice-control")
            self._start_media_pumps(self._require_media(), self._media_generation)
            self._spawn(self._provider_input_sender(), "voice-provider-input")
            self._spawn(self._provider_audio_router(), "voice-provider-audio")
            self._spawn(self._watchdog(), "voice-watchdog")
            if self._task_mailbox is not None:
                self._spawn(self._mailbox_listener(), "voice-task-mailbox")
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
        return await asyncio.shield(self._result)

    async def request_stop(self, reason: VoiceStopReason) -> None:
        if self._closed:
            return
        signal = self._stop_signal
        if signal is not None and not signal.done():
            signal.set_result(reason)
        await self._control.put(_StopRequested(reason))

    async def aclose(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            self._closed = True
            self._provider_ready.clear()
            await self._input.aclose()
            await self._output.aclose()
            try:
                async with asyncio.timeout(self._settings.stop_timeout_seconds):
                    await self._close_owned_resources()
            except TimeoutError:
                logger.warning(
                    "Voice runtime cleanup exceeded stop budget: session=%s budget_seconds=%.2f",
                    opaque_ref(str(self._context.descriptor.session_id)),
                    self._settings.stop_timeout_seconds,
                )
            finally:
                self._set_state(VoiceSessionState.CLOSED)

    async def _close_owned_resources(self) -> None:
        notices = self._proactive_task_notices
        self._proactive_task_notices = ()
        current = asyncio.current_task()
        tasks = tuple(task for task in self._tasks if task is not current)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        cleanup = [
            self._close_provider_transport(),
            self._close_media_transport(),
        ]
        if notices and self._task_mailbox is not None:
            cleanup.append(self._defer_task_notifications(notices))
        await asyncio.gather(*cleanup)

    async def _defer_task_notifications(
        self,
        notices: tuple[VoiceTaskNotification, ...],
    ) -> None:
        if self._task_mailbox is None:  # pragma: no cover - guarded by caller
            return
        try:
            await self._task_mailbox.defer(tuple(item.task_id for item in notices))
        except Exception as exc:
            logger.warning(
                "Voice task notification defer failed during cleanup: session=%s error=%s",
                opaque_ref(str(self._context.descriptor.session_id)),
                exception_kind(exc),
                exc_info=True,
            )

    async def _close_provider_transport(self) -> None:
        provider_session = self._provider_session
        if provider_session is not None:
            try:
                async with asyncio.timeout(min(0.25, self._settings.stop_timeout_seconds / 4)):
                    await provider_session.finish()
            except TimeoutError:
                logger.warning(
                    "Voice Provider finish exceeded graceful budget: session=%s",
                    opaque_ref(str(self._context.descriptor.session_id)),
                )
            except Exception as exc:
                logger.warning(
                    "Voice Provider finish failed during cleanup: session=%s error=%s",
                    opaque_ref(str(self._context.descriptor.session_id)),
                    exception_kind(exc),
                    exc_info=True,
                )
        provider = self._provider
        if provider is not None:
            try:
                await provider.aclose()
            except Exception as exc:
                logger.warning(
                    "Voice Provider close failed during cleanup: session=%s error=%s",
                    opaque_ref(str(self._context.descriptor.session_id)),
                    exception_kind(exc),
                    exc_info=True,
                )

    async def _close_media_transport(self) -> None:
        media = self._media
        pending = self._pending_media
        self._pending_media = None
        candidates: list[VoiceMediaSession] = []
        if media is not None:
            candidates.append(media)
        if pending is not None and pending is not media:
            candidates.append(pending)
        if not candidates:
            return
        if media is not None:
            try:
                async with asyncio.timeout(min(0.25, self._settings.stop_timeout_seconds / 4)):
                    await media.flush_output()
            except TimeoutError:
                logger.warning(
                    "Voice output flush exceeded cleanup budget: session=%s",
                    opaque_ref(str(self._context.descriptor.session_id)),
                )
            except Exception as exc:
                logger.warning(
                    "Voice output flush failed during cleanup: session=%s error=%s",
                    opaque_ref(str(self._context.descriptor.session_id)),
                    exception_kind(exc),
                    exc_info=True,
                )
        await asyncio.gather(*(self._close_one_media(candidate) for candidate in candidates))

    async def _close_one_media(self, media: VoiceMediaSession) -> None:
        try:
            await media.aclose()
        except Exception as exc:
            logger.warning(
                "Voice media close failed during cleanup: session=%s error=%s",
                opaque_ref(str(self._context.descriptor.session_id)),
                exception_kind(exc),
                exc_info=True,
            )

    async def _connect_provider(self) -> None:
        last_error: Exception | None = None
        for attempt in range(1, self._settings.provider_connect_attempts + 1):
            self._provider_connect_started_at = time.monotonic()
            self._stats = replace(
                self._stats,
                provider_connect_attempts=self._stats.provider_connect_attempts + 1,
            )
            provider_context = replace(
                self._context,
                recovery_context=VoiceRecoveryContext(
                    tuple(self._recovery_turns),
                    tuple(self._recovery_tasks),
                ),
            )
            provider = self._provider_builder(provider_context)
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
                logger.warning(
                    "Voice Provider connect failed: session=%s attempt=%d error=%s",
                    opaque_ref(str(self._context.descriptor.session_id)),
                    attempt,
                    exception_kind(exc),
                    exc_info=True,
                )
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
                self._stats = replace(
                    self._stats,
                    provider_connections=self._stats.provider_connections + 1,
                )
                return
        raise VoiceProviderDisconnectedError(
            "Voice Provider connection attempts exhausted"
        ) from last_error

    async def _recover_provider(self, *, playout_flushed: bool = False) -> bool:
        if self._recovery_started_at is not None:
            self._stats = replace(
                self._stats,
                provider_recovery_failures=self._stats.provider_recovery_failures + 1,
            )
        self._recovery_started_at = time.monotonic()
        self._set_state(VoiceSessionState.RECOVERING)
        if self._proactive_task_notices:
            await self._defer_pending_proactive_notifications()
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
                exc_info=True,
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
            connected = await self._connect_provider_with_recovery_budget()
        except Exception as exc:
            self._stats = replace(
                self._stats,
                provider_recovery_failures=self._stats.provider_recovery_failures + 1,
            )
            self._recovery_started_at = None
            logger.warning(
                "Voice Provider recovery exhausted: session=%s budget_seconds=%.2f error=%s",
                opaque_ref(str(self._context.descriptor.session_id)),
                self._settings.start_timeout_seconds,
                exception_kind(exc),
                exc_info=True,
            )
            return False
        if not connected:
            self._recovery_started_at = None
            signal = self._stop_signal
            reason = (
                signal.result()
                if signal is not None and signal.done() and not signal.cancelled()
                else VoiceStopReason.COMMAND
            )
            logger.info(
                "Voice Provider recovery preempted: session=%s reason=%s",
                opaque_ref(str(self._context.descriptor.session_id)),
                reason.value,
            )
            self._set_state(VoiceSessionState.CLOSING)
            self._complete(reason)
            return False
        session = self._provider_session
        if session is None:  # pragma: no cover - guarded by _connect_provider
            return False
        self._spawn(self._provider_event_pump(session), "voice-provider-events-recovered")
        return True

    async def _connect_provider_with_recovery_budget(self) -> bool:
        """Race the bounded reconnect sequence against an explicit runtime stop."""
        connect_task = asyncio.create_task(
            self._connect_provider(),
            name=f"voice-provider-reconnect:{self._context.descriptor.session_id}",
        )
        signal = self._stop_signal
        try:
            async with asyncio.timeout(self._settings.start_timeout_seconds):
                if signal is None:  # pragma: no cover - start ordering invariant
                    await connect_task
                    return True
                done, _ = await asyncio.wait(
                    {connect_task, signal},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if signal in done:
                    return False
                await connect_task
                return True
        finally:
            if not connect_task.done():
                connect_task.cancel()
            await asyncio.gather(connect_task, return_exceptions=True)

    async def _begin_media_recovery(self, event: _MediaEnded) -> None:
        recovery = self._media_recovery_task
        if recovery is not None and not recovery.done():
            return
        self._media_recovery_started_at = time.monotonic()
        self._user_speaking = False
        if self._active_response is not None:
            recovered = await self._interrupt_active_response(
                notify_provider=True,
                target_state=VoiceSessionState.RECOVERING,
                count_barge_in=False,
            )
            if not recovered or (self._result is not None and self._result.done()):
                return
        else:
            self._set_state(VoiceSessionState.RECOVERING)
            await self._output.flush()
            media = self._media
            if media is not None:
                with suppress(Exception):
                    await media.flush_output()
        try:
            await self._sessions.mark_recovering(self._context.descriptor.session_id)
        except Exception as exc:
            logger.warning(
                "Could not persist voice media recovery state: session=%s error=%s",
                opaque_ref(str(self._context.descriptor.session_id)),
                exception_kind(exc),
                exc_info=True,
            )
        await self._cancel_media_pumps()
        task = self._spawn(
            self._recover_media(event.generation, event.reason, self._require_media()),
            "voice-media-recovery",
        )
        self._media_recovery_task = task
        logger.info(
            "Voice owner media recovery started: session=%s generation=%d "
            "grace_seconds=%d reason=%s",
            opaque_ref(str(self._context.descriptor.session_id)),
            event.generation,
            self._settings.owner_leave_grace_seconds,
            event.reason.value,
        )

    async def _recover_media(
        self,
        generation: int,
        reason: VoiceMediaEndReason,
        old_media: VoiceMediaSession,
    ) -> None:
        replacement: VoiceMediaSession | None = None
        try:
            async with asyncio.timeout(self._settings.owner_leave_grace_seconds):
                await old_media.aclose()
                replacement = await self._media_gateway.open(
                    self._context.descriptor,
                    self._context.lease,
                )
            self._pending_media = replacement
            await self._control.put(_MediaRecovered(generation, replacement))
        except asyncio.CancelledError:
            if replacement is not None:
                await self._discard_replacement_media(replacement)
                if self._pending_media is replacement:
                    self._pending_media = None
            raise
        except TimeoutError:
            if replacement is not None:
                await self._discard_replacement_media(replacement)
            await self._control.put(_MediaRecoveryFailed(generation, reason, "timeout"))
        except Exception as exc:
            if replacement is not None:
                await self._discard_replacement_media(replacement)
            logger.warning(
                "Voice media recovery failed: session=%s generation=%d reason=%s error=%s",
                opaque_ref(str(self._context.descriptor.session_id)),
                generation,
                reason.value,
                exception_kind(exc),
                exc_info=True,
            )
            await self._control.put(_MediaRecoveryFailed(generation, reason, exception_kind(exc)))

    async def _discard_replacement_media(self, media: VoiceMediaSession) -> None:
        try:
            async with asyncio.timeout(min(0.25, self._settings.stop_timeout_seconds / 4)):
                await media.aclose()
        except TimeoutError:
            logger.warning(
                "Replacement voice media close exceeded cleanup budget: session=%s",
                opaque_ref(str(self._context.descriptor.session_id)),
            )
        except Exception as exc:
            logger.warning(
                "Replacement voice media close failed: session=%s error=%s",
                opaque_ref(str(self._context.descriptor.session_id)),
                exception_kind(exc),
                exc_info=True,
            )

    async def _handle_media_recovered(self, event: _MediaRecovered) -> None:
        if event.generation != self._media_generation or self._closed:
            if self._pending_media is event.media:
                self._pending_media = None
            self._spawn(self._close_one_media(event.media), "voice-stale-media-close")
            return
        self._pending_media = None
        self._media = event.media
        self._media_generation += 1
        self._media_recovery_task = None
        recovered_at = time.monotonic()
        started_at = self._media_recovery_started_at
        recovery_ms = (recovered_at - started_at) * 1000 if started_at is not None else 0.0
        self._media_recovery_started_at = None
        self._last_activity = recovered_at
        self._stats = replace(
            self._stats,
            media_reconnects=self._stats.media_reconnects + 1,
            last_media_recovery_ms=recovery_ms,
            max_media_recovery_ms=max(self._stats.max_media_recovery_ms, recovery_ms),
        )
        self._start_media_pumps(event.media, self._media_generation)
        self._set_state(
            VoiceSessionState.LISTENING
            if self._provider_ready.is_set()
            else VoiceSessionState.RECOVERING
        )
        if self._provider_ready.is_set():
            try:
                await self._sessions.mark_active(self._context.descriptor.session_id)
            except Exception as exc:
                logger.warning(
                    "Could not persist recovered voice media state: session=%s error=%s",
                    opaque_ref(str(self._context.descriptor.session_id)),
                    exception_kind(exc),
                    exc_info=True,
                )
        logger.info(
            "Voice owner media recovered: session=%s generation=%d recovery_ms=%.1f",
            opaque_ref(str(self._context.descriptor.session_id)),
            self._media_generation,
            recovery_ms,
        )

    def _handle_media_recovery_failed(self, event: _MediaRecoveryFailed) -> None:
        if event.generation != self._media_generation:
            return
        self._media_recovery_task = None
        self._media_recovery_started_at = None
        self._stats = replace(
            self._stats,
            media_recovery_failures=self._stats.media_recovery_failures + 1,
        )
        logger.info(
            "Voice owner media recovery ended: session=%s generation=%d error=%s",
            opaque_ref(str(self._context.descriptor.session_id)),
            event.generation,
            event.error_kind,
        )
        self._set_state(VoiceSessionState.CLOSING)
        self._complete(
            VoiceStopReason.OWNER_LEFT
            if event.reason
            in {VoiceMediaEndReason.OWNER_LEFT, VoiceMediaEndReason.OWNER_UNPUBLISHED}
            else VoiceStopReason.MEDIA_ENDED
        )

    async def _control_loop(self) -> None:
        while not self._closed:
            event = await self._control.get()
            if isinstance(event, _StopRequested):
                logger.info(
                    "Voice runtime stop requested: session=%s reason=%s",
                    opaque_ref(str(self._context.descriptor.session_id)),
                    event.reason.value,
                )
                self._set_state(VoiceSessionState.CLOSING)
                self._complete(event.reason)
                return
            if isinstance(event, _WatchdogExpired):
                logger.info(
                    "Voice runtime watchdog expired: session=%s reason=%s",
                    opaque_ref(str(self._context.descriptor.session_id)),
                    event.reason.value,
                )
                self._set_state(VoiceSessionState.CLOSING)
                self._complete(event.reason)
                return
            if isinstance(event, _MediaEnded):
                if event.generation != self._media_generation:
                    logger.debug(
                        "Ignoring stale voice media terminal event: session=%s event_generation=%d "
                        "current_generation=%d reason=%s",
                        opaque_ref(str(self._context.descriptor.session_id)),
                        event.generation,
                        self._media_generation,
                        event.reason.value,
                    )
                    continue
                if (
                    event.reason
                    in {VoiceMediaEndReason.OWNER_LEFT, VoiceMediaEndReason.OWNER_UNPUBLISHED}
                    and self._settings.owner_leave_grace_seconds > 0
                ):
                    await self._begin_media_recovery(event)
                    continue
                log = (
                    logger.info
                    if event.reason
                    in {VoiceMediaEndReason.OWNER_LEFT, VoiceMediaEndReason.OWNER_UNPUBLISHED}
                    else logger.warning
                )
                log(
                    "Voice media ended: session=%s generation=%d reason=%s error=%s",
                    opaque_ref(str(self._context.descriptor.session_id)),
                    event.generation,
                    event.reason.value,
                    event.error_kind or "none",
                )
                self._set_state(VoiceSessionState.CLOSING)
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
            if isinstance(event, _MediaRecovered):
                await self._handle_media_recovered(event)
                continue
            if isinstance(event, _MediaRecoveryFailed):
                if event.generation != self._media_generation:
                    continue
                self._handle_media_recovery_failed(event)
                return
            if isinstance(event, _PumpFailed):
                if (
                    event.media_generation is not None
                    and event.media_generation != self._media_generation
                ):
                    logger.debug(
                        "Ignoring stale voice pump failure: session=%s pump=%s "
                        "event_generation=%d current_generation=%d",
                        opaque_ref(str(self._context.descriptor.session_id)),
                        event.pump,
                        event.media_generation,
                        self._media_generation,
                    )
                    continue
                if event.retryable_provider:
                    logger.warning(
                        "Voice Provider pump failed; recovering: session=%s pump=%s error=%s",
                        opaque_ref(str(self._context.descriptor.session_id)),
                        event.pump,
                        event.error_kind,
                    )
                    if await self._recover_provider():
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
                    if self._result is not None and self._result.done():
                        return
                logger.error(
                    "Voice runtime pump failed; ending session: session=%s pump=%s error=%s "
                    "retryable_provider=%s media_generation=%s",
                    opaque_ref(str(self._context.descriptor.session_id)),
                    event.pump,
                    event.error_kind,
                    event.retryable_provider,
                    event.media_generation,
                )
                self._set_state(VoiceSessionState.FAILED)
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
            if isinstance(event, _MailboxAvailable):
                self._handle_mailbox_available()
                continue
            if isinstance(event, _MailboxClaimed):
                await self._handle_mailbox_claimed(event)
                continue
            if isinstance(event, _MailboxPresented):
                self._handle_mailbox_presented(event)
                continue
            if event.session is not self._provider_session:
                logger.debug(
                    "Ignoring event from stale Voice Provider session: session=%s event=%s",
                    opaque_ref(str(self._context.descriptor.session_id)),
                    type(event.event).__name__,
                )
                continue
            model_event = event.event
            self._last_activity = time.monotonic()
            if isinstance(model_event, VoiceSessionReady):
                now = time.monotonic()
                ready_ms = (now - self._provider_connect_started_at) * 1000
                recovery_started_at = self._recovery_started_at
                recovery_ms = (
                    (now - recovery_started_at) * 1000 if recovery_started_at is not None else 0.0
                )
                self._stats = replace(
                    self._stats,
                    initial_provider_ready_ms=(self._stats.initial_provider_ready_ms or ready_ms),
                    last_provider_ready_ms=ready_ms,
                    provider_reconnects=(
                        self._stats.provider_reconnects + int(recovery_started_at is not None)
                    ),
                    last_provider_recovery_ms=(
                        recovery_ms
                        if recovery_started_at is not None
                        else self._stats.last_provider_recovery_ms
                    ),
                    max_provider_recovery_ms=max(
                        self._stats.max_provider_recovery_ms,
                        recovery_ms,
                    ),
                )
                self._recovery_started_at = None
                logger.info(
                    "Voice Provider ready: session=%s model=%s reconnect=%s "
                    "ready_ms=%.1f recovery_ms=%.1f",
                    opaque_ref(str(self._context.descriptor.session_id)),
                    self._context.configuration.model.alias,
                    recovery_started_at is not None,
                    ready_ms,
                    recovery_ms,
                )
                self._set_state(
                    VoiceSessionState.RECOVERING
                    if self._media_recovery_started_at is not None
                    else (
                        VoiceSessionState.USER_SPEAKING
                        if self._user_speaking
                        else VoiceSessionState.LISTENING
                    )
                )
                self._provider_ready.set()
                self._handle_mailbox_available()
                if self._started and self._media_recovery_started_at is None:
                    try:
                        await self._sessions.mark_active(self._context.descriptor.session_id)
                    except Exception as exc:
                        logger.warning(
                            "Could not persist recovered voice active state: session=%s error=%s",
                            opaque_ref(str(self._context.descriptor.session_id)),
                            exception_kind(exc),
                            exc_info=True,
                        )
            elif isinstance(model_event, VoiceUserSpeechStarted):
                if self._user_speaking:
                    self._stats = replace(
                        self._stats,
                        duplicate_speech_started=self._stats.duplicate_speech_started + 1,
                    )
                    self._publish_status()
                    continue
                self._user_speaking = True
                if self._active_response is None:
                    self._set_state(VoiceSessionState.USER_SPEAKING)
                    continue
                await self._interrupt_active_response(
                    notify_provider=True,
                    target_state=VoiceSessionState.USER_SPEAKING,
                    count_barge_in=True,
                )
            elif isinstance(model_event, VoiceUserSpeechStopped):
                self._user_speaking = False
                self._last_user_speech_stopped = time.monotonic()
                self._set_state(VoiceSessionState.THINKING)
            elif isinstance(model_event, VoiceResponseStarted):
                await self._start_response(model_event.response_id)
                if self._proactive_task_notices:
                    notices = self._proactive_task_notices
                    self._proactive_task_notices = ()
                    self._spawn(
                        self._mark_proactive_notifications_presented(notices),
                        "voice-task-mailbox-proactive-presented",
                    )
            elif isinstance(model_event, VoiceAssistantAudio):
                await self._stage_assistant_audio(model_event)
            elif isinstance(model_event, VoiceTranscriptFinal):
                await self._handle_final_transcript(model_event)
            elif isinstance(model_event, VoiceResponseCompleted):
                self._accumulate_response_usage(model_event.response_id, model_event.usage)
                await self._complete_response_playout(model_event.response_id, model_event.usage)
            elif isinstance(model_event, VoiceResponseCancelled):
                self._accumulate_response_usage(model_event.response_id, model_event.usage)
                if model_event.response_id in self._cancelled_response_set:
                    continue
                active = self._active_response
                if active is None or active.response_id != model_event.response_id:
                    if self._proactive_task_notices:
                        await self._defer_pending_proactive_notifications()
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
            elif isinstance(model_event, (VoiceProviderFailed, VoiceProviderErrorEvent)):
                if self._proactive_task_notices:
                    await self._defer_pending_proactive_notifications()
                log = logger.warning if model_event.retryable else logger.error
                log(
                    "Voice Provider reported failure: session=%s error=%s",
                    opaque_ref(str(self._context.descriptor.session_id)),
                    model_event,
                )
                if model_event.retryable and await self._recover_provider():
                    continue
                if self._result is not None and self._result.done():
                    return
                self._set_state(VoiceSessionState.FAILED)
                self._complete(VoiceStopReason.PROVIDER_FAILED)
                return
            elif isinstance(model_event, VoiceSessionFinished):
                logger.info(
                    "Voice Provider session finished: session=%s",
                    opaque_ref(str(self._context.descriptor.session_id)),
                )
                self._set_state(VoiceSessionState.CLOSING)
                self._complete(VoiceStopReason.RUNTIME_ENDED)
                return

    async def _mailbox_listener(self) -> None:
        mailbox = self._task_mailbox
        if mailbox is None:
            return
        await self._control.put(_MailboxAvailable())
        while True:
            signalled = await mailbox.wait(
                self._context.descriptor.owner_person_id,
                self._settings.mailbox_poll_seconds,
            )
            if signalled:
                await asyncio.sleep(_NOTIFICATION_COALESCE_SECONDS)
            await self._control.put(_MailboxAvailable())

    def _handle_mailbox_available(self) -> None:
        self._mailbox_available_pending = True
        if (
            self._task_mailbox is None
            or self._mailbox_claim_in_flight
            or self._notification_in_flight
            or not self._notification_safe()
        ):
            return
        self._mailbox_available_pending = False
        self._mailbox_claim_in_flight = True
        self._spawn(self._claim_mailbox(), "voice-task-mailbox-claim")

    async def _claim_mailbox(self) -> None:
        mailbox = self._task_mailbox
        if mailbox is None:
            return
        notices: tuple[VoiceTaskNotification, ...] = ()
        error_kind = ""
        try:
            notices = await mailbox.claim(
                self._context.descriptor.session_id,
                _NOTIFICATION_BATCH_LIMIT,
            )
            await self._control.put(_MailboxClaimed(notices))
        except asyncio.CancelledError:
            if notices:
                await asyncio.shield(mailbox.defer(tuple(item.task_id for item in notices)))
            raise
        except Exception as exc:
            error_kind = exception_kind(exc)
            logger.warning(
                "Voice task mailbox claim failed: session=%s error=%s",
                opaque_ref(str(self._context.descriptor.session_id)),
                error_kind,
                exc_info=True,
            )
            await self._control.put(_MailboxClaimed((), error_kind))

    async def _handle_mailbox_claimed(self, event: _MailboxClaimed) -> None:
        self._mailbox_claim_in_flight = False
        if event.error_kind:
            logger.warning(
                "Voice task mailbox claim failed: session=%s error=%s",
                opaque_ref(str(self._context.descriptor.session_id)),
                event.error_kind,
            )
            return
        if not event.notices:
            return
        count = len(event.notices)
        self._stats = replace(
            self._stats,
            task_notifications_claimed=self._stats.task_notifications_claimed + count,
        )
        task_ids = tuple(item.task_id for item in event.notices)
        if not self._notification_safe():
            self._stats = replace(
                self._stats,
                task_notifications_deferred=self._stats.task_notifications_deferred + count,
            )
            self._spawn(self._defer_notifications(task_ids), "voice-task-mailbox-defer")
            return

        provider = self._provider
        strategy = select_task_notification_strategy(
            provider.capabilities if provider is not None else VoiceProviderCapabilities()
        )
        self._last_notification_attempt = time.monotonic()
        if strategy is VoiceTaskNotificationStrategy.INTERNAL_RESPONSE:
            self._notification_in_flight = True
            self._proactive_task_notices = event.notices
            self._spawn(
                self._request_proactive_notifications(event.notices),
                "voice-task-mailbox-proactive",
            )
            return
        if strategy is not VoiceTaskNotificationStrategy.TEXT_FALLBACK:
            self._stats = replace(
                self._stats,
                task_notifications_deferred=self._stats.task_notifications_deferred + count,
            )
            logger.info(
                "Voice proactive task notification deferred until Provider adapter support: "
                "session=%s strategy=%s tasks=%s",
                opaque_ref(str(self._context.descriptor.session_id)),
                strategy.value,
                count,
            )
            self._spawn(self._defer_notifications(task_ids), "voice-task-mailbox-capability-defer")
            return

        self._notification_in_flight = True
        self._spawn(
            self._present_text_notifications(event.notices),
            "voice-task-mailbox-text",
        )

    async def _request_proactive_notifications(
        self,
        notices: tuple[VoiceTaskNotification, ...],
    ) -> None:
        session = self._provider_session
        if session is None:
            await self._proactive_request_failed(notices, "provider_session_missing")
            return
        try:
            await session.request_proactive_response(compile_internal_task_context(notices))
        except asyncio.CancelledError:
            await asyncio.shield(
                self._require_mailbox().defer(tuple(item.task_id for item in notices))
            )
            raise
        except Exception as exc:
            logger.warning(
                "Voice proactive task notification request failed: session=%s tasks=%s error=%s",
                opaque_ref(str(self._context.descriptor.session_id)),
                len(notices),
                exception_kind(exc),
                exc_info=True,
            )
            await self._proactive_request_failed(notices, exception_kind(exc))

    async def _proactive_request_failed(
        self,
        notices: tuple[VoiceTaskNotification, ...],
        error_kind: str,
    ) -> None:
        logger.warning(
            "Voice proactive task notification request failed: session=%s tasks=%s error=%s",
            opaque_ref(str(self._context.descriptor.session_id)),
            len(notices),
            error_kind,
        )
        await self._defer_notifications(tuple(item.task_id for item in notices))
        await self._control.put(
            _MailboxPresented(
                notices,
                VoiceTaskNotificationStrategy.INTERNAL_RESPONSE,
                False,
            )
        )

    async def _mark_proactive_notifications_presented(
        self,
        notices: tuple[VoiceTaskNotification, ...],
    ) -> None:
        succeeded = False
        task_ids = tuple(item.task_id for item in notices)
        for attempt in range(1, _NOTIFICATION_PERSIST_ATTEMPTS + 1):
            try:
                await asyncio.shield(self._require_mailbox().mark_presented(task_ids))
                succeeded = True
                break
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log = logger.error if attempt == _NOTIFICATION_PERSIST_ATTEMPTS else logger.warning
                log(
                    "Could not persist started proactive notification: "
                    "session=%s tasks=%s attempt=%s error=%s",
                    opaque_ref(str(self._context.descriptor.session_id)),
                    len(notices),
                    attempt,
                    exception_kind(exc),
                    exc_info=True,
                )
                if attempt < _NOTIFICATION_PERSIST_ATTEMPTS:
                    await asyncio.sleep(_NOTIFICATION_PERSIST_RETRY_SECONDS * attempt)
        await self._control.put(
            _MailboxPresented(
                notices,
                VoiceTaskNotificationStrategy.INTERNAL_RESPONSE,
                succeeded,
            )
        )

    async def _defer_pending_proactive_notifications(self) -> None:
        notices = self._proactive_task_notices
        if not notices:
            return
        self._proactive_task_notices = ()
        await self._defer_notifications(tuple(item.task_id for item in notices))
        self._handle_mailbox_presented(
            _MailboxPresented(
                notices,
                VoiceTaskNotificationStrategy.INTERNAL_RESPONSE,
                False,
            )
        )

    async def _defer_notifications(self, task_ids: tuple[UUID, ...]) -> None:
        mailbox = self._task_mailbox
        if mailbox is None:
            return
        try:
            await asyncio.shield(mailbox.defer(task_ids))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Voice task notification defer failed: session=%s tasks=%s error=%s",
                opaque_ref(str(self._context.descriptor.session_id)),
                len(task_ids),
                exception_kind(exc),
                exc_info=True,
            )

    async def _present_text_notifications(
        self,
        notices: tuple[VoiceTaskNotification, ...],
    ) -> None:
        mailbox = self._task_mailbox
        if mailbox is None:
            return
        succeeded = False
        try:
            succeeded = await mailbox.present_text(notices)
        except asyncio.CancelledError:
            await asyncio.shield(mailbox.defer(tuple(item.task_id for item in notices)))
            raise
        except Exception as exc:
            logger.warning(
                "Voice task text fallback failed: session=%s tasks=%s error=%s",
                opaque_ref(str(self._context.descriptor.session_id)),
                len(notices),
                exception_kind(exc),
                exc_info=True,
            )
            await self._defer_notifications(tuple(item.task_id for item in notices))
        else:
            if not succeeded:
                await self._defer_notifications(tuple(item.task_id for item in notices))
        await self._control.put(
            _MailboxPresented(
                notices,
                VoiceTaskNotificationStrategy.TEXT_FALLBACK,
                succeeded,
            )
        )

    def _handle_mailbox_presented(self, event: _MailboxPresented) -> None:
        if event.strategy is VoiceTaskNotificationStrategy.INTERNAL_RESPONSE:
            self._proactive_task_notices = ()
        self._notification_in_flight = False
        self._last_notification_attempt = time.monotonic()
        self._stats = replace(
            self._stats,
            task_notifications_presented=(
                self._stats.task_notifications_presented + event.count
                if event.succeeded
                else self._stats.task_notifications_presented
            ),
            task_notifications_deferred=(
                self._stats.task_notifications_deferred
                if event.succeeded
                else self._stats.task_notifications_deferred + event.count
            ),
            task_notifications_text_fallback=(
                self._stats.task_notifications_text_fallback + event.count
                if event.succeeded and event.strategy is VoiceTaskNotificationStrategy.TEXT_FALLBACK
                else self._stats.task_notifications_text_fallback
            ),
        )
        if event.succeeded:
            for notice in event.notices:
                detail = notice.summary or notice.error_message or notice.objective
                self._recovery_tasks.append(
                    VoiceRecoveryTask(
                        notice.alias,
                        notice.status,
                        _compact_recovery_text(detail, 360),
                    )
                )
        self._publish_status()
        if self._mailbox_available_pending:
            self._handle_mailbox_available()

    def _notification_safe(self) -> bool:
        now = time.monotonic()
        return (
            self._state is VoiceSessionState.LISTENING
            and not self._user_speaking
            and self._active_response is None
            and now - self._last_user_speech_stopped >= _NOTIFICATION_SILENCE_SECONDS
            and now - self._last_notification_attempt >= _NOTIFICATION_COOLDOWN_SECONDS
        )

    def _require_mailbox(self) -> VoiceTaskMailbox:
        mailbox = self._task_mailbox
        if mailbox is None:  # pragma: no cover - guarded by notification call sites
            raise RuntimeError("Voice task mailbox is unavailable")
        return mailbox

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
        self._publish_status()
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
        self._publish_status()
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
                exc_info=True,
            )
            if not await self._recover_provider():
                if self._result is not None and self._result.done():
                    return
                self._set_state(VoiceSessionState.FAILED)
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
        self._set_state(VoiceSessionState.THINKING)

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
        if self._stats.first_provider_audio_ms == 0:
            self._stats = replace(
                self._stats,
                first_provider_audio_ms=(time.monotonic() - self._started_at) * 1000,
            )
        self._set_state(VoiceSessionState.SPEAKING)
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
        if self._stats.first_final_transcript_ms == 0:
            self._stats = replace(
                self._stats,
                first_final_transcript_ms=(time.monotonic() - self._started_at) * 1000,
            )
            self._publish_status()
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

    async def _persist_transcript(
        self,
        event: VoiceTranscriptFinal,
        *,
        usage: Mapping[str, int | float] | None = None,
    ) -> None:
        self._recovery_turns.append(
            VoiceRecoveryTurn(event.role, _compact_recovery_text(event.text, 500))
        )
        self._turn_sequence += 1
        try:
            await self._sessions.append_final_turn(
                self._context.descriptor.session_id,
                self._turn_sequence,
                VoiceTurnRole(event.role),
                event.text,
                provider_item_id=event.provider_item_id,
                usage=dict(usage or {}),
            )
        except Exception as exc:
            logger.warning(
                "Could not persist final voice transcript: session=%s sequence=%d error=%s",
                opaque_ref(str(self._context.descriptor.session_id)),
                self._turn_sequence,
                exception_kind(exc),
                exc_info=True,
            )

    async def _complete_response_playout(
        self,
        response_id: str,
        usage: Mapping[str, int | float],
    ) -> None:
        if response_id in self._cancelled_response_set:
            return
        active = self._active_response
        if active is None or active.response_id != response_id:
            return
        if active.provider_done:
            return
        active.provider_done = True
        active.usage = dict(usage)
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
            logger.error(
                "Voice output drain failed: session=%s response=%s generation=%d error=%s",
                opaque_ref(str(self._context.descriptor.session_id)),
                opaque_ref(response_id),
                generation,
                exception_kind(exc),
                exc_info=True,
            )
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
            await self._persist_transcript(transcript, usage=active.usage)
        self._stats = replace(
            self._stats,
            responses_drained=self._stats.responses_drained + 1,
        )
        self._set_state(
            VoiceSessionState.USER_SPEAKING if self._user_speaking else VoiceSessionState.LISTENING
        )
        if self._mailbox_available_pending:
            self._handle_mailbox_available()
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
            self._set_state(target_state)
            if self._mailbox_available_pending and target_state is VoiceSessionState.LISTENING:
                self._handle_mailbox_available()
            return True
        started_at = time.monotonic()
        self._set_state(VoiceSessionState.INTERRUPTING)
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
                exc_info=True,
            )
            self._set_state(VoiceSessionState.FAILED)
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
                        exc_info=True,
                    )
                    if await self._recover_provider(playout_flushed=True):
                        return True
                    if self._result is not None and self._result.done():
                        return False
                    self._set_state(VoiceSessionState.FAILED)
                    self._complete(VoiceStopReason.PROVIDER_FAILED)
                    return False
        self._set_state(target_state)
        if self._mailbox_available_pending and target_state is VoiceSessionState.LISTENING:
            self._handle_mailbox_available()
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

    def _accumulate_response_usage(
        self,
        response_id: str,
        usage: Mapping[str, int | float],
    ) -> None:
        if response_id in self._usage_response_set:
            return
        if len(self._usage_response_ids) >= _RESPONSE_USAGE_HISTORY:
            expired = self._usage_response_ids.popleft()
            self._usage_response_set.discard(expired)
        self._usage_response_ids.append(response_id)
        self._usage_response_set.add(response_id)
        self._accumulate_usage(usage)

    async def _oopz_input_pump(
        self,
        media: VoiceMediaSession,
        generation: int,
    ) -> None:
        ingress = VoiceAudioIngress()
        try:
            async for frame in media.input_frames():
                self._last_activity = time.monotonic()
                if frame.source_dropped_frames:
                    self._stats = replace(
                        self._stats,
                        source_audio_frames_dropped=(
                            self._stats.source_audio_frames_dropped + frame.source_dropped_frames
                        ),
                    )
                for packet in ingress.push(frame):
                    dropped = self._input.put(packet)
                    self._stats = replace(
                        self._stats,
                        input_packets_dropped=(self._stats.input_packets_dropped + int(dropped)),
                        max_input_queue_depth=max(
                            self._stats.max_input_queue_depth,
                            self._input.qsize,
                        ),
                    )
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
            logger.error(
                "Voice media input pump failed: session=%s generation=%d error=%s",
                opaque_ref(str(self._context.descriptor.session_id)),
                generation,
                exception_kind(exc),
                exc_info=True,
            )
            await self._control.put(
                _PumpFailed(
                    "oopz_input",
                    exception_kind(exc),
                    media_generation=generation,
                )
            )

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
            logger.warning(
                "Voice Provider input pump failed: session=%s error=%s",
                opaque_ref(str(self._context.descriptor.session_id)),
                exception_kind(exc),
                exc_info=True,
            )
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
                    routed_chunk = PcmChunk(
                        chunk.pcm,
                        chunk.format,
                        chunk.duration_ms,
                        active.generation,
                    )
                    for output_chunk in _split_pcm_chunk(
                        routed_chunk,
                        self._output.max_chunk_duration_ms,
                    ):
                        accepted = await self._output.put(output_chunk)
                        if not accepted:
                            self._stats = replace(
                                self._stats,
                                late_audio_dropped=self._stats.late_audio_dropped + 1,
                            )
                            break
                    else:
                        self._stats = replace(
                            self._stats,
                            max_output_queue_depth=max(
                                self._stats.max_output_queue_depth,
                                self._output.qsize,
                            ),
                        )
                finally:
                    self._audio_events.task_done()
        except VoiceAudioQueueClosedError:
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Voice output routing failed: session=%s error=%s",
                opaque_ref(str(self._context.descriptor.session_id)),
                exception_kind(exc),
                exc_info=True,
            )
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
            logger.warning(
                "Voice Provider event pump failed: session=%s error=%s",
                opaque_ref(str(self._context.descriptor.session_id)),
                exception_kind(exc),
                exc_info=True,
            )
            await self._control.put(
                _PumpFailed("provider_events", exception_kind(exc), retryable_provider=True)
            )

    async def _oopz_output_pump(
        self,
        media: VoiceMediaSession,
        generation: int,
    ) -> None:
        try:
            while True:
                chunk = await self._output.get()
                if chunk.generation != self._output.generation:
                    continue
                try:
                    await media.write_output(chunk)
                    if self._stats.first_oopz_output_ms == 0:
                        self._stats = replace(
                            self._stats,
                            first_oopz_output_ms=(time.monotonic() - self._started_at) * 1000,
                        )
                        self._publish_status()
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
            logger.error(
                "Voice media output pump failed: session=%s generation=%d error=%s",
                opaque_ref(str(self._context.descriptor.session_id)),
                generation,
                exception_kind(exc),
                exc_info=True,
            )
            await self._control.put(
                _PumpFailed(
                    "oopz_output",
                    exception_kind(exc),
                    media_generation=generation,
                )
            )

    async def _media_terminal_watcher(
        self,
        media: VoiceMediaSession,
        generation: int,
    ) -> None:
        try:
            terminal = await media.wait_input_closed()
            await self._control.put(_MediaEnded(generation, terminal.reason, terminal.error_kind))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Voice media terminal watcher failed: session=%s generation=%d error=%s",
                opaque_ref(str(self._context.descriptor.session_id)),
                generation,
                exception_kind(exc),
                exc_info=True,
            )
            await self._control.put(
                _MediaEnded(
                    generation,
                    VoiceMediaEndReason.TRANSPORT_LOST,
                    exception_kind(exc),
                )
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

    def _start_media_pumps(self, media: VoiceMediaSession, generation: int) -> None:
        self._spawn_media(
            self._oopz_input_pump(media, generation),
            "voice-oopz-input",
        )
        self._spawn_media(
            self._oopz_output_pump(media, generation),
            "voice-oopz-output",
        )
        self._spawn_media(
            self._media_terminal_watcher(media, generation),
            "voice-media-terminal",
        )

    async def _cancel_media_pumps(self) -> None:
        current = asyncio.current_task()
        tasks = tuple(task for task in self._media_tasks if task is not current)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _spawn_media(self, coroutine, label: str) -> asyncio.Task[None]:
        task = self._spawn(coroutine, label)
        self._media_tasks.add(task)
        task.remove_done_callback(self._task_done)
        task.add_done_callback(self._media_task_done)
        return task

    def _media_task_done(self, task: asyncio.Task[None]) -> None:
        self._media_tasks.discard(task)
        self._task_done(task)

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
        task_name = task.get_name().split(":", 1)[0]
        if task.cancelled():
            logger.debug(
                "Voice runtime background task cancelled: session=%s task=%s",
                opaque_ref(str(self._context.descriptor.session_id)),
                task_name,
            )
            return
        error = task.exception()
        if error is None:
            logger.debug(
                "Voice runtime background task completed: session=%s task=%s",
                opaque_ref(str(self._context.descriptor.session_id)),
                task_name,
            )
            return
        logger.error(
            "Voice runtime task failed unexpectedly: session=%s task=%s error=%s",
            opaque_ref(str(self._context.descriptor.session_id)),
            task_name,
            exception_kind(error),
            exc_info=(type(error), error, error.__traceback__),
        )
        self._set_state(VoiceSessionState.FAILED)
        self._complete(VoiceStopReason.PROVIDER_FAILED)

    def _set_state(self, state: VoiceSessionState) -> None:
        self._state = state
        self._publish_status()

    def _publish_status(self) -> None:
        sink = self._context.status_sink
        if sink is None:
            return
        try:
            sink.emit(VoiceRuntimeStatus(self._state, self._stats))
        except Exception as exc:
            logger.warning(
                "Voice runtime status sink failed: session=%s error=%s",
                opaque_ref(str(self._context.descriptor.session_id)),
                exception_kind(exc),
                exc_info=True,
            )

    def _complete(self, reason: VoiceStopReason) -> None:
        if self._result is not None and not self._result.done():
            logger.info(
                "Voice runtime completed: session=%s reason=%s responses=%d reconnects=%d "
                "first_oopz_output_ms=%.1f input_dropped=%d output_overflows=%d",
                opaque_ref(str(self._context.descriptor.session_id)),
                reason.value,
                self._stats.responses_drained,
                self._stats.provider_reconnects,
                self._stats.first_oopz_output_ms,
                self._stats.input_packets_dropped,
                self._stats.output_overflows,
            )
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
        task_mailbox: VoiceTaskMailbox | None = None,
    ) -> None:
        self._settings = settings
        self._media_gateway = media_gateway
        self._sessions = sessions
        self._provider_builder = provider_builder
        self._task_controls = task_controls
        self._task_mailbox = task_mailbox

    async def create(self, context: VoiceSessionRuntimeContext) -> VoiceSessionRuntime:
        return RealtimeVoiceSessionRuntimeImpl(
            context,
            self._settings,
            self._media_gateway,
            self._sessions,
            self._provider_builder,
            self._task_controls,
            self._task_mailbox,
        )
