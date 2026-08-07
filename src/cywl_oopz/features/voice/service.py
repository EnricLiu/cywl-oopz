"""Single-active-session facade for realtime voice conversations."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import suppress
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any
from uuid import UUID, uuid4

from cywl_oopz.core.observability import exception_kind, opaque_ref
from cywl_oopz.settings import VoiceSettings

from .display import NoopVoiceSessionStatusSink
from .errors import (
    VoiceBackendBusyError,
    VoiceFeatureDisabledError,
    VoiceRuntimeUnavailableError,
    VoiceSessionAlreadyActiveError,
    VoiceSessionNotActiveError,
    VoiceSessionOwnershipError,
    VoiceSessionStartCancelledError,
    VoiceSessionStartTimeoutError,
    VoiceUserNotInChannelError,
)
from .models import (
    VoiceChannelKey,
    VoiceRuntimeStats,
    VoiceRuntimeStatus,
    VoiceSessionDescriptor,
    VoiceSessionState,
    VoiceSessionStatus,
    VoiceStartRequest,
    VoiceStopReason,
)
from .ports import (
    VoiceAccessGateway,
    VoiceConfigurationRepository,
    VoiceLease,
    VoiceMemoryContextSource,
    VoiceParticipantStatus,
    VoiceSessionRepository,
    VoiceSessionRuntime,
    VoiceSessionRuntimeContext,
    VoiceSessionRuntimeFactory,
    VoiceSessionStatusSink,
)
from .settings import PersistedVoiceSessionStatus, VoiceStartConfiguration

logger = logging.getLogger(__name__)

_LEASE_RELEASE_RETRY_DELAYS = (0.1, 0.25, 0.5)
_LEASE_RELEASE_RETRY_TIMEOUT_SECONDS = 0.5
_SESSION_FINISH_RETRY_DELAYS = (0.05, 0.15, 0.3)
_SESSION_FINISH_RETRY_TIMEOUT_SECONDS = 0.25
_SESSION_FINISH_SHUTDOWN_DRAIN_SECONDS = 0.4


@dataclass(slots=True)
class _ActiveVoiceSession:
    session_id: UUID
    request: VoiceStartRequest
    started_at_monotonic: float
    state: VoiceSessionState = VoiceSessionState.STARTING
    voice_channel: VoiceChannelKey | None = None
    lease: VoiceLease | None = None
    runtime: VoiceSessionRuntime | None = None
    configuration: VoiceStartConfiguration | None = None
    persisted: bool = False
    persisted_finished: bool = False
    usage: dict[str, Any] = field(default_factory=dict)
    runtime_stats: VoiceRuntimeStats = field(default_factory=VoiceRuntimeStats)
    status_sink: VoiceSessionStatusSink = field(default_factory=NoopVoiceSessionStatusSink)
    startup_task: asyncio.Task[object] | None = None
    task: asyncio.Task[None] | None = None
    stop_requested: bool = False
    cleanup_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    done: asyncio.Event = field(default_factory=asyncio.Event)
    lease_release_task: asyncio.Task[None] | None = None
    session_finish_task: asyncio.Task[None] | None = None


class VoiceConversationService:
    """Own one realtime session and release all runtime resources deterministically."""

    def __init__(
        self,
        settings: VoiceSettings,
        access: VoiceAccessGateway,
        runtimes: VoiceSessionRuntimeFactory,
        configurations: VoiceConfigurationRepository,
        sessions: VoiceSessionRepository,
        memory_context_source: VoiceMemoryContextSource | None = None,
        participant_status: VoiceParticipantStatus | None = None,
    ) -> None:
        self._settings = settings
        self._access = access
        self._runtimes = runtimes
        self._configurations = configurations
        self._sessions = sessions
        self._memory_context_source = memory_context_source
        self._participant_status = participant_status
        self._lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._lease_release_tasks: set[asyncio.Task[None]] = set()
        self._session_finish_tasks: set[asyncio.Task[None]] = set()
        self._active: _ActiveVoiceSession | None = None
        self._closed = False

    async def start(
        self,
        request: VoiceStartRequest,
        status_sink: VoiceSessionStatusSink | None = None,
    ) -> VoiceSessionStatus:
        """Reserve the active slot and start one bounded runtime generation."""
        if not self._settings.enabled:
            raise VoiceFeatureDisabledError
        async with self._lock:
            if self._closed:
                raise VoiceRuntimeUnavailableError("Voice conversation service is closed")
            if self._active is not None:
                raise VoiceSessionAlreadyActiveError
            slot = _ActiveVoiceSession(
                uuid4(),
                request,
                time.monotonic(),
                status_sink=status_sink or NoopVoiceSessionStatusSink(),
            )
            slot.startup_task = asyncio.current_task()
            self._active = slot
        self._publish_status(slot)

        logger.info(
            "Voice conversation startup reserved: session=%s owner=%s area=%s",
            opaque_ref(str(slot.session_id)),
            opaque_ref(request.owner_person_id),
            opaque_ref(request.origin.area_id),
        )
        try:
            async with asyncio.timeout(self._settings.start_timeout_seconds):
                channel_id = await self._access.voice_channel_for_user(
                    request.origin.area_id,
                    request.owner_person_id,
                )
                if channel_id is None:
                    raise VoiceUserNotInChannelError
                channel = VoiceChannelKey(request.origin.area_id, channel_id)
                await self._update_startup(slot, VoiceSessionState.ACQUIRING_VOICE, channel)

                configuration = await self._configurations.resolve_start_configuration(
                    request.owner_person_id,
                    channel,
                )
                slot.configuration = configuration
                await self._ensure_starting(slot)
                memory_context = await self._load_memory_context(request.owner_person_id)
                await self._ensure_starting(slot)

                lease = await self._access.try_acquire(
                    channel,
                    owner_key=f"conversation:{slot.session_id}",
                )
                if lease is None:
                    raise VoiceBackendBusyError
                slot.lease = lease
                await self._ensure_starting(slot)

                descriptor = VoiceSessionDescriptor(
                    slot.session_id,
                    request.owner_person_id,
                    channel,
                    request.origin,
                )
                await self._sessions.create(descriptor, configuration)
                slot.persisted = True
                await self._ensure_starting(slot)
                await self._update_startup(slot, VoiceSessionState.CONNECTING_PROVIDER)
                runtime = await self._runtimes.create(
                    VoiceSessionRuntimeContext(
                        descriptor,
                        lease,
                        configuration,
                        _SessionRuntimeStatusRelay(self, slot),
                        memory_context,
                    )
                )
                slot.runtime = runtime
                await self._ensure_starting(slot)
                await runtime.start()
                await self._sessions.mark_active(slot.session_id)

                async with self._lock:
                    self._raise_if_start_cancelled(slot)
                    slot.state = VoiceSessionState.LISTENING
                    slot.task = asyncio.create_task(
                        self._run_session(slot),
                        name=f"voice-session:{slot.session_id}",
                    )
                    slot.startup_task = None
                    status = self._status(slot)
                self._publish_status(slot)
        except TimeoutError as exc:
            await self._cleanup(
                slot,
                VoiceSessionState.FAILED,
                VoiceStopReason.START_FAILED,
                finalize_status=False,
            )
            raise VoiceSessionStartTimeoutError from exc
        except VoiceSessionStartCancelledError:
            await self._cleanup(slot, VoiceSessionState.CLOSED, VoiceStopReason.COMMAND)
            raise
        except asyncio.CancelledError:
            await asyncio.shield(
                self._cleanup(slot, VoiceSessionState.CLOSED, VoiceStopReason.START_FAILED)
            )
            raise
        except Exception:
            await self._cleanup(
                slot,
                VoiceSessionState.FAILED,
                VoiceStopReason.START_FAILED,
                finalize_status=False,
            )
            raise

        logger.info(
            "Voice conversation active: session=%s channel=%s",
            opaque_ref(str(slot.session_id)),
            opaque_ref(channel.area_id, channel.channel_id),
        )
        return status

    async def _load_memory_context(self, owner_person_id: str) -> str:
        source = self._memory_context_source
        if source is None:
            return ""
        try:
            memory = await source.context_text(owner_person_id)
        except Exception as exc:
            logger.warning(
                "Voice memory context load failed; continuing without memory: owner=%s error=%s",
                opaque_ref(owner_person_id),
                exception_kind(exc),
            )
            return ""
        return memory.strip()[:1500]

    async def stop(self, owner_person_id: str) -> VoiceSessionStatus:
        """Stop the active session when requested by its owner."""
        normalized_owner = owner_person_id.strip()
        async with self._lock:
            slot = self._active
            if slot is None:
                raise VoiceSessionNotActiveError
            if slot.request.owner_person_id != normalized_owner:
                raise VoiceSessionOwnershipError
        await self._request_stop(slot, VoiceStopReason.COMMAND)
        await self._await_done_or_force(slot, VoiceStopReason.COMMAND)
        return self._status(slot, active=False)

    async def status(self) -> VoiceSessionStatus:
        """Return a fast in-memory status snapshot without Provider or SDK I/O."""
        async with self._lock:
            if self._active is None:
                return VoiceSessionStatus(active=False)
            return self._status(self._active)

    async def aclose(self) -> None:
        """Reject new sessions and idempotently close the current generation."""
        async with self._close_lock:
            async with self._lock:
                self._closed = True
                slot = self._active
            if slot is not None:
                await self._request_stop(slot, VoiceStopReason.SHUTDOWN)
                await self._await_done_or_force(slot, VoiceStopReason.SHUTDOWN)
            await self._drain_session_finish_tasks()
            await self._cancel_lease_release_tasks()

    async def _await_done_or_force(
        self,
        slot: _ActiveVoiceSession,
        reason: VoiceStopReason,
    ) -> None:
        grace_seconds = min(0.1, self._settings.stop_timeout_seconds / 4)
        try:
            async with asyncio.timeout(grace_seconds):
                await slot.done.wait()
        except TimeoutError:
            task: asyncio.Task[object] | None = slot.task or slot.startup_task
            if task is not None and not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
            await self._cleanup(slot, VoiceSessionState.CLOSED, reason)

    async def _run_session(self, slot: _ActiveVoiceSession) -> None:
        runtime = slot.runtime
        if runtime is None:
            await self._cleanup(slot, VoiceSessionState.FAILED, VoiceStopReason.START_FAILED)
            return
        state = VoiceSessionState.CLOSED
        reason = VoiceStopReason.RUNTIME_ENDED
        try:
            result = await runtime.wait_finished()
            reason = result.reason
            slot.usage = dict(result.usage)
            if reason in {VoiceStopReason.PROVIDER_FAILED, VoiceStopReason.MEDIA_ENDED}:
                state = VoiceSessionState.FAILED
        except asyncio.CancelledError:
            reason = VoiceStopReason.SHUTDOWN
            raise
        except Exception as exc:
            state = VoiceSessionState.FAILED
            logger.warning(
                "Voice session runtime failed: session=%s error=%s",
                opaque_ref(str(slot.session_id)),
                exception_kind(exc),
                exc_info=True,
            )
        finally:
            await asyncio.shield(self._cleanup(slot, state, reason))

    async def _request_stop(
        self,
        slot: _ActiveVoiceSession,
        reason: VoiceStopReason,
    ) -> None:
        async with self._lock:
            if slot.done.is_set():
                return
            slot.stop_requested = True
            slot.state = VoiceSessionState.CLOSING
            runtime = slot.runtime
        self._publish_status(slot)
        if runtime is not None:
            try:
                async with asyncio.timeout(min(0.1, self._settings.stop_timeout_seconds / 4)):
                    await runtime.request_stop(reason)
            except TimeoutError:
                logger.warning(
                    "Voice runtime stop request exceeded grace period: session=%s",
                    opaque_ref(str(slot.session_id)),
                )
            except Exception as exc:
                logger.warning(
                    "Voice runtime stop request failed: session=%s error=%s",
                    opaque_ref(str(slot.session_id)),
                    exception_kind(exc),
                )

    async def _update_startup(
        self,
        slot: _ActiveVoiceSession,
        state: VoiceSessionState,
        channel: VoiceChannelKey | None = None,
    ) -> None:
        async with self._lock:
            self._raise_if_start_cancelled(slot)
            if channel is not None:
                slot.voice_channel = channel
            slot.state = state
        self._publish_status(slot)

    async def _ensure_starting(self, slot: _ActiveVoiceSession) -> None:
        async with self._lock:
            self._raise_if_start_cancelled(slot)

    def _raise_if_start_cancelled(self, slot: _ActiveVoiceSession) -> None:
        if self._active is not slot or slot.stop_requested or self._closed:
            raise VoiceSessionStartCancelledError

    async def _cleanup(
        self,
        slot: _ActiveVoiceSession,
        state: VoiceSessionState,
        reason: VoiceStopReason,
        *,
        finalize_status: bool = True,
    ) -> None:
        async with slot.cleanup_lock:
            if slot.done.is_set():
                return
            runtime = slot.runtime
            if runtime is not None:
                try:
                    async with asyncio.timeout(self._settings.stop_timeout_seconds * 0.7):
                        await runtime.aclose()
                except TimeoutError:
                    logger.warning(
                        "Voice runtime force-close exceeded cleanup budget: session=%s",
                        opaque_ref(str(slot.session_id)),
                    )
                except Exception as exc:
                    logger.warning(
                        "Voice runtime cleanup failed: session=%s error=%s",
                        opaque_ref(str(slot.session_id)),
                        exception_kind(exc),
                    )
            await asyncio.gather(
                self._finish_persisted_session(slot, state, reason),
                self._release_lease(slot),
            )
            async with self._lock:
                slot.state = state
                if self._active is slot:
                    self._active = None
            self._publish_status(slot, active=False)
            await self._close_status_sink(slot, finalize_status)
            slot.done.set()
            logger.info(
                "Voice conversation closed: session=%s reason=%s state=%s",
                opaque_ref(str(slot.session_id)),
                reason.value,
                state.value,
            )

    async def _finish_persisted_session(
        self,
        slot: _ActiveVoiceSession,
        state: VoiceSessionState,
        reason: VoiceStopReason,
    ) -> None:
        if not slot.persisted or slot.persisted_finished:
            return
        persisted_status = (
            PersistedVoiceSessionStatus.FAILED
            if state is VoiceSessionState.FAILED
            else PersistedVoiceSessionStatus.ENDED
        )
        try:
            async with asyncio.timeout(self._settings.stop_timeout_seconds * 0.1):
                await self._sessions.finish(
                    slot.session_id,
                    persisted_status,
                    reason.value,
                    usage=slot.usage,
                )
        except TimeoutError:
            logger.warning(
                "Voice session persistence cleanup exceeded budget: session=%s",
                opaque_ref(str(slot.session_id)),
            )
        except Exception as exc:
            logger.warning(
                "Voice session persistence cleanup failed: session=%s error=%s",
                opaque_ref(str(slot.session_id)),
                exception_kind(exc),
            )
        else:
            slot.persisted_finished = True
            return
        self._schedule_session_finish_retry(slot, persisted_status, reason)

    def _schedule_session_finish_retry(
        self,
        slot: _ActiveVoiceSession,
        status: PersistedVoiceSessionStatus,
        reason: VoiceStopReason,
    ) -> None:
        current = slot.session_finish_task
        if current is not None and not current.done():
            return
        task = asyncio.create_task(
            self._retry_finish_session(slot, status, reason),
            name=f"voice-session-finish:{slot.session_id}",
        )
        slot.session_finish_task = task
        self._session_finish_tasks.add(task)
        task.add_done_callback(lambda completed: self._session_finish_done(slot, completed))

    async def _retry_finish_session(
        self,
        slot: _ActiveVoiceSession,
        status: PersistedVoiceSessionStatus,
        reason: VoiceStopReason,
    ) -> None:
        for attempt, delay in enumerate(_SESSION_FINISH_RETRY_DELAYS, start=1):
            await asyncio.sleep(delay)
            if slot.persisted_finished:
                return
            try:
                async with asyncio.timeout(_SESSION_FINISH_RETRY_TIMEOUT_SECONDS):
                    await self._sessions.finish(
                        slot.session_id,
                        status,
                        reason.value,
                        usage=slot.usage,
                    )
            except TimeoutError:
                logger.warning(
                    "Voice session finish retry timed out: session=%s attempt=%s",
                    opaque_ref(str(slot.session_id)),
                    attempt,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Voice session finish retry failed: session=%s attempt=%s error=%s",
                    opaque_ref(str(slot.session_id)),
                    attempt,
                    exception_kind(exc),
                )
            else:
                slot.persisted_finished = True
                logger.info(
                    "Voice session terminal state persisted after retry: session=%s attempt=%s",
                    opaque_ref(str(slot.session_id)),
                    attempt,
                )
                return
        logger.error(
            "Voice session finish retries exhausted: session=%s",
            opaque_ref(str(slot.session_id)),
        )

    def _session_finish_done(
        self,
        slot: _ActiveVoiceSession,
        task: asyncio.Task[None],
    ) -> None:
        self._session_finish_tasks.discard(task)
        if slot.session_finish_task is task:
            slot.session_finish_task = None
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error(
                "Voice session finish task failed unexpectedly: session=%s error=%s",
                opaque_ref(str(slot.session_id)),
                exception_kind(error),
            )

    async def _drain_session_finish_tasks(self) -> None:
        tasks = tuple(self._session_finish_tasks)
        if not tasks:
            return
        _, pending = await asyncio.wait(
            tasks,
            timeout=min(
                _SESSION_FINISH_SHUTDOWN_DRAIN_SECONDS,
                self._settings.stop_timeout_seconds * 0.25,
            ),
        )
        for task in pending:
            task.cancel()
        if pending:
            _, still_pending = await asyncio.wait(pending, timeout=0.1)
            if still_pending:
                logger.warning(
                    "Voice session finish tasks exceeded shutdown budget: count=%s",
                    len(still_pending),
                )

    async def _release_lease(self, slot: _ActiveVoiceSession) -> None:
        lease = slot.lease
        if lease is None or lease.released:
            return
        try:
            async with asyncio.timeout(self._settings.stop_timeout_seconds * 0.1):
                await lease.release()
        except TimeoutError:
            logger.warning(
                "Voice lease cleanup exceeded budget: session=%s",
                opaque_ref(str(slot.session_id)),
            )
        except Exception as exc:
            logger.warning(
                "Voice lease cleanup failed: session=%s error=%s",
                opaque_ref(str(slot.session_id)),
                exception_kind(exc),
            )
        if not lease.released and not self._closed:
            self._schedule_lease_release_retry(slot)

    def _schedule_lease_release_retry(self, slot: _ActiveVoiceSession) -> None:
        current = slot.lease_release_task
        if current is not None and not current.done():
            return
        task = asyncio.create_task(
            self._retry_release_lease(slot),
            name=f"voice-lease-release:{slot.session_id}",
        )
        slot.lease_release_task = task
        self._lease_release_tasks.add(task)
        task.add_done_callback(lambda completed: self._lease_release_done(slot, completed))

    async def _retry_release_lease(self, slot: _ActiveVoiceSession) -> None:
        lease = slot.lease
        if lease is None:
            return
        for attempt, delay in enumerate(_LEASE_RELEASE_RETRY_DELAYS, start=1):
            await asyncio.sleep(delay)
            if lease.released:
                return
            try:
                async with asyncio.timeout(_LEASE_RELEASE_RETRY_TIMEOUT_SECONDS):
                    await lease.release()
            except TimeoutError:
                logger.warning(
                    "Voice lease release retry timed out: session=%s attempt=%s",
                    opaque_ref(str(slot.session_id)),
                    attempt,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Voice lease release retry failed: session=%s attempt=%s error=%s",
                    opaque_ref(str(slot.session_id)),
                    attempt,
                    exception_kind(exc),
                )
            if lease.released:
                logger.info(
                    "Voice lease released after retry: session=%s attempt=%s",
                    opaque_ref(str(slot.session_id)),
                    attempt,
                )
                return
        logger.error(
            "Voice lease release retries exhausted; backend remains reserved: session=%s",
            opaque_ref(str(slot.session_id)),
        )

    def _lease_release_done(
        self,
        slot: _ActiveVoiceSession,
        task: asyncio.Task[None],
    ) -> None:
        self._lease_release_tasks.discard(task)
        if slot.lease_release_task is task:
            slot.lease_release_task = None
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error(
                "Voice lease release task failed unexpectedly: session=%s error=%s",
                opaque_ref(str(slot.session_id)),
                exception_kind(error),
            )

    async def _cancel_lease_release_tasks(self) -> None:
        tasks = tuple(self._lease_release_tasks)
        for task in tasks:
            task.cancel()
        if not tasks:
            return
        done, pending = await asyncio.wait(
            tasks,
            timeout=min(0.1, self._settings.stop_timeout_seconds * 0.1),
        )
        del done
        if pending:
            logger.warning(
                "Voice lease release tasks exceeded shutdown budget: count=%s",
                len(pending),
            )

    async def _close_status_sink(
        self,
        slot: _ActiveVoiceSession,
        finalize_status: bool,
    ) -> None:
        if not finalize_status:
            return
        try:
            async with asyncio.timeout(self._settings.stop_timeout_seconds * 0.1):
                await slot.status_sink.aclose()
        except TimeoutError:
            logger.warning(
                "Voice status display cleanup exceeded budget: session=%s",
                opaque_ref(str(slot.session_id)),
            )
        except Exception as exc:
            logger.warning(
                "Voice status display cleanup failed: session=%s error=%s",
                opaque_ref(str(slot.session_id)),
                exception_kind(exc),
            )

    def _publish_status(self, slot: _ActiveVoiceSession, *, active: bool = True) -> None:
        try:
            slot.status_sink.emit(self._status(slot, active=active))
        except Exception as exc:
            logger.warning(
                "Voice status snapshot delivery failed: session=%s error=%s",
                opaque_ref(str(slot.session_id)),
                exception_kind(exc),
            )

    def _status(
        self,
        slot: _ActiveVoiceSession,
        *,
        active: bool = True,
    ) -> VoiceSessionStatus:
        runtime_state = slot.runtime.state if active and slot.runtime is not None else slot.state
        metrics = slot.runtime_stats.as_metrics()
        if slot.runtime is not None:
            metrics = slot.runtime.stats.as_metrics()
        channel = slot.voice_channel
        if (
            active
            and channel is not None
            and self._participant_status is not None
            and self._participant_status.music_active(channel)
        ):
            metrics["audio_music_participant_active"] = 1
        configuration = slot.configuration
        return VoiceSessionStatus(
            active=active,
            session_id=slot.session_id,
            owner_person_id=slot.request.owner_person_id,
            voice_channel=slot.voice_channel,
            state=runtime_state,
            elapsed_seconds=max(0.0, time.monotonic() - slot.started_at_monotonic),
            model_display_name=(
                configuration.model.display_name if configuration is not None else ""
            ),
            metrics=MappingProxyType(dict(metrics)),
            usage=MappingProxyType(
                {
                    key: value
                    for key, value in slot.usage.items()
                    if isinstance(value, int | float) and not isinstance(value, bool)
                }
            ),
        )


class _SessionRuntimeStatusRelay:
    """Project-owned synchronous relay; display I/O remains in its own worker."""

    def __init__(self, service: VoiceConversationService, slot: _ActiveVoiceSession) -> None:
        self._service = service
        self._slot = slot

    def emit(self, status: VoiceRuntimeStatus) -> None:
        self._slot.state = status.state
        self._slot.runtime_stats = status.stats
        self._service._publish_status(self._slot)
