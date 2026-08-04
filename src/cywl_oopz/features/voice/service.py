"""Single-active-session facade for realtime voice conversations."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import suppress
from dataclasses import dataclass, field
from uuid import UUID, uuid4

from cywl_oopz.core.observability import exception_kind, opaque_ref
from cywl_oopz.settings import VoiceSettings

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
    VoiceSessionRepository,
    VoiceSessionRuntime,
    VoiceSessionRuntimeContext,
    VoiceSessionRuntimeFactory,
)
from .settings import PersistedVoiceSessionStatus, VoiceStartConfiguration

logger = logging.getLogger(__name__)


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
    startup_task: asyncio.Task[object] | None = None
    task: asyncio.Task[None] | None = None
    stop_requested: bool = False
    cleanup_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    done: asyncio.Event = field(default_factory=asyncio.Event)


class VoiceConversationService:
    """Own one realtime session and release all runtime resources deterministically."""

    def __init__(
        self,
        settings: VoiceSettings,
        access: VoiceAccessGateway,
        runtimes: VoiceSessionRuntimeFactory,
        configurations: VoiceConfigurationRepository,
        sessions: VoiceSessionRepository,
    ) -> None:
        self._settings = settings
        self._access = access
        self._runtimes = runtimes
        self._configurations = configurations
        self._sessions = sessions
        self._lock = asyncio.Lock()
        self._active: _ActiveVoiceSession | None = None
        self._closed = False

    async def start(self, request: VoiceStartRequest) -> VoiceSessionStatus:
        """Reserve the active slot and start one bounded runtime generation."""
        if not self._settings.enabled:
            raise VoiceFeatureDisabledError
        async with self._lock:
            if self._closed:
                raise VoiceRuntimeUnavailableError("Voice conversation service is closed")
            if self._active is not None:
                raise VoiceSessionAlreadyActiveError
            slot = _ActiveVoiceSession(uuid4(), request, time.monotonic())
            slot.startup_task = asyncio.current_task()
            self._active = slot

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
                    VoiceSessionRuntimeContext(descriptor, lease, configuration)
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
        except TimeoutError as exc:
            await self._cleanup(slot, VoiceSessionState.FAILED, VoiceStopReason.START_FAILED)
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
            await self._cleanup(slot, VoiceSessionState.FAILED, VoiceStopReason.START_FAILED)
            raise

        logger.info(
            "Voice conversation active: session=%s channel=%s",
            opaque_ref(str(slot.session_id)),
            opaque_ref(channel.area_id, channel.channel_id),
        )
        return status

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
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            slot = self._active
        if slot is None:
            return
        await self._request_stop(slot, VoiceStopReason.SHUTDOWN)
        await self._await_done_or_force(slot, VoiceStopReason.SHUTDOWN)

    async def _await_done_or_force(
        self,
        slot: _ActiveVoiceSession,
        reason: VoiceStopReason,
    ) -> None:
        try:
            async with asyncio.timeout(self._settings.start_timeout_seconds):
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
        if runtime is not None:
            try:
                await runtime.request_stop(reason)
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
    ) -> None:
        async with slot.cleanup_lock:
            if slot.done.is_set():
                return
            runtime = slot.runtime
            if runtime is not None:
                try:
                    await runtime.aclose()
                except Exception as exc:
                    logger.warning(
                        "Voice runtime cleanup failed: session=%s error=%s",
                        opaque_ref(str(slot.session_id)),
                        exception_kind(exc),
                    )
            if slot.persisted:
                persisted_status = (
                    PersistedVoiceSessionStatus.FAILED
                    if state is VoiceSessionState.FAILED
                    else PersistedVoiceSessionStatus.ENDED
                )
                try:
                    await self._sessions.finish(
                        slot.session_id,
                        persisted_status,
                        reason.value,
                    )
                except Exception as exc:
                    logger.warning(
                        "Voice session persistence cleanup failed: session=%s error=%s",
                        opaque_ref(str(slot.session_id)),
                        exception_kind(exc),
                    )
            lease = slot.lease
            if lease is not None and not lease.released:
                try:
                    await lease.release()
                except Exception as exc:
                    logger.warning(
                        "Voice lease cleanup failed: session=%s error=%s",
                        opaque_ref(str(slot.session_id)),
                        exception_kind(exc),
                    )
            async with self._lock:
                slot.state = state
                if self._active is slot:
                    self._active = None
            slot.done.set()
            logger.info(
                "Voice conversation closed: session=%s reason=%s state=%s",
                opaque_ref(str(slot.session_id)),
                reason.value,
                state.value,
            )

    @staticmethod
    def _status(
        slot: _ActiveVoiceSession,
        *,
        active: bool = True,
    ) -> VoiceSessionStatus:
        return VoiceSessionStatus(
            active=active,
            session_id=slot.session_id,
            owner_person_id=slot.request.owner_person_id,
            voice_channel=slot.voice_channel,
            state=slot.state,
            elapsed_seconds=max(0.0, time.monotonic() - slot.started_at_monotonic),
        )
