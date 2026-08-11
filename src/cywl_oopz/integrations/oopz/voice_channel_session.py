"""Shared music/conversation ownership for the single OOPZ voice backend."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from oopz_sdk import OopzBot, VoiceConnectionState

from cywl_oopz.core.observability import exception_kind, opaque_ref
from cywl_oopz.features.audio.errors import AudioBusFailedError, AudioSessionClosedError
from cywl_oopz.features.audio.mixer import AudioMixer
from cywl_oopz.features.audio.models import (
    AUDIO_BLOCK_FRAMES,
    AudioChannelKey,
    MixerLevels,
    VoiceParticipantKind,
    VoiceParticipantRequest,
)
from cywl_oopz.features.audio.ports import MasterPcmOutput
from cywl_oopz.features.audio.session import SharedAudioMixerBus, master_buffer_frames

logger = logging.getLogger(__name__)


class _MasterOutputFactory(Protocol):
    @property
    def max_buffer_ms(self) -> int: ...

    async def open(self) -> MasterPcmOutput: ...


class VoiceChannelSessionState(StrEnum):
    """Observable physical channel transition state."""

    JOINING = "joining"
    ACTIVE = "active"
    LEAVING = "leaving"


class VoiceParticipantTerminationReason(StrEnum):
    """Why one generation-scoped participant can no longer be used."""

    RELEASED = "released"
    MANAGER_CLOSED = "manager_closed"
    BACKEND_FAILED = "backend_failed"


@dataclass(frozen=True, slots=True)
class VoiceParticipantTermination:
    """Immutable terminal signal observed by feature-owned runtime tasks."""

    generation: int
    reason: VoiceParticipantTerminationReason


@dataclass(frozen=True, slots=True)
class VoiceChannelSessionSnapshot:
    """Immutable session view without exposing mutable participant tokens."""

    channel: AudioChannelKey
    generation: int
    state: VoiceChannelSessionState
    participants: tuple[VoiceParticipantRequest, ...]


class OopzVoiceParticipant:
    """Idempotent participant token protected from stale-generation release."""

    def __init__(
        self,
        manager: OopzVoiceChannelSessionManager,
        request: VoiceParticipantRequest,
        generation: int,
    ) -> None:
        self._manager = manager
        self.request = request
        self.generation = generation
        self._released = False
        self._termination = asyncio.get_running_loop().create_future()

    @property
    def released(self) -> bool:
        return self._released

    async def release(self) -> bool:
        if self._released:
            return False
        released = await self._manager._release(self)
        self._finish(VoiceParticipantTerminationReason.RELEASED)
        return released

    async def wait_terminated(self) -> VoiceParticipantTermination:
        """Wait until this exact participant generation becomes unusable."""
        return await asyncio.shield(self._termination)

    async def audio_bus(self) -> SharedAudioMixerBus:
        """Return this generation's session-owned lazy master bus."""
        if self._released:
            raise AudioSessionClosedError("Released participant has no audio bus")
        return await self._manager._audio_bus(self)

    async def __aenter__(self) -> OopzVoiceParticipant:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.release()

    def _finish(self, reason: VoiceParticipantTerminationReason) -> None:
        if self._released:
            return
        self._released = True
        self._termination.set_result(VoiceParticipantTermination(self.generation, reason))


class SharedVoiceChannelSession:
    """One joined OOPZ channel with at most one owner for each feature kind."""

    def __init__(self, channel: AudioChannelKey, generation: int) -> None:
        self.channel = channel
        self.generation = generation
        self._participants: dict[VoiceParticipantKind, OopzVoiceParticipant] = {}
        self._bus: SharedAudioMixerBus | None = None
        self._bus_lock = asyncio.Lock()

    @property
    def participants(self) -> tuple[OopzVoiceParticipant, ...]:
        return tuple(self._participants.values())

    def try_add(
        self,
        manager: OopzVoiceChannelSessionManager,
        request: VoiceParticipantRequest,
    ) -> tuple[OopzVoiceParticipant | None, bool]:
        if request.channel != self.channel:
            return None, False
        existing = self._participants.get(request.kind)
        if existing is not None:
            if existing.request.owner_key == request.owner_key and not existing.released:
                return existing, False
            return None, False
        participant = OopzVoiceParticipant(manager, request, self.generation)
        self._participants[request.kind] = participant
        self._sync_bus_participants()
        return participant, True

    def remove(self, participant: OopzVoiceParticipant) -> bool:
        existing = self._participants.get(participant.request.kind)
        if existing is not participant or participant.generation != self.generation:
            return False
        del self._participants[participant.request.kind]
        self._sync_bus_participants()
        return True

    async def audio_bus(
        self,
        factory: _MasterOutputFactory,
        *,
        master_target_buffer_ms: int,
        music_queue_ms: int,
        voice_queue_ms: int,
        mixer_levels: MixerLevels,
    ) -> SharedAudioMixerBus:
        async with self._bus_lock:
            if self._bus is not None and not self._bus.closed and not self._bus.failed:
                return self._bus
            if self._bus is not None and not self._bus.closed:
                await self._bus.aclose()
            try:
                master = await factory.open()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(
                    "Could not open shared audio master: channel=%s error=%s",
                    opaque_ref(self.channel.area_id, self.channel.channel_id),
                    exception_kind(exc),
                )
                raise AudioBusFailedError("Could not open shared audio master") from exc
            self._bus = SharedAudioMixerBus(
                master,
                max_buffer_frames=(
                    master_buffer_frames(factory.max_buffer_ms) + AUDIO_BLOCK_FRAMES
                ),
                master_target_buffer_ms=master_target_buffer_ms,
                music_queue_ms=music_queue_ms,
                voice_queue_ms=voice_queue_ms,
                mixer=AudioMixer(mixer_levels),
            )
            self._sync_bus_participants()
            return self._bus

    async def aclose_bus(self) -> None:
        async with self._bus_lock:
            if self._bus is not None:
                await self._bus.aclose()

    def _sync_bus_participants(self) -> None:
        if self._bus is not None:
            self._bus.update_participants(frozenset(self._participants))

    def snapshot(self, state: VoiceChannelSessionState) -> VoiceChannelSessionSnapshot:
        requests = tuple(
            participant.request
            for participant in sorted(
                self._participants.values(),
                key=lambda item: item.request.kind.value,
            )
        )
        return VoiceChannelSessionSnapshot(self.channel, self.generation, state, requests)


class OopzVoiceChannelSessionManager:
    """Serialize one physical join while allowing same-channel feature participants."""

    def __init__(
        self,
        bot: OopzBot,
        *,
        transition_wait_seconds: float = 1.0,
        allow_mixed_participants: bool = True,
        master_factory: _MasterOutputFactory | None = None,
        master_target_buffer_ms: int = 60,
        music_queue_ms: int = 500,
        voice_queue_ms: int = 60,
        mixer_levels: MixerLevels | None = None,
        health_check_interval_seconds: float = 0.5,
        health_failure_threshold: int = 2,
    ) -> None:
        if transition_wait_seconds <= 0:
            raise ValueError("Voice session transition wait must be positive")
        if health_check_interval_seconds <= 0:
            raise ValueError("Voice session health check interval must be positive")
        if health_failure_threshold <= 0:
            raise ValueError("Voice session health failure threshold must be positive")
        self._bot = bot
        self._transition_wait_seconds = transition_wait_seconds
        self._allow_mixed_participants = allow_mixed_participants
        self._master_factory = master_factory
        self._master_target_buffer_ms = master_target_buffer_ms
        self._music_queue_ms = music_queue_ms
        self._voice_queue_ms = voice_queue_ms
        self._mixer_levels = mixer_levels or MixerLevels()
        self._health_check_interval_seconds = health_check_interval_seconds
        self._health_failure_threshold = health_failure_threshold
        self._lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._session: SharedVoiceChannelSession | None = None
        self._pending_request: VoiceParticipantRequest | None = None
        self._pending_generation = 0
        self._pending_done = asyncio.Event()
        self._pending_done.set()
        self._leaving = False
        self._leaving_done = asyncio.Event()
        self._leaving_done.set()
        self._joined_without_session = False
        self._generation = 0
        self._closed = False
        self._health_task: asyncio.Task[None] | None = None

    async def try_acquire(
        self,
        request: VoiceParticipantRequest,
    ) -> OopzVoiceParticipant | None:
        """Join or share the requested channel, returning ``None`` for a conflict."""
        while True:
            join_generation: int | None = None
            wait_event: asyncio.Event | None = None
            async with self._lock:
                if self._closed:
                    raise AudioSessionClosedError("OOPZ voice channel session manager is closed")
                if self._leaving:
                    wait_event = self._leaving_done
                elif self._session is not None:
                    if self._session.channel != request.channel:
                        self._log_unavailable(request, self._session.channel)
                        return None
                    if not self._can_mix_with_active(request):
                        self._log_coexistence_disabled(request)
                        return None
                    participant, added = self._session.try_add(self, request)
                    if participant is None:
                        self._log_kind_conflict(request)
                        return None
                    if added:
                        logger.info(
                            "Added OOPZ voice participant: kind=%s channel=%s generation=%s",
                            request.kind.value,
                            self._channel_ref(request.channel),
                            self._session.generation,
                        )
                    return participant
                elif self._pending_request is not None:
                    pending = self._pending_request
                    if pending.channel != request.channel:
                        self._log_unavailable(request, pending.channel)
                        return None
                    if not self._allow_mixed_participants and pending.kind is not request.kind:
                        self._log_coexistence_disabled(request)
                        return None
                    if pending.kind is request.kind and pending.owner_key != request.owner_key:
                        self._log_kind_conflict(request)
                        return None
                    wait_event = self._pending_done
                else:
                    self._generation += 1
                    join_generation = self._generation
                    self._pending_generation = join_generation
                    self._pending_request = request
                    self._pending_done.clear()
                    logger.info(
                        "Joining shared OOPZ voice channel: kind=%s channel=%s generation=%s",
                        request.kind.value,
                        self._channel_ref(request.channel),
                        join_generation,
                    )

            if wait_event is not None:
                if not await self._wait_for_transition(wait_event):
                    logger.info(
                        "OOPZ voice session transition remained busy: requested=%s channel=%s",
                        request.kind.value,
                        self._channel_ref(request.channel),
                    )
                    return None
                continue
            if join_generation is not None:
                return await self._join_initial(request, join_generation)

    async def current(self) -> VoiceChannelSessionSnapshot | None:
        async with self._lock:
            if self._session is not None:
                state = (
                    VoiceChannelSessionState.LEAVING
                    if self._leaving
                    else VoiceChannelSessionState.ACTIVE
                )
                return self._session.snapshot(state)
            if self._pending_request is not None:
                return VoiceChannelSessionSnapshot(
                    self._pending_request.channel,
                    self._pending_generation,
                    VoiceChannelSessionState.JOINING,
                    (self._pending_request,),
                )
            return None

    def participant_active(
        self,
        kind: VoiceParticipantKind,
        channel: AudioChannelKey,
    ) -> bool:
        """Return a coherent same-event-loop view without blocking status rendering."""
        session = self._session
        return (
            session is not None
            and not self._leaving
            and session.channel == channel
            and any(participant.request.kind is kind for participant in session.participants)
        )

    async def aclose(self) -> None:
        """Close once, preserving ownership when leave fails so a later call can retry."""
        async with self._close_lock:
            async with self._lock:
                self._closed = True
            while True:
                wait_event: asyncio.Event | None = None
                session: SharedVoiceChannelSession | None = None
                orphaned_join = False
                async with self._lock:
                    if self._pending_request is not None:
                        wait_event = self._pending_done
                    elif self._leaving:
                        wait_event = self._leaving_done
                    elif self._session is not None:
                        session = self._session
                        self._begin_leave_locked()
                    elif self._joined_without_session:
                        orphaned_join = True
                        self._begin_leave_locked()
                    else:
                        return

                if wait_event is not None:
                    await wait_event.wait()
                    continue
                try:
                    await self._cancel_health_monitor()
                    if session is not None:
                        await session.aclose_bus()
                    await self._bot.voice.leave()
                except asyncio.CancelledError:
                    await self._restore_failed_leave()
                    raise
                except Exception as exc:
                    await self._restore_failed_leave()
                    logger.warning(
                        "Could not leave OOPZ voice while closing shared session: "
                        "owner=%s error=%s",
                        "orphaned_join" if orphaned_join else "participants",
                        exception_kind(exc),
                    )
                    return

                async with self._lock:
                    if session is not None and self._session is session:
                        self._session = None
                        for participant in session.participants:
                            participant._finish(VoiceParticipantTerminationReason.MANAGER_CLOSED)
                    self._joined_without_session = False
                    self._finish_leave_locked()
                logger.info("Closed shared OOPZ voice channel session")
                return

    async def _join_initial(
        self,
        request: VoiceParticipantRequest,
        generation: int,
    ) -> OopzVoiceParticipant:
        try:
            await self._bot.voice.join(
                area=request.channel.area_id,
                channel=request.channel.channel_id,
            )
        except BaseException:
            async with self._lock:
                if self._pending_generation == generation:
                    self._clear_pending_locked()
            raise

        should_leave = False
        async with self._lock:
            if (
                self._closed
                or self._pending_request != request
                or self._pending_generation != generation
            ):
                should_leave = True
                self._joined_without_session = True
                self._pending_request = None
            else:
                session = SharedVoiceChannelSession(request.channel, generation)
                participant, _added = session.try_add(self, request)
                assert participant is not None
                self._session = session
                self._start_health_monitor_locked(session)
                self._clear_pending_locked()

        if should_leave:
            try:
                await self._bot.voice.leave()
            except BaseException:
                raise
            else:
                async with self._lock:
                    self._joined_without_session = False
            finally:
                self._pending_done.set()
            raise AudioSessionClosedError("OOPZ voice session manager closed during join")

        logger.info(
            "Shared OOPZ voice channel active: kind=%s channel=%s generation=%s",
            request.kind.value,
            self._channel_ref(request.channel),
            generation,
        )
        return participant

    async def _release(self, participant: OopzVoiceParticipant) -> bool:
        while True:
            wait_event: asyncio.Event | None = None
            session: SharedVoiceChannelSession | None = None
            async with self._lock:
                if self._leaving:
                    wait_event = self._leaving_done
                else:
                    session = self._session
                    if (
                        session is None
                        or session.generation != participant.generation
                        or participant not in session.participants
                    ):
                        logger.debug(
                            "Ignored stale OOPZ voice participant release: kind=%s generation=%s",
                            participant.request.kind.value,
                            participant.generation,
                        )
                        return False
                    if len(session.participants) > 1:
                        session.remove(participant)
                        participant._finish(VoiceParticipantTerminationReason.RELEASED)
                        logger.info(
                            "Removed OOPZ voice participant: kind=%s channel=%s generation=%s",
                            participant.request.kind.value,
                            self._channel_ref(session.channel),
                            session.generation,
                        )
                        return True
                    self._begin_leave_locked()

            if wait_event is not None:
                await asyncio.shield(wait_event.wait())
                if participant.released:
                    return False
                continue

            assert session is not None
            try:
                await self._cancel_health_monitor()
                await session.aclose_bus()
                await self._bot.voice.leave()
            except BaseException:
                await self._restore_failed_leave()
                raise

            async with self._lock:
                if self._session is session:
                    session.remove(participant)
                    self._session = None
                    participant._finish(VoiceParticipantTerminationReason.RELEASED)
                self._finish_leave_locked()
            logger.info(
                "Released final OOPZ voice participant: kind=%s channel=%s generation=%s",
                participant.request.kind.value,
                self._channel_ref(session.channel),
                session.generation,
            )
            return True

    async def invalidate_backend(self, *, expected_generation: int | None = None) -> bool:
        """Discard one unhealthy physical generation and wake all participants."""
        async with self._lock:
            session = self._session
            if (
                session is None
                or self._leaving
                or (expected_generation is not None and session.generation != expected_generation)
            ):
                return False
            self._begin_leave_locked()
            for participant in session.participants:
                participant._finish(VoiceParticipantTerminationReason.BACKEND_FAILED)

        await self._cancel_health_monitor()
        try:
            await session.aclose_bus()
        except Exception as exc:
            logger.warning(
                "Could not close shared audio bus while invalidating voice backend: "
                "channel=%s generation=%s error=%s",
                self._channel_ref(session.channel),
                session.generation,
                exception_kind(exc),
            )
        try:
            await self._bot.voice.leave()
        except Exception as exc:
            logger.warning(
                "Could not leave failed OOPZ voice backend during invalidation: "
                "channel=%s generation=%s error=%s",
                self._channel_ref(session.channel),
                session.generation,
                exception_kind(exc),
            )

        async with self._lock:
            if self._session is session:
                self._session = None
            self._finish_leave_locked()
        logger.warning(
            "Invalidated unhealthy OOPZ voice generation: channel=%s generation=%s",
            self._channel_ref(session.channel),
            session.generation,
        )
        return True

    async def _audio_bus(self, participant: OopzVoiceParticipant) -> SharedAudioMixerBus:
        async with self._lock:
            session = self._session
            if (
                session is None
                or session.generation != participant.generation
                or participant not in session.participants
            ):
                raise AudioSessionClosedError("Voice participant generation is no longer active")
            factory = self._master_factory
        if factory is None:
            raise AudioSessionClosedError("Shared voice session has no master output factory")
        return await session.audio_bus(
            factory,
            master_target_buffer_ms=self._master_target_buffer_ms,
            music_queue_ms=self._music_queue_ms,
            voice_queue_ms=self._voice_queue_ms,
            mixer_levels=self._mixer_levels,
        )

    def _start_health_monitor_locked(self, session: SharedVoiceChannelSession) -> None:
        get_health = getattr(self._bot.voice, "get_health", None)
        if not callable(get_health):
            return
        if self._health_task is not None and not self._health_task.done():
            raise RuntimeError("Voice session health monitor already owns a generation")
        self._health_task = asyncio.create_task(
            self._monitor_health(session, get_health),
            name=f"oopz-voice-health:{session.generation}",
        )

    async def _monitor_health(self, session: SharedVoiceChannelSession, get_health) -> None:
        failures = 0
        try:
            while True:
                await asyncio.sleep(self._health_check_interval_seconds)
                async with self._lock:
                    if self._session is not session or self._leaving or self._closed:
                        return
                try:
                    health = await get_health()
                    healthy = (
                        health.connection_state is VoiceConnectionState.JOINED
                        and health.agora_client_joined
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    healthy = False
                    logger.warning(
                        "Could not inspect OOPZ voice health: channel=%s generation=%s error=%s",
                        self._channel_ref(session.channel),
                        session.generation,
                        exception_kind(exc),
                    )
                failures = 0 if healthy else failures + 1
                if failures < self._health_failure_threshold:
                    continue
                await self.invalidate_backend(expected_generation=session.generation)
                return
        except asyncio.CancelledError:
            raise

    async def _cancel_health_monitor(self) -> None:
        task = self._health_task
        self._health_task = None
        if task is None or task is asyncio.current_task():
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    def _begin_leave_locked(self) -> None:
        self._leaving = True
        self._leaving_done.clear()

    def _finish_leave_locked(self) -> None:
        self._leaving = False
        self._leaving_done.set()

    async def _restore_failed_leave(self) -> None:
        async with self._lock:
            self._finish_leave_locked()
            if not self._closed and self._session is not None:
                self._start_health_monitor_locked(self._session)

    def _clear_pending_locked(self) -> None:
        self._pending_request = None
        self._pending_generation = 0
        self._pending_done.set()

    async def _wait_for_transition(self, event: asyncio.Event) -> bool:
        try:
            async with asyncio.timeout(self._transition_wait_seconds):
                await asyncio.shield(event.wait())
        except TimeoutError:
            return False
        return True

    def _log_unavailable(
        self,
        request: VoiceParticipantRequest,
        active_channel: AudioChannelKey,
    ) -> None:
        logger.info(
            "OOPZ voice channel unavailable: requested=%s channel=%s active_channel=%s",
            request.kind.value,
            self._channel_ref(request.channel),
            self._channel_ref(active_channel),
        )

    def _can_mix_with_active(self, request: VoiceParticipantRequest) -> bool:
        if self._allow_mixed_participants or self._session is None:
            return True
        return all(
            participant.request.kind is request.kind for participant in self._session.participants
        )

    @staticmethod
    def _log_coexistence_disabled(request: VoiceParticipantRequest) -> None:
        logger.info(
            "OOPZ mixed voice participants are not enabled yet: requested=%s",
            request.kind.value,
        )

    @staticmethod
    def _log_kind_conflict(request: VoiceParticipantRequest) -> None:
        logger.info(
            "OOPZ voice participant kind already owned: kind=%s owner=%s",
            request.kind.value,
            opaque_ref(request.owner_key),
        )

    @staticmethod
    def _channel_ref(channel: AudioChannelKey) -> str:
        return opaque_ref(channel.area_id, channel.channel_id)
