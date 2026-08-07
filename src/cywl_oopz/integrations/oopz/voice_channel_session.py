"""Shared music/conversation ownership for the single OOPZ voice backend."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import StrEnum

from oopz_sdk import OopzBot

from cywl_oopz.core.observability import exception_kind, opaque_ref
from cywl_oopz.features.audio.errors import AudioSessionClosedError
from cywl_oopz.features.audio.models import (
    AudioChannelKey,
    VoiceParticipantKind,
    VoiceParticipantRequest,
)

logger = logging.getLogger(__name__)


class VoiceChannelSessionState(StrEnum):
    """Observable physical channel transition state."""

    JOINING = "joining"
    ACTIVE = "active"
    LEAVING = "leaving"


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

    @property
    def released(self) -> bool:
        return self._released

    async def release(self) -> bool:
        if self._released:
            return False
        released = await self._manager._release(self)
        self._released = True
        return released

    async def __aenter__(self) -> OopzVoiceParticipant:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.release()


class SharedVoiceChannelSession:
    """One joined OOPZ channel with at most one owner for each feature kind."""

    def __init__(self, channel: AudioChannelKey, generation: int) -> None:
        self.channel = channel
        self.generation = generation
        self._participants: dict[VoiceParticipantKind, OopzVoiceParticipant] = {}

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
        return participant, True

    def remove(self, participant: OopzVoiceParticipant) -> bool:
        existing = self._participants.get(participant.request.kind)
        if existing is not participant or participant.generation != self.generation:
            return False
        del self._participants[participant.request.kind]
        return True

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

    def __init__(self, bot: OopzBot, *, transition_wait_seconds: float = 1.0) -> None:
        if transition_wait_seconds <= 0:
            raise ValueError("Voice session transition wait must be positive")
        self._bot = bot
        self._transition_wait_seconds = transition_wait_seconds
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
                            participant._released = True
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
                await self._bot.voice.leave()
            except BaseException:
                await self._restore_failed_leave()
                raise

            async with self._lock:
                if self._session is session:
                    session.remove(participant)
                    self._session = None
                self._finish_leave_locked()
            logger.info(
                "Released final OOPZ voice participant: kind=%s channel=%s generation=%s",
                participant.request.kind.value,
                self._channel_ref(session.channel),
                session.generation,
            )
            return True

    def _begin_leave_locked(self) -> None:
        self._leaving = True
        self._leaving_done.clear()

    def _finish_leave_locked(self) -> None:
        self._leaving = False
        self._leaving_done.set()

    async def _restore_failed_leave(self) -> None:
        async with self._lock:
            self._finish_leave_locked()

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
