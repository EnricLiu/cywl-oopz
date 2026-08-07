"""Per-voice-channel queue state and application-owned playback workers."""

from __future__ import annotations

import asyncio
import logging
import random
from collections import OrderedDict, deque
from contextlib import suppress
from dataclasses import dataclass, field
from uuid import UUID

from cywl_oopz.core.observability import opaque_ref
from cywl_oopz.features.agent.models import AgentIdentity
from cywl_oopz.settings import MusicSettings

from .errors import (
    MusicBackendClosedError,
    MusicNotFoundError,
    MusicPlaybackError,
    MusicQueryError,
    MusicQueueFullError,
    MusicVoiceBusyError,
    MusicVoiceChannelRequiredError,
)
from .models import (
    EnqueueResult,
    MusicPlaybackEndReason,
    MusicQueueSnapshot,
    MusicTrack,
    PlaybackMode,
    PlaybackModeChange,
    PlaybackState,
    QueuedTrack,
    QueueRebuildResult,
    VoiceChannelKey,
)
from .ports import MusicCatalog, MusicPlayback, MusicVoiceGateway

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _MusicSession:
    """Mutable state protected by one voice-channel lock."""

    queue: deque[QueuedTrack] = field(default_factory=deque)
    current: QueuedTrack | None = None
    state: PlaybackState = PlaybackState.IDLE
    mode: PlaybackMode = PlaybackMode.SEQUENTIAL
    revision: int = 0
    playback: MusicPlayback | None = None
    voice_reserved: bool = False
    worker: asyncio.Task[None] | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    skip_requested: asyncio.Event = field(default_factory=asyncio.Event)
    idempotent_enqueues: OrderedDict[str, EnqueueResult] = field(default_factory=OrderedDict)


class MusicRequestService:
    """Search, enqueue, inspect, and control bounded voice-channel queues."""

    def __init__(
        self,
        settings: MusicSettings,
        catalog: MusicCatalog,
        voice: MusicVoiceGateway,
        *,
        rng: random.Random | None = None,
    ) -> None:
        self._settings = settings
        self._catalog = catalog
        self._voice = voice
        self._rng = rng or random.Random()
        self._sessions: dict[VoiceChannelKey, _MusicSession] = {}
        self._closing = False

    async def search(self, query: str, *, limit: int | None = None) -> tuple[MusicTrack, ...]:
        """Search the configured catalog with deterministic input and result bounds."""
        normalized = query.strip()
        if not normalized:
            raise MusicQueryError("Music search query must not be empty")
        if len(normalized) > self._settings.max_query_characters:
            raise MusicQueryError("Music search query is too long")
        requested_limit = min(limit or self._settings.search_limit, self._settings.search_limit)
        logger.info(
            "Music search started: query_characters=%s limit=%s",
            len(normalized),
            requested_limit,
        )
        matches = await self._catalog.search(
            normalized,
            limit=requested_limit,
        )
        logger.info("Music search completed: result_count=%s", len(matches))
        return matches

    async def enqueue(
        self,
        identity: AgentIdentity,
        query: str,
        *,
        idempotency_key: str = "",
    ) -> EnqueueResult:
        """Add the top catalog match to the caller's current voice-channel queue."""
        channel = await self._channel_for(identity)
        session = self._session(channel)
        if idempotency_key:
            async with session.lock:
                previous = session.idempotent_enqueues.get(idempotency_key)
                if previous is not None:
                    logger.info(
                        "Reused idempotent music enqueue: channel=%s position=%s",
                        self._channel_ref(channel),
                        previous.position,
                    )
                    return previous
        matches = await self.search(query, limit=1)
        if not matches:
            raise MusicNotFoundError("No music matched the query")
        item = QueuedTrack(matches[0], identity.person_id)
        async with session.lock:
            if idempotency_key:
                previous = session.idempotent_enqueues.get(idempotency_key)
                if previous is not None:
                    logger.info(
                        "Reused idempotent music enqueue after search: channel=%s position=%s",
                        self._channel_ref(channel),
                        previous.position,
                    )
                    return previous
            total = len(session.queue) + (1 if session.current is not None else 0)
            if total >= self._settings.max_queue_length:
                raise MusicQueueFullError("Music queue is full")
            await self._reserve_voice_locked(channel, session)
            session.queue.append(item)
            session.revision += 1
            session.state = PlaybackState.WAITING if session.current is None else session.state
            position = len(session.queue) + (1 if session.current is not None else 0)
            started = self._start_worker_locked(channel, session)
        logger.info(
            "Music enqueued: channel=%s position=%s playback_worker_started=%s",
            self._channel_ref(channel),
            position,
            started,
        )
        result = EnqueueResult(channel, item, position, started)
        if idempotency_key:
            async with session.lock:
                session.idempotent_enqueues[idempotency_key] = result
                while len(session.idempotent_enqueues) > 256:
                    session.idempotent_enqueues.popitem(last=False)
        return result

    async def queue(self, identity: AgentIdentity) -> MusicQueueSnapshot:
        """Return an immutable bounded view for the caller's current voice channel."""
        channel = await self._channel_for(identity)
        session = self._session(channel)
        async with session.lock:
            logger.debug(
                "Music queue inspected: channel=%s state=%s upcoming=%s",
                self._channel_ref(channel),
                session.state.value,
                len(session.queue),
            )
            return MusicQueueSnapshot(
                voice_channel=channel,
                state=session.state,
                mode=session.mode,
                current=session.current,
                upcoming=tuple(session.queue),
                revision=session.revision,
            )

    async def set_mode(
        self,
        identity: AgentIdentity,
        mode: PlaybackMode,
    ) -> PlaybackModeChange:
        """Change playback policy for the caller's current voice channel."""
        channel = await self._channel_for(identity)
        session = self._session(channel)
        async with session.lock:
            changed = session.mode is not mode
            if changed:
                session.mode = mode
                session.revision += 1
        logger.info(
            "Music playback mode selected: channel=%s mode=%s changed=%s",
            self._channel_ref(channel),
            mode.value,
            changed,
        )
        return PlaybackModeChange(channel, mode, changed)

    async def replace_queue(
        self,
        identity: AgentIdentity,
        tracks: tuple[MusicTrack, ...],
    ) -> QueueRebuildResult:
        """Replace current playback and upcoming items with one ordered track set."""
        if not tracks:
            raise MusicQueryError("A rebuilt music queue must not be empty")
        if len(tracks) > self._settings.max_queue_length:
            raise MusicQueueFullError("The rebuilt music queue is too large")
        channel = await self._channel_for(identity)
        session = self._session(channel)
        items = deque(QueuedTrack(track, identity.person_id) for track in tracks)
        async with session.lock:
            await self._reserve_voice_locked(channel, session)
            replaced_current = session.current is not None
            session.queue = items
            session.idempotent_enqueues.clear()
            if replaced_current:
                session.skip_requested.set()
            else:
                session.state = PlaybackState.WAITING
            session.revision += 1
            playback = session.playback if replaced_current else None
            started = self._start_worker_locked(channel, session)
        if playback is not None:
            try:
                await playback.stop()
            except Exception as exc:
                logger.warning(
                    "Could not stop current track while rebuilding queue: channel=%s error=%s",
                    self._channel_ref(channel),
                    type(exc).__name__,
                )
        logger.info(
            "Music queue rebuilt: channel=%s tracks=%s replaced_current=%s "
            "playback_worker_started=%s",
            self._channel_ref(channel),
            len(tracks),
            replaced_current,
            started,
        )
        return QueueRebuildResult(channel, len(tracks), replaced_current, started)

    async def skip(self, identity: AgentIdentity) -> bool:
        """Request one current track to stop; repeated calls before advance are harmless."""
        channel = await self._channel_for(identity)
        session = self._session(channel)
        async with session.lock:
            if session.current is None:
                logger.info(
                    "Music skip ignored because queue is idle: channel=%s",
                    self._channel_ref(channel),
                )
                return False
            session.skip_requested.set()
            session.revision += 1
            playback = session.playback
        if playback is not None:
            try:
                await playback.stop()
            except Exception as exc:
                logger.warning(
                    "Failed to stop OOPZ voice while skipping music: channel=%s error=%s",
                    self._channel_ref(channel),
                    type(exc).__name__,
                )
        logger.info("Music skip requested: channel=%s", self._channel_ref(channel))
        return True

    async def pause(self, identity: AgentIdentity) -> bool:
        """Pause only when this caller's voice channel owns the backend."""
        channel = await self._channel_for(identity)
        session = self._session(channel)
        async with session.lock:
            if session.playback is None or session.state is not PlaybackState.PLAYING:
                logger.info("Music pause ignored: channel=%s", self._channel_ref(channel))
                return False
            playback = session.playback
        try:
            paused = await playback.pause()
        except Exception as exc:
            logger.warning(
                "Music pause failed: channel=%s error=%s",
                self._channel_ref(channel),
                type(exc).__name__,
            )
            raise MusicPlaybackError("Failed to pause music") from exc
        if paused:
            async with session.lock:
                if session.playback is playback:
                    session.state = PlaybackState.PAUSED
                    session.revision += 1
        logger.info(
            "Music pause completed: channel=%s applied=%s",
            self._channel_ref(channel),
            paused,
        )
        return paused

    async def resume(self, identity: AgentIdentity) -> bool:
        """Resume only when this caller's voice channel owns the backend."""
        channel = await self._channel_for(identity)
        session = self._session(channel)
        async with session.lock:
            if session.playback is None or session.state is not PlaybackState.PAUSED:
                logger.info("Music resume ignored: channel=%s", self._channel_ref(channel))
                return False
            playback = session.playback
        try:
            resumed = await playback.resume()
        except Exception as exc:
            logger.warning(
                "Music resume failed: channel=%s error=%s",
                self._channel_ref(channel),
                type(exc).__name__,
            )
            raise MusicPlaybackError("Failed to resume music") from exc
        if resumed:
            async with session.lock:
                if session.playback is playback:
                    session.state = PlaybackState.PLAYING
                    session.revision += 1
        logger.info(
            "Music resume completed: channel=%s applied=%s",
            self._channel_ref(channel),
            resumed,
        )
        return resumed

    async def aclose(self) -> None:
        """Cancel all workers before closing OOPZ voice and catalog resources."""
        logger.info("Closing music service: active_channels=%s", len(self._sessions))
        self._closing = True
        workers = tuple(
            session.worker
            for session in self._sessions.values()
            if session.worker is not None and not session.worker.done()
        )
        for worker in workers:
            worker.cancel()
        if workers:
            await asyncio.gather(*workers, return_exceptions=True)
        try:
            await self._voice.aclose()
        finally:
            await self._catalog.aclose()

    async def _channel_for(self, identity: AgentIdentity) -> VoiceChannelKey:
        area_id = identity.conversation.area_id.strip()
        if not area_id:
            raise MusicVoiceChannelRequiredError(
                "Music playback requires a message in an OOPZ area"
            )
        try:
            channel_id = await self._voice.voice_channel_for_user(
                area_id,
                identity.person_id,
            )
        except Exception as exc:
            logger.warning("Could not resolve caller voice channel: error=%s", type(exc).__name__)
            raise MusicPlaybackError("Failed to locate the user's voice channel") from exc
        if not channel_id:
            raise MusicVoiceChannelRequiredError(
                "Join an OOPZ voice channel before controlling music"
            )
        return VoiceChannelKey(area_id, channel_id)

    def _session(self, channel: VoiceChannelKey) -> _MusicSession:
        session = self._sessions.get(channel)
        if session is None:
            session = _MusicSession()
            self._sessions[channel] = session
        return session

    async def _play_queue(
        self,
        channel: VoiceChannelKey,
        session: _MusicSession,
    ) -> None:
        drained_normally = False
        backend_retries: dict[UUID, int] = {}
        try:
            while True:
                async with session.lock:
                    if not session.queue:
                        session.current = None
                        session.state = PlaybackState.IDLE
                        session.revision += 1
                        session.voice_reserved = False
                        logger.info(
                            "Music playback worker idle: channel=%s",
                            self._channel_ref(channel),
                        )
                        # Keep the session lock until the SDK has actually left. An
                        # enqueue arriving in this boundary must acquire a fresh lease,
                        # not mistake the generation being released for a reservation.
                        await self._release_voice(channel)
                        if session.worker is asyncio.current_task():
                            session.worker = None
                        drained_normally = True
                        return
                    else:
                        item = self._take_next(session)
                        session.current = item
                        session.state = PlaybackState.LOADING
                        session.revision += 1
                        session.skip_requested.clear()
                playback: MusicPlayback | None = None
                completed = False
                retry_current = False
                try:
                    logger.info(
                        "Music track resolving: channel=%s",
                        self._channel_ref(channel),
                    )
                    playable = await self._catalog.resolve(item.track)
                    if session.skip_requested.is_set():
                        logger.info(
                            "Music track skipped before playback: channel=%s",
                            self._channel_ref(channel),
                        )
                        continue
                    playback = await self._voice.start_playback(channel, playable.stream_url)
                    if session.skip_requested.is_set():
                        await playback.stop()
                        continue
                    async with session.lock:
                        session.playback = playback
                        session.state = PlaybackState.PLAYING
                        session.revision += 1
                    logger.info(
                        "Music track playback started: channel=%s",
                        self._channel_ref(channel),
                    )
                    result = await playback.wait_finished()
                    completed = result.end_reason is MusicPlaybackEndReason.FINISHED
                    if (
                        result.end_reason is MusicPlaybackEndReason.BACKEND_CLOSED
                        and backend_retries.get(item.id, 0) < 1
                        and not session.skip_requested.is_set()
                    ):
                        backend_retries[item.id] = backend_retries.get(item.id, 0) + 1
                        retry_current = True
                        logger.warning(
                            "Retrying music track after shared backend failure: "
                            "channel=%s attempt=%s",
                            self._channel_ref(channel),
                            backend_retries[item.id],
                        )
                    elif not completed and result.end_reason not in {
                        MusicPlaybackEndReason.STOPPED,
                        MusicPlaybackEndReason.REPLACED,
                    }:
                        raise MusicPlaybackError(
                            f"Music playback ended with {result.end_reason.value}"
                        ) from result.terminal_error
                except asyncio.CancelledError:
                    if playback is not None:
                        with suppress(Exception):
                            await playback.stop()
                    raise
                except MusicBackendClosedError as exc:
                    if backend_retries.get(item.id, 0) < 1 and not session.skip_requested.is_set():
                        backend_retries[item.id] = backend_retries.get(item.id, 0) + 1
                        retry_current = True
                        logger.warning(
                            "Retrying music startup after shared backend failure: "
                            "channel=%s attempt=%s error=%s",
                            self._channel_ref(channel),
                            backend_retries[item.id],
                            type(exc).__name__,
                        )
                    else:
                        logger.error(
                            "Music backend recovery exhausted: channel=%s error=%s",
                            self._channel_ref(channel),
                            type(exc).__name__,
                        )
                        async with session.lock:
                            session.state = PlaybackState.FAILED
                            session.revision += 1
                except Exception as exc:
                    logger.error(
                        "Music playback failed: channel=%s error=%s",
                        self._channel_ref(channel),
                        type(exc).__name__,
                    )
                    if playback is not None:
                        with suppress(Exception):
                            await playback.stop()
                    async with session.lock:
                        session.state = PlaybackState.FAILED
                        session.revision += 1
                finally:
                    async with session.lock:
                        if not session.skip_requested.is_set() and completed:
                            self._retain_completed(session, item)
                            backend_retries.pop(item.id, None)
                        elif not session.skip_requested.is_set() and retry_current:
                            session.queue.appendleft(item)
                            session.state = PlaybackState.WAITING
                            session.revision += 1
                        else:
                            backend_retries.pop(item.id, None)
                        if session.playback is playback:
                            session.playback = None
                        session.current = None
                        session.skip_requested.clear()
                        session.revision += 1
                    logger.debug(
                        "Music playback state reset: channel=%s",
                        self._channel_ref(channel),
                    )
        finally:
            if not drained_normally:
                async with session.lock:
                    reserved = session.voice_reserved
                    session.playback = None
                    session.current = None
                    session.skip_requested.clear()
                    session.voice_reserved = False
                    if session.worker is asyncio.current_task():
                        session.worker = None
                    if session.state is not PlaybackState.IDLE:
                        session.state = PlaybackState.IDLE
                        session.revision += 1
                    if reserved:
                        await self._release_voice(channel)

    def _take_next(self, session: _MusicSession) -> QueuedTrack:
        if session.mode is PlaybackMode.SHUFFLE and len(session.queue) > 1:
            index = self._rng.randrange(len(session.queue))
            session.queue.rotate(-index)
            item = session.queue.popleft()
            session.queue.rotate(index)
            return item
        return session.queue.popleft()

    @staticmethod
    def _retain_completed(session: _MusicSession, item: QueuedTrack) -> None:
        if session.mode is PlaybackMode.REPEAT_ONE:
            session.queue.appendleft(item)
        elif session.mode is PlaybackMode.REPEAT_ALL:
            session.queue.append(item)

    async def _reserve_voice_locked(
        self,
        channel: VoiceChannelKey,
        session: _MusicSession,
    ) -> None:
        """Reserve the shared backend while the caller holds ``session.lock``."""
        if self._closing:
            raise RuntimeError("Music service is closing")
        if session.voice_reserved:
            return
        try:
            acquired = await self._voice.acquire(channel)
        except Exception as exc:
            logger.warning(
                "Could not reserve OOPZ voice for music: channel=%s error=%s",
                self._channel_ref(channel),
                type(exc).__name__,
            )
            raise MusicPlaybackError("Failed to reserve OOPZ voice for music") from exc
        if not acquired:
            logger.info(
                "Music playback rejected because OOPZ voice is busy: channel=%s",
                self._channel_ref(channel),
            )
            raise MusicVoiceBusyError("OOPZ voice is currently used by another feature")
        session.voice_reserved = True

    def _start_worker_locked(
        self,
        channel: VoiceChannelKey,
        session: _MusicSession,
    ) -> bool:
        """Create one worker while the caller holds ``session.lock``."""
        if self._closing:
            raise RuntimeError("Music service is closing")
        if session.worker is not None and not session.worker.done():
            return False
        worker = asyncio.create_task(
            self._play_queue(channel, session),
            name=f"music:{self._channel_ref(channel)}",
        )
        session.worker = worker
        worker.add_done_callback(lambda completed: self._on_worker_done(channel, completed))
        return True

    def _on_worker_done(
        self,
        channel: VoiceChannelKey,
        worker: asyncio.Task[None],
    ) -> None:
        if worker.cancelled():
            logger.debug("Music playback worker cancelled: channel=%s", self._channel_ref(channel))
            return
        try:
            worker.result()
        except Exception as exc:
            logger.error(
                "Music playback worker failed: channel=%s error=%s",
                self._channel_ref(channel),
                type(exc).__name__,
            )
        else:
            logger.debug("Music playback worker completed: channel=%s", self._channel_ref(channel))

    async def _release_voice(self, channel: VoiceChannelKey) -> None:
        try:
            released = await self._voice.release(channel)
        except Exception as exc:
            logger.warning(
                "Could not release idle OOPZ voice channel: channel=%s error=%s",
                self._channel_ref(channel),
                type(exc).__name__,
            )
            return
        logger.info(
            "Music voice channel released after queue drained: channel=%s left=%s",
            self._channel_ref(channel),
            released,
        )

    @staticmethod
    def _channel_ref(channel: VoiceChannelKey) -> str:
        return opaque_ref(channel.area_id, channel.channel_id)
