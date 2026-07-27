"""Per-voice-channel queue state and application-owned playback workers."""

from __future__ import annotations

import asyncio
import logging
import random
from collections import OrderedDict, deque
from contextlib import suppress
from dataclasses import dataclass, field

from cywl_oopz.core.observability import opaque_ref
from cywl_oopz.core.tasks import TaskSupervisor
from cywl_oopz.features.agent.models import AgentIdentity
from cywl_oopz.settings import MusicSettings

from .errors import (
    MusicNotFoundError,
    MusicPlaybackError,
    MusicQueryError,
    MusicQueueFullError,
    MusicVoiceChannelRequiredError,
)
from .models import (
    EnqueueResult,
    MusicQueueSnapshot,
    MusicTrack,
    PlaybackMode,
    PlaybackModeChange,
    PlaybackState,
    QueuedTrack,
    VoiceChannelKey,
)
from .ports import MusicCatalog, MusicVoiceGateway

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _MusicSession:
    """Mutable state protected by one voice-channel lock."""

    queue: deque[QueuedTrack] = field(default_factory=deque)
    current: QueuedTrack | None = None
    state: PlaybackState = PlaybackState.IDLE
    mode: PlaybackMode = PlaybackMode.SEQUENTIAL
    revision: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    skip_requested: asyncio.Event = field(default_factory=asyncio.Event)
    idempotent_enqueues: OrderedDict[str, EnqueueResult] = field(default_factory=OrderedDict)


class MusicRequestService:
    """Search, enqueue, inspect, and control bounded voice-channel queues."""

    _FINISHED_BACKEND_STATES = frozenset(
        {"finished", "idle", "joined", "stopped", "ended", "error"}
    )
    _COMPLETED_BACKEND_STATES = frozenset({"finished", "idle", "joined", "stopped", "ended"})

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
        self._tasks = TaskSupervisor(lambda key: f"music:{self._channel_ref(key)}")
        # oopz-sdk exposes one voice backend per bot. Queue state remains channel-scoped,
        # while this slot gives one track at a time exclusive access to that backend.
        self._voice_slot = asyncio.Lock()
        self._voice_owner: VoiceChannelKey | None = None

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
            session.queue.append(item)
            session.revision += 1
            session.state = PlaybackState.WAITING if session.current is None else session.state
            position = len(session.queue) + (1 if session.current is not None else 0)
        started = self._tasks.start(channel, self._play_queue(channel, session))
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
            owns_voice = self._voice_owner == channel
        if owns_voice:
            try:
                await self._voice.stop()
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
            if self._voice_owner != channel or session.state is not PlaybackState.PLAYING:
                logger.info("Music pause ignored: channel=%s", self._channel_ref(channel))
                return False
        try:
            paused = await self._voice.pause()
        except Exception as exc:
            logger.warning(
                "Music pause failed: channel=%s error=%s",
                self._channel_ref(channel),
                type(exc).__name__,
            )
            raise MusicPlaybackError("Failed to pause music") from exc
        if paused:
            async with session.lock:
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
            if self._voice_owner != channel or session.state is not PlaybackState.PAUSED:
                logger.info("Music resume ignored: channel=%s", self._channel_ref(channel))
                return False
        try:
            resumed = await self._voice.resume()
        except Exception as exc:
            logger.warning(
                "Music resume failed: channel=%s error=%s",
                self._channel_ref(channel),
                type(exc).__name__,
            )
            raise MusicPlaybackError("Failed to resume music") from exc
        if resumed:
            async with session.lock:
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
        await self._tasks.close()
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
        while True:
            async with self._voice_slot:
                async with session.lock:
                    if not session.queue:
                        session.current = None
                        session.state = PlaybackState.IDLE
                        session.revision += 1
                        logger.info(
                            "Music playback worker idle: channel=%s",
                            self._channel_ref(channel),
                        )
                        should_leave = True
                    else:
                        should_leave = False
                        item = self._take_next(session)
                        session.current = item
                        session.state = PlaybackState.LOADING
                        session.revision += 1
                        session.skip_requested.clear()
                if should_leave:
                    await self._leave_idle_channel(channel)
                    return
                self._voice_owner = channel
                completed_state = ""
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
                    await self._voice.play(channel, playable.stream_url)
                    async with session.lock:
                        session.state = PlaybackState.PLAYING
                        session.revision += 1
                    logger.info(
                        "Music track playback started: channel=%s",
                        self._channel_ref(channel),
                    )
                    completed_state = await self._wait_until_finished(session)
                except asyncio.CancelledError:
                    with suppress(Exception):
                        await self._voice.stop()
                    raise
                except Exception as exc:
                    logger.error(
                        "Music playback failed: channel=%s error=%s",
                        self._channel_ref(channel),
                        type(exc).__name__,
                    )
                    with suppress(Exception):
                        await self._voice.stop()
                    async with session.lock:
                        session.state = PlaybackState.FAILED
                        session.revision += 1
                finally:
                    self._voice_owner = None
                    async with session.lock:
                        if (
                            not session.skip_requested.is_set()
                            and completed_state in self._COMPLETED_BACKEND_STATES
                        ):
                            self._retain_completed(session, item)
                        session.current = None
                        session.skip_requested.clear()
                        session.revision += 1
                    logger.debug(
                        "Music playback state reset: channel=%s",
                        self._channel_ref(channel),
                    )

    async def _wait_until_finished(self, session: _MusicSession) -> str:
        while not session.skip_requested.is_set():
            await asyncio.sleep(self._settings.playback_poll_seconds)
            try:
                state = (await self._voice.state()).strip().casefold()
            except Exception as exc:
                raise MusicPlaybackError("Failed to read music playback state") from exc
            if state in self._FINISHED_BACKEND_STATES:
                logger.debug("Music backend reported finished state: state=%s", state)
                return state
        return ""

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

    async def _leave_idle_channel(self, channel: VoiceChannelKey) -> None:
        try:
            left = await self._voice.leave(channel)
        except Exception as exc:
            logger.warning(
                "Could not leave idle OOPZ voice channel: channel=%s error=%s",
                self._channel_ref(channel),
                type(exc).__name__,
            )
            return
        logger.info(
            "Music voice channel released after queue drained: channel=%s left=%s",
            self._channel_ref(channel),
            left,
        )

    @staticmethod
    def _channel_ref(channel: VoiceChannelKey) -> str:
        return opaque_ref(channel.area_id, channel.channel_id)
