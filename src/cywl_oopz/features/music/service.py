"""Per-voice-channel queue state and application-owned playback workers."""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict, deque
from contextlib import suppress
from dataclasses import dataclass, field

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
    revision: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    skip_requested: asyncio.Event = field(default_factory=asyncio.Event)
    idempotent_enqueues: OrderedDict[str, EnqueueResult] = field(default_factory=OrderedDict)


class MusicRequestService:
    """Search, enqueue, inspect, and control bounded voice-channel queues."""

    _FINISHED_BACKEND_STATES = frozenset({"finished", "idle", "stopped", "ended", "error"})

    def __init__(
        self,
        settings: MusicSettings,
        catalog: MusicCatalog,
        voice: MusicVoiceGateway,
    ) -> None:
        self._settings = settings
        self._catalog = catalog
        self._voice = voice
        self._sessions: dict[VoiceChannelKey, _MusicSession] = {}
        self._tasks = TaskSupervisor(lambda key: f"music:{key.area_id}:{key.channel_id}")
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
        return await self._catalog.search(
            normalized,
            limit=min(limit or self._settings.search_limit, self._settings.search_limit),
        )

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
                    return previous
        matches = await self.search(query, limit=1)
        if not matches:
            raise MusicNotFoundError("No music matched the query")
        item = QueuedTrack(matches[0], identity.person_id)
        async with session.lock:
            if idempotency_key:
                previous = session.idempotent_enqueues.get(idempotency_key)
                if previous is not None:
                    return previous
            total = len(session.queue) + (1 if session.current is not None else 0)
            if total >= self._settings.max_queue_length:
                raise MusicQueueFullError("Music queue is full")
            session.queue.append(item)
            session.revision += 1
            session.state = PlaybackState.WAITING if session.current is None else session.state
            position = len(session.queue) + (1 if session.current is not None else 0)
        started = self._tasks.start(channel, self._play_queue(channel, session))
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
            return MusicQueueSnapshot(
                voice_channel=channel,
                state=session.state,
                current=session.current,
                upcoming=tuple(session.queue),
                revision=session.revision,
            )

    async def skip(self, identity: AgentIdentity) -> bool:
        """Request one current track to stop; repeated calls before advance are harmless."""
        channel = await self._channel_for(identity)
        session = self._session(channel)
        async with session.lock:
            if session.current is None:
                return False
            session.skip_requested.set()
            session.revision += 1
            owns_voice = self._voice_owner == channel
        if owns_voice:
            try:
                await self._voice.stop()
            except Exception:
                logger.exception("Failed to stop OOPZ voice while skipping music")
        return True

    async def pause(self, identity: AgentIdentity) -> bool:
        """Pause only when this caller's voice channel owns the backend."""
        channel = await self._channel_for(identity)
        session = self._session(channel)
        async with session.lock:
            if self._voice_owner != channel or session.state is not PlaybackState.PLAYING:
                return False
        try:
            paused = await self._voice.pause()
        except Exception as exc:
            raise MusicPlaybackError("Failed to pause music") from exc
        if paused:
            async with session.lock:
                session.state = PlaybackState.PAUSED
                session.revision += 1
        return paused

    async def resume(self, identity: AgentIdentity) -> bool:
        """Resume only when this caller's voice channel owns the backend."""
        channel = await self._channel_for(identity)
        session = self._session(channel)
        async with session.lock:
            if self._voice_owner != channel or session.state is not PlaybackState.PAUSED:
                return False
        try:
            resumed = await self._voice.resume()
        except Exception as exc:
            raise MusicPlaybackError("Failed to resume music") from exc
        if resumed:
            async with session.lock:
                session.state = PlaybackState.PLAYING
                session.revision += 1
        return resumed

    async def aclose(self) -> None:
        """Cancel all workers before closing OOPZ voice and catalog resources."""
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
                        return
                    item = session.queue.popleft()
                    session.current = item
                    session.state = PlaybackState.LOADING
                    session.revision += 1
                    session.skip_requested.clear()
                self._voice_owner = channel
                try:
                    playable = await self._catalog.resolve(item.track)
                    if session.skip_requested.is_set():
                        continue
                    await self._voice.play(channel, playable.stream_url)
                    async with session.lock:
                        session.state = PlaybackState.PLAYING
                        session.revision += 1
                    await self._wait_until_finished(session)
                except asyncio.CancelledError:
                    with suppress(Exception):
                        await self._voice.stop()
                    raise
                except Exception:
                    logger.exception(
                        "Music playback failed: area=%s channel=%s track=%s",
                        channel.area_id,
                        channel.channel_id,
                        item.track.source_id,
                    )
                    with suppress(Exception):
                        await self._voice.stop()
                    async with session.lock:
                        session.state = PlaybackState.FAILED
                        session.revision += 1
                finally:
                    self._voice_owner = None
                    async with session.lock:
                        session.current = None
                        session.skip_requested.clear()
                        session.revision += 1

    async def _wait_until_finished(self, session: _MusicSession) -> None:
        while not session.skip_requested.is_set():
            await asyncio.sleep(self._settings.playback_poll_seconds)
            try:
                state = (await self._voice.state()).strip().casefold()
            except Exception as exc:
                raise MusicPlaybackError("Failed to read music playback state") from exc
            if state in self._FINISHED_BACKEND_STATES:
                return
