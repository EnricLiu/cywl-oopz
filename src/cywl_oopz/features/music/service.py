"""Per-voice-channel queue state and application-owned playback workers."""

from __future__ import annotations

import asyncio
import logging
import random
from collections import deque
from collections.abc import Awaitable, Callable

from cywl_oopz.conversation.models import ActorContext
from cywl_oopz.core.observability import opaque_ref
from cywl_oopz.settings import MusicSettings

from .errors import (
    MusicNotFoundError,
    MusicPlaybackError,
    MusicQueryError,
    MusicQueueFullError,
    MusicReferenceError,
    MusicVoiceBusyError,
    MusicVoiceChannelRequiredError,
)
from .models import (
    EnqueueResult,
    MusicPlaybackPolicy,
    MusicProviderHealth,
    MusicQueueClearResult,
    MusicQueueSnapshot,
    MusicSourceKind,
    MusicTrack,
    MusicTrackReference,
    PlaybackOrder,
    PlaybackPolicyChange,
    PlaybackState,
    QueuedTrack,
    QueueRebuildResult,
    RepeatPolicy,
    VoiceChannelKey,
)
from .playback import MusicPlaybackCoordinator
from .ports import MusicCatalog, MusicVoiceGateway
from .references import MusicInputParser
from .session import MusicSession

logger = logging.getLogger(__name__)


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
        self._input_parser = MusicInputParser()
        self._playback = MusicPlaybackCoordinator(catalog, voice, rng or random.Random())
        self._sessions: dict[VoiceChannelKey, MusicSession] = {}
        self._closing = False

    @property
    def default_source(self) -> MusicSourceKind:
        return self._settings.default_source

    @property
    def enabled_sources(self) -> tuple[MusicSourceKind, ...]:
        return self._settings.enabled_sources

    async def search(
        self,
        query: str,
        *,
        source: MusicSourceKind | None = None,
        limit: int | None = None,
    ) -> tuple[MusicTrack, ...]:
        """Search the configured catalog with deterministic input and result bounds."""
        normalized = self._normalize_query(query)
        normalized_source = MusicSourceKind(source) if source is not None else None
        requested_limit = min(limit or self._settings.search_limit, self._settings.search_limit)
        logger.info(
            "Music search started: source=%s query_characters=%s limit=%s",
            normalized_source.value if normalized_source is not None else "default",
            len(normalized),
            requested_limit,
        )
        matches = await self._catalog.search(
            normalized,
            limit=requested_limit,
            source=normalized_source,
        )
        logger.info(
            "Music search completed: source=%s result_count=%s",
            normalized_source.value if normalized_source is not None else "default",
            len(matches),
        )
        return matches

    async def lookup(self, reference: MusicTrackReference) -> MusicTrack:
        """Load one exact stable reference before it enters a queue or playlist."""
        return await self._catalog.lookup(reference)

    async def track_from_input(
        self,
        value: str,
        *,
        source: MusicSourceKind | None = None,
    ) -> MusicTrack:
        """Resolve a URL/locator or the top explicit-source text match into trusted metadata."""
        normalized = value.strip()
        if not normalized:
            raise MusicQueryError("Music input must not be empty")
        selected_source = MusicSourceKind(source) if source is not None else None
        parsed = self._input_parser.parse(normalized)
        if parsed is not None:
            if selected_source is not None and selected_source is not parsed.source:
                raise MusicReferenceError(
                    "Music URL source does not match the explicitly selected source"
                )
            if isinstance(parsed, MusicTrackReference):
                return await self._catalog.lookup(parsed)
            return await self._catalog.inspect(parsed)
        return await self._track_from_query(normalized, source=selected_source)

    async def health(self) -> tuple[MusicProviderHealth, ...]:
        """Return independently probed health for every enabled source."""
        return await self._catalog.health()

    async def enqueue(
        self,
        identity: ActorContext,
        query: str,
        *,
        idempotency_key: str = "",
    ) -> EnqueueResult:
        """Compatibility wrapper accepting either a query or a supported source URL."""
        return await self.enqueue_input(
            identity,
            query,
            idempotency_key=idempotency_key,
        )

    async def enqueue_query(
        self,
        identity: ActorContext,
        query: str,
        *,
        source: MusicSourceKind | None = None,
        idempotency_key: str = "",
    ) -> EnqueueResult:
        """Search one source, validate its top stable reference, and enqueue it."""
        return await self._enqueue_from(
            identity,
            lambda: self._track_from_query(query, source=source),
            idempotency_key=idempotency_key,
        )

    async def enqueue_reference(
        self,
        identity: ActorContext,
        reference: MusicTrackReference,
        *,
        idempotency_key: str = "",
    ) -> EnqueueResult:
        """Validate and enqueue one exact provider reference."""
        return await self._enqueue_from(
            identity,
            lambda: self._catalog.lookup(reference),
            idempotency_key=idempotency_key,
        )

    async def enqueue_input(
        self,
        identity: ActorContext,
        value: str,
        *,
        source: MusicSourceKind | None = None,
        idempotency_key: str = "",
    ) -> EnqueueResult:
        """Classify one user value, remotely normalize it, then enqueue trusted metadata."""
        return await self._enqueue_from(
            identity,
            lambda: self.track_from_input(value, source=source),
            idempotency_key=idempotency_key,
        )

    async def _enqueue_from(
        self,
        identity: ActorContext,
        load_track: Callable[[], Awaitable[MusicTrack]],
        *,
        idempotency_key: str,
    ) -> EnqueueResult:
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
        track = await load_track()
        item = QueuedTrack(track, identity.person_id)
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
            result = EnqueueResult(channel, item, position, started)
            if idempotency_key:
                session.idempotent_enqueues[idempotency_key] = result
                while len(session.idempotent_enqueues) > 256:
                    session.idempotent_enqueues.popitem(last=False)
        logger.info(
            "Music enqueued: channel=%s source=%s position=%s playback_worker_started=%s",
            self._channel_ref(channel),
            track.source.value,
            position,
            started,
        )
        return result

    async def _track_from_query(
        self,
        query: str,
        *,
        source: MusicSourceKind | None,
    ) -> MusicTrack:
        matches = await self.search(query, source=source, limit=1)
        if not matches:
            raise MusicNotFoundError("No music matched the query")
        return await self._catalog.lookup(matches[0].reference)

    def _normalize_query(self, query: str) -> str:
        normalized = query.strip()
        if not normalized:
            raise MusicQueryError("Music search query must not be empty")
        if len(normalized) > self._settings.max_query_characters:
            raise MusicQueryError("Music search query is too long")
        return normalized

    async def queue(self, identity: ActorContext) -> MusicQueueSnapshot:
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
                policy=session.policy,
                current=session.current,
                upcoming=tuple(session.queue),
                cycle_completed_count=len(session.cycle_history),
                revision=session.revision,
                last_failure=session.last_failure,
            )

    async def set_policy(
        self,
        identity: ActorContext,
        *,
        order: PlaybackOrder | None = None,
        repeat: RepeatPolicy | None = None,
    ) -> PlaybackPolicyChange:
        """Change playback policy for the caller's current voice channel."""
        if order is None and repeat is None:
            raise MusicQueryError("At least one music playback policy field is required")
        channel = await self._channel_for(identity)
        session = self._session(channel)
        async with session.lock:
            try:
                policy = MusicPlaybackPolicy(
                    order=order or session.policy.order,
                    repeat=repeat or session.policy.repeat,
                )
            except ValueError as exc:
                raise MusicQueryError(str(exc)) from exc
            changed = session.policy != policy
            if changed:
                session.policy = policy
                session.revision += 1
        logger.info(
            "Music playback policy selected: channel=%s order=%s repeat=%s changed=%s",
            self._channel_ref(channel),
            policy.order.value,
            policy.repeat.value,
            changed,
        )
        return PlaybackPolicyChange(channel, policy, changed)

    async def replace_queue(
        self,
        identity: ActorContext,
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
            session.cycle_history.clear()
            session.idempotent_enqueues.clear()
            session.last_failure = None
            if replaced_current:
                session.retain_skipped_for_cycle = False
                session.skip_requested.set()
                session.cancel_resolve()
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

    async def clear(self, identity: ActorContext) -> MusicQueueClearResult:
        """Stop current playback and clear every transient queue cycle item."""
        channel = await self._channel_for(identity)
        session = self._session(channel)
        async with session.lock:
            stopped_current = session.current is not None and not session.skip_requested.is_set()
            removed_count = (
                len(session.queue) + len(session.cycle_history) + (1 if stopped_current else 0)
            )
            session.queue.clear()
            session.cycle_history.clear()
            session.idempotent_enqueues.clear()
            session.policy = MusicPlaybackPolicy()
            session.last_failure = None
            playback = session.playback if stopped_current else None
            if stopped_current:
                session.retain_skipped_for_cycle = False
                session.skip_requested.set()
                session.cancel_resolve()
            elif session.voice_reserved:
                session.state = PlaybackState.WAITING
                self._start_worker_locked(channel, session)
            else:
                session.state = PlaybackState.IDLE
            session.revision += 1
        if playback is not None:
            try:
                await playback.stop()
            except Exception as exc:
                logger.warning(
                    "Could not stop current track while clearing queue: channel=%s error=%s",
                    self._channel_ref(channel),
                    type(exc).__name__,
                )
        logger.info(
            "Music queue cleared: channel=%s stopped_current=%s removed=%s",
            self._channel_ref(channel),
            stopped_current,
            removed_count,
        )
        return MusicQueueClearResult(channel, stopped_current, removed_count)

    async def skip(self, identity: ActorContext) -> bool:
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
            session.retain_skipped_for_cycle = session.policy.repeat is RepeatPolicy.ALL
            session.cancel_resolve()
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

    async def pause(self, identity: ActorContext) -> bool:
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

    async def resume(self, identity: ActorContext) -> bool:
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

    async def _channel_for(self, identity: ActorContext) -> VoiceChannelKey:
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

    def _session(self, channel: VoiceChannelKey) -> MusicSession:
        session = self._sessions.get(channel)
        if session is None:
            session = MusicSession()
            self._sessions[channel] = session
        return session

    async def _reserve_voice_locked(
        self,
        channel: VoiceChannelKey,
        session: MusicSession,
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
        session: MusicSession,
    ) -> bool:
        """Create one worker while the caller holds ``session.lock``."""
        if self._closing:
            raise RuntimeError("Music service is closing")
        if session.worker is not None and not session.worker.done():
            return False
        worker = asyncio.create_task(
            self._playback.run(channel, session),
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

    @staticmethod
    def _channel_ref(channel: VoiceChannelKey) -> str:
        return opaque_ref(channel.area_id, channel.channel_id)
