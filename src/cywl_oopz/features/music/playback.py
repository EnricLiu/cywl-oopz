"""Queue worker lifecycle and playback recovery."""

from __future__ import annotations

import asyncio
import logging
import random
from contextlib import suppress
from dataclasses import dataclass
from uuid import UUID

from cywl_oopz.core.observability import opaque_ref

from .errors import (
    MusicBackendClosedError,
    MusicCatalogError,
    MusicDecoderError,
    MusicNotFoundError,
    MusicPlaybackError,
)
from .models import (
    MusicFailureCode,
    MusicFailureScope,
    MusicPlaybackEndReason,
    MusicPlaybackPolicy,
    MusicPlaybackResult,
    PlaybackState,
    QueuedTrack,
    RepeatPolicy,
    VoiceChannelKey,
)
from .ports import MusicCatalog, MusicPlayback, MusicVoiceGateway
from .session import MusicSession
from .track_playback import TrackPlaybackRunner

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PlaybackAttempt:
    playback: MusicPlayback | None = None
    completed: bool = False
    retry_current: bool = False
    halt_after_failure: bool = False


class MusicPlaybackCoordinator:
    """Own queue advancement, retry decisions, and voice release for each worker."""

    def __init__(self, catalog: MusicCatalog, voice: MusicVoiceGateway, rng: random.Random) -> None:
        self._tracks = TrackPlaybackRunner(catalog, voice)
        self._voice = voice
        self._rng = rng

    async def run(self, channel: VoiceChannelKey, session: MusicSession) -> None:
        drained_normally = False
        backend_retries: dict[UUID, int] = {}
        media_retries: dict[UUID, int] = {}
        try:
            while True:
                item = await self._take_or_release(channel, session)
                if item is None:
                    drained_normally = True
                    return
                attempt = PlaybackAttempt()
                try:
                    await self._play_attempt(
                        channel, session, item, attempt, backend_retries, media_retries
                    )
                finally:
                    await self._finish_attempt(
                        channel, session, item, attempt, backend_retries, media_retries
                    )
                if attempt.halt_after_failure:
                    return
        finally:
            if not drained_normally:
                await self._close_session(channel, session)

    async def _take_or_release(
        self, channel: VoiceChannelKey, session: MusicSession
    ) -> QueuedTrack | None:
        async with session.lock:
            if (
                not session.queue
                and session.policy.repeat is RepeatPolicy.ALL
                and session.cycle_history
            ):
                session.queue.extend(session.cycle_history)
                session.cycle_history.clear()
                session.revision += 1
                logger.info(
                    "Music repeat cycle rebuilt: channel=%s tracks=%s order=%s",
                    opaque_ref(channel.area_id, channel.channel_id),
                    len(session.queue),
                    session.policy.order.value,
                )
            if not session.queue:
                session.current = None
                session.state = PlaybackState.RELEASING
                session.revision += 1
                logger.info(
                    "Music playback worker releasing idle channel: channel=%s",
                    opaque_ref(channel.area_id, channel.channel_id),
                )
                # Keep the session lock until the SDK has actually left. An
                # enqueue arriving in this boundary must acquire a fresh lease,
                # not mistake the generation being released for a reservation.
                released = await self.release_voice(channel)
                session.voice_reserved = not released
                if released:
                    session.state = PlaybackState.IDLE
                    session.policy = MusicPlaybackPolicy()
                    session.cycle_history.clear()
                    session.idempotent_enqueues.clear()
                else:
                    session.state = PlaybackState.FAILED
                    session.record_failure(
                        None,
                        MusicFailureCode.RELEASE_FAILED,
                        MusicFailureScope.VOICE_SESSION,
                        recoverable=True,
                        retry_count=3,
                    )
                session.revision += 1
                if session.worker is asyncio.current_task():
                    session.worker = None
                return None
            else:
                item = session.take_next(self._rng)
                session.current = item
                session.state = PlaybackState.LOADING
                session.revision += 1
                session.skip_requested.clear()
                session.retain_skipped_for_cycle = False
        return item

    async def _play_attempt(
        self,
        channel: VoiceChannelKey,
        session: MusicSession,
        item: QueuedTrack,
        attempt: PlaybackAttempt,
        backend_retries: dict[UUID, int],
        media_retries: dict[UUID, int],
    ) -> None:
        try:
            attempt.playback = await self._tracks.prepare(channel, session, item)
            if attempt.playback is None:
                return
            result = await attempt.playback.wait_finished()
            await self._handle_result(
                channel, session, item, attempt, result, backend_retries, media_retries
            )
        except asyncio.CancelledError:
            if attempt.playback is not None:
                with suppress(Exception):
                    await attempt.playback.stop()
            raise
        except MusicBackendClosedError as exc:
            await self._handle_backend_error(channel, session, item, attempt, backend_retries, exc)
        except MusicDecoderError as exc:
            await self._handle_decoder_error(channel, session, item, attempt, media_retries, exc)
        except Exception as exc:
            logger.error(
                "Music playback failed: channel=%s error=%s",
                opaque_ref(channel.area_id, channel.channel_id),
                type(exc).__name__,
            )
            if attempt.playback is not None:
                with suppress(Exception):
                    await attempt.playback.stop()
            async with session.lock:
                session.state = PlaybackState.FAILED
                scope = (
                    MusicFailureScope.CATALOG
                    if isinstance(exc, (MusicCatalogError, MusicNotFoundError))
                    else MusicFailureScope.TRACK
                )
                session.record_failure(
                    item,
                    (
                        MusicFailureCode.CATALOG_ERROR
                        if scope is MusicFailureScope.CATALOG
                        else MusicFailureCode.TRACK_ERROR
                    ),
                    scope,
                    recoverable=False,
                    retry_count=0,
                )
                session.revision += 1

    async def _handle_result(
        self,
        channel: VoiceChannelKey,
        session: MusicSession,
        item: QueuedTrack,
        attempt: PlaybackAttempt,
        result: MusicPlaybackResult,
        backend_retries: dict[UUID, int],
        media_retries: dict[UUID, int],
    ) -> None:
        attempt.completed = result.end_reason is MusicPlaybackEndReason.FINISHED
        if result.end_reason in {
            MusicPlaybackEndReason.BACKEND_CLOSED,
            MusicPlaybackEndReason.VOICE_LEFT,
        } and (backend_retries.get(item.id, 0) < 1 and not session.skip_requested.is_set()):
            backend_retries[item.id] = backend_retries.get(item.id, 0) + 1
            if result.end_reason is MusicPlaybackEndReason.VOICE_LEFT:
                attempt.retry_current = await self._reacquire_voice(channel, session)
                attempt.halt_after_failure = not attempt.retry_current
                message = "physical voice generation loss"
                failure_code = MusicFailureCode.VOICE_LEFT
            else:
                attempt.retry_current = True
                message = "shared backend failure"
                failure_code = MusicFailureCode.BACKEND_CLOSED
            async with session.lock:
                session.record_failure(
                    item,
                    failure_code,
                    MusicFailureScope.VOICE_SESSION,
                    recoverable=attempt.retry_current,
                    retry_count=backend_retries[item.id],
                )
            logger.warning(
                "Retrying music track after %s: channel=%s attempt=%s fresh_voice=%s",
                message,
                opaque_ref(channel.area_id, channel.channel_id),
                backend_retries[item.id],
                result.end_reason is MusicPlaybackEndReason.VOICE_LEFT,
            )
        elif (
            result.end_reason is MusicPlaybackEndReason.TRACK_ERROR
            and result.duration_seconds is not None
            and result.duration_seconds <= 3.0
            and media_retries.get(item.id, 0) < 1
            and not session.skip_requested.is_set()
        ):
            media_retries[item.id] = media_retries.get(item.id, 0) + 1
            attempt.retry_current = True
            async with session.lock:
                session.record_failure(
                    item,
                    MusicFailureCode.TRACK_ERROR,
                    MusicFailureScope.TRACK,
                    recoverable=True,
                    retry_count=media_retries[item.id],
                )
            logger.warning(
                "Re-resolving music after early media failure: "
                "channel=%s source=%s attempt=%s elapsed_seconds=%.3f",
                opaque_ref(channel.area_id, channel.channel_id),
                item.track.source.value,
                media_retries[item.id],
                result.duration_seconds,
            )
        elif not attempt.completed and result.end_reason not in {
            MusicPlaybackEndReason.STOPPED,
            MusicPlaybackEndReason.REPLACED,
        }:
            if result.end_reason in {
                MusicPlaybackEndReason.BACKEND_CLOSED,
                MusicPlaybackEndReason.VOICE_LEFT,
            }:
                attempt.halt_after_failure = True
                async with session.lock:
                    session.state = PlaybackState.FAILED
                    session.record_failure(
                        item,
                        MusicFailureCode(result.end_reason.value),
                        MusicFailureScope.VOICE_SESSION,
                        recoverable=False,
                        retry_count=backend_retries.get(item.id, 0),
                    )
                    session.revision += 1
            elif result.end_reason is MusicPlaybackEndReason.TRACK_ERROR:
                logger.error(
                    "Music media playback failed: channel=%s source=%s retries=%s error=%s",
                    opaque_ref(channel.area_id, channel.channel_id),
                    item.track.source.value,
                    media_retries.get(item.id, 0),
                    (
                        type(result.terminal_error).__name__
                        if result.terminal_error is not None
                        else "none"
                    ),
                )
                async with session.lock:
                    session.state = PlaybackState.FAILED
                    session.record_failure(
                        item,
                        MusicFailureCode.TRACK_ERROR,
                        MusicFailureScope.TRACK,
                        recoverable=False,
                        retry_count=media_retries.get(item.id, 0),
                    )
                    session.revision += 1
            else:
                raise MusicPlaybackError(
                    f"Music playback ended with {result.end_reason.value}"
                ) from result.terminal_error

    async def _handle_backend_error(
        self,
        channel: VoiceChannelKey,
        session: MusicSession,
        item: QueuedTrack,
        attempt: PlaybackAttempt,
        backend_retries: dict[UUID, int],
        exc: MusicBackendClosedError,
    ) -> None:
        if backend_retries.get(item.id, 0) < 1 and not session.skip_requested.is_set():
            backend_retries[item.id] = backend_retries.get(item.id, 0) + 1
            attempt.retry_current = True
            logger.warning(
                "Retrying music startup after shared backend failure: "
                "channel=%s attempt=%s error=%s",
                opaque_ref(channel.area_id, channel.channel_id),
                backend_retries[item.id],
                type(exc).__name__,
            )
        else:
            logger.error(
                "Music backend recovery exhausted: channel=%s error=%s",
                opaque_ref(channel.area_id, channel.channel_id),
                type(exc).__name__,
            )
            async with session.lock:
                session.state = PlaybackState.FAILED
                session.record_failure(
                    item,
                    MusicFailureCode.BACKEND_CLOSED,
                    MusicFailureScope.VOICE_SESSION,
                    recoverable=False,
                    retry_count=backend_retries.get(item.id, 0),
                )
                session.revision += 1
            attempt.halt_after_failure = True

    async def _handle_decoder_error(
        self,
        channel: VoiceChannelKey,
        session: MusicSession,
        item: QueuedTrack,
        attempt: PlaybackAttempt,
        media_retries: dict[UUID, int],
        exc: MusicDecoderError,
    ) -> None:
        if media_retries.get(item.id, 0) < 1 and not session.skip_requested.is_set():
            media_retries[item.id] = media_retries.get(item.id, 0) + 1
            attempt.retry_current = True
            async with session.lock:
                session.record_failure(
                    item,
                    MusicFailureCode.TRACK_ERROR,
                    MusicFailureScope.TRACK,
                    recoverable=True,
                    retry_count=media_retries[item.id],
                )
            logger.warning(
                "Re-resolving music after decoder startup failure: "
                "channel=%s source=%s attempt=%s error=%s",
                opaque_ref(channel.area_id, channel.channel_id),
                item.track.source.value,
                media_retries[item.id],
                type(exc).__name__,
            )
        else:
            logger.error(
                "Music media recovery exhausted: channel=%s source=%s error=%s",
                opaque_ref(channel.area_id, channel.channel_id),
                item.track.source.value,
                type(exc).__name__,
            )
            async with session.lock:
                session.state = PlaybackState.FAILED
                session.record_failure(
                    item,
                    MusicFailureCode.TRACK_ERROR,
                    MusicFailureScope.TRACK,
                    recoverable=False,
                    retry_count=media_retries.get(item.id, 0),
                )
                session.revision += 1

    async def _finish_attempt(
        self,
        channel: VoiceChannelKey,
        session: MusicSession,
        item: QueuedTrack,
        attempt: PlaybackAttempt,
        backend_retries: dict[UUID, int],
        media_retries: dict[UUID, int],
    ) -> None:
        async with session.lock:
            if attempt.halt_after_failure and not session.skip_requested.is_set():
                session.queue.appendleft(item)
                backend_retries.pop(item.id, None)
                media_retries.pop(item.id, None)
            elif not session.skip_requested.is_set() and attempt.completed:
                session.retain_completed(item)
                session.last_failure = None
                backend_retries.pop(item.id, None)
                media_retries.pop(item.id, None)
            elif not session.skip_requested.is_set() and attempt.retry_current:
                session.queue.appendleft(item)
                session.state = PlaybackState.WAITING
                session.revision += 1
            elif session.retain_skipped_for_cycle:
                session.cycle_history.append(item)
                backend_retries.pop(item.id, None)
                media_retries.pop(item.id, None)
            else:
                backend_retries.pop(item.id, None)
                media_retries.pop(item.id, None)
            if session.playback is attempt.playback:
                session.playback = None
            session.current = None
            session.skip_requested.clear()
            session.retain_skipped_for_cycle = False
            session.revision += 1
        logger.debug(
            "Music playback state reset: channel=%s",
            opaque_ref(channel.area_id, channel.channel_id),
        )

    async def _close_session(self, channel: VoiceChannelKey, session: MusicSession) -> None:
        async with session.lock:
            reserved = session.voice_reserved
            session.playback = None
            session.cancel_resolve()
            session.resolve_task = None
            session.current = None
            session.skip_requested.clear()
            session.retain_skipped_for_cycle = False
            session.voice_reserved = False
            if session.worker is asyncio.current_task():
                session.worker = None
            if session.state not in {PlaybackState.IDLE, PlaybackState.FAILED}:
                session.state = PlaybackState.IDLE
                session.revision += 1
            if reserved:
                released = await self.release_voice(channel)
                session.voice_reserved = not released
                if not released and session.state is not PlaybackState.FAILED:
                    session.state = PlaybackState.FAILED
                    session.record_failure(
                        None,
                        MusicFailureCode.RELEASE_FAILED,
                        MusicFailureScope.VOICE_SESSION,
                        recoverable=True,
                        retry_count=3,
                    )
                    session.revision += 1

    async def _reacquire_voice(
        self,
        channel: VoiceChannelKey,
        session: MusicSession,
    ) -> bool:
        """Replace one stale physical voice generation without holding session.lock."""
        async with session.lock:
            session.voice_reserved = False
            session.state = PlaybackState.RECOVERING
            session.revision += 1
        try:
            await self._voice.reset(channel)
            acquired = await self._voice.acquire(channel)
        except Exception as exc:
            logger.warning(
                "Could not acquire a fresh OOPZ voice generation for music: channel=%s error=%s",
                opaque_ref(channel.area_id, channel.channel_id),
                type(exc).__name__,
            )
            return False
        async with session.lock:
            session.voice_reserved = acquired
        return acquired

    async def release_voice(self, channel: VoiceChannelKey) -> bool:
        for attempt, delay in enumerate((0.1, 0.25, 0.5), start=1):
            try:
                released = await self._voice.release(channel)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Could not release idle OOPZ voice channel: channel=%s attempt=%s error=%s",
                    opaque_ref(channel.area_id, channel.channel_id),
                    attempt,
                    type(exc).__name__,
                )
                if attempt < 3:
                    await asyncio.sleep(delay)
                continue
            logger.info(
                "Music voice channel released after queue drained: channel=%s left=%s",
                opaque_ref(channel.area_id, channel.channel_id),
                released,
            )
            return True
        return False
