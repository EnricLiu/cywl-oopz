"""Single-track source resolution and playback startup."""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress

from cywl_oopz.core.observability import opaque_ref

from .models import (
    PlaybackState,
    QueuedTrack,
    VoiceChannelKey,
)
from .ports import MusicCatalog, MusicPlayback, MusicVoiceGateway
from .session import MusicSession

logger = logging.getLogger(__name__)


class TrackPlaybackRunner:
    """Resolve and start one track while honoring queue-control cancellation."""

    def __init__(self, catalog: MusicCatalog, voice: MusicVoiceGateway) -> None:
        self._catalog = catalog
        self._voice = voice

    async def prepare(
        self, channel: VoiceChannelKey, session: MusicSession, item: QueuedTrack
    ) -> MusicPlayback | None:
        logger.info(
            "Music track resolving: channel=%s source=%s",
            opaque_ref(channel.area_id, channel.channel_id),
            item.track.source.value,
        )
        async with session.lock:
            if session.skip_requested.is_set():
                return None
            resolve_task = asyncio.create_task(
                self._catalog.resolve(item.track),
                name=f"music-resolve:{opaque_ref(channel.area_id, channel.channel_id)}",
            )
            session.resolve_task = resolve_task
        try:
            playable = await resolve_task
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if (
                session.skip_requested.is_set()
                and current is not None
                and current.cancelling() == 0
            ):
                logger.info(
                    "Music track resolve cancelled by queue control: channel=%s",
                    opaque_ref(channel.area_id, channel.channel_id),
                )
                return None
            raise
        finally:
            async with session.lock:
                if session.resolve_task is resolve_task:
                    session.resolve_task = None
        if session.skip_requested.is_set():
            logger.info(
                "Music track skipped before playback: channel=%s",
                opaque_ref(channel.area_id, channel.channel_id),
            )
            return None
        playback = await self._voice.start_playback(channel, playable)
        try:
            if session.skip_requested.is_set():
                await playback.stop()
                return None
            async with session.lock:
                session.playback = playback
                session.state = PlaybackState.PLAYING
                session.revision += 1
        except BaseException:
            # The caller has not received this handle yet. Cancellation while
            # waiting for the session lock must still release the new playback.
            with suppress(Exception):
                await playback.stop()
            raise
        logger.info(
            "Music track playback started: channel=%s source=%s",
            opaque_ref(channel.area_id, channel.channel_id),
            item.track.source.value,
        )
        return playback
