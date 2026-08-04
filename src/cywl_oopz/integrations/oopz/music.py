"""OOPZ SDK adapter for project-owned music playback use cases."""

from __future__ import annotations

import asyncio
import logging

from oopz_sdk import OopzBot, VoicePlayback

from cywl_oopz.core.observability import exception_kind, opaque_ref
from cywl_oopz.features.music.models import (
    MusicPlaybackEndReason,
    MusicPlaybackResult,
    VoiceChannelKey,
)

from .voice_lease import (
    OopzVoiceLease,
    OopzVoiceLeaseManager,
    VoiceLeasePurpose,
    VoiceLeaseRequest,
)

DEFAULT_VOLUME = 2  # %

logger = logging.getLogger(__name__)


class OopzMusicPlayback:
    """Map one SDK playback handle to the project music port."""

    def __init__(self, playback: VoicePlayback) -> None:
        self._playback = playback

    @property
    def finished(self) -> bool:
        return self._playback.finished

    async def wait_finished(self) -> MusicPlaybackResult:
        result = await self._playback.wait_finished()
        return MusicPlaybackResult(
            end_reason=MusicPlaybackEndReason(result.end_reason.value),
            duration_seconds=result.duration_seconds,
            terminal_error=result.terminal_error,
        )

    async def stop(self) -> None:
        await self._playback.stop()

    async def pause(self) -> bool:
        await self._playback.pause()
        return True

    async def resume(self) -> bool:
        await self._playback.resume()
        return True


class OopzMusicVoiceGateway:
    """Translate music operations under one shared OOPZ voice lease."""

    def __init__(self, bot: OopzBot, leases: OopzVoiceLeaseManager) -> None:
        self._bot = bot
        self._leases = leases
        self._lease: OopzVoiceLease | None = None
        self._channel: VoiceChannelKey | None = None
        self._playback: OopzMusicPlayback | None = None
        self._lock = asyncio.Lock()

    async def voice_channel_for_user(self, area_id: str, person_id: str) -> str | None:
        channel = await self._bot.channels.get_voice_channel_for_user(area_id, person_id)
        logger.debug(
            "Resolved user voice channel: area=%s user=%s found=%s",
            opaque_ref(area_id),
            opaque_ref(person_id),
            channel is not None,
        )
        return channel

    async def acquire(self, channel: VoiceChannelKey) -> bool:
        async with self._lock:
            if self._lease is not None:
                return self._channel == channel and not self._lease.released
            lease = await self._leases.try_acquire(
                VoiceLeaseRequest(
                    VoiceLeasePurpose.MUSIC,
                    channel.area_id,
                    channel.channel_id,
                    owner_key=f"music:{channel.area_id}:{channel.channel_id}",
                )
            )
            if lease is None:
                return False
            self._lease = lease
            self._channel = channel
            return True

    async def start_playback(
        self,
        channel: VoiceChannelKey,
        stream_url: str,
    ) -> OopzMusicPlayback:
        async with self._lock:
            if self._lease is None or self._lease.released or self._channel != channel:
                raise RuntimeError("Music playback requires a matching active voice lease")
            if self._playback is not None and not self._playback.finished:
                raise RuntimeError("Music playback already has an active owner handle")
            logger.debug(
                "Starting typed OOPZ music playback: channel=%s",
                self._channel_ref(channel),
            )
            await self._bot.voice.set_volume(DEFAULT_VOLUME)
            playback = OopzMusicPlayback(await self._bot.voice.start_url_playback(stream_url))
            self._playback = playback
            return playback

    async def release(self, channel: VoiceChannelKey) -> bool:
        async with self._lock:
            if self._lease is None or self._channel != channel:
                return False
            playback = self._playback
            lease = self._lease
            self._playback = None
            self._lease = None
            self._channel = None
            if playback is not None and not playback.finished:
                try:
                    await playback.stop()
                except Exception as exc:
                    logger.warning(
                        "Could not stop music before releasing voice lease: error=%s",
                        exception_kind(exc),
                    )
            return await lease.release()

    async def aclose(self) -> None:
        async with self._lock:
            playback = self._playback
            lease = self._lease
            self._playback = None
            self._lease = None
            self._channel = None
            if playback is not None and not playback.finished:
                try:
                    await playback.stop()
                except Exception as exc:
                    logger.warning(
                        "Could not stop music during close: error=%s",
                        exception_kind(exc),
                    )
            if lease is not None:
                try:
                    await lease.release()
                except Exception as exc:
                    logger.warning(
                        "Could not release music voice lease during close: error=%s",
                        exception_kind(exc),
                    )
        logger.info("Closed OOPZ music gateway")

    @staticmethod
    def _channel_ref(channel: VoiceChannelKey) -> str:
        return opaque_ref(channel.area_id, channel.channel_id)
