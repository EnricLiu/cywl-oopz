"""OOPZ SDK adapter for project-owned music playback use cases."""

from __future__ import annotations

import asyncio
import logging

from oopz_sdk import OopzBot

from cywl_oopz.core.observability import exception_kind, opaque_ref
from cywl_oopz.features.music.models import VoiceChannelKey

DEFAULT_VOLUME = 2  # %

logger = logging.getLogger(__name__)


class OopzMusicVoiceGateway:
    """Translate music operations into the single voice backend owned by OopzBot."""

    def __init__(self, bot: OopzBot) -> None:
        self._bot = bot
        self._current_channel: VoiceChannelKey | None = None
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

    async def play(self, channel: VoiceChannelKey, stream_url: str) -> None:
        async with self._lock:
            if self._current_channel != channel:
                if self._current_channel is not None:
                    logger.info("Leaving OOPZ voice channel before switching playback")
                    await self._bot.voice.leave()
                    self._current_channel = None
                logger.info(
                    "Joining OOPZ voice channel for playback: channel=%s",
                    self._channel_ref(channel),
                )
                await self._bot.voice.join(
                    area=channel.area_id,
                    channel=channel.channel_id,
                )
                self._current_channel = channel

            logger.debug(
                "Starting OOPZ voice playback: channel=%s",
                self._channel_ref(channel),
            )
            await self._bot.voice.set_volume(DEFAULT_VOLUME)
            await self._bot.voice.play_url(stream_url)

    async def state(self) -> str:
        return await self._bot.voice.get_state()

    async def stop(self) -> None:
        logger.info("Stopping OOPZ voice playback")
        await self._bot.voice.stop()

    async def pause(self) -> bool:
        paused = await self._bot.voice.pause()
        logger.info("Paused OOPZ voice playback: applied=%s", paused)
        return paused

    async def resume(self) -> bool:
        resumed = await self._bot.voice.resume()
        logger.info("Resumed OOPZ voice playback: applied=%s", resumed)
        return resumed

    async def aclose(self) -> None:
        async with self._lock:
            try:
                await self._bot.voice.stop()
            except Exception as exc:
                logger.warning(
                    "Could not stop OOPZ voice during close: error=%s", exception_kind(exc)
                )
            if self._current_channel is not None:
                try:
                    await self._bot.voice.leave()
                except Exception as exc:
                    logger.warning(
                        "Could not leave OOPZ voice during close: error=%s",
                        exception_kind(exc),
                    )
            self._current_channel = None
        logger.info("Closed OOPZ voice gateway")

    @staticmethod
    def _channel_ref(channel: VoiceChannelKey) -> str:
        return opaque_ref(channel.area_id, channel.channel_id)
