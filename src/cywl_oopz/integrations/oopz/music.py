"""OOPZ SDK adapter for project-owned music playback use cases."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any

from cywl_oopz.features.music.models import VoiceChannelKey


class OopzMusicVoiceGateway:
    """Translate music operations into the single voice backend owned by OopzBot."""

    def __init__(self, bot: Any) -> None:
        self._bot = bot
        self._current_channel: VoiceChannelKey | None = None
        self._lock = asyncio.Lock()

    async def voice_channel_for_user(self, area_id: str, person_id: str) -> str | None:
        return await self._bot.channels.get_voice_channel_for_user(area_id, person_id)

    async def play(self, channel: VoiceChannelKey, stream_url: str) -> None:
        async with self._lock:
            if self._current_channel != channel:
                if self._current_channel is not None:
                    await self._bot.voice.leave()
                    self._current_channel = None
                await self._bot.voice.join(
                    area=channel.area_id,
                    channel=channel.channel_id,
                )
                self._current_channel = channel
            await self._bot.voice.play_url(stream_url)

    async def state(self) -> str:
        return await self._bot.voice.get_state()

    async def stop(self) -> None:
        await self._bot.voice.stop()

    async def pause(self) -> bool:
        return await self._bot.voice.pause()

    async def resume(self) -> bool:
        return await self._bot.voice.resume()

    async def aclose(self) -> None:
        async with self._lock:
            with suppress(Exception):
                await self._bot.voice.stop()
            if self._current_channel is not None:
                with suppress(Exception):
                    await self._bot.voice.leave()
            self._current_channel = None
