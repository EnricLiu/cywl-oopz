from __future__ import annotations

import pytest

from cywl_oopz.features.music.models import VoiceChannelKey
from cywl_oopz.integrations.oopz.music import OopzMusicVoiceGateway


class FakeChannels:
    async def get_voice_channel_for_user(self, area: str, person: str) -> str | None:
        assert (area, person) == ("area", "person")
        return "voice"


class FakeVoice:
    def __init__(self) -> None:
        self.joins: list[dict[str, str]] = []
        self.urls: list[str] = []
        self.leaves = 0
        self.stops = 0

    async def join(self, **values: str) -> None:
        self.joins.append(values)

    async def play_url(self, url: str) -> None:
        self.urls.append(url)

    async def leave(self) -> None:
        self.leaves += 1

    async def stop(self) -> None:
        self.stops += 1

    async def get_state(self) -> str:
        return "playing"

    async def pause(self) -> bool:
        return True

    async def resume(self) -> bool:
        return True


class FakeBot:
    def __init__(self) -> None:
        self.channels = FakeChannels()
        self.voice = FakeVoice()


@pytest.mark.asyncio
async def test_oopz_music_gateway_reuses_channel_and_switches_cleanly() -> None:
    bot = FakeBot()
    gateway = OopzMusicVoiceGateway(bot)
    first = VoiceChannelKey("area", "voice")
    second = VoiceChannelKey("area", "other")

    assert await gateway.voice_channel_for_user("area", "person") == "voice"
    await gateway.play(first, "https://music.example/one.mp3")
    await gateway.play(first, "https://music.example/two.mp3")
    await gateway.play(second, "https://music.example/three.mp3")

    assert bot.voice.joins == [
        {"area": "area", "channel": "voice"},
        {"area": "area", "channel": "other"},
    ]
    assert bot.voice.leaves == 1

    await gateway.aclose()
    assert bot.voice.stops == 1
    assert bot.voice.leaves == 2
