from __future__ import annotations

import asyncio

import pytest
from oopz_sdk import (
    VoicePlaybackEndReason,
    VoicePlaybackResult,
    VoicePlaybackState,
)

from cywl_oopz.features.music.models import MusicPlaybackEndReason, VoiceChannelKey
from cywl_oopz.integrations.oopz.music import OopzMusicVoiceGateway
from cywl_oopz.integrations.oopz.voice_lease import (
    OopzVoiceLeaseManager,
    VoiceLeasePurpose,
    VoiceLeaseRequest,
)


class FakeChannels:
    async def get_voice_channel_for_user(self, area: str, person: str) -> str | None:
        assert (area, person) == ("area", "person")
        return "voice"


class FakeSdkPlayback:
    def __init__(self, playback_id: int) -> None:
        self.playback_id = playback_id
        self.finished = False
        self.end_reason = VoicePlaybackEndReason.FINISHED
        self.finished_event = asyncio.Event()
        self.pause_calls = 0
        self.resume_calls = 0
        self.stop_calls = 0

    async def wait_finished(self) -> VoicePlaybackResult:
        await self.finished_event.wait()
        state = (
            VoicePlaybackState.FINISHED
            if self.end_reason is VoicePlaybackEndReason.FINISHED
            else VoicePlaybackState.STOPPED
        )
        return VoicePlaybackResult(
            playback_id=self.playback_id,
            state=state,
            end_reason=self.end_reason,
            duration_seconds=3.5,
        )

    async def stop(self) -> None:
        if self.finished:
            return
        self.stop_calls += 1
        self.finish(VoicePlaybackEndReason.STOPPED)

    async def pause(self) -> None:
        self.pause_calls += 1

    async def resume(self) -> None:
        self.resume_calls += 1

    def finish(self, reason: VoicePlaybackEndReason = VoicePlaybackEndReason.FINISHED) -> None:
        self.end_reason = reason
        self.finished = True
        self.finished_event.set()


class FakeVoice:
    def __init__(self) -> None:
        self.joins: list[dict[str, str]] = []
        self.urls: list[str] = []
        self.playbacks: list[FakeSdkPlayback] = []
        self.leaves = 0
        self.volumes: list[int] = []

    async def join(self, **values: str) -> None:
        self.joins.append(values)

    async def start_url_playback(self, url: str) -> FakeSdkPlayback:
        self.urls.append(url)
        playback = FakeSdkPlayback(len(self.playbacks) + 1)
        self.playbacks.append(playback)
        return playback

    async def set_volume(self, volume: int) -> bool:
        self.volumes.append(volume)
        return True

    async def leave(self) -> None:
        self.leaves += 1


class FakeBot:
    def __init__(self) -> None:
        self.channels = FakeChannels()
        self.voice = FakeVoice()


@pytest.mark.asyncio
async def test_oopz_music_gateway_reuses_one_lease_and_typed_playback() -> None:
    bot = FakeBot()
    leases = OopzVoiceLeaseManager(bot)
    gateway = OopzMusicVoiceGateway(bot, leases)
    channel = VoiceChannelKey("area", "voice")

    assert await gateway.voice_channel_for_user("area", "person") == "voice"
    assert await gateway.acquire(channel) is True
    assert await gateway.acquire(channel) is True
    first = await gateway.start_playback(channel, "https://music.example/one.mp3")
    assert await first.pause() is True
    assert await first.resume() is True
    bot.voice.playbacks[0].finish()
    result = await first.wait_finished()

    assert result.end_reason is MusicPlaybackEndReason.FINISHED
    assert result.duration_seconds == 3.5
    second = await gateway.start_playback(channel, "https://music.example/two.mp3")
    await second.stop()

    assert bot.voice.joins == [{"area": "area", "channel": "voice"}]
    assert bot.voice.urls == [
        "https://music.example/one.mp3",
        "https://music.example/two.mp3",
    ]
    assert bot.voice.volumes == [2, 2]
    assert await gateway.release(VoiceChannelKey("area", "other")) is False
    assert await gateway.release(channel) is True
    assert bot.voice.leaves == 1
    assert await leases.current() is None

    await gateway.aclose()
    await leases.aclose()
    assert bot.voice.leaves == 1


@pytest.mark.asyncio
async def test_oopz_music_gateway_does_not_preempt_another_lease() -> None:
    bot = FakeBot()
    leases = OopzVoiceLeaseManager(bot)
    conversation = await leases.try_acquire(
        VoiceLeaseRequest(
            VoiceLeasePurpose.CONVERSATION,
            "area",
            "voice",
            "conversation:owner",
        )
    )
    assert conversation is not None
    gateway = OopzMusicVoiceGateway(bot, leases)

    assert await gateway.acquire(VoiceChannelKey("area", "voice")) is False
    assert len(bot.voice.joins) == 1

    await conversation.release()
    await gateway.aclose()
    await leases.aclose()
