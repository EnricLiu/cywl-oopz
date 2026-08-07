from __future__ import annotations

import asyncio

import numpy as np
import pytest
from oopz_sdk import (
    VoicePlaybackEndReason,
    VoicePlaybackResult,
    VoicePlaybackState,
)

from cywl_oopz.features.audio.models import (
    AUDIO_BLOCK_FRAMES,
    AUDIO_CHANNELS,
    AudioChannelKey,
    DecodedAudioBlock,
    VoiceParticipantKind,
    VoiceParticipantRequest,
)
from cywl_oopz.features.music.models import MusicPlaybackEndReason, VoiceChannelKey
from cywl_oopz.integrations.audio.fake import FakeMasterPcmOutput
from cywl_oopz.integrations.oopz.music import OopzMusicVoiceGateway
from cywl_oopz.integrations.oopz.voice_channel_session import OopzVoiceChannelSessionManager
from cywl_oopz.settings import AudioMixerSettings


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
        self.leave_entered = asyncio.Event()
        self.leave_allowed = asyncio.Event()
        self.leave_allowed.set()
        self.leave_error: Exception | None = None

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
        self.leave_entered.set()
        await self.leave_allowed.wait()
        if self.leave_error is not None:
            raise self.leave_error
        self.leaves += 1


class FakeBot:
    def __init__(self) -> None:
        self.channels = FakeChannels()
        self.voice = FakeVoice()


class FakeDecoder:
    def __init__(self) -> None:
        self._delivered = False
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self) -> DecodedAudioBlock:
        if self._delivered:
            raise StopAsyncIteration
        self._delivered = True
        return DecodedAudioBlock(
            AUDIO_BLOCK_FRAMES,
            np.full(
                (AUDIO_BLOCK_FRAMES, AUDIO_CHANNELS),
                0.25,
                dtype=np.float32,
            ),
        )

    async def aclose(self) -> None:
        self.closed = True


class FakeDecoderFactory:
    def __init__(self) -> None:
        self.validations = 0
        self.urls: list[str] = []
        self.decoders: list[FakeDecoder] = []

    async def validate(self) -> None:
        self.validations += 1

    async def open(self, stream_url: str) -> FakeDecoder:
        self.urls.append(stream_url)
        decoder = FakeDecoder()
        self.decoders.append(decoder)
        return decoder


class FakeMasterFactory:
    max_buffer_ms = 160

    def __init__(self) -> None:
        self.outputs: list[FakeMasterPcmOutput] = []

    async def open(self) -> FakeMasterPcmOutput:
        output = FakeMasterPcmOutput(max_buffer_frames=AUDIO_BLOCK_FRAMES * 10)
        self.outputs.append(output)
        return output


def pcm_settings() -> AudioMixerSettings:
    return AudioMixerSettings.from_mapping({"CYWL_AUDIO_MIXER_ENABLED": "true"})


@pytest.mark.asyncio
async def test_oopz_music_gateway_reuses_one_lease_and_typed_playback() -> None:
    bot = FakeBot()
    leases = OopzVoiceChannelSessionManager(bot)
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
async def test_oopz_music_gateway_reuses_one_pcm_master_across_tracks() -> None:
    bot = FakeBot()
    sessions = OopzVoiceChannelSessionManager(bot)
    decoders = FakeDecoderFactory()
    masters = FakeMasterFactory()
    gateway = OopzMusicVoiceGateway(
        bot,
        sessions,
        pcm_settings(),
        decoder_factory=decoders,
        master_factory=masters,
    )
    channel = VoiceChannelKey("area", "voice")

    await gateway.validate_capabilities()
    assert await gateway.acquire(channel) is True
    first = await gateway.start_playback(channel, "https://music.example/one.mp3")
    first_result = await first.wait_finished()
    second = await gateway.start_playback(channel, "https://music.example/two.mp3")
    second_result = await second.wait_finished()

    assert first_result.end_reason is MusicPlaybackEndReason.FINISHED
    assert second_result.end_reason is MusicPlaybackEndReason.FINISHED
    assert decoders.validations == 1
    assert decoders.urls == [
        "https://music.example/one.mp3",
        "https://music.example/two.mp3",
    ]
    assert all(decoder.closed for decoder in decoders.decoders)
    assert len(masters.outputs) == 1
    assert len(masters.outputs[0].writes) == 2
    assert bot.voice.urls == []
    assert bot.voice.volumes == []

    assert await gateway.release(channel) is True
    assert masters.outputs[0].closed is True
    assert bot.voice.leaves == 1
    await gateway.aclose()
    await sessions.aclose()


@pytest.mark.asyncio
async def test_oopz_music_gateway_does_not_preempt_another_lease() -> None:
    bot = FakeBot()
    leases = OopzVoiceChannelSessionManager(bot, allow_mixed_participants=False)
    conversation = await leases.try_acquire(
        VoiceParticipantRequest(
            VoiceParticipantKind.CONVERSATION,
            AudioChannelKey("area", "voice"),
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


@pytest.mark.asyncio
async def test_oopz_music_gateway_keeps_lease_when_release_is_cancelled() -> None:
    bot = FakeBot()
    leases = OopzVoiceChannelSessionManager(bot)
    gateway = OopzMusicVoiceGateway(bot, leases)
    channel = VoiceChannelKey("area", "voice")
    assert await gateway.acquire(channel) is True
    bot.voice.leave_allowed.clear()
    releasing = asyncio.create_task(gateway.release(channel))
    await bot.voice.leave_entered.wait()

    releasing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await releasing

    assert gateway._lease is not None
    assert gateway._channel == channel
    assert await leases.current() is not None
    bot.voice.leave_allowed.set()
    assert await gateway.release(channel) is True
    assert gateway._lease is None
    assert gateway._channel is None
    assert await leases.current() is None
    assert bot.voice.leaves == 1
    await gateway.aclose()
    await leases.aclose()


@pytest.mark.asyncio
async def test_oopz_music_gateway_close_retries_failed_lease_release() -> None:
    bot = FakeBot()
    leases = OopzVoiceChannelSessionManager(bot)
    gateway = OopzMusicVoiceGateway(bot, leases)
    channel = VoiceChannelKey("area", "voice")
    assert await gateway.acquire(channel) is True
    await gateway.start_playback(channel, "https://music.example/retry.mp3")
    bot.voice.leave_error = RuntimeError("fixture leave failure")

    await gateway.aclose()

    assert gateway._lease is not None
    assert gateway._channel == channel
    assert gateway._playback is None
    assert bot.voice.playbacks[0].stop_calls == 1
    assert await leases.current() is not None
    bot.voice.leave_error = None
    await gateway.aclose()
    assert gateway._lease is None
    assert gateway._channel is None
    assert await leases.current() is None
    assert bot.voice.playbacks[0].stop_calls == 1
    assert bot.voice.leaves == 1
    await leases.aclose()
