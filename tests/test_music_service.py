from __future__ import annotations

import asyncio
import random
from dataclasses import replace

import pytest

from cywl_oopz.features.agent.models import AgentIdentity
from cywl_oopz.features.chat.models import ConversationKey
from cywl_oopz.features.music.errors import (
    MusicQueueFullError,
    MusicVoiceChannelRequiredError,
)
from cywl_oopz.features.music.models import (
    MusicTrack,
    PlayableTrack,
    PlaybackMode,
    VoiceChannelKey,
)
from cywl_oopz.features.music.service import MusicRequestService
from cywl_oopz.settings import MusicSettings


def settings(**changes: str) -> MusicSettings:
    return MusicSettings.from_mapping(
        {
            "CYWL_MUSIC_ENABLED": "true",
            "CYWL_MUSIC_CATALOG_BASE_URL": "https://music.example",
            "CYWL_MUSIC_PLAYBACK_POLL_SECONDS": "0.001",
            **changes,
        }
    )


def identity(person_id: str = "person") -> AgentIdentity:
    return AgentIdentity(
        person_id,
        ConversationKey("channel", "area", "text", person_id),
    )


class FakeCatalog:
    def __init__(self) -> None:
        self.closed = False

    async def search(self, query: str, *, limit: int) -> tuple[MusicTrack, ...]:
        return (
            MusicTrack(
                "netease",
                query,
                query,
                ("artist",),
                1000,
            ),
        )[:limit]

    async def resolve(self, track: MusicTrack) -> PlayableTrack:
        return PlayableTrack(track, f"https://music.example/{track.source_id}.mp3")

    async def aclose(self) -> None:
        self.closed = True


class FakeVoice:
    def __init__(self) -> None:
        self.channels = {"person": "voice-a", "other": "voice-b"}
        self.played: list[tuple[VoiceChannelKey, str]] = []
        self.current_finished = asyncio.Event()
        self.play_started = asyncio.Event()
        self.paused = False
        self.closed = False
        self.stop_calls = 0
        self.left: list[VoiceChannelKey] = []

    async def voice_channel_for_user(self, area_id: str, person_id: str) -> str | None:
        assert area_id == "area"
        return self.channels.get(person_id)

    async def play(self, channel: VoiceChannelKey, stream_url: str) -> None:
        self.current_finished = asyncio.Event()
        self.played.append((channel, stream_url))
        self.play_started.set()

    async def state(self) -> str:
        if self.current_finished.is_set():
            return "finished"
        return "paused" if self.paused else "playing"

    async def stop(self) -> None:
        self.stop_calls += 1
        self.current_finished.set()

    async def pause(self) -> bool:
        self.paused = True
        return True

    async def resume(self) -> bool:
        self.paused = False
        return True

    async def leave(self, channel: VoiceChannelKey) -> bool:
        self.left.append(channel)
        return True

    async def aclose(self) -> None:
        self.closed = True


async def eventually(predicate, *, attempts: int = 100) -> None:
    for _ in range(attempts):
        if predicate():
            return
        await asyncio.sleep(0.001)
    raise AssertionError("condition did not become true")


@pytest.mark.asyncio
async def test_music_service_serializes_queue_controls_and_cleans_up() -> None:
    catalog = FakeCatalog()
    voice = FakeVoice()
    service = MusicRequestService(settings(), catalog, voice)

    first = await service.enqueue(identity(), "first", idempotency_key="run:first")
    await voice.play_started.wait()
    replay = await service.enqueue(identity(), " FIRST ", idempotency_key="run:first")
    second = await service.enqueue(identity(), "second")
    snapshot = await service.queue(identity())

    assert first.position == 1
    assert replay.item.id == first.item.id
    assert second.position == 2
    assert snapshot.current is not None
    assert snapshot.current.track.title == "first"
    assert [item.track.title for item in snapshot.upcoming] == ["second"]

    assert await service.pause(identity()) is True
    assert (await service.queue(identity())).state.value == "paused"
    assert await service.resume(identity()) is True
    assert await service.skip(identity()) is True

    await eventually(lambda: len(voice.played) == 2)
    assert voice.played[1][1].endswith("/second.mp3")
    voice.current_finished.set()
    await eventually(lambda: not service._tasks.has_active(VoiceChannelKey("area", "voice-a")))
    assert (await service.queue(identity())).state.value == "idle"
    assert voice.left == [VoiceChannelKey("area", "voice-a")]

    await service.aclose()
    assert catalog.closed is True
    assert voice.closed is True


@pytest.mark.asyncio
async def test_music_service_keeps_channel_queues_isolated_and_bounds_capacity() -> None:
    voice = FakeVoice()
    service = MusicRequestService(
        settings(CYWL_MUSIC_MAX_QUEUE_LENGTH="1"),
        FakeCatalog(),
        voice,
    )

    await service.enqueue(identity(), "one")
    await voice.play_started.wait()
    with pytest.raises(MusicQueueFullError):
        await service.enqueue(identity(), "overflow")

    other = replace(identity(), person_id="other", conversation=identity("other").conversation)
    result = await service.enqueue(other, "other-song")
    assert result.voice_channel.channel_id == "voice-b"
    assert len(voice.played) == 1

    voice.current_finished.set()
    await eventually(lambda: len(voice.played) == 2)
    assert voice.played[1][0].channel_id == "voice-b"

    await service.aclose()


@pytest.mark.asyncio
async def test_music_service_repeats_one_track_until_skipped() -> None:
    voice = FakeVoice()
    service = MusicRequestService(settings(), FakeCatalog(), voice)

    selected = await service.set_mode(identity(), PlaybackMode.REPEAT_ONE)
    unchanged = await service.set_mode(identity(), PlaybackMode.REPEAT_ONE)
    assert selected.changed is True
    assert unchanged.changed is False
    assert (await service.queue(identity())).mode is PlaybackMode.REPEAT_ONE

    await service.enqueue(identity(), "loop")
    await eventually(lambda: len(voice.played) == 1)
    voice.current_finished.set()
    await eventually(lambda: len(voice.played) == 2)

    assert voice.played[0][1] == voice.played[1][1]
    assert await service.skip(identity()) is True
    await eventually(lambda: bool(voice.left))
    assert (await service.queue(identity())).state.value == "idle"

    await service.aclose()


@pytest.mark.asyncio
async def test_music_service_repeats_the_whole_queue_in_order() -> None:
    voice = FakeVoice()
    service = MusicRequestService(settings(), FakeCatalog(), voice)
    await service.set_mode(identity(), PlaybackMode.REPEAT_ALL)

    await service.enqueue(identity(), "first")
    await eventually(lambda: len(voice.played) == 1)
    await service.enqueue(identity(), "second")

    voice.current_finished.set()
    await eventually(lambda: len(voice.played) == 2)
    voice.current_finished.set()
    await eventually(lambda: len(voice.played) == 3)

    assert [url.rsplit("/", 1)[-1] for _, url in voice.played[:3]] == [
        "first.mp3",
        "second.mp3",
        "first.mp3",
    ]

    await service.aclose()


@pytest.mark.asyncio
async def test_music_service_selects_random_upcoming_tracks_in_shuffle_mode() -> None:
    voice = FakeVoice()
    service = MusicRequestService(
        settings(),
        FakeCatalog(),
        voice,
        rng=random.Random(0),
    )
    await service.set_mode(identity(), PlaybackMode.SHUFFLE)

    await service.enqueue(identity(), "first")
    await eventually(lambda: len(voice.played) == 1)
    await service.enqueue(identity(), "second")
    await service.enqueue(identity(), "third")
    voice.current_finished.set()
    await eventually(lambda: len(voice.played) == 2)

    assert voice.played[1][1].endswith("/third.mp3")

    await service.aclose()


@pytest.mark.asyncio
async def test_music_service_requires_the_real_callers_voice_channel() -> None:
    voice = FakeVoice()
    voice.channels.clear()
    service = MusicRequestService(settings(), FakeCatalog(), voice)

    with pytest.raises(MusicVoiceChannelRequiredError):
        await service.enqueue(identity(), "song")
    with pytest.raises(MusicVoiceChannelRequiredError):
        await service.queue(
            AgentIdentity(
                "person",
                ConversationKey("private", "", "", "person"),
            )
        )

    await service.aclose()
