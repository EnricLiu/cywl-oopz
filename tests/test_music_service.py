from __future__ import annotations

import asyncio
import random
from dataclasses import replace

import pytest

from cywl_oopz.features.agent.models import AgentIdentity
from cywl_oopz.features.chat.models import ConversationKey
from cywl_oopz.features.music.errors import (
    MusicBackendClosedError,
    MusicCatalogError,
    MusicQueryError,
    MusicQueueFullError,
    MusicVoiceBusyError,
    MusicVoiceChannelRequiredError,
)
from cywl_oopz.features.music.models import (
    MusicFailureCode,
    MusicFailureScope,
    MusicPlaybackEndReason,
    MusicPlaybackPolicy,
    MusicPlaybackResult,
    MusicTrack,
    PlayableTrack,
    PlaybackOrder,
    PlaybackState,
    RepeatPolicy,
    VoiceChannelKey,
)
from cywl_oopz.features.music.service import MusicRequestService
from cywl_oopz.settings import MusicSettings


def settings(**changes: str) -> MusicSettings:
    return MusicSettings.from_mapping(
        {
            "CYWL_MUSIC_ENABLED": "true",
            "CYWL_MUSIC_CATALOG_BASE_URL": "https://music.example",
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
        self.resolved: list[str] = []

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
        self.resolved.append(track.source_id)
        return PlayableTrack(track, f"https://music.example/{track.source_id}.mp3")

    async def aclose(self) -> None:
        self.closed = True


class FakeVoice:
    def __init__(self) -> None:
        self.channels = {"person": "voice-a", "other": "voice-b"}
        self.played: list[tuple[VoiceChannelKey, str]] = []
        self.current_finished = asyncio.Event()
        self.current_playback: FakePlayback | None = None
        self.play_started = asyncio.Event()
        self.paused = False
        self.closed = False
        self.stop_calls = 0
        self.left: list[VoiceChannelKey] = []
        self.acquired: VoiceChannelKey | None = None
        self.acquire_calls: list[VoiceChannelKey] = []
        self.available = True
        self.release_entered = asyncio.Event()
        self.release_allowed = asyncio.Event()
        self.release_allowed.set()
        self.release_calls = 0
        self.release_failures = 0

    async def voice_channel_for_user(self, area_id: str, person_id: str) -> str | None:
        assert area_id == "area"
        return self.channels.get(person_id)

    async def acquire(self, channel: VoiceChannelKey) -> bool:
        self.acquire_calls.append(channel)
        if self.acquired is not None:
            return self.acquired == channel
        if not self.available:
            return False
        self.acquired = channel
        return True

    async def start_playback(
        self,
        channel: VoiceChannelKey,
        stream_url: str,
    ) -> FakePlayback:
        assert self.acquired == channel
        self.current_finished = asyncio.Event()
        self.current_playback = FakePlayback(self, self.current_finished)
        self.played.append((channel, stream_url))
        self.play_started.set()
        return self.current_playback

    async def release(self, channel: VoiceChannelKey) -> bool:
        self.release_calls += 1
        if self.acquired != channel:
            return False
        self.release_entered.set()
        await self.release_allowed.wait()
        if self.release_failures:
            self.release_failures -= 1
            raise RuntimeError("fixture release failure")
        self.left.append(channel)
        self.acquired = None
        return True

    async def reset(self, channel: VoiceChannelKey) -> None:
        if self.acquired == channel:
            self.acquired = None

    async def aclose(self) -> None:
        self.closed = True


class FakePlayback:
    def __init__(self, voice: FakeVoice, finished: asyncio.Event) -> None:
        self._voice = voice
        self._finished = finished
        self._end_reason = MusicPlaybackEndReason.FINISHED

    async def wait_finished(self) -> MusicPlaybackResult:
        await self._finished.wait()
        return MusicPlaybackResult(self._end_reason, duration_seconds=1.0)

    async def stop(self) -> None:
        if not self._finished.is_set():
            self._voice.stop_calls += 1
            self._end_reason = MusicPlaybackEndReason.STOPPED
            self._finished.set()

    async def pause(self) -> bool:
        self._voice.paused = True
        return True

    async def resume(self) -> bool:
        self._voice.paused = False
        return True


class TerminalPlayback:
    def __init__(self, result: MusicPlaybackResult) -> None:
        self._result = result

    async def wait_finished(self) -> MusicPlaybackResult:
        return self._result

    async def stop(self) -> None:
        return None

    async def pause(self) -> bool:
        return False

    async def resume(self) -> bool:
        return False


class RecoveringVoice(FakeVoice):
    def __init__(self, failures: int = 1) -> None:
        super().__init__()
        self._failures_remaining = failures

    async def start_playback(
        self,
        channel: VoiceChannelKey,
        stream_url: str,
    ):
        if self._failures_remaining:
            self._failures_remaining -= 1
            self.played.append((channel, stream_url))
            self.play_started.set()
            return TerminalPlayback(MusicPlaybackResult(MusicPlaybackEndReason.BACKEND_CLOSED))
        return await super().start_playback(channel, stream_url)


class StartupRecoveringVoice(FakeVoice):
    def __init__(self, failures: int = 1) -> None:
        super().__init__()
        self._failures_remaining = failures

    async def start_playback(
        self,
        channel: VoiceChannelKey,
        stream_url: str,
    ):
        if self._failures_remaining:
            self._failures_remaining -= 1
            self.played.append((channel, stream_url))
            raise MusicBackendClosedError("fixture startup backend failure")
        return await super().start_playback(channel, stream_url)


class VoiceLeftRecoveringVoice(FakeVoice):
    def __init__(self) -> None:
        super().__init__()
        self._failed = False
        self.reset_calls = 0

    async def start_playback(
        self,
        channel: VoiceChannelKey,
        stream_url: str,
    ):
        if not self._failed:
            self._failed = True
            self.played.append((channel, stream_url))
            return TerminalPlayback(MusicPlaybackResult(MusicPlaybackEndReason.VOICE_LEFT))
        return await super().start_playback(channel, stream_url)

    async def reset(self, channel: VoiceChannelKey) -> None:
        self.reset_calls += 1
        await super().reset(channel)


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
    await eventually(lambda: service._session(VoiceChannelKey("area", "voice-a")).worker is None)
    assert (await service.queue(identity())).state.value == "idle"
    assert voice.left == [VoiceChannelKey("area", "voice-a")]

    await service.aclose()
    assert catalog.closed is True
    assert voice.closed is True


@pytest.mark.asyncio
async def test_music_service_retries_transient_voice_release_failure() -> None:
    voice = FakeVoice()
    voice.release_failures = 2
    service = MusicRequestService(settings(), FakeCatalog(), voice)

    await service.enqueue(identity(), "release-retry")
    await voice.play_started.wait()
    voice.current_finished.set()
    await eventually(lambda: voice.acquired is None, attempts=500)

    snapshot = await service.queue(identity())
    assert voice.release_calls == 3
    assert snapshot.state is PlaybackState.IDLE
    assert snapshot.last_failure is None
    await service.aclose()


@pytest.mark.asyncio
async def test_music_service_keeps_ownership_when_voice_release_retries_are_exhausted() -> None:
    voice = FakeVoice()
    voice.release_failures = 3
    service = MusicRequestService(settings(), FakeCatalog(), voice)
    channel = VoiceChannelKey("area", "voice-a")

    await service.enqueue(identity(), "release-broken")
    await voice.play_started.wait()
    voice.current_finished.set()
    await eventually(lambda: service._session(channel).worker is None, attempts=500)

    snapshot = await service.queue(identity())
    assert voice.release_calls == 3
    assert voice.acquired == channel
    assert service._session(channel).voice_reserved is True
    assert snapshot.state is PlaybackState.FAILED
    assert snapshot.last_failure is not None
    assert snapshot.last_failure.code is MusicFailureCode.RELEASE_FAILED
    assert snapshot.last_failure.scope is MusicFailureScope.VOICE_SESSION
    assert snapshot.last_failure.recoverable is True
    assert snapshot.last_failure.track_id is None

    await service.enqueue(identity(), "reuse-owned-voice")
    await eventually(lambda: len(voice.played) == 2)
    assert voice.acquire_calls == [channel]
    voice.current_finished.set()
    await eventually(lambda: voice.acquired is None)
    await service.aclose()


@pytest.mark.asyncio
async def test_music_service_reresolves_current_track_once_after_backend_failure() -> None:
    catalog = FakeCatalog()
    voice = RecoveringVoice()
    service = MusicRequestService(settings(), catalog, voice)

    await service.enqueue(identity(), "recover")
    await eventually(lambda: len(voice.played) == 2)

    assert catalog.resolved == ["recover", "recover"]
    assert voice.played[0] == voice.played[1]
    assert (await service.queue(identity())).state is PlaybackState.PLAYING
    voice.current_finished.set()
    await eventually(lambda: voice.acquired is None)
    await service.aclose()


@pytest.mark.asyncio
async def test_music_service_does_not_retry_backend_failure_more_than_once() -> None:
    catalog = FakeCatalog()
    voice = RecoveringVoice(failures=2)
    service = MusicRequestService(settings(), catalog, voice)

    await service.enqueue(identity(), "still-broken")
    await eventually(lambda: voice.acquired is None)

    assert catalog.resolved == ["still-broken", "still-broken"]
    assert len(voice.played) == 2
    snapshot = await service.queue(identity())
    assert snapshot.state is PlaybackState.FAILED
    assert [item.track.title for item in snapshot.upcoming] == ["still-broken"]
    assert snapshot.last_failure is not None
    assert snapshot.last_failure.code is MusicFailureCode.BACKEND_CLOSED
    assert snapshot.last_failure.scope is MusicFailureScope.VOICE_SESSION
    assert snapshot.last_failure.recoverable is False
    assert snapshot.last_failure.retry_count == 1
    await service.aclose()


@pytest.mark.asyncio
async def test_music_service_rejoins_after_physical_voice_generation_is_lost() -> None:
    catalog = FakeCatalog()
    voice = VoiceLeftRecoveringVoice()
    service = MusicRequestService(settings(), catalog, voice)

    await service.enqueue(identity(), "rejoin")
    await eventually(lambda: len(voice.played) == 2)

    assert voice.reset_calls == 1
    assert voice.acquire_calls == [
        VoiceChannelKey("area", "voice-a"),
        VoiceChannelKey("area", "voice-a"),
    ]
    assert catalog.resolved == ["rejoin", "rejoin"]
    assert (await service.queue(identity())).state is PlaybackState.PLAYING
    voice.current_finished.set()
    await eventually(lambda: voice.acquired is None)
    await service.aclose()


@pytest.mark.asyncio
async def test_music_service_reresolves_once_when_backend_fails_during_startup() -> None:
    catalog = FakeCatalog()
    voice = StartupRecoveringVoice()
    service = MusicRequestService(settings(), catalog, voice)

    await service.enqueue(identity(), "startup-recover")
    await eventually(lambda: len(voice.played) == 2)

    assert catalog.resolved == ["startup-recover", "startup-recover"]
    assert (await service.queue(identity())).state is PlaybackState.PLAYING
    voice.current_finished.set()
    await eventually(lambda: voice.acquired is None)
    await service.aclose()


@pytest.mark.asyncio
async def test_music_service_bounds_capacity_and_rejects_a_second_voice_channel() -> None:
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
    with pytest.raises(MusicVoiceBusyError):
        await service.enqueue(other, "other-song")
    assert len(voice.played) == 1

    voice.current_finished.set()
    await eventually(lambda: voice.acquired is None)
    result = await service.enqueue(other, "other-song")
    assert result.voice_channel.channel_id == "voice-b"
    await eventually(lambda: len(voice.played) == 2)
    voice.current_finished.set()

    await service.aclose()


@pytest.mark.asyncio
async def test_music_service_repeats_one_track_until_skipped() -> None:
    voice = FakeVoice()
    service = MusicRequestService(settings(), FakeCatalog(), voice)

    selected = await service.set_policy(identity(), repeat=RepeatPolicy.ONE)
    unchanged = await service.set_policy(identity(), repeat=RepeatPolicy.ONE)
    assert selected.changed is True
    assert unchanged.changed is False
    assert (await service.queue(identity())).policy == MusicPlaybackPolicy(repeat=RepeatPolicy.ONE)

    await service.enqueue(identity(), "loop")
    await eventually(lambda: len(voice.played) == 1)
    voice.current_finished.set()
    await eventually(lambda: len(voice.played) == 2)

    assert voice.played[0][1] == voice.played[1][1]
    assert await service.skip(identity()) is True
    await eventually(lambda: bool(voice.left))
    snapshot = await service.queue(identity())
    assert snapshot.state.value == "idle"
    assert snapshot.policy == MusicPlaybackPolicy()

    await service.aclose()


@pytest.mark.asyncio
async def test_music_service_repeats_the_whole_queue_in_order() -> None:
    voice = FakeVoice()
    service = MusicRequestService(settings(), FakeCatalog(), voice)
    await service.set_policy(identity(), repeat=RepeatPolicy.ALL)

    await service.enqueue(identity(), "first")
    await eventually(lambda: len(voice.played) == 1)
    await service.enqueue(identity(), "second")

    voice.current_finished.set()
    await eventually(lambda: len(voice.played) == 2)
    voice.current_finished.set()
    await eventually(lambda: len(voice.played) == 3)
    voice.current_finished.set()
    await eventually(lambda: len(voice.played) == 4)
    voice.current_finished.set()
    await eventually(lambda: len(voice.played) == 5)

    assert [url.rsplit("/", 1)[-1] for _, url in voice.played[:5]] == [
        "first.mp3",
        "second.mp3",
        "first.mp3",
        "second.mp3",
        "first.mp3",
    ]

    await service.aclose()


@pytest.mark.asyncio
async def test_music_service_combines_shuffle_and_repeat_all_without_replacement() -> None:
    voice = FakeVoice()
    service = MusicRequestService(
        settings(),
        FakeCatalog(),
        voice,
        rng=random.Random(7),
    )
    await service.set_policy(
        identity(),
        order=PlaybackOrder.SHUFFLE,
        repeat=RepeatPolicy.ALL,
    )
    await service.replace_queue(
        identity(),
        tuple(
            MusicTrack("netease", name, name, ("artist",)) for name in ("first", "second", "third")
        ),
    )

    for expected_count in range(1, 7):
        await eventually(lambda: len(voice.played) == expected_count)
        voice.current_finished.set()
    await eventually(lambda: len(voice.played) == 7)

    played = [url.rsplit("/", 1)[-1] for _, url in voice.played]
    expected_cycle = {"first.mp3", "second.mp3", "third.mp3"}
    assert set(played[:3]) == expected_cycle
    assert set(played[3:6]) == expected_cycle
    assert len(set(played[:3])) == 3
    assert len(set(played[3:6])) == 3
    snapshot = await service.queue(identity())
    assert snapshot.policy == MusicPlaybackPolicy(
        PlaybackOrder.SHUFFLE,
        RepeatPolicy.ALL,
    )

    await service.aclose()


@pytest.mark.asyncio
async def test_music_service_skip_in_repeat_all_returns_only_on_next_cycle() -> None:
    voice = FakeVoice()
    service = MusicRequestService(settings(), FakeCatalog(), voice)
    await service.set_policy(identity(), repeat=RepeatPolicy.ALL)

    await service.enqueue(identity(), "first")
    await eventually(lambda: len(voice.played) == 1)
    await service.enqueue(identity(), "second")
    assert await service.skip(identity()) is True
    await eventually(lambda: len(voice.played) == 2)
    assert voice.played[1][1].endswith("/second.mp3")

    voice.current_finished.set()
    await eventually(lambda: len(voice.played) == 3)
    assert voice.played[2][1].endswith("/first.mp3")

    await service.aclose()


@pytest.mark.asyncio
async def test_music_service_does_not_repeat_failed_track_in_repeat_all() -> None:
    class PartiallyBrokenCatalog(FakeCatalog):
        async def resolve(self, track: MusicTrack) -> PlayableTrack:
            if track.source_id == "broken":
                self.resolved.append(track.source_id)
                raise MusicCatalogError("fixture unavailable track")
            return await super().resolve(track)

    voice = FakeVoice()
    catalog = PartiallyBrokenCatalog()
    service = MusicRequestService(settings(), catalog, voice)
    await service.set_policy(identity(), repeat=RepeatPolicy.ALL)
    await service.replace_queue(
        identity(),
        (
            MusicTrack("netease", "broken", "Broken", ("artist",)),
            MusicTrack("netease", "working", "Working", ("artist",)),
        ),
    )

    await eventually(lambda: len(voice.played) == 1)
    voice.current_finished.set()
    await eventually(lambda: len(voice.played) == 2)

    assert catalog.resolved == ["broken", "working", "working"]
    assert all(url.endswith("/working.mp3") for _, url in voice.played)
    await service.aclose()


@pytest.mark.asyncio
async def test_music_service_updates_policy_dimensions_independently() -> None:
    service = MusicRequestService(settings(), FakeCatalog(), FakeVoice())

    repeat = await service.set_policy(identity(), repeat=RepeatPolicy.ALL)
    shuffled = await service.set_policy(identity(), order=PlaybackOrder.SHUFFLE)
    assert repeat.policy == MusicPlaybackPolicy(repeat=RepeatPolicy.ALL)
    assert shuffled.policy == MusicPlaybackPolicy(
        PlaybackOrder.SHUFFLE,
        RepeatPolicy.ALL,
    )

    with pytest.raises(MusicQueryError):
        await service.set_policy(identity(), repeat=RepeatPolicy.ONE)
    assert (await service.queue(identity())).policy == shuffled.policy

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
    await service.set_policy(identity(), order=PlaybackOrder.SHUFFLE)

    await service.enqueue(identity(), "first")
    await eventually(lambda: len(voice.played) == 1)
    await service.enqueue(identity(), "second")
    await service.enqueue(identity(), "third")
    voice.current_finished.set()
    await eventually(lambda: len(voice.played) == 2)

    assert voice.played[1][1].endswith("/third.mp3")

    await service.aclose()


@pytest.mark.asyncio
async def test_music_service_rebuilds_queue_and_interrupts_current_track() -> None:
    voice = FakeVoice()
    service = MusicRequestService(settings(), FakeCatalog(), voice)

    await service.enqueue(identity(), "old-current")
    await eventually(lambda: len(voice.played) == 1)
    await service.enqueue(identity(), "old-upcoming")

    result = await service.replace_queue(
        identity(),
        (
            MusicTrack("netease", "playlist-one", "Playlist One", ("artist",)),
            MusicTrack("netease", "playlist-two", "Playlist Two", ("artist",)),
        ),
    )

    assert result.loaded_count == 2
    assert result.replaced_current is True
    assert result.started_worker is False
    assert voice.stop_calls == 1
    await eventually(lambda: len(voice.played) == 2)
    assert voice.played[1][1].endswith("/playlist-one.mp3")
    snapshot = await service.queue(identity())
    assert snapshot.current is not None
    assert snapshot.current.track.title == "Playlist One"
    assert [item.track.title for item in snapshot.upcoming] == ["Playlist Two"]

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


@pytest.mark.asyncio
async def test_music_service_rejects_enqueue_when_another_feature_owns_voice() -> None:
    voice = FakeVoice()
    voice.available = False
    service = MusicRequestService(settings(), FakeCatalog(), voice)

    with pytest.raises(MusicVoiceBusyError):
        await service.enqueue(identity(), "song")

    snapshot = await service.queue(identity())
    assert snapshot.current is None
    assert snapshot.upcoming == ()
    assert snapshot.state.value == "idle"
    await service.aclose()


@pytest.mark.asyncio
async def test_enqueue_during_idle_release_acquires_a_fresh_lease_and_worker() -> None:
    voice = FakeVoice()
    service = MusicRequestService(settings(), FakeCatalog(), voice)
    channel = VoiceChannelKey("area", "voice-a")

    await service.enqueue(identity(), "first")
    await eventually(lambda: len(voice.played) == 1)
    voice.release_allowed.clear()
    voice.current_finished.set()
    await voice.release_entered.wait()

    enqueue_task = asyncio.create_task(service.enqueue(identity(), "second"))
    await asyncio.sleep(0)
    assert enqueue_task.done() is False

    voice.release_allowed.set()
    result = await enqueue_task
    assert result.started_worker is True
    await eventually(lambda: len(voice.played) == 2)
    assert voice.played[1][1].endswith("/second.mp3")
    assert voice.acquire_calls == [channel, channel]

    voice.current_finished.set()
    await eventually(lambda: voice.acquired is None)
    await service.aclose()
