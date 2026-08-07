from __future__ import annotations

import asyncio

import pytest

from cywl_oopz.features.audio.errors import AudioBusFailedError, AudioSessionClosedError
from cywl_oopz.features.audio.models import (
    AUDIO_BLOCK_FRAMES,
    AudioChannelKey,
    VoiceParticipantKind,
    VoiceParticipantRequest,
)
from cywl_oopz.integrations.audio.fake import FakeMasterPcmOutput
from cywl_oopz.integrations.oopz.voice_channel_session import (
    OopzVoiceChannelSessionManager,
    VoiceChannelSessionState,
)


def request(
    kind: VoiceParticipantKind,
    *,
    channel: str = "voice",
    owner: str = "owner",
) -> VoiceParticipantRequest:
    return VoiceParticipantRequest(kind, AudioChannelKey("area", channel), owner)


class FakeVoice:
    def __init__(self) -> None:
        self.joins: list[dict[str, str]] = []
        self.leaves = 0
        self.join_entered = asyncio.Event()
        self.join_allowed = asyncio.Event()
        self.join_allowed.set()
        self.join_error: Exception | None = None
        self.leave_entered = asyncio.Event()
        self.leave_allowed = asyncio.Event()
        self.leave_allowed.set()
        self.leave_error: Exception | None = None
        self.leave_calls = 0

    async def join(self, **values: str) -> None:
        self.join_entered.set()
        await self.join_allowed.wait()
        if self.join_error is not None:
            raise self.join_error
        self.joins.append(values)

    async def leave(self) -> None:
        self.leave_calls += 1
        self.leave_entered.set()
        await self.leave_allowed.wait()
        if self.leave_error is not None:
            raise self.leave_error
        self.leaves += 1


class FakeBot:
    def __init__(self) -> None:
        self.voice = FakeVoice()


class FakeMasterFactory:
    max_buffer_ms = 160

    def __init__(self) -> None:
        self.outputs: list[FakeMasterPcmOutput] = []

    async def open(self) -> FakeMasterPcmOutput:
        output = FakeMasterPcmOutput(max_buffer_frames=AUDIO_BLOCK_FRAMES * 10)
        self.outputs.append(output)
        return output


class RecoveringMasterFactory(FakeMasterFactory):
    def __init__(self) -> None:
        super().__init__()
        self.open_calls = 0

    async def open(self) -> FakeMasterPcmOutput:
        self.open_calls += 1
        if self.open_calls == 1:
            raise RuntimeError("fixture master open failure")
        return await super().open()


@pytest.mark.asyncio
async def test_same_channel_participants_share_join_until_final_release() -> None:
    bot = FakeBot()
    manager = OopzVoiceChannelSessionManager(bot)
    music = await manager.try_acquire(request(VoiceParticipantKind.MUSIC, owner="music"))
    conversation = await manager.try_acquire(
        request(VoiceParticipantKind.CONVERSATION, owner="conversation")
    )

    assert music is not None
    assert conversation is not None
    assert len(bot.voice.joins) == 1
    snapshot = await manager.current()
    assert snapshot is not None
    assert snapshot.state is VoiceChannelSessionState.ACTIVE
    assert {item.kind for item in snapshot.participants} == {
        VoiceParticipantKind.MUSIC,
        VoiceParticipantKind.CONVERSATION,
    }

    assert await music.release() is True
    assert bot.voice.leaves == 0
    assert await conversation.release() is True
    assert bot.voice.leaves == 1
    assert await manager.current() is None
    await manager.aclose()


@pytest.mark.asyncio
async def test_same_channel_participants_share_master_until_final_release() -> None:
    bot = FakeBot()
    masters = FakeMasterFactory()
    manager = OopzVoiceChannelSessionManager(bot, master_factory=masters)
    music = await manager.try_acquire(request(VoiceParticipantKind.MUSIC, owner="music"))
    conversation = await manager.try_acquire(
        request(VoiceParticipantKind.CONVERSATION, owner="conversation")
    )
    assert music is not None
    assert conversation is not None

    music_bus, voice_bus = await asyncio.gather(music.audio_bus(), conversation.audio_bus())

    assert music_bus is voice_bus
    assert len(masters.outputs) == 1
    assert await music.release() is True
    assert masters.outputs[0].closed is False
    assert bot.voice.leaves == 0
    assert await conversation.release() is True
    assert masters.outputs[0].closed is True
    assert bot.voice.leaves == 1
    await manager.aclose()


@pytest.mark.asyncio
async def test_master_open_failure_is_typed_and_same_participant_can_retry() -> None:
    bot = FakeBot()
    masters = RecoveringMasterFactory()
    manager = OopzVoiceChannelSessionManager(bot, master_factory=masters)
    participant = await manager.try_acquire(request(VoiceParticipantKind.MUSIC))
    assert participant is not None

    with pytest.raises(AudioBusFailedError, match="open shared audio master"):
        await participant.audio_bus()
    bus = await participant.audio_bus()

    assert bus.failed is False
    assert masters.open_calls == 2
    assert len(bot.voice.joins) == 1
    await manager.aclose()


@pytest.mark.asyncio
async def test_channel_and_kind_conflicts_do_not_change_active_session() -> None:
    bot = FakeBot()
    manager = OopzVoiceChannelSessionManager(bot)
    music_request = request(VoiceParticipantKind.MUSIC, owner="music")
    music = await manager.try_acquire(music_request)

    assert music is not None
    assert await manager.try_acquire(music_request) is music
    assert await manager.try_acquire(request(VoiceParticipantKind.MUSIC, owner="other")) is None
    assert (
        await manager.try_acquire(request(VoiceParticipantKind.CONVERSATION, channel="other"))
        is None
    )
    assert len(bot.voice.joins) == 1
    await manager.aclose()
    assert music.released is True


@pytest.mark.asyncio
async def test_rollout_gate_keeps_legacy_music_and_pcm_voice_exclusive() -> None:
    bot = FakeBot()
    manager = OopzVoiceChannelSessionManager(bot, allow_mixed_participants=False)
    music = await manager.try_acquire(request(VoiceParticipantKind.MUSIC, owner="music"))

    assert music is not None
    assert (
        await manager.try_acquire(request(VoiceParticipantKind.CONVERSATION, owner="conversation"))
        is None
    )
    assert len(bot.voice.joins) == 1
    await manager.aclose()


@pytest.mark.asyncio
async def test_same_channel_participant_waits_for_one_concurrent_join() -> None:
    bot = FakeBot()
    bot.voice.join_allowed.clear()
    manager = OopzVoiceChannelSessionManager(bot)
    music_task = asyncio.create_task(
        manager.try_acquire(request(VoiceParticipantKind.MUSIC, owner="music"))
    )
    await bot.voice.join_entered.wait()
    conversation_task = asyncio.create_task(
        manager.try_acquire(request(VoiceParticipantKind.CONVERSATION, owner="conversation"))
    )
    await asyncio.sleep(0)

    snapshot = await manager.current()
    assert snapshot is not None
    assert snapshot.state is VoiceChannelSessionState.JOINING
    bot.voice.join_allowed.set()
    music, conversation = await asyncio.gather(music_task, conversation_task)

    assert music is not None
    assert conversation is not None
    assert len(bot.voice.joins) == 1
    await manager.aclose()


@pytest.mark.asyncio
async def test_join_failure_wakes_waiter_and_manager_can_retry() -> None:
    bot = FakeBot()
    bot.voice.join_error = RuntimeError("join failed")
    manager = OopzVoiceChannelSessionManager(bot)

    with pytest.raises(RuntimeError, match="join failed"):
        await manager.try_acquire(request(VoiceParticipantKind.MUSIC))
    assert await manager.current() is None

    bot.voice.join_error = None
    participant = await manager.try_acquire(request(VoiceParticipantKind.CONVERSATION))
    assert participant is not None
    await manager.aclose()


@pytest.mark.asyncio
async def test_cancelled_final_release_preserves_token_for_retry() -> None:
    bot = FakeBot()
    manager = OopzVoiceChannelSessionManager(bot)
    participant = await manager.try_acquire(request(VoiceParticipantKind.CONVERSATION))
    assert participant is not None
    bot.voice.leave_allowed.clear()
    releasing = asyncio.create_task(participant.release())
    await bot.voice.leave_entered.wait()

    snapshot = await manager.current()
    assert snapshot is not None
    assert snapshot.state is VoiceChannelSessionState.LEAVING
    releasing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await releasing

    assert participant.released is False
    snapshot = await manager.current()
    assert snapshot is not None
    assert snapshot.state is VoiceChannelSessionState.ACTIVE
    bot.voice.leave_allowed.set()
    assert await participant.release() is True
    assert bot.voice.leaves == 1
    await manager.aclose()


@pytest.mark.asyncio
async def test_acquire_wait_for_leaving_is_bounded() -> None:
    bot = FakeBot()
    manager = OopzVoiceChannelSessionManager(bot, transition_wait_seconds=0.01)
    music = await manager.try_acquire(request(VoiceParticipantKind.MUSIC))
    assert music is not None
    bot.voice.leave_allowed.clear()
    releasing = asyncio.create_task(music.release())
    await bot.voice.leave_entered.wait()

    assert await manager.try_acquire(request(VoiceParticipantKind.CONVERSATION)) is None
    releasing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await releasing
    bot.voice.leave_allowed.set()
    await music.release()
    await manager.aclose()


@pytest.mark.asyncio
async def test_close_waits_for_pending_join_and_compensates_once() -> None:
    bot = FakeBot()
    bot.voice.join_allowed.clear()
    manager = OopzVoiceChannelSessionManager(bot)
    acquiring = asyncio.create_task(manager.try_acquire(request(VoiceParticipantKind.CONVERSATION)))
    await bot.voice.join_entered.wait()
    closing = asyncio.create_task(manager.aclose())
    await asyncio.sleep(0)

    bot.voice.join_allowed.set()
    with pytest.raises(AudioSessionClosedError, match="closed during join"):
        await acquiring
    await closing
    assert bot.voice.leaves == 1
    assert await manager.current() is None


@pytest.mark.asyncio
async def test_close_retries_cancelled_post_join_compensation() -> None:
    bot = FakeBot()
    bot.voice.join_allowed.clear()
    bot.voice.leave_allowed.clear()
    manager = OopzVoiceChannelSessionManager(bot)
    acquiring = asyncio.create_task(manager.try_acquire(request(VoiceParticipantKind.CONVERSATION)))
    await bot.voice.join_entered.wait()
    closing = asyncio.create_task(manager.aclose())
    await asyncio.sleep(0)
    bot.voice.join_allowed.set()
    await bot.voice.leave_entered.wait()

    acquiring.cancel()
    with pytest.raises(asyncio.CancelledError):
        await acquiring
    for _ in range(100):
        if bot.voice.leave_calls >= 2:
            break
        await asyncio.sleep(0)
    assert bot.voice.leave_calls == 2

    bot.voice.leave_allowed.set()
    await closing
    assert bot.voice.leaves == 1
    assert await manager.current() is None


@pytest.mark.asyncio
async def test_close_retries_cancelled_and_failed_leave() -> None:
    bot = FakeBot()
    manager = OopzVoiceChannelSessionManager(bot)
    participant = await manager.try_acquire(request(VoiceParticipantKind.MUSIC))
    assert participant is not None
    bot.voice.leave_allowed.clear()
    closing = asyncio.create_task(manager.aclose())
    await bot.voice.leave_entered.wait()
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing

    assert participant.released is False
    bot.voice.leave_allowed.set()
    bot.voice.leave_error = RuntimeError("fixture leave failure")
    await manager.aclose()
    assert participant.released is False

    bot.voice.leave_error = None
    await manager.aclose()
    assert participant.released is True
    assert bot.voice.leaves == 1
    with pytest.raises(AudioSessionClosedError, match="closed"):
        await manager.try_acquire(request(VoiceParticipantKind.CONVERSATION))


@pytest.mark.asyncio
async def test_stale_participant_cannot_release_new_generation() -> None:
    bot = FakeBot()
    manager = OopzVoiceChannelSessionManager(bot)
    first = await manager.try_acquire(request(VoiceParticipantKind.MUSIC, owner="first"))
    assert first is not None
    assert await first.release() is True
    second = await manager.try_acquire(request(VoiceParticipantKind.CONVERSATION, owner="second"))
    assert second is not None

    assert await first.release() is False
    snapshot = await manager.current()
    assert snapshot is not None
    assert snapshot.generation == 2
    await manager.aclose()
