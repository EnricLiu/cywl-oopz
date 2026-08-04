from __future__ import annotations

import asyncio

import pytest

from cywl_oopz.integrations.oopz.voice_lease import (
    OopzVoiceLeaseManager,
    VoiceLeasePurpose,
    VoiceLeaseRequest,
    VoiceLeaseState,
)


def request(
    purpose: VoiceLeasePurpose,
    *,
    channel: str = "voice",
    owner: str = "owner",
) -> VoiceLeaseRequest:
    return VoiceLeaseRequest(purpose, "area", channel, owner)


class FakeVoice:
    def __init__(self) -> None:
        self.joins: list[dict[str, str]] = []
        self.leaves = 0
        self.join_entered = asyncio.Event()
        self.join_allowed = asyncio.Event()
        self.join_allowed.set()
        self.join_error: Exception | None = None

    async def join(self, **values: str) -> None:
        self.join_entered.set()
        await self.join_allowed.wait()
        if self.join_error is not None:
            raise self.join_error
        self.joins.append(values)

    async def leave(self) -> None:
        self.leaves += 1


class FakeBot:
    def __init__(self) -> None:
        self.voice = FakeVoice()


@pytest.mark.asyncio
async def test_voice_lease_serializes_owners_and_ignores_stale_release() -> None:
    bot = FakeBot()
    manager = OopzVoiceLeaseManager(bot)
    first = await manager.try_acquire(request(VoiceLeasePurpose.MUSIC))

    assert first is not None
    first_snapshot = await manager.current()
    assert first_snapshot is not None
    assert first_snapshot.generation == 1
    assert first_snapshot.state is VoiceLeaseState.ACTIVE
    assert await manager.try_acquire(request(VoiceLeasePurpose.CONVERSATION)) is None
    assert await first.release() is True
    assert await first.release() is False

    second = await manager.try_acquire(request(VoiceLeasePurpose.CONVERSATION, owner="person"))
    assert second is not None
    assert second.generation == 2
    assert await first.release() is False
    snapshot = await manager.current()
    assert snapshot is not None
    assert snapshot.request.purpose is VoiceLeasePurpose.CONVERSATION
    assert snapshot.generation == 2
    assert bot.voice.leaves == 1

    assert await second.release() is True
    assert bot.voice.leaves == 2
    await manager.aclose()


@pytest.mark.asyncio
async def test_voice_lease_publishes_owner_only_after_join_succeeds() -> None:
    bot = FakeBot()
    bot.voice.join_allowed.clear()
    manager = OopzVoiceLeaseManager(bot)
    first_task = asyncio.create_task(manager.try_acquire(request(VoiceLeasePurpose.MUSIC)))
    await bot.voice.join_entered.wait()
    second_task = asyncio.create_task(manager.try_acquire(request(VoiceLeasePurpose.CONVERSATION)))

    pending = await manager.current()
    assert pending is not None
    assert pending.state is VoiceLeaseState.ACQUIRING
    bot.voice.join_allowed.set()
    first, second = await asyncio.gather(first_task, second_task)

    assert first is not None
    assert second is None
    assert len(bot.voice.joins) == 1
    await first.release()
    await manager.aclose()


@pytest.mark.asyncio
async def test_voice_lease_join_failure_leaves_manager_available() -> None:
    bot = FakeBot()
    bot.voice.join_error = RuntimeError("join failed")
    manager = OopzVoiceLeaseManager(bot)

    with pytest.raises(RuntimeError, match="join failed"):
        await manager.try_acquire(request(VoiceLeasePurpose.MUSIC))
    assert await manager.current() is None

    bot.voice.join_error = None
    lease = await manager.try_acquire(request(VoiceLeasePurpose.CONVERSATION))
    assert lease is not None
    await manager.aclose()
    assert lease.released is True
    assert bot.voice.leaves == 1
    with pytest.raises(RuntimeError, match="closed"):
        await manager.try_acquire(request(VoiceLeasePurpose.MUSIC))


@pytest.mark.asyncio
async def test_voice_lease_close_waits_for_pending_join_and_leaves_it() -> None:
    bot = FakeBot()
    bot.voice.join_allowed.clear()
    manager = OopzVoiceLeaseManager(bot)
    acquire_task = asyncio.create_task(manager.try_acquire(request(VoiceLeasePurpose.CONVERSATION)))
    await bot.voice.join_entered.wait()
    close_task = asyncio.create_task(manager.aclose())
    await asyncio.sleep(0)

    assert close_task.done() is False
    bot.voice.join_allowed.set()
    with pytest.raises(RuntimeError, match="closed during join"):
        await acquire_task
    await close_task

    assert await manager.current() is None
    assert bot.voice.leaves == 1
