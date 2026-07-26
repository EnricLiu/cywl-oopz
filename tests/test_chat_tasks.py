from __future__ import annotations

import asyncio

import pytest

from cywl_oopz.application import BotApplication
from cywl_oopz.features.chat.models import ConversationKey
from cywl_oopz.features.chat.tasks import ChatTaskSupervisor


def key() -> ConversationKey:
    return ConversationKey("channel", "area-1", "channel-1", "person-1")


@pytest.mark.asyncio
async def test_supervisor_cancels_and_awaits_owned_task() -> None:
    supervisor = ChatTaskSupervisor()
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def operation() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    assert supervisor.start(key(), operation()) is True
    await asyncio.wait_for(started.wait(), timeout=1)
    assert await supervisor.cancel(key()) is True

    assert cancelled.is_set()
    assert supervisor.has_active(key()) is False


@pytest.mark.asyncio
async def test_supervisor_rejects_duplicate_conversation_task() -> None:
    supervisor = ChatTaskSupervisor()
    release = asyncio.Event()

    async def operation() -> None:
        await release.wait()

    assert supervisor.start(key(), operation()) is True
    assert supervisor.start(key(), operation()) is False
    release.set()
    await asyncio.sleep(0)
    await supervisor.close()


@pytest.mark.asyncio
async def test_application_starts_chat_work_without_blocking_sdk_handler() -> None:
    supervisor = ChatTaskSupervisor()
    application = BotApplication.__new__(BotApplication)
    application.chat_tasks = supervisor
    started = asyncio.Event()
    release = asyncio.Event()

    class Context:
        event = type(
            "Event",
            (),
            {
                "is_private": False,
                "message": type(
                    "Message",
                    (),
                    {"sender_id": "person-1", "area": "area-1", "channel": "channel-1"},
                )(),
            },
        )()

        async def reply(self, _: str) -> None:
            raise AssertionError("No error response is expected")

    async def operation() -> None:
        started.set()
        await release.wait()

    await application._start_chat_task(Context(), operation())
    await asyncio.wait_for(started.wait(), timeout=1)
    assert supervisor.has_active(key()) is True
    release.set()
    await supervisor.close()
